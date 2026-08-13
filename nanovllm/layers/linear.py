import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类
import torch.nn.functional as F  # PyTorch 函数式接口（linear）
import torch.distributed as dist  # PyTorch 分布式通信（all_reduce）


def divide(numerator, denominator):  # 整除辅助函数：确保能整除并返回整数结果
    assert numerator % denominator == 0  # 断言可整除，防止张量并行维度错误
    return numerator // denominator  # 返回整除结果


class LinearBase(nn.Module):  # 各类线性层的公共基类：统一权重创建与并行切分接口

    def __init__(  # 构造函数
        self,  # 实例自身
        input_size: int,  # 输入特征维度
        output_size: int,  # 输出特征维度（并行模式下为本地分片维度）
        bias: bool = False,  # 是否包含偏置
        tp_dim: int | None = None,  # 张量并行切分的维度（0=输出维度切分，1=输入维度切分，None=不切）
    ):
        super().__init__()  # 调用父类初始化
        self.tp_dim = tp_dim  # 保存切分维度
        self.tp_rank = dist.get_rank()  # 当前进程的并行排名
        self.tp_size = dist.get_world_size()  # 并行总进程数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))  # 创建权重参数（shape 已按分片大小分配）
        self.weight.weight_loader = self.weight_loader  # 把加载器挂到权重参数上，供加载时分片
        if bias:  # 若需要偏置
            self.bias = nn.Parameter(torch.empty(output_size))  # 创建偏置参数
            self.bias.weight_loader = self.weight_loader  # 偏置同样挂上加载器
        else:  # 若不需要偏置
            self.register_parameter("bias", None)  # 显式注册为 None，便于访问 self.bias 不报错

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向占位方法
        raise NotImplementedError  # 由子类实现


class ReplicatedLinear(LinearBase):  # 全复制线性层：每个进程保存完整权重，不做切分

    def __init__(  # 构造函数
        self,  # 实例自身
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        bias: bool = False,  # 是否带偏置
    ):
        super().__init__(input_size, output_size, bias)  # 按完整尺寸初始化（不切分）

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):  # 加载器：完整拷贝权重
        param.data.copy_(loaded_weight)  # 整体拷贝，无需切分

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：标准线性变换
        return F.linear(x, self.weight, self.bias)  # 直接调用线性层


class ColumnParallelLinear(LinearBase):  # 列并行线性层：输出维度按并行度切分到各进程

    def __init__(  # 构造函数
        self,  # 实例自身
        input_size: int,  # 输入维度
        output_size: int,  # 总输出维度
        bias: bool = False,  # 是否带偏置
    ):
        tp_size = dist.get_world_size()  # 获取并行度
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)  # 本地只保存 output_size/tp_size 的输出，沿维度0切分

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):  # 加载器：沿切分维度截取本进程分片
        param_data = param.data  # 取目标参数数据
        shard_size = param_data.size(self.tp_dim)  # 本进程分片大小
        start_idx = self.tp_rank * shard_size  # 分片起始索引
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # 从完整权重切出分片
        param_data.copy_(loaded_weight)  # 拷贝进参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：各进程输出自己的分片
        return F.linear(x, self.weight, self.bias)  # 标准线性变换


class MergedColumnParallelLinear(ColumnParallelLinear):  # 合并列并行层：把多个子模块的权重拼成一个权重（如 gate+up）

    def __init__(  # 构造函数
        self,  # 实例自身
        input_size: int,  # 输入维度
        output_sizes: list[int],  # 各子模块的完整输出维度列表
        bias: bool = False,  # 是否带偏置
    ):
        self.output_sizes = output_sizes  # 保存子模块输出维度列表
        super().__init__(input_size, sum(output_sizes), bias)  # 按所有子模块输出之和创建合并权重

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):  # 加载器：按子模块 id 切分
        # ──────────────────────────────────────────────────────────────
        # 合并权重布局：完整权重 = [子模块0 | 子模块1 | ...]（沿输出维拼接）。
        # 而本地权重（列并行切分后）每个子模块也只留 1/tp_size。
        # 因此要两重定位：① 在合并权重里找到“哪个子模块”的分片（shard_offset）；
        #                 ② 在该子模块里再按 tp_rank 切出本进程那份（chunk）。
        # 注意：shard_offset 用本地维度（sum(前几个子模块输出)/tp_size），
        #       而 loaded_weight 是完整权重，需要按 tp_size chunk 切分。
        # ──────────────────────────────────────────────────────────────
        param_data = param.data  # 取目标参数数据
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size  # 该子模块在【本地合并权重】中的起始偏移（前几个子模块输出之和，再按并行切分）
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size  # 该子模块的【本地】分片大小
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # 定位到本地合并权重中的该子模块分片（获得写入目标视图）
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # 把【完整权重】沿切分维切成 tp_size 份，取本进程那份
        param_data.copy_(loaded_weight)  # 拷贝进参数


class QKVParallelLinear(ColumnParallelLinear):  # QKV 合并并行层：把 Q、K、V 的投影权重合并到一个权重里

    def __init__(  # 构造函数
        self,  # 实例自身
        hidden_size: int,  # 输入维度（=隐藏维度）
        head_size: int,  # 每个注意力头的维度
        total_num_heads: int,  # 全部 Q 头数
        total_num_kv_heads: int | None = None,  # 全部 KV 头数（GQA，默认等于 Q 头数）
        bias: bool = False,  # 是否带偏置
    ):
        tp_size = dist.get_world_size()  # 获取并行度
        total_num_kv_heads = total_num_kv_heads or total_num_heads  # 未指定 KV 头数时退化为 MHA（KV头=Q头）
        self.head_size = head_size  # 保存头维度
        self.num_heads = divide(total_num_heads, tp_size)  # 本进程负责的 Q 头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 本进程负责的 KV 头数
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size  # 合并后总输出维度 = Q + K + V 的维度之和
        super().__init__(hidden_size, output_size, bias)  # 以合并维度创建列并行层

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):  # 加载器：按 q/k/v 分片标识切分
        # 合并权重布局 = [Q权重 | K权重 | V权重]（沿输出维）。
        # 本地权重同样在 Q/K/V 各段内按 tp_size 切分，所以：
        #   Q 段：偏移 0，长度 num_heads*head_size；
        #   K 段：偏移 Q段长，长度 num_kv_heads*head_size；
        #   V 段：偏移 Q段长+K段长，长度 num_kv_heads*head_size。
        # loaded_shard_id 决定我们正在把哪个子权重（q_proj/k_proj/v_proj）填进来。
        param_data = param.data  # 取目标参数数据
        assert loaded_shard_id in ["q", "k", "v"]  # 断言分片标识合法
        if loaded_shard_id == "q":  # 加载 Q 权重
            shard_size = self.num_heads * self.head_size  # Q 的本地分片大小（本进程的 Q 头数 × 头维）
            shard_offset = 0  # Q 位于合并权重最前面
        elif loaded_shard_id == "k":  # 加载 K 权重
            shard_size = self.num_kv_heads * self.head_size  # K 的本地分片大小
            shard_offset = self.num_heads * self.head_size  # K 紧随 Q 之后
        else:  # 加载 V 权重
            shard_size = self.num_kv_heads * self.head_size  # V 的本地分片大小
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size  # V 位于 Q+K 之后
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # 定位到合并权重中的目标分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # 从完整权重切出本进程那份
        param_data.copy_(loaded_weight)  # 拷贝进参数


class RowParallelLinear(LinearBase):  # 行并行线性层：输入维度按并行切分，输出需要 all_reduce 汇总

    def __init__(  # 构造函数
        self,  # 实例自身
        input_size: int,  # 总输入维度
        output_size: int,  # 总输出维度
        bias: bool = False,  # 是否带偏置
    ):
        tp_size = dist.get_world_size()  # 获取并行度
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)  # 本地只保存 input_size/tp_size 的输入，沿维度1切分

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):  # 加载器：沿输入维度切分
        param_data = param.data  # 取目标参数数据
        if param_data.ndim == 1:  # 若参数是一维（如偏置）
            param_data.copy_(loaded_weight)  # 整体拷贝即可
            return  # 提前返回
        shard_size = param_data.size(self.tp_dim)  # 本进程分片大小
        start_idx = self.tp_rank * shard_size  # 分片起始索引
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # 从完整权重切出分片
        param_data.copy_(loaded_weight)  # 拷贝进参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：各进程算局部输出，再汇总
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)  # 偏置只在 rank0 加一次，避免重复加
        if self.tp_size > 1:  # 若开启并行
            dist.all_reduce(y)  # 各进程的局部输出全归约，得到完整输出
        return y  # 返回完整输出

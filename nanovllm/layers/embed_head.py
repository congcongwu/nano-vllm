import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类
import torch.nn.functional as F  # PyTorch 函数式接口（embedding/linear）
import torch.distributed as dist  # PyTorch 分布式通信（all_reduce/gather）

from nanovllm.utils.context import get_context  # 导入全局上下文读取函数


class VocabParallelEmbedding(nn.Module):  # 词表并行嵌入层：词表按张量并行维度切分到多张卡

    def __init__(  # 构造函数
        self,  # 实例自身
        num_embeddings: int,  # 总词表大小
        embedding_dim: int,  # 嵌入向量维度
    ):
        super().__init__()  # 调用父类初始化
        self.tp_rank = dist.get_rank()  # 当前进程在张量并行组中的排名
        self.tp_size = dist.get_world_size()  # 张量并行的总进程数
        assert num_embeddings % self.tp_size == 0  # 断言词表大小能被并行度整除
        self.num_embeddings = num_embeddings  # 保存总词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size  # 每个进程负责的词表数量
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank  # 本进程词表分片的起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition  # 本进程词表分片的结束索引（开区间）
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # 创建本进程分片的嵌入权重参数
        self.weight.weight_loader = self.weight_loader  # 把 weight_loader 挂到权重参数上，供加载器按分片切分权重

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):  # 权重加载器：从完整词表权重中切出本进程的分片
        param_data = param.data  # 取目标参数的底层数据
        shard_size = param_data.size(0)  # 本进程分片大小（第一维行数）
        start_idx = self.tp_rank * shard_size  # 计算在完整权重中的起始行
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)  # 从完整权重按行切出本进程分片
        param_data.copy_(loaded_weight)  # 把切好的分片拷贝进参数

    def forward(self, x: torch.Tensor):  # 前向：输入为 token id 张量 (num_tokens,)
        # ──────────────────────────────────────────────────────────────
        # 词表并行嵌入的难点：词表被切成 tp_size 份分布在各张量并行卡上，
        # 每个 token id 只有在“负责它那一份词表”的卡上才有权重。
        # 做法：本地查表后，用 mask 把“不属于本卡的 token”的嵌入置 0，
        # 再 all_reduce 把所有卡的贡献加起来 → 每个 token 的正确嵌入就位。
        # ──────────────────────────────────────────────────────────────
        if self.tp_size > 1:  # 仅当开启张量并行时需要特殊处理（单卡直接查完整表即可）
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)  # 标记哪些 token 落在本进程的词表分片内
            x = mask * (x - self.vocab_start_idx)  # 分片内 token → 本地索引；分片外 token → 0（否则查表越界）
        y = F.embedding(x, self.weight)  # 用本进程的分片权重查表得到嵌入向量 (num_tokens, embedding_dim)
        if self.tp_size > 1:  # 若开启张量并行
            y = mask.unsqueeze(1) * y  # 分片外 token 的嵌入置 0（它们本来该由别的卡提供）
            dist.all_reduce(y)  # 各卡的贡献全归约求和：本卡非零的部分 + 其他卡非零的部分 = 完整嵌入
        return y  # 返回完整嵌入向量


class ParallelLMHead(VocabParallelEmbedding):  # 并行输出头：继承词表并行嵌入，把隐藏状态映射回 logits

    def __init__(  # 构造函数
        self,  # 实例自身
        num_embeddings: int,  # 总词表大小
        embedding_dim: int,  # 隐藏状态维度（=嵌入维度）
        bias: bool = False,  # 是否带偏置（Qwen3 的 lm_head 无偏置）
    ):
        assert not bias  # 断言不支持偏置（此实现约定 lm_head 无 bias）
        super().__init__(num_embeddings, embedding_dim)  # 复用父类初始化逻辑（按词表切分权重）

    def forward(self, x: torch.Tensor):  # 前向：输入隐藏状态
        # ──────────────────────────────────────────────────────────────
        # 输出头只对“最后一个 token”产生 logits（推理时只需要预测下一个 token）。
        # prefill 阶段：从每段里挑出段尾 token 再线性变换（省算力）；
        # 张量并行下各卡只有词表分片，最后用 gather 把分片 logits 拼到 rank0。
        # ──────────────────────────────────────────────────────────────
        context = get_context()  # 获取全局上下文判断当前阶段
        if context.is_prefill:  # 预填充阶段：多个序列一次性计算，需要提取每个序列的最后一个 token
            last_indices = context.cu_seqlens_q[1:] - 1  # 由累计长度数组算出每个序列末 token 的全局下标（段边界-1）
            x = x[last_indices].contiguous()  # 只取每个序列最后一个 token 的隐藏状态（decode 阶段每个序列本来就只有 1 行，不用取）
        logits = F.linear(x, self.weight)  # 用本进程的权重做线性变换，得到本进程词表分片的 logits
        if self.tp_size > 1:  # 若开启张量并行
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None  # 仅 rank0 准备收集缓冲（其余进程传 None 给 gather 占位）
            dist.gather(logits, all_logits, 0)  # 所有进程把各自的词表分片 logits 发给 rank0
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  # rank0 沿词表维度把分片拼成完整 logits；其余进程置 None
        return logits  # 返回完整 logits（仅 rank0 有效；之后只有 rank0 会采样）

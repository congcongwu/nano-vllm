import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类


class RMSNorm(nn.Module):  # RMS 归一化层：不使用均值，只用均方根缩放，是 Llama/Qwen 系列的标配

    def __init__(  # 构造函数
        self,  # 实例自身
        hidden_size: int,  # 输入特征维度
        eps: float = 1e-6,  # 数值稳定用的小常数
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.eps = eps  # 保存 eps 常数
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放权重（初始全 1）

    @torch.compile  # 把该算子编译为融合内核，减少中间张量开销
    def rms_forward(  # 纯归一化前向（无残差融合）
        self,  # 实例自身
        x: torch.Tensor,  # 输入张量
    ) -> torch.Tensor:
        orig_dtype = x.dtype  # 记录输入原始精度（如 bf16）
        x = x.float()  # 转成 float32 计算，提高数值稳定性
        var = x.pow(2).mean(dim=-1, keepdim=True)  # 计算最后一维的均方（方差近似），保留维度便于广播
        x.mul_(torch.rsqrt(var + self.eps))  # 除以均方根（rsqrt = 1/sqrt），完成归一化
        x = x.to(orig_dtype).mul_(self.weight)  # 转回原精度并乘以可学习缩放权重
        return x  # 返回归一化结果

    @torch.compile  # 同样编译为融合内核
    def add_rms_forward(  # 融合了残差加法（residual add）的归一化前向，省一次显存往返
        self,  # 实例自身
        x: torch.Tensor,  # 当前层输出（待与残差相加）
        residual: torch.Tensor,  # 上一步的残差流
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回 (归一化结果, 更新后的残差流)
        orig_dtype = x.dtype  # 记录原始精度
        x = x.float().add_(residual.float())  # 在 float32 中把 x 与残差相加，作为新的归一化输入
        residual = x.to(orig_dtype)  # 相加后的结果即为新的残差流（转回原精度保存）
        var = x.pow(2).mean(dim=-1, keepdim=True)  # 计算均方
        x.mul_(torch.rsqrt(var + self.eps))  # 均方根归一化
        x = x.to(orig_dtype).mul_(self.weight)  # 转回原精度并缩放
        return x, residual  # 返回归一化结果与更新后的残差流

    def forward(  # 统一前向入口
        self,  # 实例自身
        x: torch.Tensor,  # 输入张量
        residual: torch.Tensor | None = None,  # 可选残差张量；为 None 表示纯归一化
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # 返回归一化结果（及可选残差）
        if residual is None:  # 没有残差输入
            return self.rms_forward(x)  # 走纯归一化路径
        else:  # 有残差输入
            return self.add_rms_forward(x, residual)  # 走融合残差路径

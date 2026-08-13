import torch  # PyTorch 主模块，提供张量运算
from torch import nn  # PyTorch 神经网络基类
import torch.nn.functional as F  # PyTorch 函数式接口（如 silu 激活函数）


class SiluAndMul(nn.Module):  # SwiGLU 激活层：把 gate_up 合并输出拆成两份，做 silu(gate) * up

    @torch.compile  # 用 torch.compile 把该算子编译成融合内核，减少内核启动开销
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：输入形状为 (..., 2*intermediate_size)
        x, y = x.chunk(2, -1)  # 沿最后一维切成两份：前一半作为 gate，后一半作为 up
        return F.silu(x) * y  # 计算 silu(gate) * up，即 SwiGLU 输出

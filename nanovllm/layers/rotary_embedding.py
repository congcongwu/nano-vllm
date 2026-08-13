from functools import lru_cache  # 缓存装饰器，避免重复创建昂贵的旋转位置编码表
import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类


def apply_rotary_emb(  # 应用旋转位置编码到 Q/K 张量
    x: torch.Tensor,  # 输入张量（query 或 key）
    cos: torch.Tensor,  # 该位置的 cos 值
    sin: torch.Tensor,  # 该位置的 sin 值
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)  # 把最后一维切成两半 (x1, x2)，在 float32 下计算
    y1 = x1 * cos - x2 * sin  # 旋转公式的前半部分
    y2 = x2 * cos + x1 * sin  # 旋转公式的后半部分
    return torch.cat((y1, y2), dim=-1).to(x.dtype)  # 拼接回原形状并转回原精度


class RotaryEmbedding(nn.Module):  # 旋转位置编码层：预计算 cos/sin 查找表

    def __init__(  # 构造函数
        self,  # 实例自身
        head_size: int,  # 每个注意力头的维度
        rotary_dim: int,  # 参与旋转的维度数
        max_position_embeddings: int,  # 模型支持的最大位置数
        base: float,  # RoPE 的频率基（如 10000 或 1000000）
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.head_size = head_size  # 保存头维度
        assert rotary_dim == head_size  # 本实现约定全维度旋转（rotary_dim 必须等于 head_size）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))  # 计算各频率的倒数（倒置频率），偶数下标采样
        t = torch.arange(max_position_embeddings, dtype=torch.float)  # 位置索引张量 0..max_position-1
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 外积得到每个位置、每个频率的角度（位置×频率矩阵）
        cos = freqs.cos()  # 计算所有位置的 cos 值
        sin = freqs.sin()  # 计算所有位置的 sin 值
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # 把 cos/sin 拼在一起，并在中间加一个维度方便后续索引
        self.register_buffer("cos_sin_cache", cache, persistent=False)  # 注册为缓冲区（随模型搬设备但不进 state_dict）

    @torch.compile  # 编译该算子
    def forward(  # 前向：对给定的位置索引返回旋转后的 Q/K
        self,  # 实例自身
        positions: torch.Tensor,  # 每个 token 的位置索引
        query: torch.Tensor,  # Q 张量
        key: torch.Tensor,  # K 张量
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的 (Q, K)
        cos_sin = self.cos_sin_cache[positions]  # 按位置索引查表，得到对应的 cos/sin 对
        cos, sin = cos_sin.chunk(2, dim=-1)  # 拆分成 cos 与 sin
        query = apply_rotary_emb(query, cos, sin)  # 旋转 Q
        key = apply_rotary_emb(key, cos, sin)  # 旋转 K
        return query, key  # 返回旋转后的 Q、K


@lru_cache(1)  # 缓存最近一次创建结果，避免每层都重建编码表
def get_rope(  # 工厂函数：按参数获取旋转编码实例
    head_size: int,  # 头维度
    rotary_dim: int,  # 旋转维度
    max_position: int,  # 最大位置
    base: float,  # 频率基
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)  # 创建旋转编码实例
    return rotary_emb  # 返回实例

import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类
import torch.distributed as dist  # PyTorch 分布式通信（获取并行度）
from transformers import Qwen3Config  # Qwen3 的官方配置类型

from nanovllm.layers.activation import SiluAndMul  # 导入 SwiGLU 激活层
from nanovllm.layers.attention import Attention  # 导入封装的 FlashAttention 注意力模块
from nanovllm.layers.layernorm import RMSNorm  # 导入 RMS 归一化层
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear  # 导入各类并行线性层
from nanovllm.layers.rotary_embedding import get_rope  # 导入旋转位置编码工厂函数
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead  # 导入词表并行嵌入与输出头


class Qwen3Attention(nn.Module):  # Qwen3 的注意力层：QKV 投影 + QK 归一化 + RoPE + FlashAttention

    def __init__(  # 构造函数
        self,  # 实例自身
        hidden_size: int,  # 隐藏层维度
        num_heads: int,  # 全部 query 头数
        num_kv_heads: int,  # 全部 KV 头数（GQA）
        max_position: int = 4096 * 32,  # 最大位置长度（默认 131072）
        head_dim: int | None = None,  # 头维度；若为 None 则由 hidden_size/num_heads 推算
        rms_norm_eps: float = 1e-06,  # QK 归一化的 eps
        qkv_bias: bool = False,  # QKV 投影是否带偏置
        rope_theta: float = 10000,  # RoPE 频率基
        rope_scaling: dict | None = None,  # RoPE 缩放配置（可能覆盖 rope_theta）
    ) -> None:
        super().__init__()  # 调用父类初始化
        tp_size = dist.get_world_size()  # 获取张量并行度
        self.total_num_heads = num_heads  # 保存全部 query 头数
        assert self.total_num_heads % tp_size == 0  # 校验 query 头数能被并行度整除
        self.num_heads = self.total_num_heads // tp_size  # 本进程负责的 query 头数
        self.total_num_kv_heads = num_kv_heads  # 保存全部 KV 头数
        assert self.total_num_kv_heads % tp_size == 0  # 校验 KV 头数能被并行度整除
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 本进程负责的 KV 头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 计算头维度（若未指定则按隐藏维度/头数推算）
        self.q_size = self.num_heads * self.head_dim  # 本进程 Q 张量的尺寸
        self.kv_size = self.num_kv_heads * self.head_dim  # 本进程单个 K 或 V 张量的尺寸
        self.scaling = self.head_dim ** -0.5  # 注意力缩放系数 = 1/sqrt(head_dim)
        self.qkv_bias = qkv_bias  # 保存是否带 QKV 偏置

        self.qkv_proj = QKVParallelLinear(  # 创建合并的 QKV 投影层（一次性投影出 Q、K、V）
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 全部 query 头数
            self.total_num_kv_heads,  # 全部 KV 头数
            bias=qkv_bias,  # 偏置标志
        )
        self.o_proj = RowParallelLinear(  # 创建输出投影层（行并行，输入是全部头的拼接）
            self.total_num_heads * self.head_dim,  # 输入维度 = 全部头数×头维度
            hidden_size,  # 输出维度回到隐藏维度
            bias=False,  # Qwen3 的 o_proj 无偏置
        )
        if isinstance(rope_scaling, dict):  # 若存在 rope 缩放配置
            rope_theta = rope_scaling.get("rope_theta", rope_theta)  # 从其中优先读取 rope_theta
        self.rotary_emb = get_rope(  # 创建旋转位置编码实例（工厂函数带缓存）
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度 = 头维度（全旋转）
            max_position=max_position,  # 最大位置
            base=rope_theta,  # 频率基
        )
        self.attn = Attention(  # 创建注意力模块
            self.num_heads,  # 本进程 query 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放系数
            self.num_kv_heads,  # 本进程 KV 头数
        )
        if not self.qkv_bias:  # 若 QKV 投影无偏置（Qwen3 无偏置时需要对 Q/K 分别做 RMSNorm）
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # 创建 Q 归一化层
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # 创建 K 归一化层

    def forward(  # 前向
        self,  # 实例自身
        positions: torch.Tensor,  # 位置索引（用于 RoPE）
        hidden_states: torch.Tensor,  # 输入隐藏状态
    ) -> torch.Tensor:  # 返回注意力输出
        # ──────────────────────────────────────────────────────────────
        # 标准因果自注意力流水线：
        #   QKV 投影 → 拆 Q/K/V → 重塑出头结构 → （可选）Q/K 归一化 →
        #   RoPE 旋转 → FlashAttention（读写 KV cache）→ 输出投影。
        # 注意 Q 头数 = num_heads，K/V 头数 = num_kv_heads（GQA），
        # 拆分成三段时 Q 段长 = q_size，K 段长 = V 段长 = kv_size。
        # ──────────────────────────────────────────────────────────────
        qkv = self.qkv_proj(hidden_states)  # 一次合并投影，得到 (token, q_size + 2*kv_size) 的 QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 沿最后一维按 [Q,K,V] 尺寸拆开
        q = q.view(-1, self.num_heads, self.head_dim)  # Q 重塑成 (token, num_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)  # K 重塑成 (token, num_kv_heads, head_dim)（GQA 头更少）
        v = v.view(-1, self.num_kv_heads, self.head_dim)  # V 重塑成 (token, num_kv_heads, head_dim)
        if not self.qkv_bias:  # Qwen3 无 bias 的配置下需要在注意力前对 Q/K 做 RMS 归一化
            q = self.q_norm(q)  # 对 Q 做 RMS 归一化（逐头归一）
            k = self.k_norm(k)  # 对 K 做 RMS 归一化
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码（Q、K 都要带位置信息）
        o = self.attn(q, k, v)  # 执行 FlashAttention（内部会先把 K/V 写入缓存，再按阶段计算注意力）
        output = self.o_proj(o.flatten(1, -1))  # 把 (token, num_heads, head_dim) 展平为 (token, num_heads*head_dim) 再经输出投影
        return output  # 返回注意力层输出


class Qwen3MLP(nn.Module):  # Qwen3 的多层感知机（SwiGLU 结构）

    def __init__(  # 构造函数
        self,  # 实例自身
        hidden_size: int,  # 输入/输出维度
        intermediate_size: int,  # 中间隐藏维度
        hidden_act: str,  # 激活函数名称（必须为 silu）
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 合并的 gate/up 投影（一次算两个）
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # gate 与 up 各自的输出维度（各为 intermediate_size）
            bias=False,  # 无偏置
        )
        self.down_proj = RowParallelLinear(  # down 投影（行并行）
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 无偏置
        )
        assert hidden_act == "silu"  # 本实现仅支持 silu 激活（SwiGLU）
        self.act_fn = SiluAndMul()  # 创建 SwiGLU 激活层

    def forward(self, x):  # 前向
        gate_up = self.gate_up_proj(x)  # 计算合并的 gate/up 投影
        x = self.act_fn(gate_up)  # 应用 SwiGLU 激活（silu(gate)*up）
        x = self.down_proj(x)  # down 投影回到隐藏维度
        return x  # 返回 MLP 输出


class Qwen3DecoderLayer(nn.Module):  # 单个解码层：注意力 + MLP + 前置/后置归一化

    def __init__(  # 构造函数
        self,
        config: Qwen3Config,  # 模型配置
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.self_attn = Qwen3Attention(  # 创建自注意力层（参数全部取自配置）
            hidden_size=config.hidden_size,  # 隐藏维度
            num_heads=config.num_attention_heads,  # query 头数
            num_kv_heads=config.num_key_value_heads,  # KV 头数
            max_position=config.max_position_embeddings,  # 最大位置
            rms_norm_eps=config.rms_norm_eps,  # 归一化 eps
            qkv_bias=getattr(config, 'attention_bias', True),  # QKV 偏置标志
            head_dim=getattr(config, 'head_dim', None),  # 头维度（可能未定义）
            rope_theta=getattr(config, "rope_theta", 1000000),  # RoPE 频率基（默认 1e6）
            rope_scaling=getattr(config, "rope_scaling", None),  # RoPE 缩放配置
        )
        self.mlp = Qwen3MLP(  # 创建 MLP 层
            hidden_size=config.hidden_size,  # 隐藏维度
            intermediate_size=config.intermediate_size,  # 中间维度
            hidden_act=config.hidden_act,  # 激活函数
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前的归一化层
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # MLP 前的归一化层

    def forward(  # 前向
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 当前层输出
        residual: torch.Tensor | None,  # 残差流
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回 (本层输出, 残差流)
        if residual is None:  # 首层（上一层无残差）
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states  # 归一化并初始化残差流
        else:  # 非首层
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 融合残差的归一化
        hidden_states = self.self_attn(positions, hidden_states)  # 自注意力
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后融合残差的归一化
        hidden_states = self.mlp(hidden_states)  # MLP
        return hidden_states, residual  # 返回输出与更新后的残差流


class Qwen3Model(nn.Module):  # 主干网络：嵌入 + N 层解码器 + 最终归一化

    def __init__(  # 构造函数
        self,
        config: Qwen3Config,  # 模型配置
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)  # 词嵌入层（词表并行）
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])  # 堆叠所有解码层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def forward(  # 前向
        self,
        input_ids: torch.Tensor,  # 输入 token id
        positions: torch.Tensor,  # 位置索引
    ) -> torch.Tensor:  # 返回最终隐藏状态
        hidden_states = self.embed_tokens(input_ids)  # 词嵌入得到初始隐藏状态
        residual = None  # 残差流初始化为空
        for layer in self.layers:  # 逐层前向
            hidden_states, residual = layer(positions, hidden_states, residual)  # 每层更新隐藏状态与残差流
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化（丢弃残差流）
        return hidden_states  # 返回最终隐藏状态


class Qwen3ForCausalLM(nn.Module):  # 对外暴露的完整因果语言模型（主干 + 输出头）
    packed_modules_mapping = {  # 参数打包映射表：把 HF 中分散的权重映射到合并权重，并给出分片标识
        "q_proj": ("qkv_proj", "q"),  # 原 q_proj -> 合并层的 "q" 分片
        "k_proj": ("qkv_proj", "k"),  # 原 k_proj -> 合并层的 "k" 分片
        "v_proj": ("qkv_proj", "v"),  # 原 v_proj -> 合并层的 "v" 分片
        "gate_proj": ("gate_up_proj", 0),  # 原 gate_proj -> "gate_up_proj" 的第 0 个子模块
        "up_proj": ("gate_up_proj", 1),  # 原 up_proj -> "gate_up_proj" 的第 1 个子模块
    }

    def __init__(  # 构造函数
        self,
        config: Qwen3Config  # 模型配置
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.model = Qwen3Model(config)  # 创建主干模型
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)  # 创建输出头（词表并行）
        if config.tie_word_embeddings:  # 若配置要求权重共享（词表与输出头共享）
            self.lm_head.weight.data = self.model.embed_tokens.weight.data  # 让 lm_head 权重与嵌入权重共享同一块的引用

    def forward(  # 前向（只返回隐藏状态，不做 logits，便于配合采样）
        self,
        input_ids: torch.Tensor,  # 输入 token id
        positions: torch.Tensor,  # 位置索引
    ) -> torch.Tensor:  # 返回隐藏状态
        return self.model(input_ids, positions)  # 主干前向

    def compute_logits(  # 计算 logits（隐藏状态 -> 词表维度）
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
    ) -> torch.Tensor:  # 返回 logits
        return self.lm_head(hidden_states)  # 输出头线性变换

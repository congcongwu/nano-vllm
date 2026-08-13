import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类
import triton  # Triton 编译器，用于编写高性能自定义 CUDA 内核
import triton.language as tl  # Triton 语言内置函数库

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # FlashAttention 提供的变长预填充与 KV-cache 解码内核
from nanovllm.utils.context import get_context  # 导入全局上下文读取函数


@triton.jit  # 用 Triton 编译这段内核代码（JIT 即时编译为 CUDA kernel）
def store_kvcache_kernel(  # 自定义 Triton 内核：把新的 K/V 张量写入分块存储的 KV cache
    # 为什么不用 PyTorch 的切片赋值？
    # 因为 KV cache 是“非连续、按槽位散列”存储的：每个 token 对应缓存里的一个任意槽号
    # （由 block_table 决定），写成 Triton 内核可以一个线程块处理一个 token，避免逐行 Python 循环。
    key_ptr,  # 本次输入的 key 张量指针（形状 [N, num_heads, head_dim]）
    key_stride,  # key 张量在第一个维度（token 维度）上的步长（用于行寻址）
    value_ptr,  # 本次输入的 value 张量指针
    value_stride,  # value 张量在第一个维度（token 维度）上的步长
    k_cache_ptr,  # KV cache 中的 K 缓存区指针（扁平化为 [num_blocks*block_size, D]）
    v_cache_ptr,  # KV cache 中的 V 缓存区指针
    slot_mapping_ptr,  # 每个 token 对应的缓存槽位（slot）映射指针，-1 表示无需写入
    D: tl.constexpr,  # 每个 token 的缓存特征维度（num_heads*head_dim），编译期常量
):
    idx = tl.program_id(0)  # 获取当前线程块的编号（每个线程块处理一个 token，共 N 个块）
    slot = tl.load(slot_mapping_ptr + idx)  # 从槽位映射中读取该 token 对应的缓存槽号
    if slot == -1: return  # 若槽号为 -1（例如 CUDA graph 的填充行、无块表序列），跳过不写
    key_offsets = idx * key_stride + tl.arange(0, D)  # 计算该 token 的 key 元素在输入张量中的偏移（第 idx 行、D 个元素）
    value_offsets = idx * value_stride + tl.arange(0, D)  # 计算该 token 的 value 元素在输入张量中的偏移
    key = tl.load(key_ptr + key_offsets)  # 按偏移加载该 token 的 key 向量（长度为 D）
    value = tl.load(value_ptr + value_offsets)  # 按偏移加载该 token 的 value 向量
    cache_offsets = slot * D + tl.arange(0, D)  # 计算该 token 在缓存中的目标偏移（slot 决定“第几行”，每行 D 个元素）
    tl.store(k_cache_ptr + cache_offsets, key)  # 把 key 写入 K 缓存对应槽位
    tl.store(v_cache_ptr + cache_offsets, value)  # 把 value 写入 V 缓存对应槽位


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):  # 包装函数：校验张量布局后启动 Triton 内核写入 KV cache
    N, num_heads, head_dim = key.shape  # 解包 key 张量形状：token数 N、头数 num_heads、头维 head_dim
    D = num_heads * head_dim  # 计算每个 token 缓存的总维度（所有头拼成一行存储）
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 断言最后维度内存连续，保证内核可向量化加载
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 断言头维度步长正确（同一 token 的头数据连续）
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # 断言缓存每行（slot）之间按 D 对齐（layout 与内核假设一致）
    assert slot_mapping.numel() == N  # 断言槽位映射数量与 token 数一致
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)  # 以 N 个线程块启动内核（grid=(N,)）


class Attention(nn.Module):  # 封装 FlashAttention 的注意力模块（区分 prefill 与 decode 两条路径）

    def __init__(  # 构造函数
        self,  # 实例自身
        num_heads,  # 本进程负责的 query 头数量
        head_dim,  # 每个注意力头的维度
        scale,  # attention 缩放系数（softmax 前的 scale）
        num_kv_heads,  # 本进程负责的 KV 头数量（GQA 用）
    ):
        super().__init__()  # 调用父类初始化
        self.num_heads = num_heads  # 保存 query 头数
        self.head_dim = head_dim  # 保存头维度
        self.scale = scale  # 保存缩放系数
        self.num_kv_heads = num_kv_heads  # 保存 KV 头数
        self.k_cache = self.v_cache = torch.tensor([])  # 预置空 KV 缓存引用，稍后由引擎分配显存并赋值

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):  # 前向：q/k/v 均为 (num_tokens, num_heads, head_dim)
        # ──────────────────────────────────────────────────────────────
        # 职责分两步：
        #   ① 先把本次算出的 K/V 写进 KV cache（store_kvcache）—— 为未来引用存档；
        #   ② 再算注意力。prefill 与 decode 走不同的 flash 内核：
        #      - prefill：一次处理整段，用 flash_attn_varlen_func；
        #        + 若命中前缀缓存，KV 直接换成整块缓存（含历史），让新 token 能看到全部历史；
        #      - decode ：每个序列只有 1 个新 token，用 flash_attn_with_kvcache 从缓存里取历史。
        # ──────────────────────────────────────────────────────────────
        context = get_context()  # 获取本次前向的全局上下文（阶段、槽位、长度、块表等）
        k_cache, v_cache = self.k_cache, self.v_cache  # 取出本层预分配的 KV 缓存视图（由 allocate_kv_cache 注入）
        if k_cache.numel() and v_cache.numel():  # 若缓存已分配（非空张量）
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # 先把本次的 K/V 写入缓存对应槽位（供未来使用）
        if context.is_prefill:  # 预填充阶段：整段 prompt 一次性计算
            if context.block_tables is not None:  # 若存在 block 表（说明启用了前缀缓存，需要把已缓存前缀也纳入注意力）
                k, v = k_cache, v_cache  # KV 换成整个缓存视图（含历史前缀），让本段 query 能 attend 到全部历史
            o = flash_attn_varlen_func(q, k, v,  # 调用变长 FlashAttention（支持前缀缓存模式）
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,  # 传入 query 的累计长度与最大值（划分每段边界）
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,  # 传入 KV 的累计长度与最大值
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)  # 指定缩放、因果掩码与块表
        else:  # 解码阶段：每序列只输入一个新 token
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,  # 使用缓存进行单 token 注意力（q 形状补上头维度）
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,  # 传入每个序列的缓存长度与块表
                                        softmax_scale=self.scale, causal=True)  # 指定缩放与因果掩码
        return o  # 返回注意力输出（prefill: 变长结果；decode: (num_seqs, num_heads, head_dim)）

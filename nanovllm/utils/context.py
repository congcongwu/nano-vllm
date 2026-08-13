from dataclasses import dataclass  # dataclass 装饰器，用于定义轻量的上下文数据结构
import torch  # PyTorch 主模块，用于张量类型注解


# 全局上下文类：在每次前向计算时，把 FlashAttention 需要的元信息暂存到全局，
# 从而避免把大量参数在函数调用链上一层层地传递。
@dataclass(slots=True)
class Context:
    is_prefill: bool = False  # 当前是否为预填充（prefill）阶段；False 表示解码（decode）阶段
    cu_seqlens_q: torch.Tensor | None = None  # 累计序列长度（query），用于变长 attention 的分界
    cu_seqlens_k: torch.Tensor | None = None  # 累计序列长度（key/value），与 cu_seqlens_q 配合
    max_seqlen_q: int = 0  # 本批次中 query 序列的最大长度（flash 变长注意力需要）
    max_seqlen_k: int = 0  # 本批次中 key/value 序列的最大长度
    slot_mapping: torch.Tensor | None = None  # 每个 token 在 KV cache 中的实际存储槽位映射
    context_lens: torch.Tensor | None = None  # 每个序列当前的缓存长度（解码阶段用于计算）
    block_tables: torch.Tensor | None = None  # 每个序列的 block 表（块索引），用于前缀缓存寻址

_CONTEXT = Context()  # 模块级单例上下文对象，默认全部字段为空

def get_context():  # 返回全局上下文对象，供注意力层等读取
    return _CONTEXT  # 返回全局单例

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):  # 重置并写入一次前向所需的全部上下文
    global _CONTEXT  # 声明修改全局变量
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)  # 用给定参数新建一个上下文实例

def reset_context():  # 清空上下文，为下一次迭代做准备
    global _CONTEXT  # 声明修改全局变量
    _CONTEXT = Context()  # 重建一个全部默认的空上下文

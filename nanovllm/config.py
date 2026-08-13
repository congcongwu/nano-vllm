import os  # 操作系统接口模块，用于判断模型目录是否存在
from dataclasses import dataclass  # dataclass 装饰器，用于自动生成 __init__ / __repr__ 等
from transformers import AutoConfig  # HF 的自动配置加载器，用来读取模型的 config.json


# @dataclass 自动根据字段生成构造函数，slots=True 可节省内存并加速属性访问
@dataclass(slots=True)
class Config:
    model: str  # 模型目录路径（必须是本地已下载好的 HF 模型文件夹）
    max_num_batched_tokens: int = 16384  # 每次 step 最多处理的 token 总数（限制单次 batch 计算量）
    max_num_seqs: int = 512  # 同一时刻最多并行运行的序列数量
    max_model_len: int = 4096  # 模型支持的最大序列长度（context + 生成的总长度上限）
    gpu_memory_utilization: float = 0.9  # KV cache 最多可占用 GPU 显存的比例（0.9 表示 90%）
    tensor_parallel_size: int = 1  # 张量并行的 GPU 数量，1 表示单卡不并行
    enforce_eager: bool = False  # 是否强制使用 eager 模式（关闭 CUDA graph 加速，便于调试）
    hf_config: AutoConfig | None = None  # 由 transformers 解析出的模型配置对象，加载后填充
    eos: int = -1  # 结束符（EOS）的 token id，初始化时由 tokenizer 填充
    kvcache_block_size: int = 256  # KV cache 每个 block 能容纳的 token 数量（按块管理显存）
    num_kvcache_blocks: int = -1  # KV cache 可用的 block 总数，在运行时根据显存动态计算

    # 初始化完成后的自动校验与补全逻辑（dataclass 会在 __init__ 末尾自动调用）
    def __post_init__(self):
        assert os.path.isdir(self.model)  # 校验：模型路径必须是一个真实存在的目录
        assert self.kvcache_block_size % 256 == 0  # 校验：block 大小必须是 256 的整数倍
        assert 1 <= self.tensor_parallel_size <= 8  # 校验：张量并行规模必须在 1~8 之间
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 从本地模型目录读取官方配置
        # 模型长度上限不能超过模型本身支持的最大位置编码长度（取二者较小值）
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

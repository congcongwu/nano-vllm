from copy import copy  # 导入浅拷贝函数（拷贝 token 列表）
from enum import Enum, auto  # 导入枚举，用于定义序列状态
from itertools import count  # 无上限计数器，生成全局唯一的序列 id

from nanovllm.sampling_params import SamplingParams  # 导入采样参数类


class SequenceStatus(Enum):  # 序列状态枚举，描述一条序列的生命周期
    WAITING = auto()  # 等待状态（已入队，尚未分配显存/调度）
    RUNNING = auto()  # 运行状态（正在被调度执行）
    FINISHED = auto()  # 结束状态（已完成生成或触发终止条件）


class Sequence:  # 序列类：一条请求的所有 token、状态与调度元信息
    block_size = 256  # 类变量：KV cache 每个 block 容纳的 token 数（引擎初始化时会被覆盖）
    counter = count()  # 类变量：全局递增计数器，用于分配唯一 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):  # 构造函数：输入 prompt 的 token 序列与采样参数
        self.seq_id = next(Sequence.counter)  # 生成全局唯一的序列 id
        self.status = SequenceStatus.WAITING  # 初始状态为 WAITING
        self.token_ids = copy(token_ids)  # 浅拷贝 prompt token 列表（避免共享外部可变对象）
        self.last_token = token_ids[-1]  # 记录最后一个 token id（解码时作为下一步输入）
        self.num_tokens = len(self.token_ids)  # 当前已拥有的 token 总数（prompt + 已生成）
        self.num_prompt_tokens = len(token_ids)  # 保存 prompt 的原始长度
        self.num_cached_tokens = 0  # 已写入 KV cache（含前缀缓存命中）的 token 数
        self.num_scheduled_tokens = 0  # 本次 step 计划处理的 token 数
        self.is_prefill = True  # 标记是否为预填充阶段（首次处理 prompt）
        self.block_table = []  # 记录该序列占用的 KV block id 列表（块表）
        self.temperature = sampling_params.temperature  # 读取采样温度
        self.max_tokens = sampling_params.max_tokens  # 读取最大生成 token 数
        self.ignore_eos = sampling_params.ignore_eos  # 读取是否忽略 EOS

    def __len__(self):  # 让 len(seq) 返回当前 token 总数
        return self.num_tokens  # 返回 token 总数

    def __getitem__(self, key):  # 让 seq[i] 可直接索引到 token
        return self.token_ids[key]  # 返回对应位置的 token id

    @property
    def is_finished(self):  # 判断序列是否已结束
        return self.status == SequenceStatus.FINISHED  # 状态是否为 FINISHED

    @property
    def num_completion_tokens(self):  # 已生成的（完成的）token 数
        return self.num_tokens - self.num_prompt_tokens  # 总数减去 prompt 数

    @property
    def prompt_token_ids(self):  # 取出原始 prompt 的 token 序列
        return self.token_ids[:self.num_prompt_tokens]  # 切片前 num_prompt_tokens 个

    @property
    def completion_token_ids(self):  # 取出生成的 token 序列
        return self.token_ids[self.num_prompt_tokens:]  # 切片 prompt 之后的部分

    @property
    def num_blocks(self):  # 该序列需要多少个 KV block（向上取整）
        return (self.num_tokens + self.block_size - 1) // self.block_size  # 向上取整计算 block 数

    @property
    def last_block_num_tokens(self):  # 最后一块里实际装的 token 数
        return self.num_tokens - (self.num_blocks - 1) * self.block_size  # 总数减去前面完整块占的 token

    def block(self, i):  # 取出第 i 个块内的 token id 列表
        assert 0 <= i < self.num_blocks  # 校验块索引有效
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]  # 按块区间切片

    def append_token(self, token_id: int):  # 生成阶段向序列尾部追加一个新 token
        self.token_ids.append(token_id)  # 追加到 token 列表
        self.last_token = token_id  # 更新最后一个 token
        self.num_tokens += 1  # 更新 token 总数

    def __getstate__(self):  # pickling 需要的序列化方法（用于进程间传递序列状态）
        last_state = self.last_token if not self.is_prefill else self.token_ids  # 解码阶段只传最后 token，prefill 阶段传完整列表（省显存带宽）
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)  # 返回可序列化的元组

    def __setstate__(self, state):  # pickling 反序列化方法（把进程收到的状态恢复到对象）
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state  # 解包所有状态字段
        if isinstance(last_state, list):  # 若传的是完整 token 列表（prefill）
            self.token_ids = last_state  # 直接恢复 token 列表
            self.last_token = self.token_ids[-1]  # 取最后一个 token
        else:  # 否则传的是单个 last_token（decode）
            self.token_ids = []  # 只保留空列表（不再维护完整序列）
            self.last_token = last_state  # 恢复最后一个 token

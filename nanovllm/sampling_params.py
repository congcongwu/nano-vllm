from dataclasses import dataclass  # dataclass 装饰器，自动生成构造器等样板代码


# 采样参数类：描述一次生成请求的采样配置（温度、生成长度、是否忽略 EOS）
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0  # 采样温度，值越大输出越发散随机，越小越保守集中
    max_tokens: int = 64  # 本次生成最多产生的 token 数量（不含 prompt）
    ignore_eos: bool = False  # 是否忽略 EOS 结束符；True 时会一直生成直到 max_tokens

    # 构造完成后自动校验参数合法性
    def __post_init__(self):
        # 校验：温度必须大于极小值，本项目故意禁止纯贪心采样（temperature==0）
        assert self.temperature > 1e-10, "greedy sampling is not permitted"

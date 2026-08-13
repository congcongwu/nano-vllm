from nanovllm.engine.llm_engine import LLMEngine  # 导入底层引擎 LLMEngine（真正的推理逻辑都在里面）


# LLM 是面向用户的对外类，直接继承 LLMEngine，不添加任何额外逻辑，
# 这样既保持了对外接口的简洁（from nanovllm import LLM），也便于未来扩展。
class LLM(LLMEngine):
    pass

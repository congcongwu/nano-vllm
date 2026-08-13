import os  # 操作系统接口，用来拼接/展开用户目录路径
from nanovllm import LLM, SamplingParams  # 导入对外 API：LLM 推理引擎与采样参数
from transformers import AutoTokenizer  # 导入 HF tokenizer，用于文本编码与 chat 模板拼接


def main():  # 主函数，演示 nano-vllm 的基本用法
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # 展开 ~ 为完整路径，得到模型所在目录
    tokenizer = AutoTokenizer.from_pretrained(path)  # 从模型目录加载 tokenizer（用于模板与解码）
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # 创建推理引擎（强制 eager、单卡）

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)  # 设定采样参数：温度0.6，最多256个token
    # prompts = [  # 原始用户问题列表
    #     "introduce yourself",  # 问题1：自我介绍
    #     "list all prime numbers within 100",  # 问题2：列出100以内的所有质数
    # ]
    prompts = [  # 原始用户问题列表
        "请你进行自我介绍",  # 问题1：自我介绍
        "请你列出100以内的所有质数",  # 问题2：列出100以内的所有质数
    ]
    prompts = [  # 用 chat 模板把原始问题包装成模型的对话输入格式
        tokenizer.apply_chat_template(  # 应用 Qwen 的对话模板
            [{"role": "user", "content": prompt}],  # 把问题构造成一条 user 消息
            tokenize=False,  # 只做模板包装，不进行 token 化（返回字符串）
            add_generation_prompt=True,  # 在末尾附加助手的生成提示符（如 <|im_start|>assistant）
        )
        for prompt in prompts  # 对每个原始问题逐个处理
    ]
    outputs = llm.generate(prompts, sampling_params)  # 调用引擎批量生成，返回结果列表

    for prompt, output in zip(prompts, outputs):  # 把 prompt 与对应输出一一配对遍历
        print("\n")  # 打印一个空行，方便分隔多组结果
        print(f"Prompt: {prompt!r}")  # 打印输入 prompt 原文
        print(f"Completion: {output['text']!r}")  # 打印模型生成的文本


if __name__ == "__main__":  # 当脚本被直接运行时（而非被 import）才执行
    main()  # 调用主函数

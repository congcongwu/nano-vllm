import os  # 操作系统接口，用来展开用户目录路径
import time  # 计时模块，用于测量吞吐量
from random import randint, seed  # 随机数：randint 生成随机整数，seed 固定随机种子保证可复现
from nanovllm import LLM, SamplingParams  # 导入 nano-vllm 的对外 API
# from vllm import LLM, SamplingParams  # 可选的 vLLM 导入，取消注释可对比基准性能


def main():  # 基准测试主函数：模拟一批随机长度的请求并统计吞吐量
    seed(0)  # 固定随机种子，保证每次运行生成同样的请求，结果可复现
    num_seqs = 256  # 模拟的请求（序列）总数
    max_input_len = 1024  # prompt（输入）token 长度的上限
    max_ouput_len = 1024  # 输出 token 长度的上限

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # 展开 ~ 得到模型目录绝对路径
    llm = LLM(path, enforce_eager=False, max_model_len=4096)  # 创建引擎：启用 CUDA graph，序列长度上限4096

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]  # 生成 num_seqs 条随机 prompt（每条 100~1024 个随机 token id）
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]  # 每条请求配一个随机输出长度（100~1024）的采样参数，忽略EOS
    # uncomment the following line for vllm  # 提示：若改用 vLLM，请取消下行注释
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]  # vLLM 需要把 prompt 包装成字典格式

    llm.generate(["Benchmark: "], SamplingParams())  # 预热（warmup）：跑一次小请求，让 CUDA graph/内核预热
    t = time.time()  # 记录基准测试开始时间
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)  # 正式批量生成全部请求（关闭进度条）
    t = (time.time() - t)  # 计算总耗时（秒）
    total_tokens = sum(sp.max_tokens for sp in sampling_params)  # 所有请求理论输出 token 数之和
    throughput = total_tokens / t  # 吞吐量 = 输出 token 数 / 耗时
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")  # 打印统计结果


if __name__ == "__main__":  # 脚本被直接运行时才执行
    main()  # 运行主函数

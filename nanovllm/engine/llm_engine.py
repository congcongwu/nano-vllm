import atexit  # 注册进程退出时的清理回调
from dataclasses import fields  # 读取 dataclass 字段名，用于筛选构造函数参数
from time import perf_counter  # 高精度计时器，统计吞吐量
from tqdm.auto import tqdm  # 进度条库
from transformers import AutoTokenizer  # 自动加载 tokenizer
import torch.multiprocessing as mp  # PyTorch 多进程模块（用于张量并行启动子进程）

from nanovllm.config import Config  # 导入配置类
from nanovllm.sampling_params import SamplingParams  # 导入采样参数
from nanovllm.engine.sequence import Sequence  # 导入序列类
from nanovllm.engine.scheduler import Scheduler  # 导入调度器
from nanovllm.engine.model_runner import ModelRunner  # 导入模型执行器


class LLMEngine:  # 推理引擎：对外协调 tokenizer、调度器、模型执行器，并管理张量并行子进程

    def __init__(self, model, **kwargs):  # 构造函数：model 为模型路径，kwargs 为可选配置参数
        # ──────────────────────────────────────────────────────────────
        # 张量并行下的进程拓扑：
        #   · 主进程自己就是 rank0 的 ModelRunner（负责执行 + 采样）。
        #   · 每额外一块 GPU 就 spawn 一个子进程，跑一个 rank>=1 的 ModelRunner。
        #   · 子进程加载完模型后进入“等待命令”的 loop()；主进程通过共享内存
        #     广播“要执行的方法名+参数”，所有进程用 NCCL 同步做同一份计算。
        # ──────────────────────────────────────────────────────────────
        config_fields = {field.name for field in fields(Config)}  # 取出 Config 的所有字段名
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}  # 只保留属于 Config 的关键字参数（过滤掉未知项）
        config = Config(model, **config_kwargs)  # 创建配置对象（内部会读 hf_config、校验参数）
        Sequence.block_size = config.kvcache_block_size  # 把全局块大小同步到序列类（静态变量，Sequence 里各处计算都要用到）
        self.ps = []  # 保存张量并行子进程句柄列表（exit 时 join）
        self.events = []  # 保存与子进程同步用的事件对象列表（写共享内存时逐个 set 通知）
        ctx = mp.get_context("spawn")  # 用 spawn 方式创建多进程上下文（避免 fork 与 CUDA 上下文冲突）
        for i in range(1, config.tensor_parallel_size):  # 为每个额外 GPU（rank=1..size-1）启动一个子进程
            event = ctx.Event()  # 创建该子进程对应的同步事件（主进程 set，子进程 wait）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 子进程入口：直接实例化 ModelRunner（构造函数里会完成加载并进入 loop）
            process.start()  # 启动子进程
            self.ps.append(process)  # 记录子进程句柄
            self.events.append(event)  # 记录子进程事件（顺序与 rank 一一对应）
        self.model_runner = ModelRunner(config, 0, self.events)  # 主进程自身作为 rank0 的 ModelRunner（事件列表用于广播命令）
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)  # 加载 fast tokenizer（编码/解码文本用）
        config.eos = self.tokenizer.eos_token_id  # 把 EOS token id 写回配置（调度器判定结束条件）
        self.scheduler = Scheduler(config)  # 创建调度器（内部按显存计算并分配 KV block）
        atexit.register(self.exit)  # 注册进程退出时自动调用 exit 清理（确保子进程被回收）

    def exit(self):  # 清理函数：通知子进程退出并回收
        self.model_runner.call("exit")  # 让 rank0 广播退出命令给所有子进程
        del self.model_runner  # 删除主进程模型执行器（释放 GPU 资源）
        for p in self.ps:  # 遍历所有子进程
            p.join()  # 等待子进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):  # 新增一条请求：支持字符串或已编码 token 列表
        if isinstance(prompt, str):  # 若输入是字符串
            prompt = self.tokenizer.encode(prompt)  # 用 tokenizer 编码为 token id 列表
        seq = Sequence(prompt, sampling_params)  # 创建序列对象
        self.scheduler.add(seq)  # 加入调度器等待队列

    def step(self):  # 执行一步推理（调度 + 模型前向 + 后处理），返回产出与吞吐量统计
        seqs, is_prefill = self.scheduler.schedule()  # 调度本步的序列与阶段
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)  # prefill 记录处理 token 数；decode 用负数记录序列数（用于区分统计）
        token_ids = self.model_runner.call("run", seqs, is_prefill)  # 让模型执行器对这批序列前向并采样出 token
        self.scheduler.postprocess(seqs, token_ids, is_prefill)  # 后处理：更新缓存、追加 token、判定结束
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]  # 收集本轮结束序列的输出
        return outputs, num_tokens  # 返回 (结束序列输出列表, token 数统计)

    def is_finished(self):  # 是否全部请求已完成
        return self.scheduler.is_finished()  # 委托给调度器判断

    def generate(  # 对外主接口：批量生成
        self,
        prompts: list[str] | list[list[int]],  # prompt 列表（字符串或 token id 列表）
        sampling_params: SamplingParams | list[SamplingParams],  # 采样参数（单个则所有请求共用，或每请求一个）
        use_tqdm: bool = True,  # 是否显示进度条
    ) -> list[str]:  # 返回结果列表
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)  # 创建进度条（总数=请求数）
        if not isinstance(sampling_params, list):  # 若传入单个采样参数
            sampling_params = [sampling_params] * len(prompts)  # 复制成等长的参数列表
        for prompt, sp in zip(prompts, sampling_params):  # 遍历每条请求及其参数
            self.add_request(prompt, sp)  # 加入请求队列
        outputs = {}  # 用字典保存 (seq_id -> 生成的 token 列表)
        prefill_throughput = decode_throughput = 0.  # 初始化预填充/解码吞吐量统计为 0
        while not self.is_finished():  # 循环直到所有请求完成
            t = perf_counter()  # 记录本步开始时间
            output, num_tokens = self.step()  # 执行一步
            if num_tokens > 0:  # prefill 步骤（正数）
                prefill_throughput = num_tokens / (perf_counter() - t)  # 更新预填充吞吐量
            else:  # decode 步骤（负数）
                decode_throughput = -num_tokens / (perf_counter() - t)  # 更新解码吞吐量
            pbar.set_postfix({  # 更新进度条后置信息
                "Prefill": f"{int(prefill_throughput)}tok/s",  # 显示预填充吞吐
                "Decode": f"{int(decode_throughput)}tok/s",  # 显示解码吞吐
            })
            for seq_id, token_ids in output:  # 遍历本轮结束的序列
                outputs[seq_id] = token_ids  # 保存其生成结果
                pbar.update(1)  # 进度条 +1
        pbar.close()  # 关闭进度条
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]  # 按 seq_id 排序，保证输出顺序与请求顺序一致
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]  # 用 tokenizer 解码文本，并打包为字典
        return outputs  # 返回结果列表

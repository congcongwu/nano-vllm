import pickle  # Python 对象序列化，用于进程间传参
import torch  # PyTorch 主模块
import torch.distributed as dist  # PyTorch 分布式通信（NCCL）
from multiprocessing.synchronize import Event  # 多进程同步事件类型
from multiprocessing.shared_memory import SharedMemory  # 进程间共享内存（传递命令与参数）

from nanovllm.config import Config  # 导入配置类
from nanovllm.engine.sequence import Sequence  # 导入序列类
from nanovllm.models.qwen3 import Qwen3ForCausalLM  # 导入 Qwen3 模型
from nanovllm.layers.sampler import Sampler  # 导入采样器
from nanovllm.utils.context import set_context, get_context, reset_context  # 导入全局上下文读写
from nanovllm.utils.loader import load_model  # 导入权重加载函数


class ModelRunner:  # 模型执行器：负责模型加载、KV cache 分配、数据预处理、CUDA graph 捕获与执行
    # 注意：同一个类既在主进程（rank0）实例化执行推理，也在子进程（rank>0）实例化后进入事件循环等待命令。

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):  # 构造函数
        self.config = config  # 保存配置
        hf_config = config.hf_config  # 取出 HF 模型配置
        self.block_size = config.kvcache_block_size  # KV block 容量
        self.enforce_eager = config.enforce_eager  # 是否强制 eager（关闭 CUDA graph）
        self.world_size = config.tensor_parallel_size  # 并行总进程数
        self.rank = rank  # 本进程在并行组中的排名
        self.event = event  # 事件：rank0 持有列表（广播用），子进程持有单个事件

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)  # 用 NCCL 初始化分布式进程组（地址硬编码）
        torch.cuda.set_device(rank)  # 每个进程绑定一块 GPU（按 rank）
        default_dtype = torch.get_default_dtype()  # 保存当前默认 dtype（结束时恢复）
        model_dtype = getattr(hf_config, "torch_dtype", default_dtype)  # 取模型权重精度（Qwen3Config 用 torch_dtype 而非 dtype）
        torch.set_default_dtype(model_dtype)  # 把默认 dtype 设为模型权重精度（如 bfloat16）
        torch.set_default_device("cuda")  # 默认设备设为 GPU，方便创建参数
        self.model = Qwen3ForCausalLM(hf_config)  # 创建 Qwen3 模型（空权重）
        load_model(self.model, config.model)  # 从磁盘加载权重（含张量并行切分）
        self.sampler = Sampler()  # 创建采样器
        self.warmup_model()  # 模型预热（触发 CUDA 内核编译，避免正式推理卡顿）
        self.allocate_kv_cache()  # 依据剩余显存计算并分配 KV cache
        if not self.enforce_eager:  # 若未强制 eager
            self.capture_cudagraph()  # 捕获 CUDA graph（大幅降低解码内核启动开销）
        torch.set_default_device("cpu")  # 恢复默认设备为 CPU（避免后续误建 GPU 张量）
        torch.set_default_dtype(default_dtype)  # 恢复默认 dtype

        if self.world_size > 1:  # 若开启张量并行
            if rank == 0:  # 主进程（rank0）
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 创建共享内存块（1MB，用于广播命令）
                dist.barrier()  # 同步等待所有子进程就绪
            else:  # 子进程（rank>0）
                dist.barrier()  # 同步等待 rank0 建好共享内存
                self.shm = SharedMemory(name="nanovllm")  # 打开 rank0 创建的共享内存
                self.loop()  # 进入命令接收循环（子进程在此阻塞等待命令）

    def exit(self):  # 退出清理
        if self.world_size > 1:  # 若开了并行
            self.shm.close()  # 关闭本进程的共享内存句柄
            dist.barrier()  # 同步所有进程
            if self.rank == 0:  # 仅 rank0
                self.shm.unlink()  # 删除共享内存文件
        if not self.enforce_eager:  # 若启用了 CUDA graph
            del self.graphs, self.graph_pool  # 释放 graph 相关资源
        torch.cuda.synchronize()  # 等待 GPU 全部完成
        dist.destroy_process_group()  # 销毁分布式进程组

    def loop(self):  # 子进程主循环：等待并执行主进程广播的命令
        while True:  # 无限循环
            method_name, args = self.read_shm()  # 从共享内存读取命令名与参数
            self.call(method_name, *args)  # 在本地执行该方法
            if method_name == "exit":  # 若收到退出命令
                break  # 跳出循环结束子进程

    def read_shm(self):  # 子进程：从共享内存读取命令
        assert self.world_size > 1 and self.rank > 0  # 仅并行模式的子进程调用
        self.event.wait()  # 阻塞等待主进程发出事件
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取前 4 字节获得数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化出方法名与参数
        self.event.clear()  # 清除事件（允许下一轮）
        return method_name, args  # 返回命令

    def write_shm(self, method_name, *args):  # 主进程：把命令写入共享内存并通知所有子进程
        assert self.world_size > 1 and self.rank == 0  # 仅并行模式的主进程调用
        data = pickle.dumps([method_name, *args])  # 序列化命令与参数
        n = len(data)  # 数据字节长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 前 4 字节写入长度
        self.shm.buf[4:n+4] = data  # 数据体写入共享内存
        for event in self.event:  # 遍历所有子进程事件
            event.set()  # 逐个触发通知

    def call(self, method_name, *args):  # 统一方法调用入口：主进程需先广播，子进程直接本地执行
        if self.world_size > 1 and self.rank == 0:  # 若是并行模式的主进程
            self.write_shm(method_name, *args)  # 先把命令广播给所有子进程
        method = getattr(self, method_name, None)  # 按名字取方法
        return method(*args)  # 本地执行并返回结果

    def warmup_model(self):  # 模型预热：用虚拟数据跑一次前向，触发 CUDA 内核编译与显存峰值统计
        torch.cuda.empty_cache()  # 清空显存缓存
        torch.cuda.reset_peak_memory_stats()  # 重置显存峰值统计（为后续 KV cache 计算做准备）
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len  # 取出批预算与模型最大长度
        seq_len = min(max_num_batched_tokens, max_model_len)  # 单序列长度取两者较小值
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)  # 虚拟序列数 = 预算/长度，且不超上限
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]  # 创建全 0 的虚拟序列
        for seq in seqs:  # 遍历每个虚拟序列
            seq.num_scheduled_tokens = seq_len  # 设置本次调度 token 数
        self.run(seqs, True)  # 以 prefill 方式跑一次前向（触发内核编译）
        torch.cuda.empty_cache()  # 跑完再清一次显存缓存

    def allocate_kv_cache(self):  # 依据剩余显存计算 KV cache 块数并分配显存
        # ──────────────────────────────────────────────────────────────
        # 目标：算出“在安全预留出模型权重和激活之后，还剩多少显存可以装 KV cache”，
        # 然后用“块数 = 可用的显存 ÷ 单个块大小”来决定缓存容量。
        # 这个计算必须在 warmup_model() 之后做（此时峰值显存才被统计出来）。
        # ──────────────────────────────────────────────────────────────
        config = self.config  # 取配置
        hf_config = config.hf_config  # 取 HF 配置
        free, total = torch.cuda.mem_get_info()  # 获取空闲显存 free 与总显存 total
        used = total - free  # 已用显存（模型权重等）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 历史峰值分配（warmup 时达到，代表激活的上限）
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前实际分配（warmup 结束后已回落）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 每进程的 KV 头数（张量并行切分后）
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)  # 头维度（配置没给则用 hidden_size/头数 推算）
        model_dtype = getattr(hf_config, "torch_dtype", torch.get_default_dtype())  # 取模型权重精度（Qwen3Config 用 torch_dtype 而非 dtype）
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * model_dtype.itemsize  # 单个 block 的字节数 = K、V 两份 × 层数 × 每块 token 数 × KV 头 × 头维 × 每个元素的字节
        # 可用显存预算 = total*利用率 - 已用权重 - 峰值激活 + 当前回落部分。
        # 其中“-peak+current”是在说：warmup 时产生的临时激活峰值，只要推理时不超过它，
        # 我们就能把这部分（peak-current）也拿去给 KV cache 用。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes  # 计算块数：预算显存里能放下几个 block
        assert config.num_kvcache_blocks > 0  # 断言至少能分配 1 个块（否则模型都无法启动）
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)  # 一次性分配整块 KV cache 显存，维度依次为：K/V 两份、层、块、每块 token 数、KV 头、头维
        layer_id = 0  # 层计数器（按模块遍历顺序给每层编号）
        for module in self.model.modules():  # 遍历模型所有子模块
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):  # 命中注意力层（只有 Attention 有这两个属性）
                module.k_cache = self.kv_cache[0, layer_id]  # 把第 layer_id 层的 K 缓存视图赋给该注意力层
                module.v_cache = self.kv_cache[1, layer_id]  # 把第 layer_id 层的 V 缓存视图赋给该注意力层
                layer_id += 1  # 层计数 +1

    def prepare_block_tables(self, seqs: list[Sequence]):  # 把序列的块表 padding 成等宽张量
        max_len = max(len(seq.block_table) for seq in seqs)  # 取本批最长的块表长度（GPU 张量必须等法宽）
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]  # 每条短块表补 -1 到等长（-1 代表无效位，flash 会自动跳过）
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # 转 GPU 张量（pin 内存 + 异步拷贝）
        return block_tables  # 返回形状 (num_seqs, max_len) 的块表张量

    def prepare_prefill(self, seqs: list[Sequence]):  # prefill 数据预处理
        # ──────────────────────────────────────────────────────────────
        # 函数目标：把一批“待 prefill 的序列”打包成模型（FlashAttention）需要的各种输入。
        # 它要回答的三个问题：
        #   ① 这次要算哪些 token？            -> 生成 input_ids
        #   ② 每个 token 的绝对位置(喂RoPE)？-> 生成 positions
        #   ③ 算出的 K/V 该写进缓存哪个槽？  -> 生成 slot_mapping + cu_seqlens + 块表
        # 注意：因为存在“前缀缓存”和“分块 prefill”，每条序列可能只算 prompt 的一部分
        #       （start 由 num_cached_tokens 决定，跳过已在缓存里的 token）。
        # ──────────────────────────────────────────────────────────────

        # ----- ① 初始化 8 个“收集器”（Python list，循环里逐个 append，最后统一转成张量） -----
        input_ids = []  # 收集批次内所有要“新算”的 token（展平后喂给模型）
        positions = []  # 收集每个 token 的绝对位置（即使跳过了前缀，位置也要对齐真实长度）
        cu_seqlens_q = [0]  # query 段的累计序列长度数组（[0] 起，flash 变长注意力据此划分每段的边界）
        cu_seqlens_k = [0]  # KV 段的累计序列长度数组（含前缀缓存，通常比 cu_seqlens_q 更长）
        max_seqlen_q = 0  # 本批 query 段的最大长度（flash 内核需要事先知道上界）
        max_seqlen_k = 0  # 本批 KV 段的最大长度
        slot_mapping = []  # 槽位映射：每个新 token 应写入 KV cache 的槽号（-1 表示不需要写）
        block_tables = None  # 前缀缓存时的块表（若本批无前缀则保持 None，flash 不会启用前缀寻址）

        # ----- ② 逐条序列收集数据 -----
        for seq in seqs:  # 遍历本批中的每条序列
            # --- 2a. 界定“本次要算哪一段” ---
            start = seq.num_cached_tokens  # 本次起始 token 位置（已缓存的前缀要跳过，不重复算）
            seqlen_q = seq.num_scheduled_tokens  # 本次前向要算的 query 长度（调度器决定，可能只是 prompt 的一部分）
            end = start + seqlen_q  # 本次结束位置（开区间）
            seqlen_k = end  # KV 的有效长度 = 到本次结束为止（含已缓存前缀），供注意力看全历史

            # --- 2b. 收集 token 与 绝对位置 ---
            input_ids.extend(seq[start:end])  # 把 [start, end) 这段 token 展平追加进输入列表
            positions.extend(range(start, end))  # 位置用“全局序列绝对位置”，保证跳过前缀后 RoPE 仍正确

            # --- 2c. 更新累计长度与最大长度 ---
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # 累加 query：本段在展平输入里的结束下标
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # 累加 KV：本段（含前缀）在缓存侧的结束下标
            max_seqlen_q = max(seqlen_q, max_seqlen_q)  # 刷新 query 最大长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k)  # 刷新 KV 最大长度

            # --- 2d. 由“块表”算出每个新 token 应写入的缓存槽号 ---
            # 缓存是扁平显存：物理块 b 占据槽位区间 [b*block_size, (b+1)*block_size)。
            # 逻辑上，序列的第 j 个 token 落在块 j//block_size，块内偏移 j%block_size。
            # 下面把本轮涉及到的块逐个映射出槽号区间（且要跳过块前部已缓存的前缀 token）。
            if not seq.block_table:  # 若序列还没有块表（例如 warmup 的假数据），无缓存可写
                continue  # 跳过槽位生成，继续下一条序列
            start_block = start // self.block_size  # 本次起始 token 所在的块下标
            end_block = (end + self.block_size - 1) // self.block_size  # 本次结束 token 所在块下标（向上取整，为开区间）
            for i in range(start_block, end_block):  # 遍历本次横跨的每个物理块
                slot_start = seq.block_table[i] * self.block_size  # 先假设从该块的开头槽写起
                if i == start_block:  # 若是起始块（块前部可能含有已缓存的前缀）
                    slot_start += start % self.block_size  # 块内偏移要跳过前缀，只从本次首个新 token 处写起
                if i != end_block - 1:  #  非最后一块，整块使用
                    slot_end = seq.block_table[i] * self.block_size + self.block_size  # 一直写到块尾
                else:  # 若是末尾块（可能没装满）
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size  # 只写到本次实际结束的位置
                slot_mapping.extend(range(slot_start, slot_end))  # 收集该块 [slot_start, slot_end) 的槽号

        # ----- ③ 判断是否开启“前缀缓存”寻址 -----
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # KV 总存量 > query 新增量，说明有序列命中前缀缓存
            block_tables = self.prepare_block_tables(seqs)  # 生成块表张量，交给 flash 从缓存里读前缀参与注意力

        # ----- ④ 把全部收集器转成 GPU 张量（pin 内存 + 异步拷贝到 CUDA） -----
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)  # 输入 token -> GPU
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)  # 位置 -> GPU
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # query 累计长度 -> GPU
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # KV 累计长度 -> GPU
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # 槽位映射 -> GPU

        # ----- ⑤ 把所有元信息塞进“全局上下文”，供注意力层读取 -----
        # 第 6 个参数是 context_lens，仅 decode 阶段用，这里传 None。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)  # 写入全局上下文（供注意力层读取）
        return input_ids, positions  # 显式返回 token 与位置；其余信息都挂在全局 Context 里

    def prepare_decode(self, seqs: list[Sequence]):  # decode 数据预处理：每条序列只取最后 token
        # ──────────────────────────────────────────────────────────────
        # decode 阶段：每条序列只生成 1 个新 token。
        # 输入不再是“一段 token”，而是每条序列的“最后一个 token”；
        # 位置是它对应的绝对位置；而历史全部 token 都在 KV cache 里，
        # 注意力通过 context_lens + block_table 去缓存里找，而不是重新前向。
        # ──────────────────────────────────────────────────────────────
        input_ids = []  # 收集每个序列的“最后 token”（作为本次输入）
        positions = []  # 收集每个 token 的绝对位置
        slot_mapping = []  # 槽位映射：新 token 的 K/V 写进缓存哪个槽
        context_lens = []  # 各序列当前已缓存的历史长度（供 flash 知道“已看到第几个 token”）
        for seq in seqs:  # 遍历每条序列
            input_ids.append(seq.last_token)  # 用上一步生成的最后 token 作为本次输入
            positions.append(len(seq) - 1)  # 新 token 的位置 = 当前长度 - 1（历史已到 len(seq)）
            context_lens.append(len(seq))  # 缓存长度 = 当前长度（还没算新 token，所以是 len(seq)）
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)  # 新 token 槽号 = 末块在缓存的起始槽 + 末块内已占偏移（last_block_num_tokens-1）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)  # 输入张量 (num_seqs,)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)  # 位置张量 (num_seqs,)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # 槽位映射张量 (num_seqs,)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # 缓存长度张量 (num_seqs,)
        block_tables = self.prepare_block_tables(seqs)  # 生成块表（flash 解码时按块表读历史 KV）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)  # 写入全局上下文（decode 阶段；cu_seqlens/max_seqlen 不需要）
        return input_ids, positions  # 返回输入与位置张量

    def prepare_sample(self, seqs: list[Sequence]):  # 采样参数预处理：收集每条序列的温度
        temperatures = [seq.temperature for seq in seqs]  # 收集温度列表
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)  # 转 GPU 张量
        return temperatures  # 返回温度张量

    @torch.inference_mode()  # 推理模式：关闭梯度与 autograd，加速并省显存
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):  # 模型前向：优先走 CUDA graph，否则 eager
        # ──────────────────────────────────────────────────────────────
        # decode 是逐 token 的“短批”操作，每次 GPU 内核启动的固定开销占比巨大，
        # CUDA graph 能把这整串内核的调度提前固化、运行时一次性重放，从而砍掉启动开销。
        # 但 graph 的形状是固定的，所以按 batch 大小分了多个档位（见 capture_cudagraph）。
        # 前向前先把真实数据“填”进对应档位预分配的缓冲，再 replay。
        # ──────────────────────────────────────────────────────────────
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:  # 以下情况不能/不必用 graph：prefill、强制 eager、batch 超过捕获上限
            return self.model.compute_logits(self.model(input_ids, positions))  # 直接 eager 前向并计算 logits
        else:  # decode 且可用 CUDA graph
            bs = input_ids.size(0)  # 当前真实批次大小
            context = get_context()  # 获取上下文（含槽位/长度/块表，这些是每次推理的“动态输入”）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 从已捕获档位里选一个 ≥ bs 的最小档位
            graph_vars = self.graph_vars  # 取 graph 的固定输入输出缓冲（捕获时就定好的内存）
            graph_vars["input_ids"][:bs] = input_ids  # 把真实 token 拷进缓冲前 bs 行
            graph_vars["positions"][:bs] = positions  # 把真实位置拷进缓冲
            graph_vars["slot_mapping"].fill_(-1)  # 槽位先全部置 -1（防止未被填写的行写入越界内存）
            graph_vars["slot_mapping"][:bs] = context.slot_mapping  # 只填真实 bs 行的槽位
            graph_vars["context_lens"].zero_()  # 缓存长度先清零
            graph_vars["context_lens"][:bs] = context.context_lens  # 只填真实 bs 行的长度
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables  # 写入块表（按本次实际块表宽度拷贝，其余保持 0）
            graph.replay()  # 重放 CUDA graph：执行捕获时固化的全部内核，输入从上面缓冲读取
            return self.model.compute_logits(graph_vars["outputs"][:bs])  # 取输出缓冲前 bs 行（真实部分）计算 logits

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:  # 整体运行：预处理 -> 前向 -> 采样 -> 返回 token
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)  # 按阶段准备输入数据
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None  # 仅 rank0 需要温度（只有它采样）
        logits = self.run_model(input_ids, positions, is_prefill)  # 模型前向得到 logits
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None  # 仅 rank0 采样得到 token 列表
        reset_context()  # 清空全局上下文
        return token_ids  # 返回采样出的 token

    @torch.inference_mode()  # 推理模式
    def capture_cudagraph(self):  # 捕获不同 batch 大小的 CUDA graph（解码加速核心）
        # ──────────────────────────────────────────────────────────────
        # 在正式推理前，用“占位缓冲”把不同 batch 大小的解码前向各捕获一遍。
        # 捕获时模型会在这些缓冲上进行真实计算，但结果会被丢弃——我们只要
        # “内核执行序列”被固化下来。运行时（run_model）把真实数据填进同名缓冲，
        # 再 replay，即可用极小的启动开销完成一次解码前向。
        # 注意事项：
        #   ① graph 的形状固定，因此按 batch 大小分档（graph_bs）；
        #   ② 缓冲要在捕获期间就分配好，运行时不分配新显存；
        #   ③ 捕获顺序从大到小，且用共享 pool，保证各档显存可复用、地址稳定。
        # ──────────────────────────────────────────────────────────────
        config = self.config  # 取配置
        hf_config = config.hf_config  # 取 HF 配置
        max_bs = min(self.config.max_num_seqs, 512)  # graph 支持的最大 batch（受 CUDA graph 与显存限制，上限 512）
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 单序列最多需要的块数（决定块表缓冲宽度）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)  # 输入 token 缓冲（固定内存，运行时只改值）
        positions = torch.zeros(max_bs, dtype=torch.int64)  # 位置缓冲
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)  # 槽位缓冲（-1 为无效行）
        context_lens = torch.zeros(max_bs, dtype=torch.int32)  # 缓存长度缓冲
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # 块表缓冲（等宽）
        outputs = torch.zeros(max_bs, hf_config.hidden_size)  # 输出隐藏状态缓冲
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # 预设的 batch 档位（1,2,4,8,16,32,...），运行时可向上就近取档
        self.graphs = {}  # 存储每个档位对应的 graph 对象
        self.graph_pool = None  # 共享内存池（让各档 graph 的临时内存可复用，节省显存）

        for bs in reversed(self.graph_bs):  # 从大到小逐个档位捕获（保证 pool 内存分配顺序稳定，避免地址重叠）
            graph = torch.cuda.CUDAGraph()  # 创建 CUDA graph 对象
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])  # 设置上下文（让注意力层读到缓冲的前 bs 行）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 先 eager 跑一次，触发该形状的内存分配（graph 捕获期间禁止分配）
            with torch.cuda.graph(graph, self.graph_pool):  # 进入 CUDA graph 捕获上下文
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 把“完整解码前向”固化进 graph
            if self.graph_pool is None:  # 首次捕获后
                self.graph_pool = graph.pool()  # 取回内存池，供后续档位复用同一块显存
            self.graphs[bs] = graph  # 登记该档位的 graph
            torch.cuda.synchronize()  # 同步确保捕获完成
            reset_context()  # 清空上下文，避免污染下一档捕获

        self.graph_vars = dict(  # 保存所有 graph 的输入输出缓冲，供运行时填充
            input_ids=input_ids,  # 输入 token 缓冲
            positions=positions,  # 位置缓冲
            slot_mapping=slot_mapping,  # 槽位映射缓冲
            context_lens=context_lens,  # 缓存长度缓冲
            block_tables=block_tables,  # 块表缓冲
            outputs=outputs,  # 输出缓冲
        )

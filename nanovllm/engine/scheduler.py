from collections import deque  # 双端队列，充当等待队列与运行队列

from nanovllm.config import Config  # 导入配置类
from nanovllm.engine.sequence import Sequence, SequenceStatus  # 导入序列类与状态枚举
from nanovllm.engine.block_manager import BlockManager  # 导入块管理器


class Scheduler:  # 调度器：决定每个 step 运行哪些序列、预填充还是解码、抢占哪些序列

    def __init__(self, config: Config):  # 构造函数
        self.max_num_seqs = config.max_num_seqs  # 单步最多处理的序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单步最多处理的 token 总数
        self.eos = config.eos  # 结束符 token id
        self.block_size = config.kvcache_block_size  # KV block 容量
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)  # 创建块管理器（按显存算出的总块数）
        self.waiting: deque[Sequence] = deque()  # 等待队列：存放“prompt 排队中/分块prefill算到一半/被抢占打回的序列，调度器对其做prefill
        self.running: deque[Sequence] = deque()  # 运行队列：存放“prompt 已处理完”的序列，调度器对其做 decode

    def is_finished(self):  # 判断是否所有请求都已处理完
        return not self.waiting and not self.running  # 等待与运行队列都为空即为结束

    def add(self, seq: Sequence):  # 把新请求加入等待队列
        self.waiting.append(seq)  # 入队

    def schedule(self) -> tuple[list[Sequence], bool]:  # 调度一步：返回 (被调度的序列列表, 是否 prefill 阶段)
        # ──────────────────────────────────────────────────────────────
        # 一次 schedule() 只能产生一个 batch：要么是 prefill batch（一次算很多 token），
        # 要么是 decode batch（每条序列只算 1 个 token）。原因：GPU 一次前向的形状
        # 是固定的，prefill（长序列×大段）和 decode（多序列×1 token）的输入形状不同。
        # 所以流程是：先全力凑一个 prefill batch；凑不到任何 prefill 时，再退而求其次
        # 凑一个 decode batch。
        # ──────────────────────────────────────────────────────────────
        scheduled_seqs = []  # 收集本次被调度的序列
        num_batched_tokens = 0  # 累计本次已用的 batch token 预算

        # ════════════ 阶段一：PREFILL（优先） ════════════
        # prefill优先，每次前向只能执行prefill或decioding任务
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:  # 等待队列非空且未达到序列数上限
            seq = self.waiting[0]  # “偷看”队首序列（暂不移除：下面的 break 分支会放弃它，但队列不能丢数据）
            remaining = self.max_num_batched_tokens - num_batched_tokens  # 剩余 token 预算（本轮还能放多少）
            if remaining == 0:  # 预算已用尽
                break  # 停止调度
            if not seq.block_table:  # 序列尚未分配块（全新 prompt，首次 prefill）
                num_cached_blocks = self.block_manager.can_allocate(seq)  # ① 查前缀缓存：返回可复用块数，-1 表示显存不够
                if num_cached_blocks == -1:  # 显存不足，无法分配
                    break  # 停止调度本轮（本轮 prefill 到此为止，下一轮再试）
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # 本次需新算的 token 数 = 总长 - 前缀缓存命中长度
            else:  # 已有块表（说明是上一轮“分块 prefill”没算完、继续处理同一条序列）
                num_tokens = seq.num_tokens - seq.num_cached_tokens  # 还需处理的 token 数 = 总长 - 已缓存长度
            if remaining < num_tokens and scheduled_seqs:  # 预算放不下这条序列，且本批已调度了别的序列
                break  # 则不强行塞入，留给下一轮（避免尾大不掉拖垮整批）
            if not seq.block_table:  # 若该序列还未分配块，走到这说明不管有没有分配过block的seq，预算都是够的。
                self.block_manager.allocate(seq, num_cached_blocks)  # ② 正式分配块（复用前缀命中块 + 新建其余块）
            seq.num_scheduled_tokens = min(num_tokens, remaining)  # ③ 本次实际调度 token 数 = min(待算量, 剩余预算) → 预算不够时“分块 prefill”
            num_batched_tokens += seq.num_scheduled_tokens  # 累加预算占用
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:  # ④ 已有 + 本轮 == 全部 prompt → 本条序列 prefill 完成
                seq.status = SequenceStatus.RUNNING  # 状态设为运行中（可以开始逐 token 解码了）
                self.waiting.popleft()  # 真正移出等待队列（只有此刻才敢 pop，因为已确定处理完）
                self.running.append(seq)  # 移入运行队列（后续 decode 阶段会把它捞出来）
            # 若 ④ 不成立：序列本次只 prefill 了一部分，仍留在 waiting 队首，下一轮循环继续处理剩余部分
            scheduled_seqs.append(seq)  # 加入本次调度结果（无论完整与否，本轮都要参与前向）

        if scheduled_seqs:  # prefill 阶段调度到序列
            return scheduled_seqs, True  # 返回 (序列列表, is_prefill=True)

        # ════════════ 阶段二：DECODE（当凑不到任何 prefill 时执行） ════════════
        # decode：当没有新 prefill 可做时，为跑着的序列生成一步
        while self.running and len(scheduled_seqs) < self.max_num_seqs:  # 运行队列非空且未满
            seq = self.running.popleft()  # 取运行队列队首序列（临时移除，若本轮成功会放回）
            while not self.block_manager.can_append(seq):  # 若“追加 1 个 token”会撞到块边界且无空闲块 → 需要先腾空间
                if self.running:  # 运行队列里还有别的序列
                    self.preempt(self.running.pop())  # 抢占运行队列尾部最后一条序列，回收它的块（牺牲它，保住当前这条）
                else:  # 没有其他可抢占序列
                    self.preempt(seq)  # 只能抢占当前序列自己（打回 waiting，释放其块）
                    break  # 退出内层 while（注意：此时走的是 break，下面 else 不执行，本轮不调度它）
            else:  # 内层 while 正常结束（即 can_append 变 True：空间足够/已腾出）才执行
                seq.num_scheduled_tokens = 1  # 解码阶段每序列每轮只算 1 个 token
                seq.is_prefill = False  # 标记为解码阶段（影响数据准备路径）
                self.block_manager.may_append(seq)  # 若 1 个新 token 恰好跨入新块，则先分配新块
                scheduled_seqs.append(seq)  # 加入本次调度
        assert scheduled_seqs  # 断言本次解码至少调度了一条（否则会死循环）
        self.running.extendleft(reversed(scheduled_seqs))  # 把被调度序列放回运行队列前端（reversed 抵消 extendleft 的反转，保持原顺序）
        return scheduled_seqs, False  # 返回 (序列列表, is_prefill=False)

    def preempt(self, seq: Sequence):  # 抢占：把一条运行中序列回退为等待，并释放其块
        seq.status = SequenceStatus.WAITING  # 状态改为等待
        # 为什么 preempt 要把 is_prefill 置 True？
        # is_prefill 有两处作用：
        #   ① prepare_prefill/prepare_decode 选择数据准备路径（决定输入形状）；
        #   ② Sequence.__getstate__ 决定进程间只传 last_token 还是完整 token 列表。
        # 被抢占的序列下次被调度时，它的 num_cached_tokens 已被 deallocate 清零，
        # 之前算过的 KV 也全部释放，相当于要“从头重新 prefill 整个 prompt”，
        # 所以必须标记回 prefill（True），否则会用 decode 路径处理一条没有缓存的新 prompt → 出错。
        seq.is_prefill = True  # 标记为 prefill（下次重跑被抢占的部分）
        self.block_manager.deallocate(seq)  # 释放其 KV 块（供其他序列使用）
        self.waiting.appendleft(seq)  # 插回到等待队列前端，保证尽快被重新调度

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):  # 调度后处理：更新块哈希、追加 token、判定结束
        for seq, token_id in zip(seqs, token_ids):  # 每条已调度序列对应它本次产出的 token
            self.block_manager.hash_blocks(seq)  # 更新本次完成块的哈希与内容（为前缀缓存做准备）
            seq.num_cached_tokens += seq.num_scheduled_tokens  # 累加已缓存 token 数
            seq.num_scheduled_tokens = 0  # 清零本次调度数
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:  # 若是 prefill 且尚未处理完整个 prompt（分块 prefill）
                continue  # 跳过以下逻辑，等待下一次 step 继续
            seq.append_token(token_id)  # 把生成的 token 追加进序列
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:  # 若遇到 EOS 结束，或生成的 token 数已达上限
                seq.status = SequenceStatus.FINISHED  # 标记为完成
                self.block_manager.deallocate(seq)  # 释放其 KV 块
                self.running.remove(seq)  # 从运行队列移除该序列

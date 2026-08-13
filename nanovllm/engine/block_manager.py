from collections import deque  # 双端队列，作为空闲 block 的池子（快速取头部）
import xxhash  # 快速哈希库，用于计算前缀块的内容哈希（实现前缀缓存）
import numpy as np  # 数值库，用于把 token 序列转为 bytes 参与哈希

from nanovllm.engine.sequence import Sequence  # 导入序列类（类型提示用）


class Block:  # KV cache 中的一个物理块

    def __init__(self, block_id):  # 构造函数
        self.block_id = block_id  # 块在缓存中的全局编号
        self.ref_count = 0  # 引用计数：被多少条序列共享（前缀缓存时多条序列可共享同一块）
        self.hash = -1  # 该块内容的哈希值（-1 表示未计算），用于前缀匹配
        self.token_ids = []  # 该块当前存储的 token 列表（用于校验前缀是否真正一致）

    def update(self, hash: int, token_ids: list[int]):  # 写入该块内容及其哈希
        self.hash = hash  # 记录内容哈希
        self.token_ids = token_ids  # 记录实际 token（用于后续内容比对）

    def reset(self):  # 把块恢复为被分配后的初始状态（引用计数为 1，内容清空）
        self.ref_count = 1  # 分配后默认被 1 条序列引用
        self.hash = -1  # 哈希重置为未计算
        self.token_ids = []  # 内容清空


class BlockManager:  # 块管理器：负责 KV block 的分配、释放、前缀缓存复用

    def __init__(self, num_blocks: int, block_size: int):  # 构造函数
        self.block_size = block_size  # 每个 block 的 token 容量
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 预创建全部物理块
        self.hash_to_block_id: dict[int, int] = dict()  # 哈希 -> 块号 的映射，用于前缀缓存命中
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块编号池
        self.used_block_ids: set[int] = set()  # 已分配块编号集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):  # 计算一个块的内容哈希（可级联前缀哈希）
        h = xxhash.xxh64()  # 创建 xxhash64 哈希器（快速、低冲突）
        if prefix != -1:  # 若提供了前缀哈希
            h.update(prefix.to_bytes(8, "little"))  # 把“上一块的哈希”也喂进去 → 链式哈希
        # 链式哈希的意义：第 i 块的哈希 = f(第0块…第i块的全部内容)。
        # 这样两条序列只要第 k 块的哈希相同，就说明它们的前 k 块内容完全相同
        # （哈希不同则前缀必然不同），于是可以安全地逐块向后匹配“最长公共前缀”。
        h.update(np.array(token_ids).tobytes())  # 把块内 token 转成字节并入哈希
        return h.intdigest()  # 返回 64 位整型哈希值

    def _allocate_block(self) -> int:  # 从空闲池分配一个新块
        block_id = self.free_block_ids.popleft()  # 从空闲池头部取出一个块号
        block = self.blocks[block_id]  # 取到对应的块对象
        assert block.ref_count == 0  # 校验该块确属空闲（引用计数为 0）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:  # 若该块曾被登记过哈希且映射仍指向它自己
            del self.hash_to_block_id[block.hash]  # 从哈希映射中移除旧记录（马上要写新内容，旧索引会误导前缀匹配）
        block.reset()  # 重置块状态（ref_count=1，清空内容）
        self.used_block_ids.add(block_id)  # 标记为已使用
        return block_id  # 返回新分配的块号

    def _deallocate_block(self, block_id: int):  # 释放一个块（引用计数已为 0）
        assert self.blocks[block_id].ref_count == 0  # 校验确实无人引用
        self.used_block_ids.remove(block_id)  # 从已使用集合移除
        self.free_block_ids.append(block_id)  # 归还到空闲池

    def can_allocate(self, seq: Sequence) -> int:  # 判断能否为该序列分配显存，并返回可复用的前缀块数
        # ──────────────────────────────────────────────────────────────
        # “只读探测”函数：检查显存是否够 + 能命中几块前缀缓存，但不真正分配。
        # 返回 -1 表示显存不够；否则返回可复用的前缀块数 num_cached_blocks。
        # 前缀匹配做法：逐块算链式哈希查表，能连续命中几块就复用几块；
        # 一旦某块不命中就停下（前缀必须从开头连续，中间不能断）。
        # ──────────────────────────────────────────────────────────────
        h = -1  # 级联哈希初始化为 -1（表示无前缀）
        num_cached_blocks = 0  # 可复用的前缀块数量
        num_new_blocks = seq.num_blocks  # 需要新增的块数（先假设全部新建）
        for i in range(seq.num_blocks - 1):  # 遍历除最后一个块外的所有块（最后一块是“尾巴”，不满一整块，不参与前缀匹配）
            token_ids = seq.block(i)  # 取出第 i 块的 token
            h = self.compute_hash(token_ids, h)  # 计算到第 i 块为止的链式哈希
            block_id = self.hash_to_block_id.get(h, -1)  # 查表：这段前缀内容是否有人缓存过
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # 未命中；或命中但内容不同（哈希碰撞）
                break  # 前缀到此为止，停止向后匹配
            num_cached_blocks += 1  # 该块可复用，缓存块数 +1
            if block_id in self.used_block_ids:  # 若命中的块正被别的序列占用
                num_new_blocks -= 1  # 该块可以“共享”（只加引用计数），所以真正要新建的块数 -1
        if len(self.free_block_ids) < num_new_blocks:  # 若空闲块不够覆盖所需新建的块
            return -1  # 返回 -1 表示无法分配（调度器会据此 stop 本轮 prefill）
        return num_cached_blocks  # 返回可复用的前缀块数（调度器用它算出“本次真正要算的 token 数”）

    def allocate(self, seq: Sequence, num_cached_blocks: int):  # 为序列真正分配块（复用前缀 + 新建剩余块）
        # 注意：分配时只能“沿用 can_allocate 探测出的前缀块数”，因为探测时显存是够的，
        # 而这里若临时再算哈希可能因状态变化出现不一致，所以直接用传进来的 num_cached_blocks。
        assert not seq.block_table  # 校验该序列尚未分配块
        h = -1  # 级联哈希初始化
        for i in range(num_cached_blocks):  # ① 处理可复用的前缀块（把命中块挂进本序列块表）
            token_ids = seq.block(i)  # 取第 i 块 token
            h = self.compute_hash(token_ids, h)  # 重算链式哈希（与探测时保持一致，才能定位到同一缓存块）
            block_id = self.hash_to_block_id[h]  # 由哈希取到缓存的块号
            block = self.blocks[block_id]  # 取块对象
            if block_id in self.used_block_ids:  # 若块正被其他序列使用 → 属于“共享”
                block.ref_count += 1  # 引用计数 +1（多一条序列引用它）
            else:  # 若块在空闲池（被某序列释放过，但哈希表仍登记着内容）
                block.ref_count = 1  # 设置引用计数为 1（从空闲变占用）
                self.free_block_ids.remove(block_id)  # 从空闲池移除
                self.used_block_ids.add(block_id)  # 标记为已使用
            seq.block_table.append(block_id)  # 把该块号加入序列的块表
        for i in range(num_cached_blocks, seq.num_blocks):  # ② 处理剩余需要新建的块（前缀没有命中到的部分）
            seq.block_table.append(self._allocate_block())  # 逐个分配新块加入块表
        seq.num_cached_tokens = num_cached_blocks * self.block_size  # ③ 记录“已命中前缀”的 token 数（调度器据此跳过重复计算）

    def deallocate(self, seq: Sequence):  # 释放序列占用的所有块（preempt 或结束时调用）
        for block_id in reversed(seq.block_table):  # 倒序遍历块表
            block = self.blocks[block_id]  # 取块对象
            block.ref_count -= 1  # 引用计数 -1
            if block.ref_count == 0:  # 若不再被任何序列引用
                self._deallocate_block(block_id)  # 归还到空闲池
        seq.num_cached_tokens = 0  # 清除序列的已缓存 token 数
        seq.block_table.clear()  # 清空块表

    def can_append(self, seq: Sequence) -> bool:  # 判断解码阶段追加 1 个 token 是否还够空闲块
        # 解码每轮只追加 1 个 token。只有当下一个 token 恰好“跨入一个新的空块”时才真正需要
        # 占用一个新的物理块：即当前 len(seq) 恰好是 block_size 的整数倍时。
        # len(seq) % block_size == 1 求值为 True/False，作为需要的块数（0 或 1）参与比较。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)  # 空闲块数 ≥ 需要的新块数

    def may_append(self, seq: Sequence):  # 若本次追加会开启新块，则先分配新块
        if len(seq) % self.block_size == 1:  # 当前 token 数恰好是 block_size 的整数倍（+1 将进入新块）
            seq.block_table.append(self._allocate_block())  # 为新块分配并加入块表（can_append 已保证有空闲块）

    def hash_blocks(self, seq: Sequence):  # 为本次已处理的块更新哈希与内容（用于后续前缀缓存）
        # ──────────────────────────────────────────────────────────────
        # 在每轮前向之后被调用：把“本轮已填满的块”登记进哈希表，这样未来其他序列
        # 只要前缀内容相同，就能通过 can_allocate 命中并复用这些块。
        # 只处理“完整填满的块”（start..end 区间内的块），因为只有内容定型的块才能被安全复用；
        # 最后那个没填满的“尾巴块”不登记。
        # ──────────────────────────────────────────────────────────────
        start = seq.num_cached_tokens // self.block_size  # 本次处理前，已缓存到第几个块（=起始块下标）
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size  # 本轮结束后填满到第几个块（开区间）
        if start == end: return  # 若本轮没有新填满任何一个整块（全是尾巴），无需登记
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1  # 级联起点：取上一块的哈希（首块用 -1）
        for i in range(start, end):  # 遍历本轮填满的每个块
            block = self.blocks[seq.block_table[i]]  # 取物理块对象（block_table[i] 是该块在缓存里的编号）
            token_ids = seq.block(i)  # 取该块内的 token 内容（用于后续哈希冲突时的内容比对）
            h = self.compute_hash(token_ids, h)  # 计算“第 0..i 块”的链式哈希（注意依赖上一块哈希 → 必须从头级联）
            block.update(h, token_ids)  # 把哈希与内容写进块对象
            self.hash_to_block_id[h] = block.block_id  # 登记哈希 -> 块号，供未来的前缀命中查找

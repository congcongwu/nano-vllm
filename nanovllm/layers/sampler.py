import torch  # PyTorch 主模块
from torch import nn  # PyTorch 神经网络基类


class Sampler(nn.Module):  # 采样器：把 logits 转换为最终输出的 token id

    @torch.compile  # 编译该算子，融合温度缩放与采样逻辑
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):  # 前向：输入每个序列的 logits 与温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # 按温度缩放 logits（logits/温度），温度越大越平滑
        probs = torch.softmax(logits, dim=-1)  # 转为概率分布
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)  # 用 Gumbel 技巧做有放回采样：除以指数噪声后取 argmax，等价于按概率分布采样
        return sample_tokens  # 返回采样出的 token id 张量

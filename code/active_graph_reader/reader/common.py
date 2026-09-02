"""base/attention reader 共享的小工具（消除两份 _stable_probs 拷贝）。"""

import torch
import torch.nn.functional as F


def stable_probs(scores: torch.Tensor, temperature: float = 1.0,
                 min_prob: float = 1e-3) -> torch.Tensor:
    """softmax 后托底：防极端尖锐分布下概率精确下溢为 0（log(0)=-inf → loss NaN）。

    正常训练时概率均 > min_prob，clamp 完全不生效；仅在下溢边缘托底。
    """
    probs = F.softmax(scores / temperature, dim=0)
    probs = probs.clamp(min=min_prob)
    return probs / probs.sum()

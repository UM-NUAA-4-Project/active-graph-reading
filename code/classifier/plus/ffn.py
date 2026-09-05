"""GNN+ 组件⑤前馈网络 FFN（feed-forward network）。

论文 Eq.10 忠实实现：FFN(h) = BN(σ(h·W₁)·W₂ + h)——两层变换、
**自带残差**、收尾 BatchNorm；dropout 施加在隐层激活后（W₂ 之前）。
作为逐层管线的最后一级插入（Eq.11：MP→Norm→σ→Dropout→残差→FFN）。

关闭时 encoder 用 nn.Identity 占位（零参数零键，旧 checkpoint 键集不变）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import make_norm

FFN_HIDDEN_MULT = 2  # 论文未固定扩张比；×2 为 Transformer 惯例


class FFN(nn.Module):
    """h → BN( σ(h·W₁)·W₂ + h )，W₁: dim→mult·dim，W₂: mult·dim→dim。"""

    def __init__(self, dim: int, dropout: float = 0.0,
                 hidden_mult: int = FFN_HIDDEN_MULT):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_mult * dim)
        self.fc2 = nn.Linear(hidden_mult * dim, dim)
        self.bn = make_norm(dim, 'batchnorm')
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h2 = self.fc2(F.dropout(F.relu(self.fc1(h)),
                                p=self.dropout, training=self.training))
        return self.bn(h2 + h)

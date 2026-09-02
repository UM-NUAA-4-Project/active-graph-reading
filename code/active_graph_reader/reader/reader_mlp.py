"""ReaderMLP: 将 frontier embeddings 映射为标量 scores。

支持逐层 hidden/dropout 列表；标量形式（hidden_dim + 3 层）与旧版
state_dict 键名完全一致（mlp.0 / mlp.3 / mlp.6）。
"""

from typing import List, Optional

import torch.nn as nn


class ReaderMLP(nn.Module):
    """MLP scorer，供 LearnableReader 使用。

    两种参数形式：
    - 旧式标量：hidden_dim + dropout（3 个 uniform 隐层）
    - 新式列表：hidden_dims[i] / dropouts[i]（逐层独立）
    结构：[Linear(hi) ReLU Dropout]*k → Linear(→1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1,
                 hidden_dims: Optional[List[int]] = None,
                 dropouts: Optional[List[float]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * 3
        if dropouts is None:
            dropouts = [dropout] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError("dropouts 与 hidden_dims 长度不一致")

        layers: list = []
        prev = input_dim
        for i, (h, p) in enumerate(zip(hidden_dims, dropouts)):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)

    def forward(self, frontier_embeds):
        """(N, input_dim) -> (N,)"""
        return self.mlp(frontier_embeds).squeeze(-1)

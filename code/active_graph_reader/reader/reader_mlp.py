"""ReaderMLP: 将 frontier embeddings 映射为标量 scores。"""

import torch.nn as nn


class ReaderMLP(nn.Module):
    """小型 MLP scorer，供 LearnableReader 和 POMDPReader 共用。

    Parameters
    ----------
    input_dim : int   输入维度（LearnableReader=128, POMDPReader=169）
    hidden_dim : int  隐层维度，默认 128
    dropout : float   Dropout，默认 0.1
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frontier_embeds):
        """(N, input_dim) -> (N,)"""
        return self.mlp(frontier_embeds).squeeze(-1)

"""LearnableReader: 可学习的图阅读器。

G_t → FrontierNodesEncoder → frontier_embeds → ReaderMLP → scores → softmax → argmax → action
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_reader import BaseReader
from .frontier_encoder import FrontierNodesEncoder
from ..engine.graph_state import GraphState


class ReaderMLP(nn.Module):
    """小型 MLP：将 frontier node embeddings 映射为标量 scores。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frontier_embeds: torch.Tensor) -> torch.Tensor:
        return self.mlp(frontier_embeds).squeeze(-1)


class LearnableReader(BaseReader, nn.Module):
    """可学习的图阅读器。

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度（默认 128）
    n_layers : int       FrontierEncoder 的 GNN 层数（默认 2）
    dropout : float      Dropout 概率（默认 0.1）
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        BaseReader.__init__(self)
        nn.Module.__init__(self)

        self.frontier_encoder = FrontierNodesEncoder(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.reader_mlp = ReaderMLP(hidden_dim)

    def forward(
        self, graph_state: GraphState
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """核心可微管线。Returns: (scores, probs, frontier_nodes)."""
        frontier_embeds, frontier_nodes = self.frontier_encoder(graph_state)

        if len(frontier_nodes) == 0:
            return torch.empty(0), torch.empty(0), []

        scores = self.reader_mlp(frontier_embeds)
        probs = F.softmax(scores, dim=0)
        return scores, probs, frontier_nodes

    def select(self, graph_state: GraphState) -> Optional[int]:
        """确定性 argmax 选择，兼容 RevealLoop。"""
        scores, probs, frontier_nodes = self.forward(graph_state)

        if len(frontier_nodes) == 0:
            return None

        best_idx = probs.argmax().item()
        return frontier_nodes[best_idx]

    @property
    def name(self) -> str:
        return 'learnable'

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self._training = mode
        return self

    def eval(self):
        nn.Module.eval(self)
        self._training = False
        return self

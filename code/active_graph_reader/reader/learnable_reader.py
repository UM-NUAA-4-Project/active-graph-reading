"""LearnableReader: 可学习的图阅读器。

G_t → FrontierGraphBuilder → FrontierSubgraph → FrontierNodesEncoder → frontier_embeds
     → ReaderMLP → scores → softmax → (argmax | sample) → action

stochastic=False（默认）：argmax 确定性选择
stochastic=True：从概率分布采样，探索用
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reader_mlp import ReaderMLP
from .gsae import compute_structural_features

from ..core.base_reader import BaseReader
from ..gnn.frontier.graph_builder import FrontierGraphBuilder
from ..gnn.frontier.encoder import FrontierNodesEncoder
from ..engine.graph_state import GraphState




class LearnableReader(BaseReader, nn.Module):
    """可学习的图阅读器。

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度（默认 128）
    n_layers : int       FrontierEncoder 的 GNN 层数（默认 2）
    dropout : float      Dropout 概率（默认 0.1）
    stochastic : bool    是否从概率分布采样（默认 False=argmax）
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        stochastic: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.stochastic = stochastic
        self.temperature = temperature

        self.frontier_builder = FrontierGraphBuilder()
        self.frontier_encoder = FrontierNodesEncoder(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.reader_mlp = ReaderMLP(input_dim=hidden_dim + 1 + 3, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self, graph_state: GraphState, sample: bool = False
    ):
        """核心可微管线。

        sample=False: Returns (scores, probs, frontier_nodes)
        sample=True:  Returns (scores, probs, frontier_nodes, action_idx, log_prob)
        """
        subgraph = self.frontier_builder.build(graph_state)
        frontier_embeds, frontier_nodes = self.frontier_encoder(subgraph)

        if len(frontier_nodes) == 0:
            if sample:
                return torch.empty(0), torch.empty(0), [], None, None
            return torch.empty(0), torch.empty(0), []

        remaining = graph_state.get_budget_ratio_remaining()
        remaining_tensor = torch.full((len(frontier_nodes), 1), remaining,
                                      device=frontier_embeds.device, dtype=frontier_embeds.dtype)

        struct_all = compute_structural_features(subgraph.g, subgraph.n_revealed)
        struct_frontier = struct_all[subgraph.n_revealed:].to(
            device=frontier_embeds.device, dtype=frontier_embeds.dtype)

        frontier_embeds = torch.cat([frontier_embeds, remaining_tensor, struct_frontier], dim=-1)

        scores = self.reader_mlp(frontier_embeds)
        probs = F.softmax(scores, dim=0)

        if sample:
            if self.temperature != 1.0:
                probs = F.softmax(scores / self.temperature, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            return scores, probs, frontier_nodes, action_idx, log_prob

        return scores, probs, frontier_nodes

    def select(self, graph_state: GraphState) -> Optional[int]:
        """选择节点。stochastic=False 时 argmax，True 时从概率分布采样。"""
        scores, probs, frontier_nodes = self.forward(graph_state)

        if len(frontier_nodes) == 0:
            return None

        if self.stochastic:
            if self.temperature != 1.0:
                probs = F.softmax(scores / self.temperature, dim=0)
            idx = torch.distributions.Categorical(probs).sample().item()
        else:
            idx = probs.argmax().item()
        return frontier_nodes[idx]

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

"""PrototypeReader: belief-aware 策略网络。

架构（169-dim 状态空间）：
  G_t → FrontierGraphBuilder → FrontierNodesEncoder → frontier_embeds (128-dim)
      → SubgraphBuilder → g_φ → bank.query → b_t (37-dim)
      → concat[b_t(37) ‖ frontier_embeds(128) ‖ budget_ratio(1) ‖ struct(3)]
      → ReaderMLP(169→128→1) → scores → argmax → next_node

与 LearnableReader (132-dim) 的核心差异：额外注入当前信念 b_t，
让策略知道"当前子图像哪一类"，基于此决定下一步揭示哪个节点。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .reader_mlp import ReaderMLP
from .gsae import compute_structural_features
from ..core.base_reader import BaseReader
from ..gnn.frontier.graph_builder import FrontierGraphBuilder
from ..gnn.frontier.encoder import FrontierNodesEncoder
from ..gnn.subgraph_builder import SubgraphBuilder
from ..engine.graph_state import GraphState


class PrototypeReader(BaseReader, nn.Module):
    """Belief-aware 原型阅读器。

    每步查询原型库获取当前信念 b_t，将其作为 frontier 节点选择的上下文信号。

    Parameters
    ----------
    edge_feat_dim : int   边特征维度（PPA: 7）
    hidden_dim : int      隐层维度（默认 128）
    n_classes : int       类别数（默认 37）
    n_layers : int        FrontierEncoder GNN 层数（默认 2）
    dropout : float       Dropout 概率（默认 0.1）
    stochastic : bool     推理时是否从概率分布采样（默认 False=argmax）
    temperature : float   采样温度（默认 1.0）
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        n_classes: int = 37,
        n_layers: int = 2,
        dropout: float = 0.1,
        stochastic: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.stochastic = stochastic
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.frontier_builder = FrontierGraphBuilder()
        self.subgraph_builder = SubgraphBuilder()
        self.frontier_encoder = FrontierNodesEncoder(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.reader_mlp = ReaderMLP(
            input_dim=n_classes + hidden_dim + 1 + 3,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 外部注入（不由本类管理参数）
        self._g_phi = None
        self._bank = None

    # ------------------------------------------------------------------
    # 外部依赖注入
    # ------------------------------------------------------------------

    def set_encoder(self, g_phi):
        """注入子图编码器 g_φ（需有 get_embedding(g, nids)→(B, hidden) 接口）。"""
        self._g_phi = g_phi

    def set_bank(self, bank):
        """注入原型库（需有 query(emb)→(B, n_classes) 接口）。"""
        self._bank = bank

    @property
    def g_phi(self):
        if self._g_phi is None:
            raise RuntimeError("g_φ not set. Call set_encoder() first.")
        return self._g_phi

    @property
    def bank(self):
        if self._bank is None:
            raise RuntimeError("PrototypeBank not set. Call set_bank() first.")
        return self._bank

    # ------------------------------------------------------------------
    # 核心管线
    # ------------------------------------------------------------------

    def _query_belief(self, graph_state: GraphState):
        """从当前 GraphState 构建最终子图，查询信念 b_t。"""
        sub_g, _orig_ids = self.subgraph_builder.build(graph_state)
        if sub_g.num_nodes() == 0:
            device = next(self.parameters()).device
            return torch.zeros(self.n_classes, device=device)
        sub_g = dgl.add_self_loop(sub_g)
        device = next(self.g_phi.parameters()).device
        sub_g = sub_g.to(device)
        nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            emb = self.g_phi.get_embedding(sub_g, nids)
            if emb.dim() == 2:
                emb = emb.squeeze(0)
            b_t = self.bank.query(emb.unsqueeze(0)).squeeze(0)
        return b_t.detach()

    def forward(self, graph_state: GraphState, sample: bool = False):
        """核心可微管线。

        sample=False: Returns (scores, probs, frontier_nodes)
        sample=True:  Returns (scores, probs, frontier_nodes, action_idx, log_prob)
        """
        # 1. frontier 子图 → frontier_embeds
        frontier_sub = self.frontier_builder.build(graph_state)
        frontier_embeds, frontier_nodes = self.frontier_encoder(frontier_sub)

        if len(frontier_nodes) == 0:
            if sample:
                return torch.empty(0), torch.empty(0), [], None, None
            return torch.empty(0), torch.empty(0), []

        device = frontier_embeds.device
        dtype = frontier_embeds.dtype

        # 2. 信念 b_t
        b_t = self._query_belief(graph_state).to(device=device, dtype=dtype)
        b_t_expanded = b_t.unsqueeze(0).expand(len(frontier_nodes), -1)

        # 3. budget ratio + structural features
        remaining = graph_state.get_budget_ratio_remaining()
        remaining_tensor = torch.full(
            (len(frontier_nodes), 1), remaining, device=device, dtype=dtype,
        )

        struct_all = compute_structural_features(
            frontier_sub.g, frontier_sub.n_revealed,
        )
        struct_frontier = struct_all[frontier_sub.n_revealed:].to(
            device=device, dtype=dtype,
        )

        # 4. 拼接状态 → ReaderMLP 打分
        feat = torch.cat(
            [b_t_expanded, frontier_embeds, remaining_tensor, struct_frontier],
            dim=-1,
        )

        scores = self.reader_mlp(feat)
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
        """选择下一个揭示节点。stochastic=False 时 argmax。"""
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
        return 'prototype-reader'

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self._training = mode
        return self

    def eval(self):
        nn.Module.eval(self)
        self._training = False
        return self

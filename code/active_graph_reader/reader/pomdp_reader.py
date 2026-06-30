"""POMDPReader: 证据信念 POMDP 图阅读器（简化版）。

组件：
  gsae             — GSAE (2 层 GNN + attention pooling)
  evidence_net     — 证据头 v → e, b, u
  belief_acc       — SimpleDirichletAccumulator（α += e_t）
  policy_mlp       — ReaderMLP(128 + K + 1 + 3 + 1)
  final_classifier — LSTM 证据序列 + 最终子图 Dempster 融合
  frontier_builder — FrontierGraphBuilder（复用）
  frontier_proj    — Linear(300, 128)

每步推理：
  G_t → FrontierGraphBuilder → FrontierSubgraph
      → GSAE → v, frontier_embeds(300), struct(3)
      → EvidenceNetwork → e_t, b_raw, u_raw
      → 缓存 e_t, 累加 Dirichlet → b_acc, u_acc
      → remaining = 1 - t/B
      → [frontier_128 || b_acc || remaining || struct || u_acc] → ReaderMLP → scores → probs

B 步后：
  o_B = frontier_builder.build(G_B)
  o_B → GSAE → EvidenceNetwork → e_B（加入缓存）
  FinalClassifier([e_0, ..., e_B], o_B, gsae, evidence_net) → y_hat
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_reader import BaseReader
from ..engine.graph_state import GraphState
from ..gnn.frontier.graph_builder import FrontierGraphBuilder

from .reader_mlp import ReaderMLP
from .gsae import GSAE
from .evidence_network import EvidenceNetwork
from .belief_updater import SimpleDirichletAccumulator
from .final_classifier import FinalClassifier


class POMDPReader(BaseReader, nn.Module):
    """简化版证据信念 POMDP Reader。

    Parameters
    ----------
    edge_feat_dim : int    边特征维度（PPA: 7）
    hidden_dim : int       GNN 隐层维度，默认 300
    k_state : int          保留参数（兼容），不再使用
    k_evidence : int       证据维度（类别数），默认 37
    n_layers : int         GNN 层数，默认 2
    dropout : float        Dropout，默认 0.5
    use_struct : bool      是否使用结构特征 + 结构注意力池化（默认 True）
    stochastic : bool      是否从概率分布采样选择节点（默认 False=argmax）
    """

    def __init__(self, edge_feat_dim=7, hidden_dim=300, k_state=128,
                 k_evidence=37, n_layers=2, dropout=0.5, use_struct=True,
                 stochastic=False):
        super().__init__()

        self.stochastic = stochastic
        self.k_evidence = k_evidence

        self.frontier_builder = FrontierGraphBuilder()

        self.gsae = GSAE(
            edge_feat_dim=edge_feat_dim, hidden_dim=hidden_dim,
            k_state=k_state, n_layers=n_layers, dropout=dropout,
            use_struct=use_struct,
        )

        self.evidence_net = EvidenceNetwork(
            hidden_dim=hidden_dim, k_evidence=k_evidence, dropout=dropout)

        self.belief_acc = SimpleDirichletAccumulator(k_evidence=k_evidence)

        self.frontier_proj = nn.Linear(hidden_dim, 128)  # 300 → 128

        # policy input: frontier_128(128) + b_t(K) + remaining(1) + struct(3) + u_t(1)
        policy_input_dim = 128 + k_evidence + 1 + 3 + 1
        self.policy_mlp = ReaderMLP(
            input_dim=policy_input_dim, hidden_dim=128, dropout=dropout)

        self.final_classifier = FinalClassifier(k_evidence=k_evidence)

        self._evidence_cache: List[torch.Tensor] = []
        self._last_frontier_ids: List[int] = []
        self._last_gsae_out = None

    def reset(self):
        """每图开始调用。"""
        self.belief_acc.reset()
        self._evidence_cache = []
        self._last_frontier_ids = []
        self._last_gsae_out = None

    @property
    def name(self) -> str:
        return 'pomdp'

    def forward(self, graph_state: GraphState, sample: bool = False):
        """单步推理。

        sample=False: Returns (scores, probs, frontier_nodes)
        sample=True:  Returns (scores, probs, frontier_nodes, action_idx, log_prob)
        """
        subgraph = self.frontier_builder.build(graph_state)
        self._last_gsae_out = self.gsae(subgraph)
        self._last_frontier_ids = self._last_gsae_out.frontier_original_ids

        if len(self._last_frontier_ids) == 0:
            if sample:
                return torch.empty(0), torch.empty(0), [], None, None
            return torch.empty(0), torch.empty(0), []

        # Evidence from current graph state
        ev_out = self.evidence_net(self._last_gsae_out.v)
        e_t = ev_out.e.squeeze(0)  # (K,)

        # Cache evidence
        self._evidence_cache.append(e_t)

        # Simple Dirichlet accumulation
        b_acc, u_acc = self.belief_acc.update(e_t)

        device = e_t.device

        # Frontier embeddings + remaining budget + structural + belief
        frontier_128 = self.frontier_proj(self._last_gsae_out.frontier_embeds)  # (N_f, 128)
        struct_v = self._last_gsae_out.frontier_struct.to(device)                # (N_f, 3)
        remaining = graph_state.get_budget_ratio_remaining()

        n_f = frontier_128.size(0)
        b_expanded = b_acc.unsqueeze(0).expand(n_f, -1)            # (N_f, K)
        remaining_expanded = torch.full((n_f, 1), remaining,
                                        device=device, dtype=frontier_128.dtype)
        u_expanded = u_acc.view(1, 1).expand(n_f, -1)              # (N_f, 1)

        policy_input = torch.cat([
            frontier_128, b_expanded, remaining_expanded, struct_v, u_expanded,
        ], dim=-1)

        scores = self.policy_mlp(policy_input)
        probs = F.softmax(scores, dim=0)

        if sample:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            return scores, probs, self._last_frontier_ids, action_idx, log_prob

        return scores, probs, self._last_frontier_ids

    def select(self, graph_state: GraphState) -> Optional[int]:
        """选择节点。"""
        if self.stochastic:
            scores, probs, frontier_nodes, action_idx, _ = \
                self.forward(graph_state, sample=True)
            if len(frontier_nodes) == 0:
                return None
            return frontier_nodes[action_idx.item()]

        scores, probs, frontier_nodes = self.forward(graph_state)
        if len(frontier_nodes) == 0:
            return None

        best_idx = probs.argmax().item()
        return frontier_nodes[best_idx]

    def get_terminal_belief(self, graph_state: GraphState):
        """B 步后调用：计算 e_B 加入缓存，LSTM + 最终子图 Dempster 融合分类。

        Returns: FinalOutput (b_total, u_total, y_hat)
        """
        o_B = self.frontier_builder.build(graph_state)

        # 计算 e_B 并加入缓存
        gsae_out = self.gsae(o_B)
        ev_out = self.evidence_net(gsae_out.v)
        e_B = ev_out.e.squeeze(0)  # (K,)
        self._evidence_cache.append(e_B)

        return self.final_classifier(
            self._evidence_cache, o_B, self.gsae, self.evidence_net)

    def init_from_fclass(self, checkpoint_path: str, device='cpu'):
        """从 f_class checkpoint 初始化 GSAE 的 2 层 GNN。"""
        self.gsae.init_from_fclass(checkpoint_path, device)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self._training = mode
        return self

    def eval(self):
        nn.Module.eval(self)
        self._training = False
        return self

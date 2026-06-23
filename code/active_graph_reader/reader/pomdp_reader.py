"""POMDPReader: 证据信念 POMDP 图阅读器（顶层编排）。

组件：
  gsae             — GSAE (2 层 GNN + encoder + decoder)
  evidence_net     — 证据头
  belief_updater   — DRBPN LSTM 循环信念
  policy_mlp       — ReaderMLP(128 + k_evidence, 128)
  final_classifier — 双路径 Dirichlet 融合分类
  frontier_builder — FrontierGraphBuilder（复用）
  frontier_proj    — Linear(300, 128)，降维 frontier embedding

每步推理：
  G_t → FrontierGraphBuilder → FrontierSubgraph
      → GSAE → z, v, v_hat, frontier_embeds(300)
      → EvidenceNetwork → e, b, u
      → LSTMBeliefUpdater → b_t
      → [frontier_embeds(128) || b_t] → ReaderMLP → scores → probs

B 步后：
  FinalClassifier(b_B, o_B) → y_hat, R
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_reader import BaseReader
from ..engine.graph_state import GraphState
from ..gnn.frontier.graph_builder import FrontierGraphBuilder

from .reader_mlp import ReaderMLP
from .gsae import GSAE
from .evidence_network import EvidenceNetwork
from .belief_updater import LSTMBeliefUpdater
from .final_classifier import FinalClassifier


class POMDPReader(BaseReader, nn.Module):
    """证据信念 POMDP Reader。

    Parameters
    ----------
    edge_feat_dim : int    边特征维度（PPA: 7）
    hidden_dim : int       GNN 隐层维度，默认 300
    k_state : int          语义潜在维度，默认 128
    k_evidence : int       证据维度（类别数），默认 37
    n_layers : int         GNN 层数，默认 2
    dropout : float        Dropout，默认 0.5
    """

    def __init__(self, edge_feat_dim=7, hidden_dim=300, k_state=128,
                 k_evidence=37, n_layers=2, dropout=0.5):
        super().__init__()

        self.k_evidence = k_evidence

        self.frontier_builder = FrontierGraphBuilder()

        self.gsae = GSAE(
            edge_feat_dim=edge_feat_dim, hidden_dim=hidden_dim,
            k_state=k_state, n_layers=n_layers, dropout=dropout,
        )

        self.evidence_net = EvidenceNetwork(k_state=k_state, k_evidence=k_evidence)

        self.belief_updater = LSTMBeliefUpdater(k_evidence=k_evidence, action_emb_dim=128)

        self.frontier_proj = nn.Linear(hidden_dim, 128)  # 300 -> 128

        self.policy_mlp = ReaderMLP(input_dim=128 + k_evidence + 3, hidden_dim=128, dropout=dropout)

        self.final_classifier = FinalClassifier()

        self._prev_action_emb = torch.zeros(128)
        self._current_belief = torch.ones(k_evidence) / k_evidence
        self._last_frontier_ids = []
        self._last_gsae_out = None

    def reset(self):
        """每图开始调用。"""
        self._prev_action_emb = torch.zeros(128)
        self._current_belief = torch.ones(self.k_evidence) / self.k_evidence
        self.belief_updater.reset()
        self._last_frontier_ids = []
        self._last_gsae_out = None

    @property
    def name(self) -> str:
        return 'pomdp'

    def forward(self, graph_state: GraphState, sample: bool = False):
        """单步推理。

        When sample=False (default):
            Returns: (scores, probs, frontier_nodes)
        When sample=True:
            Returns: (scores, probs, frontier_nodes, action_idx, log_prob)
        """
        subgraph = self.frontier_builder.build(graph_state)
        self._last_gsae_out = self.gsae(subgraph)
        self._last_frontier_ids = self._last_gsae_out.frontier_original_ids

        if len(self._last_frontier_ids) == 0:
            if sample:
                return torch.empty(0), torch.empty(0), [], None, None
            return torch.empty(0), torch.empty(0), []

        ev_out = self.evidence_net(self._last_gsae_out.z)

        device = ev_out.e.device
        b_prev = self._current_belief.to(device)
        a_prev = self._prev_action_emb.to(device)

        b_t = self.belief_updater(
            b_prev.unsqueeze(0),
            ev_out.e, ev_out.u,
            a_prev.unsqueeze(0),
        )
        self._current_belief = b_t.squeeze(0).detach().cpu()

        frontier_128 = self.frontier_proj(self._last_gsae_out.frontier_embeds)  # (N_f, 128)

        struct_v = self._last_gsae_out.frontier_struct.to(device)  # (N_f, 3)

        b_expanded = b_t.expand(frontier_128.size(0), -1)  # (N_f, k_evidence)
        policy_input = torch.cat([frontier_128, b_expanded, struct_v], dim=-1)  # (N_f, 128 + K + 3)

        scores = self.policy_mlp(policy_input)
        probs = F.softmax(scores, dim=0)

        if sample:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            # cache action embedding for sampled action
            emb_300 = self._last_gsae_out.frontier_embeds[action_idx:action_idx + 1]
            self._prev_action_emb = self.frontier_proj(emb_300).detach().squeeze(0).cpu()
            return scores, probs, self._last_frontier_ids, action_idx, log_prob

        return scores, probs, self._last_frontier_ids

    def select(self, graph_state: GraphState) -> Optional[int]:
        """确定性 argmax 选择 + 缓存 action embedding。"""
        scores, probs, frontier_nodes = self.forward(graph_state)

        if len(frontier_nodes) == 0:
            return None

        best_idx = probs.argmax().item()
        chosen = frontier_nodes[best_idx]

        if self._last_gsae_out is not None:
            emb_300 = self._last_gsae_out.frontier_embeds[best_idx:best_idx + 1]
            self._prev_action_emb = self.frontier_proj(emb_300).detach().squeeze(0).cpu()

        return chosen

    def get_terminal_belief(self, graph_state: GraphState):
        """B 步后调用：双路径 Dirichlet 融合 + 最终分类。

        Returns: FinalOutput (b_total, u_total, y_hat)
        """
        o_B = self.frontier_builder.build(graph_state)
        alpha_lstm = self.belief_updater.alpha_accum
        return self.final_classifier(o_B, self.gsae, self.evidence_net, alpha_lstm)

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

"""AttentionReader (attention variant + Belief State, no classification head).

G_t → FrontierGraphBuilder → FrontierSubgraph → FrontierNodesEncoder → frontier_embeds
     → FrontierPointer (self-attn + belief cross-attn) → scores → softmax → action

Same structure and logic as the base MLP LearnableReader (reader/learnable_reader.py),
except:
  - Uses FrontierPointer (self-attention + belief cross-attention) for scoring
  - Maintains a cross-step BeliefState (NewInfoEncoder encodes newly revealed info)
  - No ClassificationHead — classification is delegated to an external classifier
  - forward_states returns 3-tuple (scores, probs, frontier_nodes)
"""

from typing import List, Optional, Tuple

import dgl
import torch
import torch.nn as nn

from active_graph_reader.core.base_reader import BaseReader
from active_graph_reader.config import ReaderConfig
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder
from active_graph_reader.gnn.frontier.encoder import FrontierNodesEncoder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.reader.gsae import compute_structural_features
from active_graph_reader.reader.common import stable_probs as _stable_probs
from .frontier_pointer import FrontierPointer
from .belief_state import BeliefState
from .new_info_encoder import NewInfoEncoder


class AttentionReader(BaseReader, nn.Module):
    """Attention-enhanced selector reader with belief state (no classification head).

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度（默认 128）
    n_layers : int       FrontierEncoder 的 GNN 层数（默认 2）
    dropout : float      Dropout 概率（默认 0.1）
    stochastic : bool    是否从概率分布采样（默认 False=argmax）
    m : int              Belief token 数量（默认 4），同时用于选节点和分类
    node_feat_dim : int  原始图节点特征维度（默认 0=无节点特征，如 PPA）
    """

    def __init__(
        self,
        edge_feat_dim: int = 7,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        stochastic: bool = False,
        temperature: float = 1.0,
        m: int = 4,
        n_heads: int = 4,
        node_feat_dim: int = 0,
        config: Optional[ReaderConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = ReaderConfig.from_legacy(
                kind='attention', edge_feat_dim=edge_feat_dim,
                node_feat_dim=node_feat_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, dropout=dropout, m=m, n_heads=n_heads,
                stochastic=stochastic, temperature=temperature)
        config.validate()
        self.config = config
        self.stochastic = config.stochastic
        self.temperature = config.temperature
        self.hidden_dim = config.hidden_dim   # = d_model（encoder 须同宽，validate 已保证）

        self.frontier_builder = FrontierGraphBuilder()
        self.frontier_encoder = FrontierNodesEncoder(
            edge_feat_dim=config.edge_feat_dim,
            hidden_dims=config.encoder_hidden_dims,
            dropouts=config.encoder_dropouts,
            node_feat_dim=config.node_feat_dim,
        )
        d_in = self.hidden_dim + 1 + 3  # GNN embeds + budget_ratio + structural
        self.frontier_pointer = FrontierPointer(
            d_in=d_in, d_model=self.hidden_dim, n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Phase 2: belief state
        self.belief_state = BeliefState(
            d_model=self.hidden_dim, m=config.m,
            n_heads=config.n_heads, dropout=config.dropout,
        )

        # Full frontier subgraph GNN for encoding new reveal info (local + global)
        self.new_info_encoder = NewInfoEncoder(
            config.edge_feat_dim, self.hidden_dim, config.dropout)
        self.node_feat_proj = (
            nn.Linear(config.node_feat_dim, self.hidden_dim)
            if config.node_feat_dim > 0 else None
        )

        # Stateful belief for select() eval path
        self._belief = None
        self._prev_frontier = None
        self._last_selected = None

    # ------------------------------------------------------------------
    # Batched forward — used by cross-graph lockstep
    # ------------------------------------------------------------------

    def forward_states(
        self,
        states: List[GraphState],
        temperature: float = 1.0,
        beliefs: torch.Tensor = None,
        subgraphs: list = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """Batch encode + score multiple graph states.

        Args:
            states:      list of GraphState (all active this lockstep step)
            temperature: softmax temperature
            beliefs:     (N, m, d) belief tokens for cross-attention scoring
            subgraphs:   pre-built FrontierSubgraph list (reuse from prev step),
                         or None to build internally

        Returns:
            List of (scores, probs, frontier_nodes).
            States with empty frontier receive (empty, empty, []).
        """
        if not states:
            return []

        if subgraphs is None:
            subgraphs = [self.frontier_builder.build(s) for s in states]
        batch_results = self.frontier_encoder.forward_batch(subgraphs)

        results: List = []
        for k, (frontier_embeds, frontier_nodes) in enumerate(batch_results):
            if not frontier_nodes:
                results.append((torch.empty(0), torch.empty(0), []))
                continue

            sg = subgraphs[k]
            combined = self._build_combined(frontier_embeds, sg, states[k])

            scores = self.frontier_pointer(combined, belief=beliefs[k])
            probs = _stable_probs(scores, temperature)

            results.append((scores, probs, frontier_nodes))

        return results

    # ------------------------------------------------------------------
    # Belief management helpers (called from training loop, not internally)
    # ------------------------------------------------------------------

    def init_beliefs(self, n: int) -> torch.Tensor:
        """Return (n, m, d) initial belief tokens for n states."""
        return self.belief_state.init_beliefs(n)

    def update_belief(
        self, h: torch.Tensor, new_info: torch.Tensor,
    ) -> torch.Tensor:
        """h: (1, m, d), new_info: (1, 1, d) → (1, m, d) updated belief."""
        return self.belief_state.update_belief(h, new_info)

    def _node_embedding(self, state: GraphState, nodes: List[int],
                        device: torch.device) -> torch.Tensor:
        """Get initial node embeddings for a list of node indices.

        If the dataset has node features, project them; otherwise return zeros
        (frontier nodes are always encoded as zero, matching FrontierNodesEncoder).
        """
        if not (state.has_node_feat and state.node_feat is not None
                and self.node_feat_proj is not None):
            return torch.zeros(len(nodes), self.hidden_dim, device=device)
        feats = torch.stack([state.node_feat[v].to(device) for v in nodes])
        return self.node_feat_proj(feats)

    def encode_new_info(
        self, state: GraphState, subgraph, v_revealed: int, device: torch.device,
    ) -> torch.Tensor:
        """Encode full frontier subgraph → (1, 1, d) via 2-layer GIN.

        The subgraph covers all revealed + frontier nodes after the reveal.
        v_revealed is the original graph ID of the just-revealed node.
        """
        g = subgraph.g.to(device)
        n_total = g.num_nodes()
        n_revealed = subgraph.n_revealed

        # v's position in subgraph: revealed nodes sorted by original ID
        revealed = state.get_revealed_sorted()
        v_idx = revealed.index(v_revealed)

        # Type embeddings: revealed=0, frontier=1
        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        if n_total > n_revealed:
            node_types[n_revealed:] = 1
        h = self.frontier_encoder.node_type_emb(node_types)

        # Override v with real node features if available
        if self.node_feat_proj is not None:
            node_v = self._node_embedding(state, [v_revealed], device)
            if node_v.abs().sum() > 0:
                h[v_idx] = node_v[0]

        frontier_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        if n_total > n_revealed:
            frontier_mask[n_revealed:] = True

        edge_feats = g.edata['feat']
        pooled = self.new_info_encoder(g, h, edge_feats, v_idx, frontier_mask)
        return pooled.unsqueeze(0).unsqueeze(0)

    def encode_new_info_batch(
        self, states: list, subgraphs: list, v_revealed_list: list,
        device: torch.device, state_src: list = None,
    ) -> torch.Tensor:
        """Batch-encode multiple frontier subgraphs → (N, 1, d).

        Batches all subgraphs into one dgl.batch, runs shared GIN layers,
        then pools per-subgraph.  Much faster than N individual encodes.
        """
        n = len(states)
        g_list = [sg.g.to(device) for sg in subgraphs]
        batch_g = dgl.batch(g_list)

        node_counts = [g.num_nodes() for g in g_list]
        offsets = [0]
        for c in node_counts:
            offsets.append(offsets[-1] + c)

        # Node type embeddings: revealed=0, frontier=1
        n_total = batch_g.num_nodes()
        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        for i in range(n):
            start = offsets[i]
            n_rev = subgraphs[i].n_revealed
            n_fr = node_counts[i] - n_rev
            if n_fr > 0:
                node_types[start + n_rev:start + node_counts[i]] = 1

        h = self.frontier_encoder.node_type_emb(node_types)

        # Build per-graph v_indices (global) and frontier_masks
        v_indices = []
        frontier_masks = []
        for i in range(n):
            revealed = states[i].get_revealed_sorted()
            v_idx_local = revealed.index(v_revealed_list[i])
            v_indices.append(offsets[i] + v_idx_local)

            n_rev = subgraphs[i].n_revealed
            n_fr = node_counts[i] - n_rev
            f_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
            if n_fr > 0:
                f_mask[offsets[i] + n_rev:offsets[i] + node_counts[i]] = True
            frontier_masks.append(f_mask)

        edge_feats = batch_g.edata['feat']
        pooled = self.new_info_encoder.forward_batch(
            batch_g, h, edge_feats, v_indices, frontier_masks,
        )  # (N, d)
        if not torch.isfinite(pooled).all():
            print('\n' + '=' * 60)
            print('[NaN 监测] new_info_encoder 输出 NaN（首个 NaN 事件?）')
            print(f'  edge_feats: shape={tuple(edge_feats.shape)} '
                  f'finite={torch.isfinite(edge_feats).all().item()} '
                  f'min={edge_feats.min().item():.3e} max={edge_feats.max().item():.3e}')
            print(f'  h(type_emb): finite={torch.isfinite(h).all().item()} '
                  f'max={h.abs().max().item():.3e}')
            print(f'  frontier mask 计数: {[int(m.sum().item()) for m in frontier_masks]}')
            print(f'  node_counts: {node_counts}')
            print(f'  n_revealed: {[sg.n_revealed for sg in subgraphs]}')
            for i, cnt in enumerate(frontier_masks):
                if int(cnt.sum()) == 0:
                    st = states[i]
                    src = state_src[i] if state_src else ('?', '?')
                    print(f'  [mask=0 state #{i}] 图index={src[0]} seed_i={src[1]} '
                          f'seed_node={st.seed_node}, '
                          f'n_nodes_total={st.n_nodes_total}, '
                          f'n_revealed={len(st.V_revealed)}, '
                          f'frontier_edges={len(st.frontier_edges)}, '
                          f'get_frontier_nodes()={len(st.get_frontier_nodes())}, '
                          f'budget={st._budget_limit}, used={st.get_budget_used()}, '
                          f'reveal_history={st.reveal_history}')
            print('=' * 60)
            raise RuntimeError('NaN 监测: new_info_encoder 输出 NaN')
        return pooled.unsqueeze(1)  # (N, 1, d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_combined(self, frontier_embeds, subgraph, graph_state):
        """Concatenate GNN embeds + budget_ratio + structural features."""
        n_f = frontier_embeds.shape[0]
        remaining = graph_state.get_budget_ratio_remaining()
        remaining_tensor = torch.full(
            (n_f, 1), remaining,
            device=frontier_embeds.device, dtype=frontier_embeds.dtype,
        )
        struct_all = compute_structural_features(subgraph.g, subgraph.n_revealed)
        struct_frontier = struct_all[subgraph.n_revealed:].to(
            device=frontier_embeds.device, dtype=frontier_embeds.dtype,
        )
        return torch.cat([frontier_embeds, remaining_tensor, struct_frontier], dim=-1)

    # ------------------------------------------------------------------
    # Phase 2 stateful select() — for RevealLoop eval compat
    # ------------------------------------------------------------------

    def select(self, graph_state: GraphState) -> Optional[int]:
        """Belief-aware node selection.  Manages beliefs internally across calls.

        Call reset() before each new rollout.
        """
        if self._belief is None:
            self._belief = self.belief_state.init_belief().to(
                next(self.parameters()).device)

        # Update belief from previous reveal (skip on first call)
        if self._prev_frontier is not None and self._last_selected is not None:
            current_frontier = set(graph_state.get_frontier_nodes())
            new_frontier = list(current_frontier - self._prev_frontier)
            if new_frontier:
                device = self._belief.device
                subgraph = self.frontier_builder.build(graph_state)
                new_info = self.encode_new_info(
                    graph_state, subgraph, self._last_selected, device,
                )
                self._belief = self.update_belief(self._belief, new_info)

        self._prev_frontier = set(graph_state.get_frontier_nodes())

        subgraph = self.frontier_builder.build(graph_state)
        frontier_embeds, frontier_nodes = self.frontier_encoder(subgraph)

        if len(frontier_nodes) == 0:
            return None

        combined = self._build_combined(frontier_embeds, subgraph, graph_state)
        scores = self.frontier_pointer(combined, belief=self._belief.squeeze(0))
        probs = _stable_probs(scores, self.temperature) if self.temperature != 1.0 \
                else _stable_probs(scores)

        if self.stochastic:
            idx = torch.distributions.Categorical(probs).sample().item()
        else:
            idx = probs.argmax().item()
        self._last_selected = frontier_nodes[idx]
        return self._last_selected

    def reset(self):
        self._belief = None
        self._prev_frontier = None
        self._last_selected = None

    @property
    def name(self) -> str:
        return 'attention'

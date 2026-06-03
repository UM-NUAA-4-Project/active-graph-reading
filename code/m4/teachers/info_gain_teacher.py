"""T2: Information Gain Teacher.

At each reveal step, scores frontier candidates by the entropy reduction
they would produce if revealed: IG(v) = H(p_t) - H(p_{t|v}).

Overrides generate_trajectory() because scoring is interleaved with
reveal (not pre-computed). Candidates are batched into a single
classifier forward per step for efficiency.

Reference: BALD (Houlsby et al., NeurIPS 2011); Fuchsgruber et al.,
"Uncertainty for Active Learning on Graphs", ICML 2024.
"""

import logging
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from m4.teachers.base import Teacher
from m4.trajectory import Trajectory

# Ensure active_graph_reader imports work.
import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent.parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder

logger = logging.getLogger(__name__)


class InfoGainTeacher(Teacher):
    """T2: Scores frontier nodes by simulated entropy reduction.

    Parameters
    ----------
    temperature : float
        Softmax temperature. τ < 1 sharpens the distribution and
        amplifies information gain signals (default 1.0 = no sharpening).
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @property
    def name(self) -> str:
        return 'info-gain'

    def _params_dict(self) -> dict:
        return {'temperature': self.temperature}

    # ------------------------------------------------------------------
    # score_nodes (not the primary interface for T2)
    # ------------------------------------------------------------------

    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: nn.Module,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """Not used by T2 -- scoring is per-step in generate_trajectory()."""
        raise NotImplementedError(
            "T2 scores frontier nodes per-step. Use generate_trajectory()."
        )

    # ------------------------------------------------------------------
    # generate_trajectory (overrides base)
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        graph: dgl.DGLGraph,
        classifier: nn.Module,
        graph_idx: int,
        label: int,
        seed_node: int,
        budget: int,
        subgraph_builder: SubgraphBuilder,
        edge_key_to_id: Optional[dict] = None,
    ) -> Trajectory:
        """Generate trajectory using per-step information gain selection."""
        state = GraphState(graph, seed_node=seed_node, edge_key_to_id=edge_key_to_id)
        n_nodes = graph.num_nodes()

        # Record G_0
        sub_g, orig_ids = subgraph_builder.build(state)
        h_current, conf = self._entropy_and_conf(classifier, sub_g)
        node_scores = torch.zeros(n_nodes)
        node_scores[seed_node] = 1.0  # dummy score for seed

        traj = Trajectory(
            graph_idx=graph_idx,
            seed_node=seed_node,
            budget_used=0,
            reveal_order=[],
            scores_per_step=[node_scores[seed_node].item()],
            confidence_per_step=[conf],
            entropy_per_step=[h_current],
            teacher_type=self.name,
            teacher_params=self._params_dict(),
            label=label,
        )

        # Main loop
        while traj.budget_used < budget:
            if state.is_frontier_empty:
                break

            frontier_nodes = list({rv for (_ru, rv) in state.frontier_edges})
            if not frontier_nodes:
                break

            # Build candidate subgraphs: one per frontier node
            candidate_subs = []
            for v in frontier_nodes:
                cand_state = _copy_state(state)
                reveal_node(cand_state, v)
                sub_g, _ = subgraph_builder.build(cand_state)
                candidate_subs.append(sub_g)

            # Batch forward
            batch_g = dgl.batch(candidate_subs).to(next(classifier.parameters()).device)
            batch_node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=batch_g.device)
            with torch.no_grad():
                batch_logits = classifier(batch_g, batch_node_ids)
            entropies = self._batched_entropy(batch_logits)  # (K,)

            # Information gain: H(current) - H(after reveal)
            ig_values = h_current - entropies

            # Select best (highest IG; exclude negative IG candidates)
            best_idx = int(ig_values.argmax().item())
            if ig_values[best_idx] <= 0:
                # No candidate reduces entropy; pick highest-scoring anyway
                pass

            best_v = frontier_nodes[best_idx]
            best_ig = float(ig_values[best_idx].item())
            best_entropy = float(entropies[best_idx].item())

            # Execute actual reveal
            reveal_node(state, best_v)
            traj.reveal_order.append(best_v)
            traj.budget_used += 1
            traj.scores_per_step.append(best_ig)
            traj.entropy_per_step.append(best_entropy)

            # Update current state entropy
            sub_g, orig_ids = subgraph_builder.build(state)
            h_current, conf = self._entropy_and_conf(classifier, sub_g)
            traj.confidence_per_step.append(conf)

            # Store IG as node score for bookkeeping
            node_scores[best_v] = best_ig

        # Final prediction
        sub_g, orig_ids = subgraph_builder.build(state)
        device = next(classifier.parameters()).device
        sub_g = sub_g.to(device)
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = classifier(sub_g, node_ids)
            probs = F.softmax(logits, dim=1)
            pred = int(logits.argmax(dim=1).item())
            traj.cross_entropy = float(-torch.log(probs[0, label] + 1e-12).item())
        traj.final_pred = pred
        traj.is_correct = (pred == label)

        return traj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy_and_conf(classifier, sub_g):
        """Return (entropy, confidence) for a single subgraph."""
        device = next(classifier.parameters()).device
        sub_g = sub_g.to(device)
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = classifier(sub_g, node_ids)
            probs = F.softmax(logits, dim=1)
            conf = float(probs.max(dim=1).values.item())
            ent = float(-(probs * probs.log()).sum(dim=1).item())
        return ent, conf

    def _batched_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute per-graph entropy from batched logits.

        Parameters
        ----------
        logits : (batch_size, n_classes)

        Returns
        -------
        torch.Tensor of shape (batch_size,)
        """
        probs = F.softmax(logits / self.temperature, dim=1)
        # Use natural log for entropy in nats
        return -(probs * probs.log()).sum(dim=1)


# ------------------------------------------------------------------
# Fast GraphState copy (bypasses __init__ to avoid re-running _init_g0)
# ------------------------------------------------------------------

def _copy_state(state: GraphState) -> GraphState:
    """Create a shallow copy of GraphState with copied mutable collections."""
    new_state = GraphState.__new__(GraphState)
    for slot in GraphState.__slots__:
        val = getattr(state, slot)
        setattr(new_state, slot, val)
    # Deep-copy mutable collections (shared DGLGraph reference is safe)
    new_state.V_revealed = set(state.V_revealed)
    new_state.frontier_edges = set(state.frontier_edges)
    new_state.revealed_edge_keys = set(state.revealed_edge_keys)
    new_state._frontier_degree = defaultdict(int, state._frontier_degree)
    new_state._frontier_by_unrevealed = defaultdict(
        set,
        {k: set(v) for k, v in state._frontier_by_unrevealed.items()}
    )
    new_state.reveal_history = list(state.reveal_history)
    # 必须深拷贝排序列表——reveal_node 会就地修改它们
    new_state._sorted_revealed = list(state._sorted_revealed)
    new_state._sorted_frontier_nodes = list(state._sorted_frontier_nodes)
    new_state._revealed_edge_ids = list(state._revealed_edge_ids)
    return new_state

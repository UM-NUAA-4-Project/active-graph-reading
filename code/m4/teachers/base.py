"""Teacher abstract interface and shared trajectory generation.

Each teacher implements score_nodes() to produce per-node importance
scores from the full graph. The base class provides a shared
generate_trajectory() that simulates the reveal process guided by
those scores, reusing M3's GraphState, reveal_node, and SubgraphBuilder.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# Ensure myworkplace/code/ is on sys.path for active_graph_reader imports.
_script_dir = Path(__file__).resolve().parent.parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from m4.trajectory import Trajectory
from m4.graph_meta import GraphMeta

class Teacher(ABC):
    """Abstract teacher that generates graph reading trajectories.

    Subclasses must implement score_nodes() to provide per-node
    importance scores computed from the full graph (and optionally
    the frozen classifier).
    """

    @abstractmethod
    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: nn.Module,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute per-node importance scores from the full graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The complete graph (all nodes/edges visible).
        classifier : nn.Module
            Frozen GraphClassifier for forward/backward queries.
        label : int or None
            Ground-truth label. May be None for teachers that don't
            use label information.

        Returns
        -------
        torch.Tensor
            Per-node scores, shape (|V|,). Higher = more important.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique teacher identifier (e.g. 'gradient', 'structural-degree')."""
        ...

    # ------------------------------------------------------------------
    # Shared trajectory generation
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
    ) -> Trajectory:
        """Simulate a reveal sequence guided by pre-computed node scores.

        1. Compute full-graph node scores via score_nodes().
        2. Reveal seed → repeatedly pick highest-score frontier node
           until budget exhausted or frontier empty.
        3. At each step, run classifier on current subgraph and record
           confidence / entropy.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Complete input graph.
        classifier : nn.Module
            Frozen classifier in eval mode.
        graph_idx : int
            Dataset graph index (for bookkeeping).
        label : int
            Ground-truth class label.
        seed_node : int
            Starting seed node ID.
        budget : int
            Max number of nodes to reveal (excluding seed).
        subgraph_builder : SubgraphBuilder
            Converts GraphState → DGLGraph for classifier forward.

        Returns
        -------
        Trajectory
        """
        # 1. Pre-compute node scores
        node_scores = self.score_nodes(graph, classifier, label)

        # 2. Initialize reveal state (G_0 with seed already revealed)
        state = GraphState(graph, seed_node=seed_node)

        # Record G_0 classifier output
        sub_g, original_ids = subgraph_builder.build(state)
        conf, ent = self._classifier_forward(classifier, sub_g, original_ids)

        traj = Trajectory(
            graph_idx=graph_idx,
            seed_node=seed_node,
            budget_used=0,
            reveal_order=[],
            scores_per_step=[],
            confidence_per_step=[conf],
            entropy_per_step=[ent],
            teacher_type=self.name,
            teacher_params=self._params_dict(),
            label=label,
        )

        # 3. Main reveal loop
        while traj.budget_used < budget:
            if state.is_frontier_empty:
                break

            # Pick frontier node with highest pre-computed score
            next_node = self._pick_best_frontier(state, node_scores)
            if next_node is None:
                break

            # Execute reveal
            reveal_node(state, next_node)
            traj.reveal_order.append(next_node)
            traj.budget_used += 1

            # Record teacher score for this step
            traj.scores_per_step.append(float(node_scores[next_node].item()))

            # Classifier forward on updated subgraph
            sub_g, original_ids = subgraph_builder.build(state)
            conf, ent = self._classifier_forward(classifier, sub_g, original_ids)
            traj.confidence_per_step.append(conf)
            traj.entropy_per_step.append(ent)

        # 4. Final prediction check
        sub_g, original_ids = subgraph_builder.build(state)
        pred = self._predict(classifier, sub_g, original_ids)
        traj.final_pred = pred
        traj.is_correct = (pred == label)

        return traj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best_frontier(
        state: 'GraphState',
        node_scores: torch.Tensor,
    ) -> Optional[int]:
        """Select the frontier node with highest score (vectorized).

        Parameters
        ----------
        state : GraphState
            Current reveal state; uses get_frontier_nodes() for unique frontier.
        node_scores : tensor of shape (|V|,).

        Returns
        -------
        int or None
        """
        frontier_nodes = state.get_frontier_nodes()
        if not frontier_nodes:
            return None
        frontier_t = torch.tensor(frontier_nodes, dtype=torch.long)
        scores = node_scores[frontier_t]
        best_idx = int(scores.argmax().item())
        return frontier_nodes[best_idx]

    @staticmethod
    def _classifier_forward(classifier, sub_g, original_ids):
        """Run classifier on subgraph, return (confidence, entropy)."""
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=sub_g.device)
        with torch.no_grad():
            logits = classifier(sub_g, node_ids)
            probs = F.softmax(logits, dim=1)
            conf = float(probs.max(dim=1).values.item())
            ent = float(-(probs * probs.log()).sum(dim=1).item())
        return conf, ent

    @staticmethod
    def _predict(classifier, sub_g, original_ids) -> int:
        """Return predicted class index."""
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=sub_g.device)
        with torch.no_grad():
            logits = classifier(sub_g, node_ids)
            return int(logits.argmax(dim=1).item())

    def generate_trajectory_from_cache(
        self,
        meta: 'GraphMeta',
        graph: dgl.DGLGraph,
        scores: torch.Tensor,
        seed_node: int,
        budget: int,
        subgraph_builder,
        teacher_name: str,
    ):
        """Generate trajectory using pre-computed scores. No GNN calls.

        Parameters
        ----------
        meta : GraphMeta
            Pre-computed graph metadata (label, edge_key_to_id, etc.).
        graph : dgl.DGLGraph
            Full graph with self-loops added.
        scores : torch.Tensor
            Per-node importance scores, shape (n_nodes,).
        seed_node : int
            Starting seed node ID.
        budget : int
            Max number of nodes to reveal (excluding seed).
        subgraph_builder : SubgraphBuilder
        teacher_name : str
            Teacher identifier for bookkeeping.

        Returns
        -------
        (Trajectory, sub_g) — trajectory without final_pred/is_correct yet,
        and the final DGL subgraph for batched classifier evaluation.
        """
        from m4.trajectory import Trajectory

        state = GraphState(
            graph,
            seed_node=seed_node,
            edge_key_to_id=meta.edge_key_to_id,
        )

        reveal_order = []
        for _ in range(budget):
            if state.is_frontier_empty:
                break
            next_node = self._pick_best_frontier(state, scores)
            if next_node is None:
                break
            reveal_node(state, next_node)
            reveal_order.append(next_node)

        sub_g, _ = subgraph_builder.build(state)

        traj = Trajectory(
            graph_idx=meta.graph_idx,
            seed_node=seed_node,
            budget_used=len(reveal_order),
            reveal_order=reveal_order,
            teacher_type=teacher_name,
            teacher_params={},
            label=meta.label,
        )
        return traj, sub_g

    def _params_dict(self) -> dict:
        """Return teacher-specific parameters for bookkeeping."""
        return {}

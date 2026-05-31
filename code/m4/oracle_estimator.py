"""Oracle upper-bound estimation.

Estimates the best achievable accuracy under budget B by running
label-aware (Type I) and gradient-aware (Type II) oracle teachers.
Compares against baseline strategies (random, BFS) to compute
headroom — the gap between current baselines and theoretical maxima.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import dgl

from m4.trajectory import Trajectory

logger = logging.getLogger(__name__)


class OracleEstimator:
    """Estimates oracle performance upper bounds.

    Parameters
    ----------
    classifier : nn.Module
        Frozen GraphClassifier in eval mode.
    subgraph_builder : SubgraphBuilder
    device : str
    """

    def __init__(self, classifier, subgraph_builder, device='cpu'):
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.subgraph_builder = subgraph_builder
        self.device = device

    # ------------------------------------------------------------------
    # Type I: Label-aware oracle
    # ------------------------------------------------------------------

    def estimate_type1(
        self,
        graph: dgl.DGLGraph,
        label: int,
        seed_node: int,
        budget: int,
    ) -> Tuple[bool, float]:
        """Label-aware oracle: at each step, pick the frontier node that
        maximizes the correct-class probability.

        This is the absolute upper bound — it uses label information
        that would not be available at test time.

        Returns
        -------
        (is_correct, final_confidence)
        """
        import sys
        from pathlib import Path
        _script_dir = Path(__file__).resolve().parent.parent
        if str(_script_dir) not in sys.path:
            sys.path.insert(0, str(_script_dir))

        from active_graph_reader.engine.graph_state import GraphState
        from active_graph_reader.engine.reveal_action import reveal_node

        state = GraphState(graph, seed_node=seed_node)

        for _ in range(budget):
            if state.is_frontier_empty:
                break

            frontier_nodes = list({rv for (_ru, rv) in state.frontier_edges})
            if not frontier_nodes:
                break

            # Simulate each candidate: build subgraph, get correct-class prob
            best_v = None
            best_prob = -1.0

            for v in frontier_nodes:
                from m4.teachers.info_gain_teacher import _copy_state
                cand_state = _copy_state(state)
                reveal_node(cand_state, v)

                sub_g, orig_ids = self.subgraph_builder.build(cand_state)
                prob = self._correct_class_prob(sub_g, label)

                if prob > best_prob:
                    best_prob = prob
                    best_v = v

            if best_v is None:
                break

            reveal_node(state, best_v)

        # Final evaluation
        sub_g, orig_ids = self.subgraph_builder.build(state)
        prob = self._correct_class_prob(sub_g, label)
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long)
        with torch.no_grad():
            logits = self.classifier(sub_g, node_ids)
            pred = int(logits.argmax(dim=1).item())

        return (pred == label), prob

    # ------------------------------------------------------------------
    # Type II: Gradient-aware oracle (delegates to T1 teacher)
    # ------------------------------------------------------------------

    def estimate_type2(
        self,
        trajectories: List[Trajectory],
        teacher_name: str = 'gradient',
    ) -> Dict[int, float]:
        """Estimate learnable upper bound from gradient teacher trajectories.

        Groups trajectories by budget, computes mean accuracy.

        Returns
        -------
        dict of {budget: accuracy}
        """
        by_budget = defaultdict(list)
        for traj in trajectories:
            if traj.teacher_type == teacher_name and traj.is_correct:
                by_budget[traj.budget_used].append(1.0)
            elif traj.teacher_type == teacher_name:
                by_budget[traj.budget_used].append(0.0)

        return {
            budget: np.mean(accs)
            for budget, accs in sorted(by_budget.items())
        }

    # ------------------------------------------------------------------
    # Headroom analysis
    # ------------------------------------------------------------------

    def headroom_analysis(
        self,
        oracle_acc: Dict[int, float],
        bfs_acc: Dict[int, float],
        random_acc: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """Compute headroom gaps between oracle and baselines.

        Parameters
        ----------
        oracle_acc : {budget: accuracy}
            From estimate_type2().
        bfs_acc : {budget: accuracy}
            BFS baseline accuracy per budget.
        random_acc : {budget: accuracy} or None
            Random baseline accuracy per budget.

        Returns
        -------
        dict with keys: 'headroom_bfs', 'headroom_random', 'budgets',
                        'max_headroom_bfs', 'max_headroom_random', 'verdict'
        """
        budgets = sorted(oracle_acc.keys())
        headroom_bfs = {}
        headroom_random = {}

        for b in budgets:
            if b in bfs_acc:
                headroom_bfs[b] = oracle_acc[b] - bfs_acc[b]
            if random_acc and b in random_acc:
                headroom_random[b] = oracle_acc[b] - random_acc[b]

        max_bfs = max(headroom_bfs.values()) if headroom_bfs else 0.0
        max_random = max(headroom_random.values()) if headroom_random else 0.0

        if max_bfs > 0.05:
            verdict = "significant_headroom"
        elif max_bfs > 0.02:
            verdict = "moderate_headroom"
        else:
            verdict = "limited_headroom"

        return {
            'headroom_bfs': headroom_bfs,
            'headroom_random': headroom_random,
            'budgets': budgets,
            'max_headroom_bfs': max_bfs,
            'max_headroom_random': max_random,
            'verdict': verdict,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _correct_class_prob(self, sub_g: dgl.DGLGraph, label: int) -> float:
        """Return softmax probability of the correct class."""
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long)
        with torch.no_grad():
            logits = self.classifier(sub_g, node_ids)
            probs = F.softmax(logits, dim=1)
            return float(probs[0, label].item())

"""Trajectory quality scorer.

Computes 4-dimensional quality scores for each trajectory:
1. Final Accuracy (binary: correct or not)
2. Budget Efficiency (accuracy per unit budget, normalized by graph size)
3. Information Gain Sum (cumulative entropy reduction)
4. Stability (Jaccard similarity of node sets across same-graph trajectories)

Composite score: Score = Acc × (w1·BE + w2·IG + w3·Stability)
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from m4.trajectory import Trajectory

logger = logging.getLogger(__name__)

EPS = 1e-6


class TrajectoryScorer:
    """Scores trajectories on 4 quality dimensions.

    Parameters
    ----------
    w_acc_budget : float
        Weight for budget efficiency in composite score.
    w_ig : float
        Weight for information gain sum in composite score.
    w_stability : float
        Weight for stability in composite score.
    """

    def __init__(
        self,
        w_be: float = 0.5,
        w_ig: float = 0.3,
        w_stability: float = 0.2,
    ):
        self.w_be = w_be
        self.w_ig = w_ig
        self.w_stability = w_stability

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all(
        self,
        trajectories: List[Trajectory],
        graph_sizes: Dict[int, int] = None,
    ) -> List[Dict]:
        """Score all trajectories and return scored records.

        Each returned dict contains:
          graph_idx, teacher_type, budget_used, is_correct,
          final_acc, budget_efficiency, ig_sum, ig_sum_norm,
          stability, total_score, plus the original Trajectory under 'traj'.

        Parameters
        ----------
        trajectories : list of Trajectory
        graph_sizes : dict of {graph_idx: n_nodes_total}
            Needed for normalized budget efficiency. If None, inferred
            from trajectory reveal data (may be approximate).

        Returns
        -------
        list of dict
        """
        # Compute per-trajectory scores (except stability)
        scored = []
        for traj in trajectories:
            acc = 1.0 if traj.is_correct else 0.0
            n_nodes = graph_sizes.get(traj.graph_idx, traj.budget_used + 1) if graph_sizes else traj.budget_used + 1
            be = acc / (traj.budget_used / max(n_nodes, 1) + EPS)

            ig_sum = self._compute_ig_sum(traj)
            ig_norm = ig_sum / max(traj.budget_used, 1)

            scored.append({
                'graph_idx': traj.graph_idx,
                'teacher_type': traj.teacher_type,
                'budget_used': traj.budget_used,
                'is_correct': traj.is_correct,
                'final_acc': acc,
                'budget_efficiency': be,
                'ig_sum': ig_sum,
                'ig_sum_norm': ig_norm,
                'stability': 0.0,  # filled in next
                'total_score': 0.0,  # filled in next
                'traj': traj,
            })

        # Compute stability: group by graph_idx
        by_graph = defaultdict(list)
        for i, rec in enumerate(scored):
            by_graph[rec['graph_idx']].append(i)

        for graph_idx, indices in by_graph.items():
            if len(indices) < 2:
                # Single trajectory for this graph → perfect stability
                for idx in indices:
                    scored[idx]['stability'] = 1.0
                continue

            # Compute pairwise Jaccard of revealed node sets
            node_sets = []
            for idx in indices:
                traj = scored[idx]['traj']
                nodes = set(traj.reveal_order) | {traj.seed_node}
                node_sets.append(nodes)

            for i, idx in enumerate(indices):
                stabilities = []
                for j in range(len(indices)):
                    if i == j:
                        continue
                    inter = len(node_sets[i] & node_sets[j])
                    union = len(node_sets[i] | node_sets[j])
                    jac = inter / max(union, 1)
                    stabilities.append(jac)
                scored[idx]['stability'] = float(np.mean(stabilities)) if stabilities else 1.0

        # Compute composite score
        for rec in scored:
            if rec['final_acc'] == 0:
                rec['total_score'] = 0.0
            else:
                ig_norm_clipped = min(rec['ig_sum_norm'], 5.0)  # cap outliers
                # Normalize IG to [0, 1] range (typical IG per step is 0-2 nats)
                ig_scaled = ig_norm_clipped / 5.0

                rec['total_score'] = (
                    self.w_be * rec['budget_efficiency']
                    + self.w_ig * ig_scaled
                    + self.w_stability * rec['stability']
                )

        return scored

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ig_sum(traj: Trajectory) -> float:
        """Sum of entropy reductions across the trajectory.

        IG_sum = Σ_{t=1}^{B} [H(p_{t-1}) - H(p_t)]
        """
        if len(traj.entropy_per_step) < 2:
            return 0.0

        ig_total = 0.0
        for t in range(1, len(traj.entropy_per_step)):
            drop = traj.entropy_per_step[t - 1] - traj.entropy_per_step[t]
            if drop > 0:
                ig_total += drop
            # Negative drops (entropy increased) are ignored

        return ig_total


def compute_stability_matrix(trajectories: List[Trajectory]) -> np.ndarray:
    """Compute pairwise Jaccard stability matrix for trajectories.

    Returns (N, N) matrix where entry (i,j) = Jaccard(reveal_i, reveal_j).
    """
    n = len(trajectories)
    node_sets = [
        set(t.reveal_order) | {t.seed_node}
        for t in trajectories
    ]

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                inter = len(node_sets[i] & node_sets[j])
                union = len(node_sets[i] | node_sets[j])
                matrix[i, j] = inter / max(union, 1)

    return matrix

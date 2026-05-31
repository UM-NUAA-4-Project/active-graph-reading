"""Distillation feasibility analysis.

Computes the Distillation Feasibility Index (DFI) to determine whether
behavior cloning from teacher trajectories is viable for M5, or whether
end-to-end training is a better route.

DFI = 0.4 × relative_oracle_gain + 0.3 × stability + 0.3 × (1 - variance)
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from m4.trajectory import Trajectory

logger = logging.getLogger(__name__)

EPS = 1e-6


class FeasibilityAnalyzer:
    """Analyzes whether trajectory distillation is viable for M5.

    Parameters
    ----------
    dfi_threshold_high : float
        DFI above this → BC recommended.
    dfi_threshold_low : float
        DFI below this → skip BC, go end-to-end.
    """

    def __init__(
        self,
        dfi_threshold_high: float = 0.5,
        dfi_threshold_low: float = 0.3,
    ):
        self.dfi_threshold_high = dfi_threshold_high
        self.dfi_threshold_low = dfi_threshold_low

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        oracle_acc: Dict[int, float],
        bfs_acc: Dict[int, float],
        scored_trajectories: Optional[List[Dict]] = None,
        trajectories: Optional[List[Trajectory]] = None,
    ) -> Dict:
        """Compute DFI and M5 route recommendation.

        Parameters
        ----------
        oracle_acc : {budget: accuracy}
            Oracle Type II accuracy per budget.
        bfs_acc : {budget: accuracy}
            BFS baseline accuracy per budget.
        scored_trajectories : list of dict or None
            From TrajectoryScorer.score_all(). Used for stability/variance.
        trajectories : list of Trajectory or None
            Raw trajectories. Used for stability when scored not available.

        Returns
        -------
        dict with keys: 'dfi', 'recommendation', 'components', 'details'
        """
        # 1. Relative oracle gain
        rel_gain = self._relative_gain(oracle_acc, bfs_acc)

        # 2. Teacher stability
        stability = self._compute_stability(scored_trajectories, trajectories)

        # 3. Accuracy variance across trajectories (1 - variance)
        acc_variance = self._compute_acc_variance(scored_trajectories, trajectories)
        low_variance_score = 1.0 - min(acc_variance, 1.0)

        # 4. DFI
        dfi = (
            0.4 * rel_gain
            + 0.3 * stability
            + 0.3 * low_variance_score
        )

        # 5. Recommendation
        if dfi > self.dfi_threshold_high:
            recommendation = (
                "BC_PRETRAIN_RECOMMENDED: Trajectory distillation is viable. "
                "Use BC pretraining (M5a) followed by Gumbel-Softmax fine-tuning (M5b)."
            )
        elif dfi > self.dfi_threshold_low:
            recommendation = (
                "BC_FEASIBLE_WITH_FILTERING: BC pretraining may work with strict "
                "trajectory filtering. Consider training on top-1 trajectory per graph only."
            )
        else:
            recommendation = (
                "SKIP_BC_GO_END_TO_END: Trajectory distillation unlikely to help. "
                "Skip BC pretraining and go directly to end-to-end training (M5b)."
            )

        result = {
            'dfi': round(dfi, 4),
            'recommendation': recommendation,
            'components': {
                'relative_gain': round(rel_gain, 4),
                'stability': round(stability, 4),
                'acc_variance': round(acc_variance, 4),
                'low_variance_score': round(low_variance_score, 4),
            },
            'details': {
                'dfi_threshold_high': self.dfi_threshold_high,
                'dfi_threshold_low': self.dfi_threshold_low,
                'oracle_acc': oracle_acc,
                'bfs_acc': bfs_acc,
            },
        }

        logger.info("DFI analysis complete: DFI=%.4f → %s", dfi, recommendation.split(':')[0])
        return result

    # ------------------------------------------------------------------
    # Component computations
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_gain(
        oracle_acc: Dict[int, float],
        bfs_acc: Dict[int, float],
    ) -> float:
        """Compute relative gain: (Oracle - BFS) / max(BFS, ε), averaged over budgets."""
        gains = []
        for b, o_acc in oracle_acc.items():
            b_acc = bfs_acc.get(b, 0.0)
            gain = (o_acc - b_acc) / max(b_acc, EPS)
            gains.append(gain)
        return float(np.mean(gains)) if gains else 0.0

    @staticmethod
    def _compute_stability(
        scored: Optional[List[Dict]],
        trajectories: Optional[List[Trajectory]],
    ) -> float:
        """Average stability score across all trajectories."""
        if scored:
            stabilities = [r.get('stability', 0.0) for r in scored]
            return float(np.mean(stabilities)) if stabilities else 0.0

        if trajectories:
            # Compute Jaccard per graph
            by_graph = defaultdict(list)
            for t in trajectories:
                by_graph[t.graph_idx].append(t)

            jaccards = []
            for graph_idx, trajs in by_graph.items():
                if len(trajs) < 2:
                    jaccards.append(1.0)
                    continue
                node_sets = [set(t.reveal_order) | {t.seed_node} for t in trajs]
                for i in range(len(trajs)):
                    for j in range(i + 1, len(trajs)):
                        inter = len(node_sets[i] & node_sets[j])
                        union = len(node_sets[i] | node_sets[j])
                        jaccards.append(inter / max(union, 1))

            return float(np.mean(jaccards)) if jaccards else 0.0

        return 0.0

    @staticmethod
    def _compute_acc_variance(
        scored: Optional[List[Dict]],
        trajectories: Optional[List[Trajectory]],
    ) -> float:
        """Variance of accuracy across trajectories of the same graph."""
        if scored:
            by_graph = defaultdict(list)
            for r in scored:
                by_graph[r['graph_idx']].append(1.0 if r['is_correct'] else 0.0)

            variances = []
            for graph_idx, accs in by_graph.items():
                if len(accs) > 1:
                    variances.append(float(np.var(accs)))
            return float(np.mean(variances)) if variances else 0.0

        if trajectories:
            by_graph = defaultdict(list)
            for t in trajectories:
                by_graph[t.graph_idx].append(1.0 if t.is_correct else 0.0)

            variances = []
            for graph_idx, accs in by_graph.items():
                if len(accs) > 1:
                    variances.append(float(np.var(accs)))
            return float(np.mean(variances)) if variances else 0.0

        return 0.0

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def save_report(self, result: Dict, path: str):
        """Save DFI analysis report as JSON."""
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info("DFI report saved to %s", path)

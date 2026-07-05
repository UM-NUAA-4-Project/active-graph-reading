"""Trajectory filter: quality-based selection with diversity guarantees.

Filters scored trajectories to keep only high-quality ones for M5
behavior cloning training. Ensures each graph contributes diverse
trajectories from different teacher types.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set

from m4.trajectory import Trajectory

logger = logging.getLogger(__name__)


class TrajectoryFilter:
    """Filters scored trajectories: discard errors, keep top-K per graph.

    Parameters
    ----------
    top_k : int
        Maximum trajectories to keep per graph.
    require_diversity : bool
        If True, ensure at least one trajectory from each distinct
        teacher_type is retained (up to top_k limit).
    """

    def __init__(self, top_k: int = 3, require_diversity: bool = True):
        self.top_k = top_k
        self.require_diversity = require_diversity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, scored_trajectories: List[Dict]) -> List[Trajectory]:
        """Filter scored trajectories and return clean Trajectory list.

        Parameters
        ----------
        scored_trajectories : list of dict
            Output from TrajectoryScorer.score_all(). Each dict must have
            keys: 'graph_idx', 'teacher_type', 'final_acc', 'total_score',
            and 'traj'.

        Returns
        -------
        list of Trajectory
        """
        # Step 1: Discard trajectories with wrong final prediction
        correct = [r for r in scored_trajectories if r['final_acc'] > 0]
        n_discarded = len(scored_trajectories) - len(correct)
        if n_discarded > 0:
            logger.info(
                "Discarded %d trajectories with incorrect predictions (%.1f%%)",
                n_discarded,
                100.0 * n_discarded / max(len(scored_trajectories), 1),
            )

        # Step 2: Group by graph_idx
        by_graph = defaultdict(list)
        for rec in correct:
            by_graph[rec['graph_idx']].append(rec)

        # Step 3 & 4: Sort by score, keep top-K with diversity
        kept: List[Trajectory] = []
        diversity_stats = {'diverse': 0, 'single_teacher': 0}

        for graph_idx, recs in by_graph.items():
            # Sort by total_score descending
            recs.sort(key=lambda r: r['total_score'], reverse=True)
            selected = self._select_top_k(recs)
            kept.extend(selected)

            # Track diversity
            teacher_types = {r['teacher_type'] for r in recs[:len(selected)]}
            if len(teacher_types) > 1:
                diversity_stats['diverse'] += 1
            else:
                diversity_stats['single_teacher'] += 1

        logger.info(
            "Filter result: %d trajectories kept from %d graphs. "
            "Diverse: %d, Single-teacher: %d",
            len(kept), len(by_graph),
            diversity_stats['diverse'], diversity_stats['single_teacher'],
        )

        return kept

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def _select_top_k(self, scored_recs: List[Dict]) -> List[Trajectory]:
        """Select up to top_k trajectories with diversity guarantee.

        Strategy:
        1. Always include the top-scoring trajectory.
        2. Try to include at least one from each teacher type.
        3. Fill remaining slots with next-highest-scoring trajectories.
        """
        if len(scored_recs) <= self.top_k:
            return [r['traj'] for r in scored_recs]

        selected: List[Dict] = []
        used_teachers: Set[str] = set()
        remaining = list(scored_recs)

        # Always take the top scorer first
        selected.append(remaining.pop(0))
        used_teachers.add(selected[0]['teacher_type'])

        # Diversity pass: try to include one from each remaining teacher type
        if self.require_diversity:
            all_types = {r['teacher_type'] for r in remaining}
            for ttype in sorted(all_types):
                if len(selected) >= self.top_k:
                    break
                if ttype in used_teachers:
                    continue
                # Find highest-scoring trajectory of this type
                for i, r in enumerate(remaining):
                    if r['teacher_type'] == ttype:
                        selected.append(remaining.pop(i))
                        used_teachers.add(ttype)
                        break

        # Fill remaining slots with next-highest-scoring
        while len(selected) < self.top_k and remaining:
            selected.append(remaining.pop(0))

        return [r['traj'] for r in selected]

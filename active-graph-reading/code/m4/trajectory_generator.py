"""Batch trajectory generator with checkpoint/resume support.

Orchestrates multiple teachers across many graphs to produce raw
reading trajectories for M4 analysis and M5 training.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import dgl
from tqdm import tqdm

from m4.teachers.base import Teacher
from m4.trajectory import Trajectory

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Generates trajectories from multiple teachers on a dataset.

    Parameters
    ----------
    teachers : list of Teacher
        Teacher instances to generate trajectories from.
    classifier : nn.Module
        Frozen GraphClassifier in eval mode.
    subgraph_builder : SubgraphBuilder
        Converts GraphState to DGLGraph.
    dataset : PPADataset
        OGBG-PPA dataset wrapper.
    device : str or torch.device
        Device for classifier forward/backward.
    """

    def __init__(
        self,
        teachers: List[Teacher],
        classifier,
        subgraph_builder,
        dataset,
        device='cpu',
    ):
        self.teachers = teachers
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.subgraph_builder = subgraph_builder
        self.dataset = dataset
        self.device = device

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate_all(
        self,
        indices: Sequence[int],
        budgets: Sequence[int],
        seeds_per_graph: int = 3,
        output_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        save_interval: int = 500,
        seed_offset: int = 0,
    ) -> List[Trajectory]:
        """Generate trajectories for a set of graphs.

        Parameters
        ----------
        indices : sequence of int
            Dataset graph indices to process.
        budgets : sequence of int
            Budget values to use (each teacher × each budget).
        seeds_per_graph : int
            Number of random seed nodes per graph.
        output_dir : str or None
            Directory for saving raw trajectories incrementally.
        checkpoint_file : str or None
            JSON manifest for resume support.
        save_interval : int
            Save checkpoint every N trajectories.
        seed_offset : int
            Offset added to random seed for determinism control.

        Returns
        -------
        list of Trajectory
        """
        # Resume from checkpoint if available
        completed = _load_checkpoint(checkpoint_file) if checkpoint_file else set()
        all_trajs: List[Trajectory] = []

        n_total = len(indices)
        n_teachers = len(self.teachers)
        n_budgets = len(budgets)
        est_total = n_total * n_teachers * n_budgets * seeds_per_graph

        logger.info(
            "Generating trajectories: %d graphs × %d teachers × %d budgets × %d seeds "
            "≈ %d total (resuming from %d completed)",
            n_total, n_teachers, n_budgets, seeds_per_graph,
            est_total, len(completed),
        )

        pbar = tqdm(total=est_total, desc="trajectories")
        generated = 0
        last_save = 0

        for graph_idx in indices:
            # Add self-loops (matching full-graph training convention)
            g_raw, label = self.dataset[graph_idx]
            g = dgl.add_self_loop(g_raw)

            n_nodes = g.num_nodes()
            rng = random.Random(graph_idx * 31 + seed_offset * 17)

            for seed_i in range(seeds_per_graph):
                seed_node = rng.randint(0, max(n_nodes - 1, 0))

                for teacher in self.teachers:
                    for budget in budgets:
                        traj_key = (
                            graph_idx, seed_node, budget,
                            teacher.name, str(teacher._params_dict())
                        )
                        if traj_key in completed:
                            pbar.update(1)
                            continue

                        try:
                            traj = teacher.generate_trajectory(
                                graph=g,
                                classifier=self.classifier,
                                graph_idx=graph_idx,
                                label=label,
                                seed_node=seed_node,
                                budget=budget,
                                subgraph_builder=self.subgraph_builder,
                            )
                        except Exception as e:
                            logger.error(
                                "Trajectory failed: graph=%d teacher=%s budget=%d: %s",
                                graph_idx, teacher.name, budget, e,
                            )
                            pbar.update(1)
                            continue

                        all_trajs.append(traj)
                        completed.add(traj_key)
                        generated += 1
                        pbar.update(1)

                        # Incremental save
                        if output_dir and generated - last_save >= save_interval:
                            _save_trajs(all_trajs, output_dir)
                            if checkpoint_file:
                                _save_checkpoint(completed, checkpoint_file)
                            last_save = generated

        pbar.close()

        # Final save
        if output_dir:
            _save_trajs(all_trajs, output_dir)
            if checkpoint_file:
                _save_checkpoint(completed, checkpoint_file)

        logger.info("Generated %d trajectories total", len(all_trajs))
        return all_trajs


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def _load_checkpoint(path: str) -> set:
    """Load set of completed trajectory keys from JSON manifest."""
    if not path or not Path(path).exists():
        return set()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Keys are stored as lists; convert to tuples
        return {tuple(k) for k in data.get('completed', [])}
    except (json.JSONDecodeError, KeyError):
        logger.warning("Corrupt checkpoint file, starting fresh")
        return set()


def _save_checkpoint(completed: set, path: str):
    """Save completed trajectory keys to JSON manifest."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'completed': [list(k) for k in completed], 'count': len(completed)}, f)


def _save_trajs(trajs: List[Trajectory], output_dir: str):
    """Save trajectories to disk as a .pt file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fname = out_path / f"raw_trajectories_{timestamp}.pt"
    torch.save(trajs, fname)
    logger.debug("Saved %d trajectories to %s", len(trajs), fname)

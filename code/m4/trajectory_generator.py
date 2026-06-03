"""Batch trajectory generator with checkpoint/resume support.

Provides two generation modes:
1. generate_all() — legacy: per-trajectory teacher calls with per-step GNN
2. generate_all_phased() — optimized 3-phase pipeline with:
   - Phase 1: Pre-compute GraphMeta (shared scores)
   - Phase 2: Generate + batch-evaluate in chunks
   - Phase 3: Filter top-k per graph
   - Phase 3b: Retry deficient graphs (more seeds, larger budget) until
     min_correct_per_graph is reached (padded with incorrect if exhausted)
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm
from m4.graph_meta import GraphMeta
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

    # ==================================================================
    # Legacy: per-trajectory generation (backward compatible)
    # ==================================================================

    def generate_all(
        self,
        indices: Sequence[int],
        budgets: Sequence[int],
        seeds_per_graph: int = 3,
        output_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        save_interval: int = 500,
    ) -> List[Trajectory]:
        """Generate trajectories for a set of graphs (legacy mode).

        Parameters
        ----------
        indices : sequence of int
            Dataset graph indices to process.
        budgets : sequence of int
            Budget values to use (each teacher x each budget).
        seeds_per_graph : int
            Number of random seed nodes per graph.
        output_dir : str or None
            Directory for saving raw trajectories incrementally.
        checkpoint_file : str or None
            JSON manifest for resume support.
        save_interval : int
            Save checkpoint every N trajectories.

        Returns
        -------
        list of Trajectory
        """
        completed = _load_checkpoint(checkpoint_file) if checkpoint_file else set()
        all_trajs: List[Trajectory] = []

        n_total = len(indices)
        n_teachers = len(self.teachers)
        n_budgets = len(budgets)
        est_total = n_total * n_teachers * n_budgets * seeds_per_graph

        logger.info(
            "Generating trajectories: %d graphs x %d teachers x %d budgets x %d seeds "
            "= %d total (resuming from %d completed)",
            n_total, n_teachers, n_budgets, seeds_per_graph,
            est_total, len(completed),
        )

        pbar = tqdm(total=est_total, desc="trajectories")
        generated = 0
        last_save = 0

        max_budget = max(budgets)

        for graph_idx in indices:
            g_raw, label = self.dataset[graph_idx]
            g = dgl.add_self_loop(g_raw)

            n_nodes = g.num_nodes()
            rng = random.Random(graph_idx * 31)

            for seed_i in range(seeds_per_graph):
                seed_node = rng.randint(0, max(n_nodes - 1, 0))

                for teacher in self.teachers:
                    traj_key = (
                        graph_idx, seed_node, max_budget,
                        teacher.name, str(teacher._params_dict())
                    )
                    if traj_key in completed:
                        pbar.update(len(budgets))
                        continue

                    try:
                        full_traj = teacher.generate_trajectory(
                            graph=g,
                            classifier=self.classifier,
                            graph_idx=graph_idx,
                            label=label,
                            seed_node=seed_node,
                            budget=max_budget,
                            subgraph_builder=self.subgraph_builder,
                        )
                    except Exception as e:
                        logger.error(
                            "Trajectory failed: graph=%d teacher=%s budget=%d: %s",
                            graph_idx, teacher.name, max_budget, e,
                        )
                        pbar.update(len(budgets))
                        continue

                    all_trajs.append(full_traj)
                    completed.add(traj_key)
                    generated += 1
                    pbar.update(1)

                    for budget in budgets:
                        if budget >= max_budget:
                            continue
                        try:
                            truncated = self._truncate_trajectory(
                                full_traj, budget, g, seed_node,
                            )
                            all_trajs.append(truncated)
                            generated += 1
                            pbar.update(1)
                        except Exception as e:
                            logger.error(
                                "Truncate failed: graph=%d teacher=%s budget=%d: %s",
                                graph_idx, teacher.name, budget, e,
                            )
                            pbar.update(1)

                    if output_dir and generated - last_save >= save_interval:
                        _save_trajs(all_trajs, output_dir)
                        if checkpoint_file:
                            _save_checkpoint(completed, checkpoint_file)
                        last_save = generated

        pbar.close()

        if output_dir:
            _save_trajs(all_trajs, output_dir)
            if checkpoint_file:
                _save_checkpoint(completed, checkpoint_file)

        logger.info("Generated %d trajectories total", len(all_trajs))
        return all_trajs

    def _truncate_trajectory(
        self,
        full_traj: Trajectory,
        budget: int,
        graph: dgl.DGLGraph,
        seed_node: int,
    ) -> Trajectory:
        """Truncate a max-budget trajectory to a smaller budget.

        Replays the reveal sequence up to ``budget`` steps and runs one
        classifier forward to get final_pred/is_correct/cross_entropy at
        the intermediate state.
        """
        from active_graph_reader.engine.graph_state import GraphState
        from active_graph_reader.engine.reveal_action import reveal_node

        trimmed = full_traj.reveal_order[:budget]

        state = GraphState(graph, seed_node=seed_node)
        for v in trimmed:
            reveal_node(state, v)

        sub_g, _ = self.subgraph_builder.build(state)
        sub_g = sub_g.to(self.device)
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.classifier(sub_g, node_ids)
            probs = F.softmax(logits, dim=1)
            pred = int(logits.argmax(dim=1).item())
            ce = float(-torch.log(probs[0, full_traj.label] + 1e-12).item())

        return Trajectory(
            graph_idx=full_traj.graph_idx,
            seed_node=seed_node,
            budget_used=budget,
            reveal_order=list(trimmed),
            scores_per_step=full_traj.scores_per_step[:budget],
            confidence_per_step=full_traj.confidence_per_step[:budget + 1],
            entropy_per_step=full_traj.entropy_per_step[:budget + 1],
            final_pred=pred,
            is_correct=(pred == full_traj.label),
            cross_entropy=ce,
            teacher_type=full_traj.teacher_type,
            teacher_params=full_traj.teacher_params,
            label=full_traj.label,
        )

    # ==================================================================
    # Phased pipeline (optimized)
    # ==================================================================

    def generate_all_phased(
        self,
        indices: List[int],
        budget_ratio: float = 0.15,
        seeds_per_graph: int = 5,
        top_k: int = 3,
        min_correct_per_graph: Optional[int] = None,
        max_retries: int = 3,
        seeds_increment: int = 3,
        output_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        chunk_size: int = 500,
        skip_t2: bool = True,
        resume_from_cache: Optional[str] = None,
        cache_dir: Optional[str] = None,
        precomputed_metas: Optional[Dict[int, 'GraphMeta']] = None,
    ) -> List[Trajectory]:
        """Generate trajectories using the 3-phase optimized pipeline.

        Phase 1: Pre-compute GraphMeta (T1 gradient + T3 structural scores).
        Phase 2: Generate + batch-evaluate trajectories in chunks.
        Phase 3: Filter top-k per graph by CrossEntropy.
        Phase 3b: Retry deficient graphs with more seeds/larger budget until
                  min_correct_per_graph correct trajectories are obtained.

        Parameters
        ----------
        indices : list of int
            Dataset graph indices to process.
        budget_ratio : float
            Budget as fraction of |V| (e.g. 0.15 = 15% of nodes).
        seeds_per_graph : int
            Number of random seed nodes per graph (initial round).
        top_k : int
            Max trajectories per graph after filtering.
        min_correct_per_graph : int or None
            Minimum correct trajectories required per graph. Default: top_k.
        max_retries : int
            Max retry rounds for graphs below min_correct_per_graph.
        seeds_increment : int
            Additional seeds per graph per retry round.
        output_dir : str or None
            Directory for saving raw/filtered trajectories.
        checkpoint_file : str or None
            JSON manifest for resume support.
        chunk_size : int
            Graphs per chunk for batched GPU classifier evaluation.
        skip_t2 : bool
            Skip T2 (info-gain) teacher.
        resume_from_cache : str or None
            Path to saved GraphMeta cache.pt for resuming pre-computation.
        cache_dir : str or None
            Directory for saving/loading GraphMeta cache.

        Returns
        -------
        list of Trajectory (filtered top-k per graph, padded with incorrect if needed).
        """
        from m4.graph_meta import GraphMetaCache
        from m4.teachers.info_gain_teacher import InfoGainTeacher

        # ------------------------------------------------------------------
        # Phase 1: Pre-compute GraphMeta (or reuse precomputed_metas)
        # ------------------------------------------------------------------
        if precomputed_metas is not None:
            logger.info("Phase 1: Using %d pre-computed GraphMeta entries", len(precomputed_metas))
            metas = precomputed_metas
        else:
            logger.info("=" * 60)
            logger.info("Phase 1: Pre-computing GraphMeta for %d graphs", len(indices))
            logger.info("=" * 60)

            cache = GraphMetaCache(self.dataset, self.classifier, self.device)

            # Resume from saved cache if available
            resume_metas = None
            if resume_from_cache:
                try:
                    resume_metas = cache.load(resume_from_cache)
                    logger.info("Resumed %d metas from %s", len(resume_metas), resume_from_cache)
                except FileNotFoundError:
                    logger.warning("Cache file not found: %s, starting fresh", resume_from_cache)

            # Determine which scores to compute based on teachers
            teacher_names = {t.name for t in self.teachers}
            need_t1 = 'gradient' in teacher_names or 'hybrid' in teacher_names
            need_t3_degree = 'structural-degree' in teacher_names or 'hybrid' in teacher_names
            need_t3_pagerank = 'structural-pagerank' in teacher_names

            metas = cache.precompute(
                indices,
                skip_t1=not need_t1,
                skip_t3_degree=not need_t3_degree,
                skip_t3_pagerank=not need_t3_pagerank,
                resume_from=resume_metas,
            )

            # Save cache for future resume
            if cache_dir:
                cache_path = Path(cache_dir) / 'graph_meta_cache.pt'
                cache.save(str(cache_path))

        # ------------------------------------------------------------------
        # Phase 2: Generate + batch-evaluate in chunks
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Phase 2: Generating trajectories (chunk_size=%d)", chunk_size)
        logger.info("=" * 60)

        completed = _load_checkpoint(checkpoint_file) if checkpoint_file else set()

        all_trajs: List[Trajectory] = []
        chunk_files: List[str] = []

        # Chunk indices for memory management
        chunk_indices_list = [
            indices[i:i + chunk_size]
            for i in range(0, len(indices), chunk_size)
        ]

        for chunk_i, chunk_indices in enumerate(tqdm(chunk_indices_list, desc="chunks", unit="chunk")):
            # Skip completed graphs in checkpoint
            remaining = [
                idx for idx in chunk_indices
                if idx not in completed
            ]
            if not remaining:
                continue

            try:
                chunk_trajs = self._process_chunk(
                    chunk_indices=remaining,
                    metas=metas,
                    budget_ratio=budget_ratio,
                    seeds_per_graph=seeds_per_graph,
                    skip_t2=skip_t2,
                )
            except Exception as e:
                logger.error("Chunk %d failed: %s", chunk_i, e)
                continue

            all_trajs.extend(chunk_trajs)

            # Mark graphs as completed
            for idx in remaining:
                completed.add(idx)

            # Incremental save: raw chunk
            if output_dir and chunk_trajs:
                raw_dir = Path(output_dir) / 'raw'
                raw_dir.mkdir(parents=True, exist_ok=True)
                chunk_path = raw_dir / f'chunk_{chunk_i:04d}.pt'
                torch.save(chunk_trajs, chunk_path)
                chunk_files.append(str(chunk_path))

                if checkpoint_file:
                    _save_checkpoint(completed, checkpoint_file)

        logger.info("Phase 2 complete: %d raw trajectories", len(all_trajs))

        # ------------------------------------------------------------------
        # Phase 3: Filter top-k per graph
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Phase 3: Filtering top-%d per graph", top_k)
        logger.info("=" * 60)

        filtered = filter_top_k(all_trajs, top_k)

        # ------------------------------------------------------------------
        # Phase 3b: Retry deficient graphs until min_correct_per_graph reached
        # ------------------------------------------------------------------
        if min_correct_per_graph is None:
            min_correct_per_graph = top_k

        self._last_retry_stats: Dict[int, int] = {}  # graph_idx -> retry rounds needed

        for retry_round in range(max_retries):
            correct_per_graph = defaultdict(int)
            for t in filtered:
                if t.is_correct:
                    correct_per_graph[t.graph_idx] += 1

            deficient = [
                idx for idx in indices
                if idx in metas and correct_per_graph.get(idx, 0) < min_correct_per_graph
            ]

            if not deficient:
                logger.info(
                    "All %d graphs have >= %d correct trajectories",
                    len(set(t.graph_idx for t in filtered)), min_correct_per_graph,
                )
                break

            # Record retry stats: first time each graph needs a retry
            for idx in deficient:
                if idx not in self._last_retry_stats:
                    self._last_retry_stats[idx] = retry_round + 1

            logger.info(
                "Retry %d/%d: %d graphs still below %d correct trajectories",
                retry_round + 1, max_retries, len(deficient), min_correct_per_graph,
            )

            retry_seeds = seeds_increment
            off = (retry_round + 1) * 10000

            for i in range(0, len(deficient), chunk_size):
                chunk = deficient[i:i + chunk_size]
                try:
                    chunk_trajs = self._process_chunk(
                        chunk_indices=chunk,
                        metas=metas,
                        budget_ratio=budget_ratio,
                        seeds_per_graph=retry_seeds,
                        skip_t2=skip_t2,
                        seed_offset=off,
                    )
                    all_trajs.extend(chunk_trajs)
                except Exception as e:
                    logger.error("Retry chunk [%d:%d] failed: %s", i, i + len(chunk), e)

            # Re-filter with incorrect fallback to fill gaps
            filtered = filter_top_k(all_trajs, top_k, allow_incorrect_fallback=True)

            pct = 100.0 * sum(1 for t in all_trajs if t.is_correct) / max(len(all_trajs), 1)
            logger.info(
                "After retry %d: %d trajectories covering %d graphs (%.1f%% correct)",
                retry_round + 1, len(filtered),
                len(set(t.graph_idx for t in filtered)), pct,
            )

        pct_correct = (
            100.0 * sum(1 for t in all_trajs if t.is_correct) / max(len(all_trajs), 1)
        )
        n_correct = sum(1 for t in filtered if t.is_correct)
        n_total = len(filtered)
        logger.info(
            "Filtered: %d trajectories from %d graphs (%.1f%% correct overall, %d/%d correct)",
            n_total,
            len(set(t.graph_idx for t in filtered)),
            pct_correct,
            n_correct,
            n_total,
        )

        # Save filtered
        if output_dir:
            filtered_dir = Path(output_dir) / 'filtered'
            filtered_dir.mkdir(parents=True, exist_ok=True)
            torch.save(filtered, filtered_dir / 'filtered_trajectories.pt')
            logger.info("Filtered trajectories saved to %s", filtered_dir)

        return filtered

    # ------------------------------------------------------------------
    # Chunk processing
    # ------------------------------------------------------------------

    def _process_chunk(
        self,
        chunk_indices: List[int],
        metas: dict,
        budget_ratio: float,
        seeds_per_graph: int,
        skip_t2: bool,
        seed_offset: int = 0,
    ) -> List[Trajectory]:
        """Process one chunk: generate all trajectories, batch-evaluate on GPU.

        Steps:
        1. For each graph, generate trajectories using cached scores (no GNN).
        2. Batch all final subgraphs into one classifier forward pass.
        3. Fill in final_pred, is_correct, cross_entropy for each trajectory.
        """
        from m4.graph_meta import GraphMetaCache
        from m4.teachers.info_gain_teacher import InfoGainTeacher

        cache = GraphMetaCache(self.dataset, self.classifier, self.device)
        cache._cache = metas  # share pre-computed metas

        all_trajs: List[Trajectory] = []
        all_subgraphs: List[dgl.DGLGraph] = []

        for graph_idx in chunk_indices:
            meta = metas.get(graph_idx)
            if meta is None:
                logger.warning("Graph %d: no meta, skipping", graph_idx)
                continue

            # Load full graph once per graph
            g_raw, _ = self.dataset[graph_idx]
            g = dgl.add_self_loop(g_raw)

            budget = max(1, int(meta.n_nodes * budget_ratio))
            teacher_scores = cache.get_teacher_scores(meta)

            rng = random.Random(graph_idx * 31 + seed_offset)
            seed_nodes = [
                rng.randint(0, max(meta.n_nodes - 1, 0))
                for _ in range(seeds_per_graph)
            ]

            for seed_node in seed_nodes:
                for teacher_name, scores in teacher_scores.items():
                    try:
                        base_teacher = self.teachers[0]  # any Teacher works for static method
                        traj, sub_g = base_teacher.generate_trajectory_from_cache(
                            meta=meta,
                            graph=g,
                            scores=scores,
                            seed_node=seed_node,
                            budget=budget,
                            subgraph_builder=self.subgraph_builder,
                            teacher_name=teacher_name,
                        )
                        all_trajs.append(traj)
                        all_subgraphs.append(sub_g)
                    except Exception as e:
                        logger.warning(
                            "Trajectory failed: graph=%d teacher=%s seed=%d: %s",
                            graph_idx, teacher_name, seed_node, e,
                        )

            # T2 (info-gain): separate per-step path
            if not skip_t2:
                t2_teacher = None
                for t in self.teachers:
                    if isinstance(t, InfoGainTeacher):
                        t2_teacher = t
                        break

                if t2_teacher is not None:
                    g_raw, _ = self.dataset[graph_idx]
                    g = dgl.add_self_loop(g_raw)
                    for seed_node in seed_nodes:
                        try:
                            traj = t2_teacher.generate_trajectory(
                                graph=g,
                                classifier=self.classifier,
                                graph_idx=graph_idx,
                                label=meta.label,
                                seed_node=seed_node,
                                budget=budget,
                                subgraph_builder=self.subgraph_builder,
                                edge_key_to_id=meta.edge_key_to_id,
                            )
                            all_trajs.append(traj)
                        except Exception as e:
                            logger.debug(
                                "T2 trajectory failed: graph=%d: %s",
                                graph_idx, e,
                            )

        # Batch classifier evaluation on GPU
        if all_subgraphs:
            batch_g = dgl.batch(all_subgraphs).to(self.device)
            node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.classifier(batch_g, node_ids)
            probs = F.softmax(logits, dim=1)

            for i, traj in enumerate(all_trajs[:len(all_subgraphs)]):
                traj.final_pred = int(logits[i].argmax(dim=0).item())
                traj.is_correct = (traj.final_pred == traj.label)
                traj.cross_entropy = float(
                    -torch.log(probs[i, traj.label] + 1e-12).item()
                )

        return all_trajs


# ======================================================================
# Filtering
# ======================================================================

def filter_top_k(trajectories: List[Trajectory], top_k: int,
                 allow_incorrect_fallback: bool = False) -> List[Trajectory]:
    """Per graph: keep correctly classified, sort by CrossEntropy asc, take top-k.

    If allow_incorrect_fallback, graphs with fewer than top_k correct trajectories
    are padded with the best (lowest CE) incorrect trajectories to reach top_k.
    """
    by_graph_correct = defaultdict(list)
    by_graph_all = defaultdict(list)
    for t in trajectories:
        by_graph_all[t.graph_idx].append(t)
        if t.is_correct:
            by_graph_correct[t.graph_idx].append(t)

    result = []
    for graph_idx in sorted(by_graph_all):
        correct = sorted(by_graph_correct.get(graph_idx, []),
                         key=lambda t: t.cross_entropy)
        if len(correct) >= top_k or not allow_incorrect_fallback:
            result.extend(correct[:top_k])
        else:
            result.extend(correct)
            remaining = top_k - len(correct)
            incorrect = [t for t in by_graph_all[graph_idx] if not t.is_correct]
            incorrect.sort(key=lambda t: t.cross_entropy)
            result.extend(incorrect[:remaining])

    n_graphs = len(set(t.graph_idx for t in result))
    n_correct = sum(1 for t in result if t.is_correct)
    logger.info(
        "Filtered to %d trajectories: %d graphs, top-%d each (%d correct)",
        len(result), n_graphs, top_k, n_correct,
    )
    return result


# ======================================================================
# Checkpoint helpers
# ======================================================================

def _load_checkpoint(path: str) -> set:
    """Load set of completed graph indices from JSON manifest."""
    if not path or not Path(path).exists():
        return set()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return set(data.get('completed', []))
    except (json.JSONDecodeError, KeyError):
        logger.warning("Corrupt checkpoint file, starting fresh")
        return set()


def _save_checkpoint(completed: set, path: str):
    """Save completed graph indices to JSON manifest."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'completed': sorted(list(completed)),
            'count': len(completed),
        }, f)


def _save_trajs(trajs: List[Trajectory], output_dir: str):
    """Save trajectories to disk as a timestamped .pt file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fname = out_path / f"raw_trajectories_{timestamp}.pt"
    torch.save(trajs, fname)
    logger.debug("Saved %d trajectories to %s", len(trajs), fname)

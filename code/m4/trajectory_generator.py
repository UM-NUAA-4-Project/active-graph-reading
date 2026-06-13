"""TrajectoryGenerator — teacher 自选 top-n seed → 生成轨迹 → 过滤正确分类。

三重优化：
  1. 每图梯度传播一次，多个 gradient 系 teacher 共享
  2. generate_trajectory 返回最终 GraphState，验证时不重建
  3. 跨图 mini-batch：合并多图轨迹的最终 state 一次过分类器
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from .teachers.base import Teacher, Trajectory
from .teachers.gradient_teacher import GradientTeacher
from .teachers.structural_teacher import StructuralTeacher
from .graph_meta import GraphMeta, precompute_graph_meta

logger = logging.getLogger(__name__)

# 模块级复用，避免每张图新建 GradientTeacher 实例
_grad_teacher = GradientTeacher()
_struct_pagerank_teacher = StructuralTeacher(method='pagerank')


class TrajectoryGenerator:
    """在给定 reveal_ratio 下，为每张图的每个 teacher 生成轨迹。

    每个 teacher 用 score_nodes() 分数选 top-n_seeds 个 seed，
    每 seed 生成一条轨迹，验证阶段跨图 mini-batch 合并推理。
    """

    def __init__(
        self,
        teachers: List[Teacher],
        classifier: torch.nn.Module,
        subgraph_builder: SubgraphBuilder,
        device: str = 'cpu',
        graph_batch_size: int = 32,
    ):
        self.teachers = teachers
        self.classifier = classifier
        self.subgraph_builder = subgraph_builder
        self.device = device
        self.graph_batch_size = graph_batch_size
        self.classifier.to(device)
        self.classifier.eval()

        # 分析哪些分数需要预计算
        teacher_names = [t.name for t in teachers]
        self._needs_gradient = any(
            n == 'gradient' or n.startswith('hybrid-') for n in teacher_names
        )
        self._needs_pagerank = 'structural-pagerank' in teacher_names
        self._hybrid_alphas = [
            float(n.split('-')[1].replace('a', ''))
            for n in teacher_names if n.startswith('hybrid-')
        ]

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def generate_all(
        self,
        dataset,
        train_idx: List[int],
        reveal_ratio: float,
        n_seeds: int = 5,
    ) -> Dict[str, List[List[Trajectory]]]:
        """在给定 reveal_ratio 下生成所有图 × teacher 的轨迹。"""
        self.classifier.eval()

        n_graphs = len(train_idx)
        logger.info("预计算 %d 张图的 GraphMeta...", n_graphs)
        metas = [precompute_graph_meta(dataset[g_idx][0]) for g_idx in tqdm(range(n_graphs), desc="GraphMeta")]

        result: Dict[str, List[List[Trajectory]]] = {
            t.name: [[] for _ in range(n_graphs)] for t in self.teachers
        }

        total_pending = 0
        for batch_start in tqdm(range(0, n_graphs, self.graph_batch_size), desc="Batches"):
            batch_end = min(batch_start + self.graph_batch_size, n_graphs)
            # (teacher_name, result_g_idx, traj, state, label)
            pending: List[Tuple[str, int, Trajectory, GraphState, int]] = []

            for g_idx in range(batch_start, batch_end):
                idx = train_idx[g_idx]
                g, lbl = dataset[g_idx]
                label = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                meta = metas[g_idx]
                budget = max(1, int(reveal_ratio * meta.n_nodes))
                if budget < 1 or budget >= meta.n_nodes:
                    continue

                score_cache = self._compute_score_cache(g, label, meta)

                for teacher in self.teachers:
                    scores = score_cache.get(teacher.name)
                    trajs_states = self._generate_for_graph(
                        teacher, g, label, idx, meta, budget, n_seeds, scores=scores,
                    )
                    for traj, state in trajs_states:
                        pending.append((teacher.name, g_idx, traj, state, label))

            # 跨图批量子图评估
            if pending:
                total_pending += len(pending)
                self._validate_batch(pending)

                for teacher_name, g_idx, traj, _state, _label in pending:
                    if traj.is_correct:
                        result[teacher_name][g_idx].append(traj)

        logger.info("Total trajectory candidates validated: %d", total_pending)
        return result

    # ------------------------------------------------------------------
    # 共享分数计算
    # ------------------------------------------------------------------

    def _compute_score_cache(
        self, graph: dgl.DGLGraph, label: int, meta: GraphMeta,
    ) -> Dict[str, torch.Tensor]:
        """每图一次梯度传播，多个 gradient 系 teacher 共享。"""
        cache: Dict[str, torch.Tensor] = {}

        if self._needs_gradient:
            grad_scores = _grad_teacher.score_nodes(graph, self.classifier, label)
            cache['gradient'] = grad_scores

            # Hybrid teachers: α·norm(grad) + (1-α)·norm(degree)
            deg = meta.degrees.float()
            max_deg = deg.max()
            deg_scores = deg / max_deg if max_deg > 0 else deg
            cache['structural-degree'] = deg_scores

            grad_norm = self._minmax_norm(grad_scores)
            for alpha in self._hybrid_alphas:
                key = f'hybrid-a{alpha:.2f}'
                cache[key] = alpha * grad_norm + (1.0 - alpha) * deg_scores
        else:
            # 没有 gradient 系 teacher，但仍可能需要 structural 分数
            deg = meta.degrees.float()
            max_deg = deg.max()
            cache['structural-degree'] = deg / max_deg if max_deg > 0 else deg

        if self._needs_pagerank:
            cache['structural-pagerank'] = _struct_pagerank_teacher.score_nodes(graph)

        return cache

    @staticmethod
    def _minmax_norm(t: torch.Tensor) -> torch.Tensor:
        t_min, t_max = t.min(), t.max()
        denom = t_max - t_min
        if denom > 0:
            return (t - t_min) / denom
        return torch.zeros_like(t)

    # ------------------------------------------------------------------
    # 单图轨迹生成
    # ------------------------------------------------------------------

    def _generate_for_graph(
        self,
        teacher: Teacher,
        graph: dgl.DGLGraph,
        label: int,
        idx: int,
        meta: GraphMeta,
        budget: int,
        n_seeds: int,
        scores: torch.Tensor = None,
    ) -> List[Tuple[Trajectory, GraphState]]:
        """每个 teacher 用 score_nodes 选 top-n 个 seed，生成 (traj, state)。

        scores 传入则跳过 teacher.score_nodes()（共享分数优化）。
        """
        if scores is None:
            scores = teacher.score_nodes(graph, self.classifier, label)

        top_seeds = scores.argsort(descending=True)[:n_seeds].tolist()

        results: List[Tuple[Trajectory, GraphState]] = []
        for seed_node in top_seeds:
            traj, state = teacher.generate_trajectory(
                graph, scores, seed_node, budget, edge_key_to_id=meta.edge_key_to_id,
            )
            traj.graph_idx = idx
            results.append((traj, state))

        return results

    # ------------------------------------------------------------------
    # 批量验证
    # ------------------------------------------------------------------

    def _validate_batch(
        self, pending: List[Tuple[str, int, Trajectory, GraphState, int]],
    ):
        """跨图批量分类器评估：一次 forward 处理 batch 内所有轨迹的最终子图。"""
        states = [item[3] for item in pending]
        labels = [item[4] for item in pending]

        batched_g, _ = self.subgraph_builder.build_batch(states)
        batched_g = batched_g.to(self.device)
        node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.classifier(batched_g, node_ids)

        label_t = torch.tensor(labels, device=self.device)
        for i, (teacher_name, g_idx, traj, _state, label) in enumerate(pending):
            logit_i = logits[i:i + 1]
            probs = F.softmax(logit_i, dim=-1)
            traj.final_pred = int(probs.argmax(dim=-1).item())
            traj.is_correct = (traj.final_pred == label)
            traj.ce = float(F.cross_entropy(logit_i, label_t[i:i + 1]))

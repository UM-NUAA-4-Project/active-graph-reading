"""TrajectoryGenerator — 生成→过滤→选 top-k→重试循环。"""

import logging
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from .teachers.base import Teacher, Trajectory
from .graph_meta import GraphMeta, precompute_graph_meta, precompute_graph_meta_from_tensors

logger = logging.getLogger(__name__)


def _deterministic_seed(graph_index: int, seed_i: int, n_nodes: int) -> int:
    return (graph_index * 31 + seed_i * 17 + 7) % n_nodes


def _precompute_meta_worker(payload) -> GraphMeta:
    """模块级 worker，接收 (src, dst, degrees) CPU tensor，避免跨进程传输 DGL 图。"""
    src, dst, degrees = payload
    return precompute_graph_meta_from_tensors(src, dst, degrees)


class TrajectoryGenerator:
    """在给定 reveal_ratio 下，为每张图的每个 teacher 生成 top-k 正确轨迹。

    流程（per graph, per teacher）:
      1. 从 n seed 生成轨迹 → 过滤 is_correct
      2. 按 CE 升序选 top-k
      3. 不够 → 重试新 seed（最多 100 次）
      4. 仍不够 → 退出
    """

    def __init__(
        self,
        teachers: List[Teacher],
        classifier: torch.nn.Module,
        subgraph_builder: SubgraphBuilder,
        device: str = 'cpu',
    ):
        self.teachers = teachers
        self.classifier = classifier
        self.subgraph_builder = subgraph_builder
        self.device = device
        self.classifier.to(device)
        self.classifier.eval()

    def generate_all(
        self,
        dataset,
        train_idx: List[int],
        reveal_ratio: float,
        k_top: int = 3,
        max_attempts: int = 100,
    ) -> Dict[str, List[List[Trajectory]]]:
        """在给定 reveal_ratio 下生成所有图 × teacher 的 top-k 轨迹。
        Returns
        -------
        Dict[teacher_name, List[List[Trajectory]]]
        """
        self.classifier.eval()

        logger.info("预计算 %d 张图的 GraphMeta (并行)...", len(train_idx))
        payloads = []
        for i in train_idx:
            g = dataset[int(i)][0]
            src, dst = g.edges()
            payloads.append((
                src.cpu().clone(),
                dst.cpu().clone(),
                g.in_degrees().cpu().clone(),
            ))
        with Pool() as pool:
            metas = list(tqdm(
                pool.imap(_precompute_meta_worker, payloads),
                total=len(payloads), desc="GraphMeta",
            ))

        result: Dict[str, List[List[Trajectory]]] = {
            t.name: [] for t in self.teachers
        }

        for g_idx, idx in enumerate(tqdm(train_idx, desc="Graphs")):
            g, lbl = dataset[int(idx)]
            label = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
            meta = metas[g_idx]
            budget = max(1, int(reveal_ratio * meta.n_nodes))
            for teacher in self.teachers:
                trajs = self._generate_for_graph(
                    teacher, g, label, idx, meta, budget, k_top, max_attempts,
                )
                result[teacher.name].append(trajs)

        return result

    def _generate_for_graph(
        self,
        teacher: Teacher,
        graph: dgl.DGLGraph,
        label: int,
        idx: int,
        meta: GraphMeta,
        budget: int,
        k_top: int,
        max_attempts: int,
    ) -> List[Trajectory]:
        """为单张图、单个 teacher、单个 budget 生成 top-k 轨迹。"""
        n_nodes = meta.n_nodes

        if budget < 1 or budget >= n_nodes:
            return []

        scores = teacher.score_nodes(graph, self.classifier, label)

        correct_trajs: List[Tuple[Trajectory, float]] = []
        tried_seeds: set = set()
        attempts = 0

        while len(correct_trajs) < k_top and attempts < max_attempts:
            batch_remaining = k_top - len(correct_trajs)
            batch_size = min(max(batch_remaining * 3, 5), max_attempts - attempts, 20)

            batch_trajs = self._generate_batch(
                teacher, graph, idx, meta, scores, label,
                budget, batch_size, tried_seeds,
            )
            correct_trajs.extend(batch_trajs)
            attempts += batch_size

        correct_trajs.sort(key=lambda x: x[1])
        top_trajs = [t for t, _ in correct_trajs[:k_top]]

        return top_trajs

    def _generate_batch(
        self,
        teacher: Teacher,
        graph: dgl.DGLGraph,
        idx: int,
        meta: GraphMeta,
        scores: torch.Tensor,
        label: int,
        budget: int,
        batch_size: int,
        tried_seeds: set,
    ) -> List[Tuple[Trajectory, float]]:
        """生成一批 trajectory，批量评估 classifier，返回正确分类的 (traj, CE)。"""
        n_nodes = meta.n_nodes
        edge_key_to_id = meta.edge_key_to_id

        raw_trajs: List[Trajectory] = []
        for i in range(batch_size):
            seed_i = len(tried_seeds) + i
            seed_node = _deterministic_seed(idx, seed_i, n_nodes)
            traj = teacher.generate_trajectory(
                graph, scores, seed_node, budget, edge_key_to_id=edge_key_to_id,
            )
            traj.graph_idx = idx
            raw_trajs.append(traj)
            tried_seeds.add(seed_i)

        # 批量构建子图 + 推理
        states = []
        for traj in raw_trajs:
            state = GraphState(graph, traj.seed_node, edge_key_to_id=edge_key_to_id)
            for node in traj.reveal_order:
                reveal_node(state, node)
            states.append(state)

        batched_g, _ = self.subgraph_builder.build_batch(states)
        batched_g = batched_g.to(self.device)
        node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.classifier(batched_g, node_ids)

        results: List[Tuple[Trajectory, float]] = []
        for i, traj in enumerate(raw_trajs):
            logit_i = logits[i:i + 1]
            probs = F.softmax(logit_i, dim=-1)
            pred = int(probs.argmax(dim=-1).item())
            is_correct = (pred == label)
            ce = float(F.cross_entropy(logit_i, torch.tensor([label], device=self.device)))

            traj.final_pred = pred
            traj.is_correct = is_correct

            if is_correct:
                results.append((traj, ce))

        return results

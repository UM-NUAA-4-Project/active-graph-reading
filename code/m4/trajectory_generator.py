"""TrajectoryGenerator — teacher 自选 top-n seed → 生成轨迹 → 过滤正确分类。"""

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from .teachers.base import Teacher, Trajectory
from .graph_meta import GraphMeta, precompute_graph_meta

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """在给定 reveal_ratio 下，为每张图的每个 teacher 生成轨迹。

    每个 teacher 用 score_nodes() 分数选 top-n_seeds 个 seed，
    每 seed 生成一条轨迹，只保留 classifier 正确分类的。
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
        n_seeds: int = 5,
    ) -> Dict[str, List[List[Trajectory]]]:
        """在给定 reveal_ratio 下生成所有图 × teacher 的轨迹。
        Returns
        -------
        Dict[teacher_name, List[List[Trajectory]]]
        """
        self.classifier.eval()

        logger.info("预计算 %d 张图的 GraphMeta...", len(train_idx))
        metas = [precompute_graph_meta(dataset[int(i)][0]) for i in tqdm(train_idx, desc="GraphMeta")]

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
                    teacher, g, label, idx, meta, budget, n_seeds,
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
        n_seeds: int,
    ) -> List[Trajectory]:
        """每个 teacher 用 score_nodes 选 top-n 个 seed，生成轨迹并过滤正确分类的。"""
        n_nodes = meta.n_nodes

        if budget < 1 or budget >= n_nodes:
            return []

        scores = teacher.score_nodes(graph, self.classifier, label)
        top_seeds = scores.argsort(descending=True)[:n_seeds].tolist()

        # 生成 n_seeds 条轨迹
        raw_trajs: List[Trajectory] = []
        for seed_node in top_seeds:
            traj = teacher.generate_trajectory(
                graph, scores, seed_node, budget, edge_key_to_id=meta.edge_key_to_id,
            )
            traj.graph_idx = idx
            raw_trajs.append(traj)

        # 批量重建 GraphState + 分类器验证
        states = []
        for traj in raw_trajs:
            state = GraphState(graph, traj.seed_node, edge_key_to_id=meta.edge_key_to_id)
            for node in traj.reveal_order:
                reveal_node(state, node)
            states.append(state)

        batched_g, _ = self.subgraph_builder.build_batch(states)
        batched_g = batched_g.to(self.device)
        node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.classifier(batched_g, node_ids)

        for i, traj in enumerate(raw_trajs):
            logit_i = logits[i:i + 1]
            probs = F.softmax(logit_i, dim=-1)
            traj.final_pred = int(probs.argmax(dim=-1).item())
            traj.is_correct = (traj.final_pred == label)
            traj.ce = float(F.cross_entropy(logit_i, torch.tensor([label], device=self.device)))

        return [t for t in raw_trajs if t.is_correct]

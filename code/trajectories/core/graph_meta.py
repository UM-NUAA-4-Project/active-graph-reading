"""GraphMeta — 每张图的预计算缓存，teacher 复用避免重复构建（自 m4 迁入）。"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class GraphMeta:
    """预计算的图元数据缓存。"""

    edge_key_to_id: Dict[Tuple[int, int], int]
    degrees: torch.Tensor  # shape (|V|,) in-degree

    @property
    def n_nodes(self) -> int:
        return self.degrees.shape[0]


def precompute_graph_meta(graph) -> GraphMeta:
    """从 DGL 图预计算 edge_key_to_id 和度数。"""
    edge_key_to_id: Dict[Tuple[int, int], int] = {}
    has_feat = 'feat' in graph.edata
    if has_feat:
        src = graph.edges()[0]
        dst = graph.edges()[1]
        for eid in range(graph.num_edges()):
            u, v = int(src[eid]), int(dst[eid])
            key = (min(u, v), max(u, v))
            if key not in edge_key_to_id:
                edge_key_to_id[key] = eid

    degrees = graph.in_degrees().clone().detach().cpu()

    return GraphMeta(edge_key_to_id=edge_key_to_id, degrees=degrees)

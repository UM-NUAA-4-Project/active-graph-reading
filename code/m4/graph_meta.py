"""GraphMeta — 每张图的预计算缓存，teacher 复用避免重复构建。"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import dgl


@dataclass
class GraphMeta:
    """预计算的图元数据缓存。"""

    edge_key_to_id: Dict[Tuple[int, int], int]
    degrees: torch.Tensor  # shape (|V|,) in-degree

    @property
    def n_nodes(self) -> int:
        return self.degrees.shape[0]


def precompute_graph_meta(graph: dgl.DGLGraph) -> GraphMeta:
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


def precompute_graph_meta_from_tensors(
    src: torch.Tensor,
    dst: torch.Tensor,
    degrees: torch.Tensor,
) -> GraphMeta:
    """从轻量 tensor 构建 GraphMeta — 仅供 multiprocessing worker 使用。

    避免跨进程传输 DGL 图（spawn 模式下 DGL C++ backend 序列化会崩溃）。
    """
    edge_key_to_id: Dict[Tuple[int, int], int] = {}
    for eid in range(src.shape[0]):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in edge_key_to_id:
            edge_key_to_id[key] = eid
    return GraphMeta(edge_key_to_id=edge_key_to_id, degrees=degrees)

"""T3: 结构 Teacher — 度中心性或 PageRank 作为节点重要性。"""

import logging
from typing import Optional

import torch
import dgl
import dgl.function as fn

from .base import Teacher

logger = logging.getLogger(__name__)


class StructuralTeacher(Teacher):
    """T3 (structural): 选 degree / pagerank 最高的 frontier node。

    纯拓扑计算，不依赖 GNN。
    """

    VALID_METHODS = ('degree', 'pagerank')

    def __init__(self, method: str = 'degree'):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method 必须是 {self.VALID_METHODS}，实际: {method}")
        self.method = method

    @property
    def name(self) -> str:
        return f'structural-{self.method}'

    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: Optional[torch.nn.Module] = None,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        if self.method == 'degree':
            return self._score_degree(graph)
        return self._score_pagerank(graph)

    @staticmethod
    def _score_degree(graph: dgl.DGLGraph) -> torch.Tensor:
        deg = graph.in_degrees().float()
        max_deg = deg.max()
        if max_deg > 0:
            return deg / max_deg
        return deg

    @staticmethod
    def _score_pagerank(
        graph: dgl.DGLGraph,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        n = graph.num_nodes()
        g = graph.local_var()
        g.ndata['pr'] = torch.ones(n) / n
        out_deg = g.out_degrees().float().clamp(min=1)

        for _ in range(max_iter):
            old_pr = g.ndata['pr'].clone()
            g.update_all(fn.copy_u('pr', 'm'), fn.sum('m', 'new_pr'))
            g.ndata['pr'] = damping * g.ndata['new_pr'] / out_deg + (1.0 - damping) / n
            if (g.ndata['pr'] - old_pr).abs().sum() < tol:
                break

        pr = g.ndata['pr']
        pr_max = pr.max()
        if pr_max > 0:
            return pr / pr_max
        return pr

"""T3: Structural Teacher.

Computes per-node importance from graph topology alone:
- degree centrality (DGL native, O(|E|))
- PageRank (networkx, iterative)
- betweenness centrality (networkx, O(|V|×|E|), small graphs only)

No classifier needed -- scores are purely structural.
Independent of any GNN, can be pre-computed offline.

Reference: Ying et al., "GNNExplainer", NeurIPS 2019.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import dgl
import networkx as nx

from m4.teachers.base import Teacher

logger = logging.getLogger(__name__)

_VALID_METHODS = {'degree', 'pagerank', 'betweenness'}


class StructuralTeacher(Teacher):
    """T3: Scores nodes by graph topology metrics.

    Parameters
    ----------
    method : str
        One of 'degree', 'pagerank', 'betweenness'.
    """

    def __init__(self, method: str = 'degree'):
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {sorted(_VALID_METHODS)}."
            )
        self.method = method

    @property
    def name(self) -> str:
        return f'structural-{self.method}'

    def _params_dict(self) -> dict:
        return {'method': self.method}

    # ------------------------------------------------------------------
    # score_nodes
    # ------------------------------------------------------------------

    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: Optional[nn.Module] = None,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute topology-based importance scores.

        classifier and label are ignored -- structural scores are
        independent of any learned model.
        """
        n = graph.num_nodes()
        if n <= 1:
            return torch.ones(n)

        if self.method == 'degree':
            return self._degree_scores(graph, n)

        # PageRank and betweenness require networkx conversion
        nx_g = dgl.to_networkx(graph, node_attrs=None, edge_attrs=None).to_undirected()

        if self.method == 'pagerank':
            return self._pagerank_scores(nx_g, n)

        if self.method == 'betweenness':
            return self._betweenness_scores(nx_g, n)

        raise RuntimeError(f"Unreachable: method={self.method}")

    # ------------------------------------------------------------------
    # Score implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _degree_scores(graph: dgl.DGLGraph, n: int) -> torch.Tensor:
        """Degree centrality: deg(v) / (|V|-1)."""
        deg = graph.in_degrees().float()
        return deg / max(n - 1, 1)

    @staticmethod
    def _pagerank_scores(nx_g: nx.Graph, n: int) -> torch.Tensor:
        """PageRank with damping factor α=0.85."""
        pr_dict = nx.pagerank(nx_g, alpha=0.85, max_iter=200)
        scores = torch.tensor([pr_dict.get(i, 0.0) for i in range(n)], dtype=torch.float32)
        return scores

    @staticmethod
    def _betweenness_scores(nx_g: nx.Graph, n: int) -> torch.Tensor:
        """Betweenness centrality. Only called for graphs with < 200 nodes."""
        if n > 200:
            raise ValueError(
                f"Betweenness too expensive for {n} nodes (limit: 200). "
                "Use 'degree' or 'pagerank' instead."
            )
        # Use k=min(n, 50) random pivots for speed on mid-sized graphs
        k = min(n, 50)
        bc_dict = nx.betweenness_centrality(nx_g, k=k, normalized=True)
        scores = torch.tensor([bc_dict.get(i, 0.0) for i in range(n)], dtype=torch.float32)
        return scores

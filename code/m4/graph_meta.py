"""GraphMeta: per-graph pre-computed scores and metadata.

Enables one-time computation of T1 (gradient) and T3 (structural) scores
that are shared across all seeds and budgets for a graph, eliminating
redundant forward/backward passes and NetworkX conversions.

Also provides a GraphMetaCache for batch pre-computation with
checkpoint/resume support.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dgl
import networkx as nx
import torch
from tqdm import tqdm

from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher

logger = logging.getLogger(__name__)


@dataclass
class GraphMeta:
    """Pre-computed per-graph data shared across all trajectories.

    Attributes
    ----------
    graph_idx : int
        Dataset index.
    label : int
        Ground-truth class label.
    n_nodes : int
        Number of nodes in the full graph.
    edge_key_to_id : Dict[Tuple[int, int], int]
        Canonical (min(u,v), max(u,v)) → edge_id mapping for O(1) lookup.
    t1_scores : torch.Tensor or None
        Gradient-attribution scores, shape (n_nodes,).
    t3_degree_scores : torch.Tensor or None
        Degree centrality scores, shape (n_nodes,).
    t3_pagerank_scores : torch.Tensor or None
        PageRank scores, shape (n_nodes,).
    """

    graph_idx: int
    label: int
    n_nodes: int
    edge_key_to_id: Dict[Tuple[int, int], int]
    t1_scores: Optional[torch.Tensor] = None
    t3_degree_scores: Optional[torch.Tensor] = None
    t3_pagerank_scores: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        """Serialize to a dict for torch.save (excludes cached DGLGraph)."""
        return {
            'graph_idx': self.graph_idx,
            'label': self.label,
            'n_nodes': self.n_nodes,
            'edge_key_to_id': self.edge_key_to_id,
            't1_scores': self.t1_scores,
            't3_degree_scores': self.t3_degree_scores,
            't3_pagerank_scores': self.t3_pagerank_scores,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GraphMeta':
        """Deserialize from dict."""
        return cls(
            graph_idx=d['graph_idx'],
            label=d['label'],
            n_nodes=d['n_nodes'],
            edge_key_to_id=d['edge_key_to_id'],
            t1_scores=d['t1_scores'],
            t3_degree_scores=d['t3_degree_scores'],
            t3_pagerank_scores=d['t3_pagerank_scores'],
        )


class GraphMetaCache:
    """Pre-computes and caches per-graph scores for trajectory generation.

    Parameters
    ----------
    dataset : PPADataset
    classifier : nn.Module
        Frozen GraphClassifier for T1 gradient computation.
    device : str or torch.device
    """

    def __init__(self, dataset, classifier, device='cpu'):
        self.dataset = dataset
        self.classifier = classifier
        self.device = device
        self._cache: Dict[int, GraphMeta] = {}

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def precompute(
        self,
        indices: List[int],
        skip_t1: bool = False,
        skip_t3_degree: bool = False,
        skip_t3_pagerank: bool = False,
        resume_from: Optional[dict] = None,
    ) -> Dict[int, GraphMeta]:
        """Pre-compute GraphMeta for all given indices.

        Parameters
        ----------
        indices : list of int
            Dataset graph indices to process.
        skip_t1 : bool
            Skip gradient score computation.
        skip_t3_degree : bool
            Skip degree score computation.
        skip_t3_pagerank : bool
            Skip pagerank score computation.
        resume_from : dict or None
            Previously saved {graph_idx: GraphMeta} dict to resume from.

        Returns
        -------
        dict mapping graph_idx → GraphMeta
        """
        if resume_from:
            self._cache.update(resume_from)
            logger.info("Resumed %d cached metas", len(resume_from))

        remaining = [i for i in indices if i not in self._cache]
        if not remaining:
            logger.info("All %d graphs already cached", len(indices))
            return {i: self._cache[i] for i in indices if i in self._cache}

        logger.info(
            "Pre-computing GraphMeta for %d graphs (T1=%s, T3-degree=%s, T3-pagerank=%s)...",
            len(remaining),
            not skip_t1,
            not skip_t3_degree,
            not skip_t3_pagerank,
        )

        # Create lazy teacher instances as needed
        grad_teacher = GradientTeacher(aggregate='mean') if not skip_t1 else None

        for idx in tqdm(remaining, desc="pre-compute meta", unit="graph"):
            try:
                meta = self._precompute_one(
                    idx, grad_teacher,
                    skip_t3_degree=skip_t3_degree,
                    skip_t3_pagerank=skip_t3_pagerank,
                )
                self._cache[idx] = meta
            except Exception as e:
                logger.error("Failed to pre-compute graph %d: %s", idx, e)
                continue

        logger.info("Pre-computed %d GraphMeta entries", len(self._cache))
        return {i: self._cache[i] for i in indices if i in self._cache}

    def _precompute_one(
        self,
        idx: int,
        grad_teacher=None,
        skip_t3_degree: bool = False,
        skip_t3_pagerank: bool = False,
    ) -> GraphMeta:
        """Pre-compute scores for a single graph.

        Loads the DGL graph from dataset once, adds self-loops, builds
        edge_key_to_id, and computes all requested scores.
        """
        g_raw, label = self.dataset[idx]
        g = dgl.add_self_loop(g_raw)
        n_nodes = g.num_nodes()

        # Build edge_key_to_id (canonical undirected key → edge index)
        edge_key_to_id = _build_edge_key_to_id(g)

        meta = GraphMeta(
            graph_idx=idx,
            label=label,
            n_nodes=n_nodes,
            edge_key_to_id=edge_key_to_id,
        )

        # T1: gradient attribution (need encoder params to require grad)
        if grad_teacher is not None:
            g_dev = g.to(self.device)
            # Temporarily enable gradients on encoder for attribution
            encoder = self.classifier.encoder
            was_frozen = [p.requires_grad for p in encoder.parameters()]
            for p in encoder.parameters():
                p.requires_grad = True
            try:
                meta.t1_scores = grad_teacher.score_nodes(g_dev, self.classifier, label).cpu()
            finally:
                for p, was in zip(encoder.parameters(), was_frozen):
                    p.requires_grad = was
                self.classifier.zero_grad()

        # T3: degree (DGL native, fast)
        if not skip_t3_degree:
            meta.t3_degree_scores = StructuralTeacher._degree_scores(g, n_nodes)

        # T3: pagerank (NetworkX, moderate)
        if not skip_t3_pagerank:
            nx_g = dgl.to_networkx(g, node_attrs=None, edge_attrs=None).to_undirected()
            meta.t3_pagerank_scores = StructuralTeacher._pagerank_scores(nx_g, n_nodes)

        return meta

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all cached GraphMeta to disk (as dict of dicts)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {idx: meta.to_dict() for idx, meta in self._cache.items()}
        torch.save(data, path)
        logger.info("Saved %d GraphMeta entries to %s", len(data), path)

    def load(self, path: str) -> Dict[int, GraphMeta]:
        """Load cached GraphMeta from disk."""
        data = torch.load(path, map_location='cpu', weights_only=False)
        loaded = {}
        for idx, d in data.items():
            loaded[idx] = GraphMeta.from_dict(d)
        self._cache.update(loaded)
        logger.info("Loaded %d GraphMeta entries from %s", len(loaded), path)
        return loaded

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def get_teacher_scores(self, meta: GraphMeta) -> Dict[str, torch.Tensor]:
        """Return dict mapping teacher name → (n_nodes,) score tensor."""
        scores = {}
        if meta.t1_scores is not None:
            scores['gradient'] = meta.t1_scores
        if meta.t3_degree_scores is not None:
            scores['structural-degree'] = meta.t3_degree_scores
        if meta.t3_pagerank_scores is not None:
            scores['structural-pagerank'] = meta.t3_pagerank_scores
        # T4 hybrid variants — alpha values MUST match teacher instances in
        # generate_trajectories.py (see HybridTeacher constructors).
        if meta.t1_scores is not None and meta.t3_degree_scores is not None:
            scores['hybrid-0.3'] = _compute_hybrid_scores(
                meta.t1_scores, meta.t3_degree_scores, alpha=0.3
            )
            scores['hybrid-0.7'] = _compute_hybrid_scores(
                meta.t1_scores, meta.t3_degree_scores, alpha=0.7
            )
        return scores

    def get_scores_for(
        self, meta: GraphMeta, teacher_name: str
    ) -> Optional[torch.Tensor]:
        """Get scores for a specific teacher by name."""
        return self.get_teacher_scores(meta).get(teacher_name)

    def __contains__(self, idx: int) -> bool:
        return idx in self._cache

    def __getitem__(self, idx: int) -> GraphMeta:
        return self._cache[idx]

    def __len__(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_edge_key_to_id(g: dgl.DGLGraph) -> Dict[Tuple[int, int], int]:
    """Build canonical undirected edge key → edge index mapping."""
    edge_key_to_id: Dict[Tuple[int, int], int] = {}
    if 'feat' not in g.edata:
        return edge_key_to_id
    src = g.edges()[0]
    dst = g.edges()[1]
    for eid in range(g.num_edges()):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in edge_key_to_id:
            edge_key_to_id[key] = eid
    return edge_key_to_id


def minmax_norm(scores: torch.Tensor) -> torch.Tensor:
    """Normalize scores to [0, 1] via min-max scaling."""
    smin = scores.min()
    smax = scores.max()
    if smax - smin < 1e-8:
        return torch.ones_like(scores) * 0.5
    return (scores - smin) / (smax - smin)


def _compute_hybrid_scores(
    t1: torch.Tensor,
    t3: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute T4 hybrid scores: α × t1_norm + (1-α) × t3_norm."""
    t1_norm = minmax_norm(t1)
    t3_norm = minmax_norm(t3)
    return alpha * t1_norm + (1.0 - alpha) * t3_norm

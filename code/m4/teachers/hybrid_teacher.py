"""T4: Hybrid Teacher.

Combines T1 (gradient attribution) and T3 (structural topology) scores
via weighted combination:

    s_v = α · s_v^{grad} + (1-α) · s_v^{struct}

Alpha can be fixed or randomly sampled per graph for diversity.

Reference: M4-implementation.md §3.2 Teacher T4.
"""

import logging
import random
from typing import Optional

import torch
import torch.nn as nn
import dgl

from m4.teachers.base import Teacher
from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.graph_meta import minmax_norm

logger = logging.getLogger(__name__)


class HybridTeacher(Teacher):
    """T4: Weighted combination of gradient and structural teacher scores.

    Parameters
    ----------
    gradient_teacher : GradientTeacher
        T1 instance for gradient-based scores.
    structural_teacher : StructuralTeacher
        T3 instance for topology-based scores.
    alpha : float or tuple of (min, max) or None
        - float (e.g. 0.5): fixed weight for gradient component.
        - tuple (e.g. (0.3, 0.7)): uniformly sampled per score_nodes() call.
        - None: defaults to fixed 0.5.
    """

    def __init__(
        self,
        gradient_teacher: Optional[GradientTeacher] = None,
        structural_teacher: Optional[StructuralTeacher] = None,
        alpha=None,
    ):
        self._grad_teacher = gradient_teacher or GradientTeacher(aggregate='mean')
        self._struct_teacher = structural_teacher or StructuralTeacher(method='degree')

        if alpha is None:
            alpha = 0.5
        if isinstance(alpha, (int, float)):
            self._alpha_fixed = float(alpha)
            self._alpha_range = None
        else:
            self._alpha_fixed = None
            self._alpha_range = (float(alpha[0]), float(alpha[1]))

    @property
    def name(self) -> str:
        return 'hybrid'

    def _params_dict(self) -> dict:
        alpha_val = self._alpha_fixed if self._alpha_fixed is not None else f'{self._alpha_range[0]}-{self._alpha_range[1]}'
        return {
            'alpha': alpha_val,
            'gradient': self._grad_teacher.name,
            'structural': self._struct_teacher.name,
        }

    # ------------------------------------------------------------------
    # score_nodes
    # ------------------------------------------------------------------

    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: nn.Module,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute hybrid scores: α × gradient + (1-α) × structural.

        Returns tensor of shape (|V|,).
        """
        # Resolve alpha
        if self._alpha_range is not None:
            alpha = random.uniform(*self._alpha_range)
        else:
            alpha = self._alpha_fixed

        # Compute component scores
        grad_scores = self._grad_teacher.score_nodes(graph, classifier, label)
        struct_scores = self._struct_teacher.score_nodes(graph, classifier=None, label=None)

        # Normalize each to [0, 1] before combining (avoids scale mismatch)
        grad_norm = minmax_norm(grad_scores)
        struct_norm = minmax_norm(struct_scores).to(device=grad_norm.device)

        combined = alpha * grad_norm + (1.0 - alpha) * struct_norm

        logger.debug(
            "Hybrid: α=%.2f, grad_range=[%.3f, %.3f], struct_range=[%.3f, %.3f]",
            alpha,
            float(grad_scores.min()), float(grad_scores.max()),
            float(struct_scores.min()), float(struct_scores.max()),
        )

        return combined

    # ------------------------------------------------------------------
    # Cache-based scoring (used by phased pipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_scores_from_cache(meta, alpha: float) -> torch.Tensor:
        """Compute hybrid scores from pre-computed GraphMeta.

        Uses t1_scores and t3_degree_scores from the cache.
        """
        t1_norm = minmax_norm(meta.t1_scores)
        t3_norm = minmax_norm(meta.t3_degree_scores)
        return alpha * t1_norm + (1.0 - alpha) * t3_norm

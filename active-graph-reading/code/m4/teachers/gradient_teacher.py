"""T1: Gradient Attribution Teacher.

Computes per-node importance by back-propagating the correct-class
logit through a frozen GIN classifier and measuring gradient norms
of each node's intermediate embeddings.

Reference: Grad-CAM (Selvaraju et al., ICCV 2017) generalized to GNNs
(Pope et al., CVPR 2019).
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from m4.teachers.base import Teacher

logger = logging.getLogger(__name__)


class GradientTeacher(Teacher):
    """T1: Scores nodes by ||∂ŷ_c/∂h_v||₂ averaged across GIN layers.

    Uses register_forward_hook on each GINELayer to capture per-layer
    node embeddings without modifying the encoder's forward() interface.
    """

    def __init__(self, aggregate: str = 'mean'):
        """
        Parameters
        ----------
        aggregate : str
            How to combine per-layer gradient norms: 'mean' or 'max'.
        """
        self._aggregate = aggregate
        self._embeddings: List[torch.Tensor] = []
        self._handles: List = []

    @property
    def name(self) -> str:
        return 'gradient'

    def _params_dict(self) -> dict:
        return {'aggregate': self._aggregate}

    # ------------------------------------------------------------------
    # score_nodes
    # ------------------------------------------------------------------

    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: nn.Module,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute gradient-attribution scores for all nodes.

        Returns a tensor of shape (|V|,) with higher values indicating
        nodes whose embeddings most influence the correct-class prediction.
        """
        n_nodes = graph.num_nodes()
        encoder = classifier.encoder
        n_layers = len(encoder.convs)

        # 1. Register forward hooks on each GINELayer
        self._register_hooks(encoder)

        # 2. Forward pass on full graph
        node_ids = torch.zeros(n_nodes, dtype=torch.long, device=graph.device)
        logits = classifier(graph, node_ids)

        if label is None:
            label = int(logits.argmax(dim=1).item())

        # 3. Compute gradients of correct-class logit w.r.t. captured embeddings
        try:
            grads = torch.autograd.grad(
                outputs=logits[0, label],
                inputs=self._embeddings,
                retain_graph=False,
                create_graph=False,
            )
        except RuntimeError as e:
            logger.warning("Gradient computation failed: %s. Falling back to uniform.", e)
            self._remove_hooks()
            return torch.ones(n_nodes)

        # 4. Per-layer gradient norm for each node
        layer_norms = []
        for layer_idx, grad in enumerate(grads):
            if grad is None:
                logger.debug("Layer %d: gradient is None, skipping", layer_idx)
                continue
            # grad shape: (n_nodes, hidden_dim) or (n_batched_nodes, hidden_dim)
            node_norms = grad.norm(p=2, dim=1)  # (n_nodes,)
            layer_norms.append(node_norms)

        if not layer_norms:
            self._remove_hooks()
            return torch.ones(n_nodes)

        # 5. Aggregate across layers
        stacked = torch.stack(layer_norms, dim=0)  # (n_layers, n_nodes)
        if self._aggregate == 'max':
            scores = stacked.max(dim=0).values
        else:
            scores = stacked.mean(dim=0)

        # 6. Cleanup
        self._remove_hooks()
        classifier.zero_grad()

        # Verify gradient signal has variance
        std = scores.std().item()
        if std < 1e-8:
            logger.warning(
                "Gradient scores have near-zero variance (std=%.2e). "
                "Check if classifier parameters are frozen or gradients are detached.",
                std,
            )

        return scores

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self, encoder):
        """Register forward hooks on all GINELayer conv modules."""
        self._remove_hooks()
        self._embeddings = []

        for i, conv in enumerate(encoder.convs):
            handle = conv.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._embeddings = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input_, output):
            # output is post-MLP GIN embedding, pre-BN/ReLU/Dropout/Adapter
            # Retain gradient for torch.autograd.grad
            if output.requires_grad:
                output.retain_grad()
            self._embeddings.append(output)
        return hook_fn

"""
Graph classifier (OGB-official architecture)
=============================================

GINEEncoder + single Linear classification head.
Matches OGB's GNN wrapper for ogbg-ppa.

Supports pretrained weight loading.
"""

from typing import Dict, Any
import logging

import torch
import torch.nn as nn

from .gin_encoder import GINEEncoder

logger = logging.getLogger(__name__)


class GraphClassifier(nn.Module):
    """Graph-level classifier = GINEEncoder + Linear head.

    Parameters
    ----------
    edge_feat_dim : int
        Raw edge feature dimension (PPA: 7).
    n_classes : int
        Number of classes (PPA: 37).
    hidden_dim : int
        Hidden dimension (default 300).
    n_layers : int
        Number of GIN conv layers (default 5).
    dropout : float
        Dropout probability (default 0.5).
    node_feat_dim : int
        Raw node feature dimension. 0 (default) keeps Embedding behavior
        (PPA: no node features); >0 uses a Linear projection (e.g.
        CIFAR10-SuperPixel: 5).
    """

    def __init__(
        self,
        edge_feat_dim: int,
        n_classes: int,
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
        virtual_node: bool = False,
        residual: bool = False,
        node_feat_dim: int = 0,
    ):
        super().__init__()

        n_classes = int(n_classes)
        edge_feat_dim = int(edge_feat_dim)

        self.encoder = GINEEncoder(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            virtual_node=virtual_node,
            residual=residual,
            node_feat_dim=node_feat_dim,
        )

        self.head = nn.Linear(hidden_dim, n_classes)

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_pretrained(self, path: str, device: str = 'cpu') -> Dict[str, Any]:
        """Load pretrained weights from best_model.pt.

        Returns the checkpoint metadata dict.
        """
        ckpt = torch.load(path, map_location=device)
        missing, unexpected = self.load_state_dict(ckpt['model_state_dict'], strict=True)
        if missing:
            logger.warning("Missing keys in pretrained weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in pretrained weights: %s", unexpected)
        logger.info("Loaded pretrained weights from %s (val_acc=%.4f)",
                    path, ckpt.get('val_acc', 0.0))
        return ckpt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        g,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        g_emb = self.encoder(g, node_ids)
        return self.head(g_emb)

    def get_embedding(
        self,
        g,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(g, node_ids)


# ------------------------------------------------------------------
# Factory: load frozen f_class from checkpoint
# ------------------------------------------------------------------

def load_fclass(
    ckpt_path: str,
    device: torch.device | str = 'cpu',
    edge_feat_dim: int = 7,
    n_classes: int = 37,
    hidden_dim: int = 300,
    n_layers: int = 5,
    dropout: float = 0.5,
) -> GraphClassifier:
    """加载冻结的 f_class 全图分类器（eval 模式，requires_grad=False）。"""
    device = torch.device(device)
    clf = GraphClassifier(
        edge_feat_dim=edge_feat_dim,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    clf.load_state_dict(ckpt['model_state_dict'], strict=True)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    logger.info("f_class loaded (val_acc=%.4f), frozen.",
                ckpt.get('val_acc', float('nan')))
    return clf

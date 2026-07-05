"""
Graph classifier (OGB-official architecture)
=============================================

GINEEncoder + single Linear classification head.
Matches OGB's GNN wrapper for ogbg-ppa.

Supports fine-tuning with freeze control, bottleneck adapters,
and pretrained weight loading.
"""

from typing import Set, Dict, Any
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
    """

    def __init__(
        self,
        edge_feat_dim: int,
        n_classes: int,
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()

        n_classes = int(n_classes)
        edge_feat_dim = int(edge_feat_dim)

        self.encoder = GINEEncoder(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.head = nn.Linear(hidden_dim, n_classes)

    # ------------------------------------------------------------------
    # Fine-tuning API
    # ------------------------------------------------------------------

    def load_pretrained(self, path: str, device: str = 'cpu') -> Dict[str, Any]:
        """Load pretrained weights (from train_full_graph.py's best_model.pt).

        Must be called BEFORE apply_finetune_strategy so that adapter keys
        don't exist yet and strict=True loading succeeds.

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

    def apply_finetune_strategy(
        self,
        freeze_layers: Set[int],
        adapter_layers: Set[int],
        adapter_bottleneck_ratio: int = 16,
        freeze_node_emb: bool = True,
    ):
        """Configure the encoder for fine-tuning. Head always stays trainable.

        Parameters
        ----------
        freeze_layers : set of int
            GIN layer indices to freeze.
        adapter_layers : set of int
            GIN layer indices to add bottleneck adapters after.
        adapter_bottleneck_ratio : int
            Bottleneck compression ratio.
        freeze_node_emb : bool
            Whether to freeze the node embedding (default True — PPA has no
            node features so the embedding is generic).
        """
        self.encoder.apply_finetune_strategy(
            freeze_layers=freeze_layers,
            adapter_layers=adapter_layers,
            adapter_bottleneck_ratio=adapter_bottleneck_ratio,
        )
        if freeze_node_emb:
            self.encoder.freeze_node_emb()

        # Ensure head is trainable
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        """Generator over parameters that require gradients."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p

    def log_trainable(self):
        """Log a summary of trainable vs. frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("Trainable: %d / %d params (%.1f%%)",
                    trainable, total, 100.0 * trainable / max(total, 1))
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug("  trainable: %s  (%d params)", name, p.numel())

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

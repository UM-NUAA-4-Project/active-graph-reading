"""SubgraphEncoder g_phi: 2-layer GINE for subgraph embedding.

Used by PrototypeReader to encode subgraph -> 300-dim embedding -> PCA -> bank query.
Initialized from f_class first 2 layers (warm-start), fine-tuned for subgraph distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import dgl

from .gin_encoder import GINEEncoder
from .classifier import GraphClassifier

logger = logging.getLogger(__name__)

# PPA constants
PPA_HIDDEN_DIM = 300
PPA_EDGE_FEAT_DIM = 7
PPA_DROPOUT = 0.5


class SubgraphEncoder(nn.Module):
    """g_phi: 2-layer GINE, outputs 300-dim graph embedding.

    Wraps GINEEncoder(n_layers=2). Supports warm-start from f_class (5-layer).
    """

    def __init__(
        self,
        edge_feat_dim: int = PPA_EDGE_FEAT_DIM,
        hidden_dim: int = PPA_HIDDEN_DIM,
        dropout: float = PPA_DROPOUT,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = GINEEncoder(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    # ------------------------------------------------------------------
    # warm-start from f_class
    # ------------------------------------------------------------------

    def init_from_fclass(self, fclass_encoder: GINEEncoder):
        """Copy first n_layers weights from f_class GINEEncoder.

        Copies: node_emb, convs[0..n_layers-1], bns[0..n_layers-1].
        Pool (AvgPooling) is parameter-free and kept as-is.
        """
        assert isinstance(fclass_encoder, GINEEncoder)
        assert fclass_encoder.n_layers >= self.n_layers

        self.encoder.node_emb.load_state_dict(
            fclass_encoder.node_emb.state_dict())
        for i in range(self.n_layers):
            self.encoder.convs[i].load_state_dict(
                fclass_encoder.convs[i].state_dict())
            self.encoder.bns[i].load_state_dict(
                fclass_encoder.bns[i].state_dict())

        logger.info("SubgraphEncoder init from f_class first %d layers.",
                    self.n_layers)

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------

    def get_embedding(self, g: dgl.DGLGraph, node_ids: torch.Tensor) -> torch.Tensor:
        """sub_g -> (1, hidden_dim) graph embedding."""
        return self.encoder(g, node_ids)

    def forward(self, g: dgl.DGLGraph, node_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(g, node_ids)


def build_g_phi_from_fclass(
    fclass_ckpt_path: str | Path,
    device: torch.device | str = 'cpu',
    n_layers: int = 2,
    freeze_node_emb: bool = False,
) -> SubgraphEncoder:
    """Build g_phi and init from f_class checkpoint.

    Parameters
    ----------
    fclass_ckpt_path : path to f_class best_model.pt
    device : target device
    n_layers : GIN layers (default 2)
    freeze_node_emb : freeze the shared node embedding
    """
    device = torch.device(device)

    fclass = GraphClassifier(
        edge_feat_dim=PPA_EDGE_FEAT_DIM, n_classes=37,
        hidden_dim=PPA_HIDDEN_DIM, n_layers=5, dropout=PPA_DROPOUT,
    ).to(device)
    ckpt = torch.load(str(fclass_ckpt_path), map_location=device,
                      weights_only=False)
    fclass.load_state_dict(ckpt['model_state_dict'], strict=True)
    fclass.eval()
    for p in fclass.parameters():
        p.requires_grad = False

    g_phi = SubgraphEncoder(
        edge_feat_dim=PPA_EDGE_FEAT_DIM,
        hidden_dim=PPA_HIDDEN_DIM,
        dropout=PPA_DROPOUT,
        n_layers=n_layers,
    ).to(device)
    g_phi.init_from_fclass(fclass.encoder)

    if freeze_node_emb:
        g_phi.encoder.node_emb.requires_grad_(False)

    trainable = sum(p.numel() for p in g_phi.parameters() if p.requires_grad)
    total = sum(p.numel() for p in g_phi.parameters())
    logger.info("g_phi trainable: %d / %d params (%.1f%%)",
                trainable, total, 100.0 * trainable / max(total, 1))
    return g_phi

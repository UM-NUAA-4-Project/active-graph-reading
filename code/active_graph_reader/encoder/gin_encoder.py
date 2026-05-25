"""
GIN Encoder (OGB-official architecture)
========================================

Exact match with OGB's reference GIN for ogbg-ppa.
Supports fine-tuning strategies through per-layer freeze control and
optional bottleneck adapters (Houlsby et al., ICML 2019).
"""

import logging
from typing import List, Optional, Set

import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import BottleneckAdapter

logger = logging.getLogger(__name__)


class GINELayer(nn.Module):
    """Single GIN layer with edge features (per-layer edge encoder).

    Matches OGB's GINConv:
      edge_emb      = Linear(edge_feat_dim, hidden_dim)(edge_feat)
      message       = ReLU(src_h + edge_emb)
      aggregation   = sum
      h_new         = MLP((1+eps) * h + aggregated_messages)
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, mlp_hidden_mult: int = 2):
        super().__init__()
        bottleneck_dim = hidden_dim * mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        with g.local_scope():
            g.ndata['h'] = h
            edge_emb = self.edge_encoder(edge_feat)
            g.edata['e'] = edge_emb

            g.apply_edges(
                lambda edges: {'m': F.relu(edges.src['h'] + edges.data['e'])}
            )
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'neigh'))

            h_new = (1.0 + self.eps) * h + g.ndata['neigh']
            return self.mlp(h_new)


class GINEEncoder(nn.Module):
    """GIN encoder matching OGB official architecture with adapter support.

    Parameters
    ----------
    edge_feat_dim : int
        Raw edge feature dimension (PPA: 7).
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
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.node_emb = nn.Embedding(1, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GINELayer(hidden_dim, edge_feat_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Placeholder adapters — replaced by real ones when strategy is applied.
        # nn.Identity has zero parameters and passes through unchanged.
        self.adapters = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        self.pool = dglnn.AvgPooling()

    # ------------------------------------------------------------------
    # Fine-tuning strategy API
    # ------------------------------------------------------------------

    def apply_finetune_strategy(
        self,
        freeze_layers: Set[int],
        adapter_layers: Set[int],
        adapter_bottleneck_ratio: int = 16,
    ):
        """Configure encoder for fine-tuning.

        Parameters
        ----------
        freeze_layers : set of int
            Indices of GIN layers to freeze (conv + bn set to requires_grad=False).
        adapter_layers : set of int
            Indices of GIN layers to inject bottleneck adapters after.
        adapter_bottleneck_ratio : int
            Bottleneck ratio for adapters (hidden_dim // ratio).
        """
        bottleneck_dim = max(1, self.hidden_dim // adapter_bottleneck_ratio)

        for i in range(self.n_layers):
            # Freeze / unfreeze conv + bn
            requires = i not in freeze_layers
            for p in self.convs[i].parameters():
                p.requires_grad = requires
            for p in self.bns[i].parameters():
                p.requires_grad = requires

            # Adapter
            if i in adapter_layers:
                self.adapters[i] = BottleneckAdapter(self.hidden_dim, bottleneck_dim)
                logger.info("Layer %d: %s, adapter added (bottleneck=%d)",
                            i,
                            'frozen' if i in freeze_layers else 'trainable',
                            bottleneck_dim)
            else:
                self.adapters[i] = nn.Identity()
                logger.info("Layer %d: %s, no adapter",
                            i,
                            'frozen' if i in freeze_layers else 'trainable')

    def freeze_node_emb(self):
        """Freeze the shared node embedding (generic, not specific to partial graphs)."""
        self.node_emb.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        g: dgl.DGLGraph,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_emb(node_ids)

        edge_feat = g.edata.get('feat', None)
        if edge_feat is None:
            edge_feat_dim = self.convs[0].edge_encoder.in_features
            edge_feat = torch.zeros(g.num_edges(), edge_feat_dim, device=h.device)

        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Apply adapter if present (skip Identity placeholders)
            if not isinstance(self.adapters[i], nn.Identity):
                h = h + self.adapters[i](h)

        return self.pool(g, h)

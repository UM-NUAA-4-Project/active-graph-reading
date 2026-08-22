"""
GIN Encoder (OGB-official architecture)
========================================

Exact match with OGB's reference GIN for ogbg-ppa.
Supports optional Virtual Node and residual connections.
"""

import logging
from typing import List, Optional

import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """GIN encoder matching OGB official architecture.

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
    virtual_node : bool
        Enable Virtual Node (OGB gin-virtual). Default False for
        backward compatibility.
    residual : bool
        Enable residual connections between GIN layers. Default False.
    node_feat_dim : int
        Raw node feature dimension. 0 (default) uses Embedding(1, hidden_dim)
        (PPA: no node features); >0 uses a Linear projection from node
        features in ``g.ndata['feat']`` (e.g. CIFAR10-SuperPixel: 5).
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
        virtual_node: bool = False,
        residual: bool = False,
        node_feat_dim: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.virtual_node = virtual_node
        self.residual = residual

        self.node_feat_dim = int(node_feat_dim)
        if node_feat_dim > 0:
            self.node_proj = nn.Linear(int(node_feat_dim), hidden_dim)
        else:
            self.node_emb = nn.Embedding(1, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GINELayer(hidden_dim, edge_feat_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Virtual Node (optional, matches OGB gin-virtual)
        if virtual_node:
            self.vn_embedding = nn.Embedding(1, hidden_dim)
            nn.init.constant_(self.vn_embedding.weight.data, 0)
            self.vn_mlps = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.vn_mlps.append(nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.BatchNorm1d(2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ))

        self.pool = dglnn.AvgPooling()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        g: dgl.DGLGraph,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.node_feat_dim > 0:
            h = self.node_proj(g.ndata['feat'].to(node_ids.device))
        else:
            h = self.node_emb(node_ids)

        edge_feat = g.edata.get('feat', None)
        if edge_feat is None:
            edge_feat_dim = self.convs[0].edge_encoder.in_features
            edge_feat = torch.zeros(g.num_edges(), edge_feat_dim, device=h.device)

        # Virtual Node init
        if self.virtual_node:
            num_graphs = g.batch_size
            vn_h = self.vn_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=h.device)
            )
            batch_num_nodes = g.batch_num_nodes()
            node_to_graph = torch.arange(
                num_graphs, device=h.device
            ).repeat_interleave(batch_num_nodes)

        h_list = [h]
        for i in range(self.n_layers):
            # Add Virtual Node embedding to node features
            if self.virtual_node:
                h_list[i] = h_list[i] + vn_h[node_to_graph]

            h = self.convs[i](g, h_list[i], edge_feat)
            h = self.bns[i](h)

            if i == self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)

            if self.residual:
                h = h + h_list[i]

            h_list.append(h)

            # Update Virtual Node
            if self.virtual_node and i < self.n_layers - 1:
                g.ndata['_vn_tmp'] = h
                vn_agg = dgl.sum_nodes(g, '_vn_tmp')
                vn_h = vn_h + F.dropout(
                    self.vn_mlps[i](vn_agg + vn_h),
                    p=self.dropout, training=self.training,
                )

        return self.pool(g, h)

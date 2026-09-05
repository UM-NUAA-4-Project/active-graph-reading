"""GIN backbone（OGB 官方架构），自 active_graph_reader/gnn/gin_encoder.py 迁出。

新增：逐层 hidden_dims / dropouts 列表（旧标量参数自动展开，state_dict 键名
与旧 checkpoint 完全一致，可直接加载）。virtual_node / residual 仅支持
各层同宽（维度对齐限制），不同宽时构造即报错。
"""

import logging

import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .norm import make_norm

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

    两种参数形式（二选一，列表优先）：
    - 旧式标量：hidden_dim + n_layers + dropout（逐层统一）
    - 新式列表：hidden_dims / dropouts（逐层独立）

    state_dict 键名与旧 checkpoint 完全一致。
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
        hidden_dims: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        gnn_norm: Optional[str] = None,
        readout: str = 'mean',
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [int(hidden_dim)] * int(n_layers)
        if dropouts is None:
            dropouts = [float(dropout)] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError(
                f"dropouts 长度 {len(dropouts)} != hidden_dims 长度 {len(hidden_dims)}")
        if (virtual_node or residual) and len(set(hidden_dims)) > 1:
            raise ValueError("virtual_node / residual 要求各层同宽（维度对齐）")
        if readout not in ('mean', 'meanmax', 'sum'):
            raise ValueError(f"未知读出池化 '{readout}'，可选: mean, meanmax, sum")

        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropouts = [float(p) for p in dropouts]
        self.n_layers = len(self.hidden_dims)
        self.hidden_dim = self.hidden_dims[0]   # 兼容旧属性访问
        self.dropout = self.dropouts[0]
        self.virtual_node = virtual_node
        self.residual = residual
        self.readout = readout
        # meanmax 读出输出维度翻倍（GraphClassifier 以此定 head 输入维度）
        self.out_dim = self.hidden_dims[-1] * (2 if readout == 'meanmax' else 1)
        # 历史语义：GIN 层间恒为 BatchNorm（state_dict 键名 bns.i.*），
        # gnn_norm=None 保持兼容；'none'/'layernorm' 显式覆盖
        self.norm_type = gnn_norm if gnn_norm is not None else 'batchnorm'

        self.node_feat_dim = int(node_feat_dim)
        if node_feat_dim > 0:
            self.node_proj = nn.Linear(int(node_feat_dim), self.hidden_dims[0])
        else:
            self.node_emb = nn.Embedding(1, self.hidden_dims[0])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(GINELayer(self.hidden_dims[i], edge_feat_dim))
            self.bns.append(make_norm(self.hidden_dims[i], self.norm_type))

        # Virtual Node (optional, matches OGB gin-virtual)
        if virtual_node:
            self.vn_embedding = nn.Embedding(1, self.hidden_dims[0])
            nn.init.constant_(self.vn_embedding.weight.data, 0)
            self.vn_mlps = nn.ModuleList()
            for _ in range(self.n_layers - 1):
                h = self.hidden_dims[0]
                self.vn_mlps.append(nn.Sequential(
                    nn.Linear(h, 2 * h),
                    nn.BatchNorm1d(2 * h),
                    nn.ReLU(),
                    nn.Linear(2 * h, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                ))

        self.pool = dglnn.AvgPooling()
        if readout == 'meanmax':
            self.pool_max = dglnn.MaxPooling()
        elif readout == 'sum':
            self.pool = dglnn.SumPooling()

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
                h = F.dropout(h, p=self.dropouts[i], training=self.training)
            else:
                h = F.dropout(F.relu(h), p=self.dropouts[i], training=self.training)

            if self.residual:
                h = h + h_list[i]

            h_list.append(h)

            # Update Virtual Node
            if self.virtual_node and i < self.n_layers - 1:
                g.ndata['_vn_tmp'] = h
                vn_agg = dgl.sum_nodes(g, '_vn_tmp')
                vn_h = vn_h + F.dropout(
                    self.vn_mlps[i](vn_agg + vn_h),
                    p=self.dropouts[i], training=self.training,
                )

        if self.readout == 'meanmax':
            return torch.cat([self.pool(g, h), self.pool_max(g, h)], dim=-1)
        return self.pool(g, h)

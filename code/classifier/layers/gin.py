"""GIN backbone（OGB 官方架构）+ GNN+ 逐层管线（arXiv 2502.09263, Eq.11）。

逐层管线：MP(含边特征整合①) → 层间归一化② → σ → Dropout③ → 残差④ →
FFN⑤(收尾)；位置编码⑥在输入端注入；读出在末层后。各积木由
ClassifierConfig 字段控制插拔（见 plus/__init__ 的映射表），**默认档
与旧 checkpoint 的数值和 state_dict 键集逐位一致**。

virtual_node 为历史遗留（非 GNN+ 组件；全局聚合在揭示子图上分布漂移，
子图评估会退化——见 result/cifar10/classifier/GIN-res-vn/subgraph_eval.json），
保留原实现与键名（vn_embedding/vn_mlps）。

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

from ..plus.edge import EDGE_ENCODER_KINDS, make_edge_encoder
from ..plus.ffn import FFN
from ..plus.norm import make_norm
from ..plus.pos_enc import random_walk_structural_encoding
from ..plus.readout import make_readout

logger = logging.getLogger(__name__)


class GINELayer(nn.Module):
    """Single GIN layer with edge features (per-layer edge encoder).

    Matches OGB's GINConv:
      edge_emb      = edge_encoder(edge_feat)      ← plus/edge.py 积木
      message       = ReLU(src_h + edge_emb)       ← 边特征整合进聚合（Eq.6）
      aggregation   = sum
      h_new         = MLP((1+eps) * h + aggregated_messages)
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, mlp_hidden_mult: int = 2,
                 edge_encoder_kind: str = 'linear'):
        super().__init__()
        bottleneck_dim = hidden_dim * mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_encoder = make_edge_encoder(edge_feat_dim, hidden_dim,
                                              edge_encoder_kind)

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
    """GIN encoder，GNN+ 逐层管线骨架。

    两种参数形式（二选一，列表优先）：
    - 旧式标量：hidden_dim + n_layers + dropout（逐层统一）
    - 新式列表：hidden_dims / dropouts（逐层独立）

    plus/ 积木 kwargs（末尾追加，均有默认值=原行为）：
      readout / edge_encoder_kind / ffn / ffn_dropout / pos_enc / pos_enc_ksteps

    state_dict 键名与旧 checkpoint 完全一致（默认档下零新增键）。
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
        edge_encoder_kind: str = 'linear',
        ffn: bool = False,
        ffn_dropout: float = 0.0,
        pos_enc: str = 'none',
        pos_enc_ksteps: int = 8,
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
        if edge_encoder_kind not in EDGE_ENCODER_KINDS:
            raise ValueError(
                f"未知边编码器 '{edge_encoder_kind}'，可选: {EDGE_ENCODER_KINDS}")
        if pos_enc not in ('none', 'rwse'):
            raise ValueError(f"未知位置编码 '{pos_enc}'，可选: none, rwse")

        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropouts = [float(p) for p in dropouts]
        self.n_layers = len(self.hidden_dims)
        self.hidden_dim = self.hidden_dims[0]   # 兼容旧属性访问
        self.dropout = self.dropouts[0]
        self.virtual_node = virtual_node
        self.residual = residual
        # 历史语义：GIN 层间恒为 BatchNorm（state_dict 键名 bns.i.*），
        # gnn_norm=None 保持兼容；'none'/'layernorm' 显式覆盖
        self.norm_type = gnn_norm if gnn_norm is not None else 'batchnorm'

        # ⑥ 位置编码：开启时输入维 = node_feat_dim + ksteps（Eq.12 拼接投影）
        self.pos_enc = pos_enc
        self.pos_enc_ksteps = int(pos_enc_ksteps)
        self.node_feat_dim = int(node_feat_dim)
        proj_in = self.node_feat_dim + (self.pos_enc_ksteps
                                        if pos_enc == 'rwse' else 0)
        if pos_enc == 'rwse' or self.node_feat_dim > 0:
            self.node_proj = nn.Linear(proj_in, self.hidden_dims[0])
        else:
            self.node_emb = nn.Embedding(1, self.hidden_dims[0])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(GINELayer(self.hidden_dims[i], edge_feat_dim,
                                        edge_encoder_kind=edge_encoder_kind))
            self.bns.append(make_norm(self.hidden_dims[i], self.norm_type))

        # ⑤ FFN：关闭时 Identity 占位（零参数零键）
        if ffn:
            self.ffns = nn.ModuleList([FFN(d, ffn_dropout)
                                       for d in self.hidden_dims])
        else:
            self.ffns = nn.ModuleList([nn.Identity()
                                       for _ in range(self.n_layers)])

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

        # 读出（plus/readout.py；meanmax → 两池化拼接，out_dim 翻倍）
        self.pools = nn.ModuleList(make_readout(readout)[0])
        self.out_dim = self.hidden_dims[-1] * make_readout(readout)[1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        g: dgl.DGLGraph,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        # ⑥ 位置编码注入（Eq.12：[x ‖ x_RWSE] → 投影）
        if self.pos_enc == 'rwse':
            pe = random_walk_structural_encoding(
                g, self.pos_enc_ksteps).to(node_ids.device)
            if self.node_feat_dim > 0:
                x = torch.cat([g.ndata['feat'].to(node_ids.device), pe], dim=-1)
            else:
                x = torch.cat([torch.ones(pe.shape[0], 1,
                                          device=node_ids.device), pe], dim=-1)
            h = self.node_proj(x)
        elif self.node_feat_dim > 0:
            h = self.node_proj(g.ndata['feat'].to(node_ids.device))
        else:
            h = self.node_emb(node_ids)

        edge_feat = g.edata.get('feat', None)
        if edge_feat is None:
            edge_feat_dim = self.convs[0].edge_encoder.in_features \
                if hasattr(self.convs[0].edge_encoder, 'in_features') \
                else self.convs[0].edge_encoder[0].in_features
            edge_feat = torch.zeros(g.num_edges(), edge_feat_dim,
                                    device=h.device)

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

            h = self.convs[i](g, h_list[i], edge_feat)      # ① MP + 边特征整合
            h = self.bns[i](h)                              # ② 层间归一化
            h = h if i == self.n_layers - 1 else F.relu(h)  # σ（末层不激活）
            h = F.dropout(h, p=self.dropouts[i],            # ③ Dropout
                          training=self.training)
            if self.residual:                               # ④ 残差
                h = h + h_list[i]
            h = self.ffns[i](h)                             # ⑤ FFN（关闭=恒等）

            h_list.append(h)

            # Update Virtual Node
            if self.virtual_node and i < self.n_layers - 1:
                g.ndata['_vn_tmp'] = h
                vn_agg = dgl.sum_nodes(g, '_vn_tmp')
                vn_h = vn_h + F.dropout(
                    self.vn_mlps[i](vn_agg + vn_h),
                    p=self.dropouts[i], training=self.training,
                )

        # 读出（meanmax → 拼接）
        pooled = [pool(g, h) for pool in self.pools]
        return torch.cat(pooled, dim=-1) if len(pooled) > 1 else pooled[0]

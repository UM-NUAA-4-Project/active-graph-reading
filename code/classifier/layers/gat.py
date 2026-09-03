"""GAT backbone（多头图注意力，含边特征），与 GINEEncoder 接口对齐。

接口与 GINEEncoder 完全一致：
    __init__(edge_feat_dim, hidden_dims, dropouts, node_feat_dim,
             virtual_node, residual, n_heads=None, ...)
    forward(g, node_ids) -> (num_graphs, hidden_dims[-1])

注意力设计（在标准 GAT 上引入边特征，与 v1 协议"边特征是唯一
前沿候选区分通道"的立场一致）：
    logits_ij = LeakyReLU( a_l·(W h_i) + a_r·(W h_j) + edge_score(e_ij) )
    alpha_ij  = softmax_j(logits_ij)          （按头独立）
    m_ij      = (W h_j) + edge_encoder(e_ij)  （边信息进 value，同 GINE 思路）
    h_i'      = ELU( concat_heads( Σ_j alpha_ij * m_ij ) )

约束：hidden_dims[i] 必须能被 n_heads[i] 整除（concat heads）。
virtual_node / residual 仅支持各层同宽，同 GINEEncoder。
"""

import dgl
import dgl.nn as dglnn
import dgl.function as fn
import dgl.ops as dgl_ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

from .norm import make_norm


class GATLayer(nn.Module):
    """多头 GAT 层（边特征同时参与注意力打分与消息 value）。

    输入宽度 in_dim（各层同宽时即 hidden_dim），输出宽度
    n_heads * out_per_head = out_dim（由 encoder 保证 == hidden_dims[i]）。
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int,
                 edge_feat_dim: int, negative_slope: float = 0.2):
        super().__init__()
        if out_dim % n_heads != 0:
            raise ValueError(
                f"out_dim {out_dim} 不能被 n_heads {n_heads} 整除（concat heads）")
        self.n_heads = n_heads
        self.out_per_head = out_dim // n_heads
        self.negative_slope = negative_slope

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 注意力向量（每头独立）
        self.attn_l = nn.Parameter(torch.zeros(1, n_heads, self.out_per_head))
        self.attn_r = nn.Parameter(torch.zeros(1, n_heads, self.out_per_head))
        # 边特征 → 每头一个注意力偏置；边特征 → value 空间
        self.edge_score = nn.Linear(edge_feat_dim, n_heads)
        self.edge_encoder = nn.Linear(edge_feat_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            H = self.fc(h).view(-1, self.n_heads, self.out_per_head)
            g.ndata['h'] = H
            # 注意力 logits：src/dst 投影 + 边特征打分
            g.ndata['el'] = (H * self.attn_l).sum(-1, keepdim=True)
            g.ndata['er'] = (H * self.attn_r).sum(-1, keepdim=True)
            g.edata['es'] = self.edge_score(edge_feat).view(
                -1, self.n_heads, 1)
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            logits = F.leaky_relu(g.edata['e'] + g.edata['es'],
                                  self.negative_slope)
            alpha = dgl_ops.edge_softmax(g, logits)
            # value 消息：dst 邻居投影 + 边编码（edge-dependent value）
            g.edata['em'] = self.edge_encoder(edge_feat).view(
                -1, self.n_heads, self.out_per_head)
            g.apply_edges(fn.u_add_e('h', 'em', 'm'))
            g.edata['m'] = alpha * g.edata['m']
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_new'))
            return g.ndata['h_new'].reshape(-1, self.n_heads * self.out_per_head)


class GATEncoder(nn.Module):
    """多头 GAT encoder，构造/前向接口与 GINEEncoder 一致。

    两种参数形式（二选一，列表优先）：
    - 旧式标量：hidden_dim + n_layers + dropout（逐层统一）
    - 新式列表：hidden_dims / dropouts / n_heads（逐层独立）

    state_dict 键名（encoder.convs.{i}.fc.weight / .attn_l / .edge_encoder.weight）
    可被 tools/migrate_legacy_classifier_ckpt.py 识别（仅 legacy 迁移用）；
    正常流程下结构以 checkpoint 内嵌 config 为准。
    """

    DEFAULT_HEADS = 4

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
        n_heads: Optional[Union[int, List[int]]] = None,
        gnn_norm: Optional[str] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [int(hidden_dim)] * int(n_layers)
        if dropouts is None:
            dropouts = [float(dropout)] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError(
                f"dropouts 长度 {len(dropouts)} != hidden_dims 长度 {len(hidden_dims)}")

        if n_heads is None:
            heads = [self.DEFAULT_HEADS] * len(hidden_dims)
        elif isinstance(n_heads, int):
            heads = [n_heads] * len(hidden_dims)
        else:
            heads = [int(x) for x in n_heads]
        if len(heads) != len(hidden_dims):
            raise ValueError(
                f"n_heads 长度 {len(heads)} != hidden_dims 长度 {len(hidden_dims)}")
        for i, (hd, nh) in enumerate(zip(hidden_dims, heads)):
            if hd % nh != 0:
                raise ValueError(
                    f"hidden_dims[{i}]={hd} 不能被 n_heads[{i}]={nh} 整除"
                    f"（GAT concat heads，可调整 hidden 或 heads）")
        if (virtual_node or residual) and len(set(hidden_dims)) > 1:
            raise ValueError("virtual_node / residual 要求各层同宽（维度对齐）")

        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropouts = [float(p) for p in dropouts]
        self.n_heads = heads
        self.n_layers = len(self.hidden_dims)
        self.hidden_dim = self.hidden_dims[0]   # 兼容旧属性访问
        self.dropout = self.dropouts[0]
        self.virtual_node = virtual_node
        self.residual = residual
        # 历史语义：GAT 层间原本无归一化，gnn_norm=None 保持兼容
        self.norm_type = gnn_norm if gnn_norm is not None else 'none'

        self.node_feat_dim = int(node_feat_dim)
        if node_feat_dim > 0:
            self.node_proj = nn.Linear(int(node_feat_dim), self.hidden_dims[0])
        else:
            self.node_emb = nn.Embedding(1, self.hidden_dims[0])

        self.convs = nn.ModuleList()
        in_dim = self.hidden_dims[0]
        for i in range(self.n_layers):
            self.convs.append(GATLayer(
                in_dim, self.hidden_dims[i], heads[i], edge_feat_dim))
            in_dim = self.hidden_dims[i]
        self.bns = nn.ModuleList(
            make_norm(d, self.norm_type) for d in self.hidden_dims)

        # Virtual Node（与 GINEEncoder 相同的模式）
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

        if self.virtual_node:
            num_graphs = g.batch_size
            vn_h = self.vn_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=h.device)
            )
            node_to_graph = torch.arange(
                num_graphs, device=h.device
            ).repeat_interleave(g.batch_num_nodes())

        for i in range(self.n_layers):
            if self.virtual_node:
                h = h + vn_h[node_to_graph]

            h_prev = h
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)

            if i == self.n_layers - 1:
                h = F.dropout(h, p=self.dropouts[i], training=self.training)
            else:
                h = F.dropout(F.elu(h), p=self.dropouts[i], training=self.training)

            if self.residual:
                h = h + h_prev

            if self.virtual_node and i < self.n_layers - 1:
                g.ndata['_vn_tmp'] = h
                vn_agg = dgl.sum_nodes(g, '_vn_tmp')
                vn_h = vn_h + F.dropout(
                    self.vn_mlps[i](vn_agg + vn_h),
                    p=self.dropouts[i], training=self.training,
                )

        return self.pool(g, h)

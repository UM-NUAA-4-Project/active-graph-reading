"""FrontierNodesEncoder：在扩展子图上运行 GNN 以编码前沿节点。

扩展子图 = V_revealed ∪ frontier_nodes + revealed→revealed 边（双向）+ revealed→frontier 边（单向）
节点特征使用可学习 type embedding（revealed=0, frontier=1），
边特征使用原图边特征（这是区分前沿节点的关键特征）。

优化：使用向量化 tensor 操作代替 Python 循环，与 SubgraphBuilder 模式一致。
"""

from typing import List, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.graph_state import GraphState
from ..encoder.gin_encoder import GINELayer


class FrontierNodesEncoder(nn.Module):
    """在扩展子图上运行 GNN，编码前沿节点。

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度（默认 128）
    n_layers : int       GNN 层数（默认 2）
    dropout : float      Dropout 概率（默认 0.1）
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.node_type_emb = nn.Embedding(2, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GINELayer(hidden_dim, edge_feat_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(
        self, graph_state: GraphState
    ) -> Tuple[torch.Tensor, List[int]]:
        """编码前沿节点。Returns: (frontier_embeds, frontier_nodes_original_ids)."""
        revealed = graph_state.get_revealed_sorted()
        frontier = graph_state.get_frontier_nodes()

        if len(frontier) == 0:
            return torch.empty(0, self.hidden_dim), []

        n_revealed = len(revealed)
        n_total = n_revealed + len(frontier)

        # ---- 向量化 old→new ID 映射（同 SubgraphBuilder 模式） ----
        max_old_id = max(revealed[-1], frontier[-1])
        old_to_new = torch.full((max_old_id + 1,), -1, dtype=torch.long)
        old_to_new[revealed] = torch.arange(n_revealed)
        old_to_new[frontier] = torch.arange(n_revealed, n_total)

        src_parts = []
        dst_parts = []
        feat_parts = []

        # ---- revealed ↔ revealed 边（双向）----
        if graph_state.revealed_edge_keys:
            rk_list = list(graph_state.revealed_edge_keys)
            rk_t = torch.tensor(rk_list, dtype=torch.long)  # (E_r, 2)
            edge_ids_r = [graph_state._edge_key_to_id[k] for k in rk_list]
            feat_r = graph_state.edge_feat[edge_ids_r]  # (E_r, D)

            u_r = old_to_new[rk_t[:, 0]]  # (E_r,)
            v_r = old_to_new[rk_t[:, 1]]  # (E_r,)

            src_parts.append(u_r)
            src_parts.append(v_r)  # reverse
            dst_parts.append(v_r)
            dst_parts.append(u_r)  # reverse
            feat_parts.append(feat_r)
            feat_parts.append(feat_r)  # repeat for reverse

        # ---- revealed → frontier 边（单向）----
        if graph_state.frontier_edges:
            # 批量获取 frontier edge 的 edge IDs
            fr_keys = list(graph_state.frontier_edges)
            fr_ids = [graph_state._edge_key_to_id[(min(u, v), max(u, v))] for u, v in fr_keys]
            feat_f = graph_state.edge_feat[fr_ids]  # (F_e, D)

            fr_t = torch.tensor(fr_keys, dtype=torch.long)  # (F_e, 2)
            u_f = old_to_new[fr_t[:, 0]]  # revealed → frontier (directional)
            v_f = old_to_new[fr_t[:, 1]]

            src_parts.append(u_f)
            dst_parts.append(v_f)
            feat_parts.append(feat_f)

        # ---- 构建 DGL 图 ----
        if len(src_parts) == 0:
            g = dgl.graph(([], []), num_nodes=n_total)
            edge_feat = torch.empty(0, graph_state.edge_feat_dim)
        else:
            src = torch.cat(src_parts, dim=0)
            dst = torch.cat(dst_parts, dim=0)
            edge_feat = torch.cat(feat_parts, dim=0)
            g = dgl.graph((src, dst), num_nodes=n_total)

        # ---- GNN 前向传播 ----
        node_types = torch.zeros(n_total, dtype=torch.long)
        node_types[n_revealed:] = 1
        h = self.node_type_emb(node_types)

        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

        frontier_embeds = h[n_revealed:]
        return frontier_embeds, frontier

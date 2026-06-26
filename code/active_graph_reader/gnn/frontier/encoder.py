"""FrontierNodesEncoder：在扩展子图上运行 GNN 以编码前沿节点。

扩展子图 = V_revealed ∪ frontier_nodes + revealed→revealed 边（双向）+ revealed→frontier 边（单向）
节点特征使用可训练 type embedding（revealed=0, frontier=1），
边特征使用原图边特征（这是区分前沿节点的关键特征）。

图构建逻辑已抽取至 frontier_graph_builder.py 的 FrontierGraphBuilder。
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import FrontierSubgraph
from ..gin_encoder import GINELayer


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
        self, subgraph: FrontierSubgraph
    ) -> Tuple[torch.Tensor, List[int]]:
        """编码前沿节点。Returns: (frontier_embeds, frontier_nodes_original_ids)."""
        g = subgraph.g
        n_revealed = subgraph.n_revealed
        frontier = subgraph.frontier_original_ids

        if len(frontier) == 0:
            return torch.empty(0, self.hidden_dim), []

        device = self.node_type_emb.weight.device
        g = subgraph.g.to(device)
        n_total = g.num_nodes()
        edge_feat = g.edata['feat']

        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        node_types[n_revealed:] = 1
        h = self.node_type_emb(node_types)

        # GNN 前向传播
        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

        frontier_embeds = h[n_revealed:]
        return frontier_embeds, frontier

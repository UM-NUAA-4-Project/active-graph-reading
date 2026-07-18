"""FrontierNodesEncoder：在扩展子图上运行 GNN 以编码前沿节点。

扩展子图 = V_revealed ∪ frontier_nodes + revealed→revealed 边（双向）+ revealed→frontier 边（单向）
节点特征使用可训练 type embedding（revealed=0, frontier=1），
边特征使用原图边特征（这是区分前沿节点的关键特征）。

图构建逻辑已抽取至 frontier_graph_builder.py 的 FrontierGraphBuilder。
"""

from typing import List, Tuple

import dgl
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

        # 节点初始特征：revealed 有原图特征则用，否则 type embedding；frontier 恒为零
        if 'feat' in g.ndata:
            node_feat = g.ndata['feat'].to(device)
            if node_feat.shape[1] != self.hidden_dim:
                node_feat = F.pad(node_feat, (0, self.hidden_dim - node_feat.shape[1]))
            h = node_feat
        else:
            node_types = torch.zeros(n_total, dtype=torch.long, device=device)
            node_types[n_revealed:] = 1
            h = self.node_type_emb(node_types)
            h[n_revealed:] = 0  # frontier 恒为零

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

    def forward_batch(
        self, subgraphs: List[FrontierSubgraph],
    ) -> List[Tuple[torch.Tensor, List[int]]]:
        """一次 GNN forward 编码多个扩展子图，返回每个子图的 frontier embedding。

        将多个小图的 GNN kernel launch 合并为一次，减少 GPU kernel overhead。
        """
        if not subgraphs:
            return []

        device = self.node_type_emb.weight.device
        batch_g = dgl.batch([sg.g.to(device) for sg in subgraphs])
        n_total = batch_g.num_nodes()

        # 计算每个子图在 batch 中的偏移
        node_counts = [sg.g.num_nodes() for sg in subgraphs]
        offsets = [0]
        for c in node_counts:
            offsets.append(offsets[-1] + c)

        # 构建 node type embedding，子图内前 n_revealed 为 revealed(0)，后为 frontier(1)
        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        frontier_starts: List[int] = []
        frontier_counts: List[int] = []
        for i, sg in enumerate(subgraphs):
            start = offsets[i]
            n_rev = sg.n_revealed
            n_fr = len(sg.frontier_original_ids)
            frontier_starts.append(start + n_rev)
            frontier_counts.append(n_fr)
            if n_fr > 0:
                node_types[start + n_rev:start + n_rev + n_fr] = 1

        h = self.node_type_emb(node_types)
        for start, cnt in zip(frontier_starts, frontier_counts):
            if cnt > 0:
                h[start:start + cnt] = 0  # frontier 初始化为零

        edge_feat = batch_g.edata.get('feat',
                      torch.empty(0, 7, device=device))

        # GNN（一次 forward）
        for i in range(self.n_layers):
            h = self.convs[i](batch_g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 拆分回每个子图的 frontier embeddings
        results: List[Tuple[torch.Tensor, List[int]]] = []
        for i, sg in enumerate(subgraphs):
            start = frontier_starts[i]
            cnt = frontier_counts[i]
            if cnt == 0:
                results.append((torch.empty(0, self.hidden_dim, device=device), []))
            else:
                results.append((h[start:start + cnt], sg.frontier_original_ids))

        return results

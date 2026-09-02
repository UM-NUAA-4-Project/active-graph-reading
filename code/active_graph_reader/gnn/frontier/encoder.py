"""FrontierNodesEncoder：在扩展子图上运行 GNN 以编码前沿节点。

扩展子图 = V_revealed ∪ frontier_nodes + revealed→revealed 边（双向）+ revealed→frontier 边（单向）
节点特征使用可训练 type embedding（revealed=0, frontier=1），
边特征使用原图边特征（这是区分前沿节点的关键特征）。

图构建逻辑已抽取至 frontier_graph_builder.py 的 FrontierGraphBuilder。

支持逐层 hidden_dims/dropouts（旧标量自动展开，state_dict 键名不变）。
注意：node_type_emb 维度 = hidden_dims[0]；空前沿返回宽度 = hidden_dims[-1]。
"""

from typing import List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.layers.gin import GINELayer
from .graph_builder import FrontierSubgraph


class FrontierNodesEncoder(nn.Module):
    """在扩展子图上运行 GNN，编码前沿节点。

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度（默认 128，uniform 形式）
    n_layers : int       GNN 层数（默认 2）
    dropout : float      Dropout 概率（默认 0.1）
    node_feat_dim : int  原始节点特征维度（0=无节点特征，用 type embedding）
    hidden_dims/dropouts : 逐层列表（提供时覆盖标量形式）
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        node_feat_dim: int = 0,
        hidden_dims: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [int(hidden_dim)] * int(n_layers)
        if dropouts is None:
            dropouts = [float(dropout)] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError("dropouts 与 hidden_dims 长度不一致")

        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropouts = [float(p) for p in dropouts]
        self.n_layers = len(self.hidden_dims)
        self.hidden_dim = self.hidden_dims[-1]   # 输出宽度（兼容旧属性）
        self.dropout = self.dropouts[0]

        self.node_type_emb = nn.Embedding(2, self.hidden_dims[0])
        self.node_feat_proj = (
            nn.Linear(node_feat_dim, self.hidden_dims[0]) if node_feat_dim > 0 else None
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(GINELayer(self.hidden_dims[i], edge_feat_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dims[i]))

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

        # 节点初始特征：
        #   有原始节点特征 → 投影 revealed，frontier 恒为零
        #   无原始节点特征 → type embedding（revealed=0），frontier 恒为零
        if 'feat' in g.ndata and self.node_feat_proj is not None:
            h = self.node_feat_proj(g.ndata['feat'].to(device))
            h[n_revealed:] = 0
        else:
            node_types = torch.zeros(n_total, dtype=torch.long, device=device)
            node_types[n_revealed:] = 1
            h = self.node_type_emb(node_types)
            h[n_revealed:] = 0

        # GNN 前向传播
        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropouts[i], training=self.training)
            else:
                h = F.dropout(h, p=self.dropouts[i], training=self.training)

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

        frontier_starts: List[int] = []
        frontier_counts: List[int] = []

        # 节点初始特征：
        #   有原始节点特征 → 投影 revealed，frontier 恒为零
        #   无原始节点特征 → type embedding（revealed=0），frontier 恒为零
        if 'feat' in batch_g.ndata and self.node_feat_proj is not None:
            h = self.node_feat_proj(batch_g.ndata['feat'].to(device))
            for i, sg in enumerate(subgraphs):
                start = offsets[i]
                n_rev = sg.n_revealed
                n_fr = len(sg.frontier_original_ids)
                frontier_starts.append(start + n_rev)
                frontier_counts.append(n_fr)
                if n_fr > 0:
                    h[start + n_rev:start + n_rev + n_fr] = 0
        else:
            node_types = torch.zeros(n_total, dtype=torch.long, device=device)
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
                    h[start:start + cnt] = 0

        edge_feat = batch_g.edata.get('feat',
                      torch.empty(0, self.convs[0].edge_encoder.in_features,
                                  device=device))

        # GNN（一次 forward）
        for i in range(self.n_layers):
            h = self.convs[i](batch_g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropouts[i], training=self.training)
            else:
                h = F.dropout(h, p=self.dropouts[i], training=self.training)

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

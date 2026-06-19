"""GSAE (Graph Semantic AutoEncoder): 3 层 GNN + encoder + decoder。

从 f_class 预训练权重初始化前 3 层 GNN，通过 GSAE encoder 将 300 维图嵌入
压缩到 128 维语义潜在向量 z，decoder 重构回 v̂。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gnn.gin_encoder import GINELayer
from ..gnn.frontier.graph_builder import FrontierSubgraph


@dataclass
class GSAEOutput:
    z: torch.Tensor              # (1, 128) 语义潜在向量
    v: torch.Tensor              # (1, 300) 图嵌入（pool V_revealed）
    v_hat: torch.Tensor          # (1, 300) decoder 重构
    frontier_embeds: torch.Tensor  # (N_f, 300) frontier 节点嵌入（未降维）
    frontier_original_ids: list   # frontier 节点原图 ID


class GSAE(nn.Module):
    """3 层 GIN GNN + autoencoder 压缩。

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度，默认 300
    k_state : int        语义潜在维度，默认 128
    n_layers : int       GNN 层数，默认 3
    dropout : float      Dropout，默认 0.5
    """

    def __init__(self, edge_feat_dim=7, hidden_dim=300, k_state=128,
                 n_layers=3, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_state = k_state
        self.n_layers = n_layers
        self.dropout = dropout

        self.node_type_emb = nn.Embedding(2, hidden_dim)  # 0=revealed, 1=frontier

        self.convs = nn.ModuleList([
            GINELayer(hidden_dim, edge_feat_dim) for _ in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])

        self.encoder = nn.Linear(hidden_dim, k_state)   # 300 -> 128
        self.decoder = nn.Linear(k_state, hidden_dim)    # 128 -> 300

    def init_from_fclass(self, checkpoint_path: str, device='cpu'):
        """从 f_class checkpoint 加载前 n_layers 层 GNN 权重。

        映射规则：encoder.convs.{i}.* -> convs.{i}.*
                 encoder.bns.{i}.*   -> bns.{i}.*
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict']

        own_state = self.state_dict()
        for own_key in own_state:
            ckpt_key = 'encoder.' + own_key
            if ckpt_key in state:
                own_state[own_key] = state[ckpt_key]

        self.load_state_dict(own_state, strict=False)
        loaded = sum(1 for k in own_state if 'encoder.' + k in state)
        print(f"  GSAE: loaded {loaded}/{len(own_state)} params from f_class")

    def forward(self, subgraph: FrontierSubgraph) -> GSAEOutput:
        """编码 FrontierSubgraph，返回节点嵌入和图嵌入。"""
        g = subgraph.g
        n_revealed = subgraph.n_revealed
        frontier = subgraph.frontier_original_ids
        n_total = g.num_nodes()

        if n_total == 0:
            device = self.node_type_emb.weight.device
            return GSAEOutput(
                z=torch.zeros(1, self.k_state, device=device),
                v=torch.zeros(1, self.hidden_dim, device=device),
                v_hat=torch.zeros(1, self.hidden_dim, device=device),
                frontier_embeds=torch.empty(0, self.hidden_dim),
                frontier_original_ids=[],
            )

        device = self.node_type_emb.weight.device
        g = g.to(device)
        edge_feat = g.edata['feat']

        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        node_types[n_revealed:] = 1
        h = self.node_type_emb(node_types)

        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 图级 pooling（仅 V_revealed 节点取 mean）
        v = h[:n_revealed].mean(dim=0, keepdim=True)  # (1, 300)

        z = self.encoder(v)
        v_hat = self.decoder(z)

        frontier_embeds = h[n_revealed:]

        return GSAEOutput(
            z=z, v=v, v_hat=v_hat,
            frontier_embeds=frontier_embeds,
            frontier_original_ids=frontier,
        )

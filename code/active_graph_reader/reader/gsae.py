"""GSAE (Graph Semantic AutoEncoder): 2 层 GNN + attention pooling。

从 f_class 预训练权重初始化前 2 层 GNN，通过结构感知 attention pooling
将节点嵌入聚合为 300 维图嵌入 v。encoder/decoder 已移除，编码功能合并到
EvidenceNetwork 中。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gnn.gin_encoder import GINELayer
from ..gnn.frontier.graph_builder import FrontierSubgraph


@dataclass
class GSAEOutput:
    z: torch.Tensor = None              # 保留字段，恒为 None（encoder 已移除）
    v: torch.Tensor = None              # (1, 300) 图嵌入（pool V_revealed）
    v_hat: torch.Tensor = None          # 保留字段，恒为 None（decoder 已移除）
    frontier_embeds: torch.Tensor = None  # (N_f, 300) frontier 节点嵌入（未降维）
    frontier_original_ids: list = None   # frontier 节点原图 ID
    frontier_struct: torch.Tensor = None  # (N_f, 3) frontier 节点结构特征


def compute_structural_features(g, n_revealed: int):
    """计算子图中所有节点的 3 维结构特征。

    三个特征：
      维度0 — 归一化度 deg_in_revealed（与已揭示子图的连接度）
      维度1 — 归一化前沿度 (out_deg - in_deg) / max_abs → [-1, 1]
              正值=revealed 节点连接 frontier，负值=frontier 节点（收而不发）
      维度2 — 局部聚类系数（已揭示邻居互连程度）

    Returns: struct_feat: (n_total, 3), 归一化在子图内完成
    """
    n_total = g.num_nodes()
    in_deg = g.in_degrees().float()
    out_deg = g.out_degrees().float()

    deg_revealed = in_deg.clone()
    deg_frontier = out_deg - in_deg  # +: revealed 连接 frontier, 0: 无连接, -: frontier 节点

    norm_deg = deg_revealed / deg_revealed.max().clamp(min=1)
    norm_fr_deg = deg_frontier / deg_frontier.abs().max().clamp(min=1)  # 归一到 [-1, 1]

    clustering = torch.zeros(n_total)
    if n_total > 0 and n_revealed > 0:
        adj = g.adjacency_matrix().to_dense()  # 小子图，稠密化开销可忽略
        for v in range(n_revealed):
            d = int(deg_revealed[v].item())
            if d < 2:
                continue
            nbr_mask = (adj[v] > 0) & (torch.arange(n_total) < n_revealed)
            nbr = nbr_mask.nonzero(as_tuple=True)[0]
            nbr_count = nbr.numel()
            if nbr_count < 2:
                continue
            e_v = adj[nbr][:, nbr].sum().item() / 2.0
            clustering[v] = 2.0 * e_v / (nbr_count * (nbr_count - 1))

    return torch.stack([norm_deg, norm_fr_deg, clustering], dim=-1)


class StructAttentionPool(nn.Module):
    """单头结构感知 attention pooling。

    仅对 revealed 节点做加权求和，attention 分数由节点嵌入和结构特征共同决定。
    直觉：高度、高聚类系数、前沿附近的节点获得更高注意力。

    Parameters
    ----------
    dim_h : int     节点嵌入维度（300）
    dim_struct : int 结构特征维度（3）
    """

    def __init__(self, dim_h=300, dim_struct=3):
        super().__init__()
        self.W_q = nn.Linear(dim_h, dim_h)
        self.W_s = nn.Linear(dim_struct, dim_struct)
        self.a = nn.Linear(dim_h + dim_struct, 1)

    def forward(self, h, struct_feat, mask):
        """h: (n, dim_h), struct_feat: (n, dim_struct), mask: (n,) bool (True=revealed)"""
        q = self.W_q(h)
        s = self.W_s(struct_feat)
        combined = torch.cat([q, s], dim=-1)
        scores = self.a(F.leaky_relu(combined)).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(scores, dim=0)
        v_t = (alpha.unsqueeze(-1) * h).sum(dim=0)
        return v_t


class GSAE(nn.Module):
    """2-3 层 GIN GNN + attention pooling。

    encoder/decoder 已移除，编码功能合并到 EvidenceNetwork（Linear 300→128→37）。
    GSAE 现在只输出图嵌入 v（300 维）和 frontier 节点嵌入。

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度，默认 300
    k_state : int        保留参数（兼容旧 checkpoint），不再使用
    n_layers : int       GNN 层数，默认 3
    dropout : float      Dropout，默认 0.5
    """

    def __init__(self, edge_feat_dim=7, hidden_dim=300, k_state=128,
                 n_layers=3, dropout=0.5, use_struct=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_state = k_state
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_struct = use_struct

        self.node_type_emb = nn.Embedding(2, hidden_dim)  # 保留兼容，forward 中未使用
        self.struct_proj = nn.Linear(3, hidden_dim)        # 保留兼容

        self.convs = nn.ModuleList([
            GINELayer(hidden_dim, edge_feat_dim) for _ in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])

        self.attn_pool = StructAttentionPool(dim_h=hidden_dim, dim_struct=3)

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
        """编码 FrontierSubgraph，返回节点嵌入和图嵌入。

        改动 A：输入端增加 3 维结构特征（归一化度 + 归一化前沿度 + 局部聚类系数）
        改动 B：结构感知 attention pooling 替代 mean
        """
        g = subgraph.g
        n_revealed = subgraph.n_revealed
        frontier = subgraph.frontier_original_ids
        n_total = g.num_nodes()

        if n_total == 0:
            device = self.node_type_emb.weight.device
            return GSAEOutput(
                z=None,
                v=torch.zeros(1, self.hidden_dim, device=device),
                v_hat=None,
                frontier_embeds=torch.empty(0, self.hidden_dim, device=device),
                frontier_original_ids=[],
            )

        device = self.node_type_emb.weight.device

        if self.use_struct:
            struct_feat = compute_structural_features(g, n_revealed)  # CPU 上计算
        else:
            struct_feat = torch.zeros(n_total, 3)

        g = g.to(device)
        edge_feat = g.edata['feat']
        struct_feat = struct_feat.to(device)

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
            # revealed→emb[0], frontier→emb[1]，均参与梯度学习

        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.use_struct:
            mask = torch.ones(n_total, dtype=torch.bool, device=device)
            v = self.attn_pool(h, struct_feat, mask).unsqueeze(0)  # (1, 300) 全节点 attention 池化
        else:
            v = h.mean(dim=0, keepdim=True)  # (1, 300) 全节点均值池化

        frontier_embeds = h[n_revealed:]
        frontier_struct = struct_feat[n_revealed:]

        return GSAEOutput(
            z=None, v=v, v_hat=None,
            frontier_embeds=frontier_embeds,
            frontier_original_ids=frontier,
            frontier_struct=frontier_struct,
        )

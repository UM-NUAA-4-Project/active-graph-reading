"""GSAE (Graph Semantic AutoEncoder): 2 层 GNN + encoder + decoder。

从 f_class 预训练权重初始化前 2 层 GNN，通过 GSAE encoder 将 300 维图嵌入
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


def compute_structural_features(g, n_revealed: int):
    """计算子图中所有节点的 3 维结构特征。

    三个特征从不同尺度互补地刻画节点的结构角色：
      维度0 — 归一化度（deg_in_revealed）
      维度1 — 归一化前沿度（deg_in_frontier / 探索潜力）
      维度2 — 局部聚类系数（邻居互连程度 / 社区结构信号）

    利用 FrontierSubgraph 的边方向性高效计算：
      revealed 节点：in_deg = deg_in_revealed，out_deg - in_deg = deg_in_frontier
      frontier 节点：deg_in_revealed = in_deg（仅接收），deg_in_frontier = 0

    Returns: struct_feat: (n_total, 3), 归一化在子图内完成
    """
    n_total = g.num_nodes()
    in_deg = g.in_degrees().float()
    out_deg = g.out_degrees().float()

    deg_revealed = in_deg.clone()
    deg_frontier = torch.zeros(n_total)
    deg_frontier[:n_revealed] = (out_deg[:n_revealed] - in_deg[:n_revealed]).clamp(min=0)

    norm_deg = deg_revealed / deg_revealed.max().clamp(min=1)
    norm_fr_deg = deg_frontier / deg_frontier.max().clamp(min=1)

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
        self.struct_proj = nn.Linear(3, hidden_dim)        # 3 维结构特征 → hidden_dim

        self.convs = nn.ModuleList([
            GINELayer(hidden_dim, edge_feat_dim) for _ in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])

        self.encoder = nn.Linear(hidden_dim, k_state)   # 300 -> 128
        self.decoder = nn.Linear(k_state, hidden_dim)    # 128 -> 300

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
        改动 B：结构感知 attention pooling 替代 mean（仅对 revealed 节点加权求和）
        """
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

        # 结构特征在 CPU 上计算（小子图，避免 GPU launch overhead）
        struct_feat = compute_structural_features(g, n_revealed)

        g = g.to(device)
        edge_feat = g.edata['feat']
        struct_feat = struct_feat.to(device)

        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        node_types[n_revealed:] = 1
        h = self.node_type_emb(node_types) + self.struct_proj(struct_feat)

        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 结构感知 attention pooling（仅 V_revealed 节点）
        mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        mask[:n_revealed] = True
        v = self.attn_pool(h, struct_feat, mask).unsqueeze(0)  # (1, 300)

        z = self.encoder(v)
        v_hat = self.decoder(z)

        frontier_embeds = h[n_revealed:]

        return GSAEOutput(
            z=z, v=v, v_hat=v_hat,
            frontier_embeds=frontier_embeds,
            frontier_original_ids=frontier,
        )

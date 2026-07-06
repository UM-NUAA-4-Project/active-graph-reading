"""子图节点结构特征计算。

提供 compute_structural_features —— 计算子图中所有节点的 3 维结构特征
（归一化度 + 归一化前沿度 + 局部聚类系数），供 LearnableReader 使用。
"""

import torch


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

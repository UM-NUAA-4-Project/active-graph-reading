"""GNN+ 组件⑥位置编码（positional encoding）：RWSE（随机游走结构编码）。

论文（arXiv 2502.09263）选用 RWSE（"one of the most efficient PE methods"，
且在大图数据集上关闭 PE）。注入方式与 Eq.12 同构：[x ‖ x_RWSE] → node_proj。

选 RWSE 而非 Laplacian PE 的另一原因：LapPE 特征向量符号/基不唯一，
图被切除（揭示子图）后完全改变，与本仓库"子图鲁棒"目标冲突；
RWSE 逐子图可计算且相对稳定。
"""

import torch


def random_walk_structural_encoding(g, ksteps: int) -> torch.Tensor:
    """batched DGL 图的 RWSE：逐图计算 M=A_hat 的幂对角线，k=1..ksteps。

    A_hat = D⁻¹(A + I)（行归一化，含自环；多重边按重数累计）。
    返回 (total_nodes, ksteps) float32，节点顺序与 batched 图一致。

    逐图稠密幂适合超像素级小图（每图 ≤~1000 节点）；更大图请勿启用
    pos_enc（GNN+ 论文在 COCO-SP/ogbg-code2 上同样关闭 PE）。
    """
    device = g.device
    eu, ev = g.edges()
    bn = g.batch_num_nodes().tolist()
    be = g.batch_num_edges().tolist()
    outs = []
    off_n = off_e = 0
    for n, m in zip(bn, be):
        u = (eu[off_e:off_e + m] - off_n).long()
        v = (ev[off_e:off_e + m] - off_n).long()
        a = torch.zeros(n, n, device=device)
        a.index_put_((u, v), torch.ones(m, device=device), accumulate=True)
        a = a + torch.eye(n, device=device)
        m_hat = a / a.sum(dim=1, keepdim=True).clamp(min=1.0)
        diag = []
        p = m_hat
        for _ in range(ksteps):
            diag.append(p.diagonal())
            p = p @ m_hat
        outs.append(torch.stack(diag, dim=-1))   # (n, ksteps)
        off_n += n
        off_e += m
    if not outs:
        return torch.zeros(0, ksteps, device=device)
    return torch.cat(outs, dim=0)

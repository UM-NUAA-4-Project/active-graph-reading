"""EvidenceNetwork: v → Linear(300,128) + ReLU → Linear(128,37) + ReLU → e → Dirichlet → (b, u)。

将原 GSAE encoder (300→128) 合并到证据头中，形成两层投影。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EvidenceOutput:
    e: torch.Tensor       # (*, K) 证据向量（非负）
    alpha: torch.Tensor   # (*, K) Dirichlet 参数 = 1 + e
    b: torch.Tensor       # (*, K) belief mass = alpha / S
    u: torch.Tensor       # (*,)   不确定度 = K / S
    S: torch.Tensor       # (*,)   Dirichlet 强度


class EvidenceNetwork(nn.Module):
    """从图嵌入 v 投影证据，做 Dirichlet 参数化。

    两层投影：hidden_dim → 128 → k_evidence（吸收原 GSAE encoder）

    Parameters
    ----------
    hidden_dim : int    图嵌入维度，默认 300
    k_evidence : int    证据维度（类别数），默认 37
    """

    def __init__(self, hidden_dim=300, k_evidence=37):
        super().__init__()
        self.k_evidence = k_evidence
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k_evidence),
            nn.ReLU(),
        )

    def forward(self, v: torch.Tensor) -> EvidenceOutput:
        """v: (*, hidden_dim) 图嵌入"""
        e = self.evidence_head(v)                      # (*, K)
        alpha = 1.0 + e                                 # (*, K)
        S = alpha.sum(dim=-1, keepdim=True)             # (*, 1)
        b = e / S                                       # (*, K)  信仰质量 = e / S
        u = self.k_evidence / S.squeeze(-1)             # (*,)    不确定度 = K / S
        return EvidenceOutput(e=e, alpha=alpha, b=b, u=u, S=S.squeeze(-1))

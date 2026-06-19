"""EvidenceNetwork: z → e → Dirichlet → (b, u)。"""

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
    """从语义潜在向量 z 投影证据，做 Dirichlet 参数化。

    Parameters
    ----------
    k_state : int      语义潜在维度，默认 128
    k_evidence : int   证据维度（类别数），默认 37
    """

    def __init__(self, k_state=128, k_evidence=37):
        super().__init__()
        self.k_evidence = k_evidence
        self.evidence_head = nn.Sequential(
            nn.Linear(k_state, k_evidence),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> EvidenceOutput:
        e = self.evidence_head(z)                      # (*, K)
        alpha = 1.0 + e                                 # (*, K)
        S = alpha.sum(dim=-1, keepdim=True)             # (*, 1)
        b = e / S                                       # (*, K)  信仰质量 = e / S
        u = self.k_evidence / S.squeeze(-1)             # (*,)    不确定度 = K / S
        return EvidenceOutput(e=e, alpha=alpha, b=b, u=u, S=S.squeeze(-1))

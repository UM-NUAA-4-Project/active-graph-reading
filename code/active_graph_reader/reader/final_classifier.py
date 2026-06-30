"""FinalClassifier: LSTM 证据序列 + 最终子图 → Dempster 融合分类。

路径 A：LSTM 整合时序证据序列 [e_0, ..., e_B] → 信念
路径 B：最终前沿子图 → GSAE → EvidenceNetwork → 信念
Dempster 组合规则融合两条路径。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FinalOutput:
    b_total: torch.Tensor   # (1, K) 融合后信念
    u_total: torch.Tensor   # (1,)   融合后不确定度
    y_hat: torch.Tensor     # (1,)   预测类别


class EvidenceSequenceLSTM(nn.Module):
    """LSTM 整合时序证据序列。

    [e_0, e_1, ..., e_B] → LSTM → last_hidden → Linear → Softplus → e_lstm

    Parameters
    ----------
    k_evidence : int  证据维度（类别数），默认 37
    hidden : int      LSTM hidden size，默认 128
    """

    def __init__(self, k_evidence=37, hidden=128):
        super().__init__()
        self.k_evidence = k_evidence
        self.hidden = hidden
        self.lstm = nn.LSTM(k_evidence, hidden, batch_first=True)
        self.head = nn.Linear(hidden, k_evidence)

    def forward(self, e_seq: torch.Tensor):
        """e_seq: (B+1, K) → evidence: (K,)"""
        out, _ = self.lstm(e_seq.unsqueeze(0))   # (1, B+1, hidden)
        last = out[:, -1, :]                       # (1, hidden)
        return F.softplus(self.head(last)).squeeze(0)  # (K,)


class FinalClassifier(nn.Module):
    """双路径 Dempster 融合分类器。

    路径 A：EvidenceSequenceLSTM 整合时序证据
    路径 B：最终前沿子图 → GSAE → EvidenceNetwork

    Dempster 组合：
      K = (1-u1)(1-u2) - b1·b2
      b_k = (b1_k·b2_k + b1_k·u2 + b2_k·u1) / (1-K)
      u   = (u1·u2) / (1-K)
    """

    def __init__(self, k_evidence=37):
        super().__init__()
        self.k_evidence = k_evidence
        self.evidence_lstm = EvidenceSequenceLSTM(k_evidence)

    def forward(self, evidence_sequence, o_B, gsae, evidence_net):
        """双路径融合并分类。

        Parameters
        ----------
        evidence_sequence : List[Tensor]  长度 B+1 的证据序列，每个 (K,)
        o_B : FrontierSubgraph            最终前沿子图
        gsae : GSAE                       GSAE 模块
        evidence_net : EvidenceNetwork    证据头

        Returns
        -------
        FinalOutput
        """
        Kc = self.k_evidence

        # 路径 A：LSTM over evidence sequence
        e_seq = torch.stack(evidence_sequence)              # (B+1, K)
        e_lstm = self.evidence_lstm(e_seq)                  # (K,)
        alpha1 = 1.0 + e_lstm                               # (K,)
        S1 = alpha1.sum()
        b1 = (alpha1 - 1.0) / S1.clamp(min=1e-6)            # (K,)
        u1 = Kc / S1.clamp(min=1e-6)                        # scalar

        # 路径 B：最终子图证据
        gsae_out = gsae(o_B)
        ev_out = evidence_net(gsae_out.v)
        b2 = ev_out.b.squeeze(0)                            # (K,)
        u2 = ev_out.u                                       # scalar

        # Dempster 组合
        conflict = (1 - u1) * (1 - u2) - (b1 * b2).sum()
        denom = 1.0 - conflict.clamp(max=0.999)
        b_total = (b1 * b2 + b1 * u2 + b2 * u1) / denom    # (K,)
        u_total = (u1 * u2) / denom                          # scalar

        b_total = b_total.unsqueeze(0)                       # (1, K)
        u_total = u_total.unsqueeze(0)                       # (1,)
        y_hat = b_total.argmax(dim=-1)                       # (1,)

        return FinalOutput(b_total=b_total, u_total=u_total, y_hat=y_hat)

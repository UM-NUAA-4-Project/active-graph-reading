"""FinalClassifier: 双路径 Dirichlet 融合 + 最终分类。"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FinalOutput:
    b_total: torch.Tensor   # (*, K) 融合后信念
    u_total: torch.Tensor   # (*,)   融合后不确定度
    y_hat: torch.Tensor     # (*,)   预测类别


class FinalClassifier(nn.Module):
    """双路径 Dirichlet 融合（Dempster 组合规则）。

    路径 A：LSTM 累积 → 意见 ω1 = (b_lstm, u_lstm)
    路径 B：最终子图 → 意见 ω2 = (b_final, u_final)

    Dempster 组合（解析，无参数）：
      K = (1-u1)(1-u2) - b1·b2        （冲突质量）
      b_k = (b1_k·b2_k + b1_k·u2 + b2_k·u1) / (1-K)
      u   = (u1 · u2) / (1-K)

    参考：TMC (Han et al., ICLR'21 / TPAMI'22)
    """

    def __init__(self):
        super().__init__()

    def forward(self, o_B, gsae, evidence_net, alpha_lstm=None):
        """双路径融合并分类。

        Parameters
        ----------
        o_B : FrontierSubgraph  最终子图
        gsae : GSAE             GSAE 模块（共享参数）
        evidence_net : EvidenceNetwork  证据头（共享参数）
        alpha_lstm : Tensor or None  LSTM 累积的 Dirichlet 参数 (K,)
                                      None 表示仅用路径 B

        Returns
        -------
        FinalOutput
        """
        gsae_out = gsae(o_B)
        ev_out = evidence_net(gsae_out.v)

        if alpha_lstm is not None:
            Kc = alpha_lstm.shape[0]

            # 路径 A：LSTM 意见
            e_lstm = alpha_lstm - 1.0
            S_lstm = alpha_lstm.sum()
            b1 = e_lstm / S_lstm.clamp(min=1)
            u1 = Kc / S_lstm.clamp(min=1)

            # 路径 B：Final 意见
            b2 = ev_out.b.squeeze(0)
            u2 = ev_out.u

            # Dempster 组合规则
            K = (1 - u1) * (1 - u2) - (b1 * b2).sum()   # 冲突质量
            denom = 1.0 - K.clamp(max=0.999)

            b_total = (b1 * b2 + b1 * u2 + b2 * u1) / denom  # (K,)
            u_total = (u1 * u2) / denom                       # scalar

            # 重建 Dirichlet 参数（供需要 α 的下游使用）
            S_total = Kc / u_total.clamp(min=1e-6)
            e_total = b_total * S_total
            alpha_total = e_total + 1.0

            b_total = b_total.unsqueeze(0)                   # (1, K)
            u_total = u_total.unsqueeze(0)                   # (1,)
        else:
            alpha_total = ev_out.alpha
            S_total = alpha_total.sum(dim=-1, keepdim=True)
            b_total = (alpha_total - 1.0) / S_total
            u_total = (ev_out.u.unsqueeze(0) if ev_out.u.dim() == 0
                       else ev_out.u)

        y_hat = b_total.argmax(dim=-1)

        return FinalOutput(b_total=b_total, u_total=u_total, y_hat=y_hat)

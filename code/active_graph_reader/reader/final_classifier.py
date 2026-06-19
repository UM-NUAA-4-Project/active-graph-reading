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
    """双路径 Dirichlet 加法融合。

    路径 A：LSTM 累积的 alpha_lstm（包含时序信息）
    路径 B：最终子图 o_B 再过一次 GSAE+证据头 → alpha_final（静态信息）

    融合公式（解析，无参数）：
      alpha_total = alpha_lstm + alpha_final - 1
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
        ev_out = evidence_net(gsae_out.z)

        if alpha_lstm is not None:
            alpha_total = alpha_lstm.unsqueeze(0) + ev_out.alpha - 1.0
        else:
            alpha_total = ev_out.alpha

        S_total = alpha_total.sum(dim=-1, keepdim=True)
        b_total = (alpha_total - 1.0) / S_total     # 信仰质量 = e / S
        u_total = evidence_net.k_evidence / S_total.squeeze(-1)
        y_hat = b_total.argmax(dim=-1)

        return FinalOutput(b_total=b_total, u_total=u_total, y_hat=y_hat)

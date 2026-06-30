"""LSTMBeliefUpdater: DRBPN 式 LSTM 门控 + Dirichlet 闭式累加。"""

import torch
import torch.nn as nn


class LSTMBeliefUpdater(nn.Module):
    """DRBPN 式两阶段更新。

    阶段1（LSTM 门控，有参数）：学习 evidence 权重 gate_τ
    阶段2（Dirichlet 累加，无参数）：α_t = α_0 + Σ e_τ ⊙ gate_τ

    输入（4路拼接，203维）：
      [b_{t-1} (37), e_t (37), u_t (1), emb(a_{t-1}) (128)]

    Parameters
    ----------
    k_evidence : int       类别数（37）
    lstm_hidden : int      LSTM hidden size（128）
    action_emb_dim : int   动作 embedding 维度（128）
    """

    def __init__(self, k_evidence=37, lstm_hidden=128, action_emb_dim=128):
        super().__init__()
        self.k_evidence = k_evidence
        input_dim = k_evidence * 2 + 1 + action_emb_dim  # 37+37+1+128 = 203
        self.lstm = nn.LSTMCell(input_dim, lstm_hidden)
        self.gate = nn.Linear(lstm_hidden, k_evidence)       # tanh gate ∈ (-1, 1)
        self._alpha_accum = None
        self._h = None
        self._c = None

    def reset(self):
        """每图开始时调用：重置 LSTM state 和 Dirichlet 累加器。"""
        self._alpha_accum = torch.ones(self.k_evidence)  # uniform prior
        self._h = torch.zeros(1, self.lstm.hidden_size)
        self._c = torch.zeros(1, self.lstm.hidden_size)

    def forward(self, b_prev, e, u, emb_a_prev):
        """单步信念更新。

        Parameters
        ----------
        b_prev : Tensor (1, K)      上一步信念
        e : Tensor (1, K)           当前证据
        u : Tensor (1,) or scalar   当前不确定度
        emb_a_prev : Tensor (1, D)  上一步动作 embedding

        Returns
        -------
        b_t : Tensor (1, K)  更新后的信念
        """
        # 确保 LSTM 隐状态与输入在同一设备（reset 后 _h/_c 在 CPU 上）
        self._h = self._h.to(b_prev.device)
        self._c = self._c.to(b_prev.device)

        u_t = u.unsqueeze(0).unsqueeze(-1) if u.dim() == 0 else u.view(1, -1)
        lstm_input = torch.cat([b_prev, e, u_t, emb_a_prev], dim=-1)  # (1, 203)

        h, c = self.lstm(lstm_input, (self._h, self._c))
        self._h, self._c = h, c

        gate = torch.tanh(self.gate(h))                  # (1, K) gate ∈ (-1, 1)

        # DRBPN 逐类不确定性调制：
        # Var(b_k) = α_k·(S-α_k) / (S²·(S+1))  — Dirichlet 边际方差
        # κ_k = σ_k 直接用作增益（不除以 u）：
        #   - α_k≈S/2 时 σ_k 取峰值 → 竞争中的类别最开放
        #   - α_k→0 或 α_k→S 时 σ_k→0 → 边缘类和决胜类都保守
        #   - σ_k ∝ 1/√S → 总证据越多整体越谨慎
        self._alpha_accum = self._alpha_accum.to(e.device)

        S_prev = self._alpha_accum.sum()
        var_k = (self._alpha_accum * (S_prev - self._alpha_accum)
                 / (S_prev * S_prev * (S_prev + 1) + 1e-6))
        sigma_k = var_k.sqrt()                                     # (K,) 逐类不确定性
        kappa = sigma_k                                            # (K,) 逐类 Kalman 增益
        self._alpha_accum = self._alpha_accum + kappa * e.squeeze(0) * gate.squeeze(0)
        self._alpha_accum = torch.clamp(self._alpha_accum, min=1e-6)  # Dirichlet 参数需 > 0

        S = self._alpha_accum.sum()
        b_t = self._alpha_accum / S
        return b_t.unsqueeze(0)

    @property
    def alpha_accum(self):
        """当前 Dirichlet 累加参数（供 FinalClassifier 路径 A 使用）。"""
        return self._alpha_accum

    @property
    def current_u(self):
        """当前不确定度 u_t = K / S_t。"""
        S = self._alpha_accum.sum()
        return self.k_evidence / S.clamp(min=1)

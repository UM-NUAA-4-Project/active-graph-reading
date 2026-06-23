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
        # DRBPN gate: i_τ ⊙ g̃_τ ∈ (-1, 1)，允许正/负证据
        self.gate_i = nn.Linear(lstm_hidden, k_evidence)   # input gate (sigmoid)
        self.gate_g = nn.Linear(lstm_hidden, k_evidence)   # candidate (tanh)
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
        u_t = u.unsqueeze(0).unsqueeze(-1) if u.dim() == 0 else u.view(1, -1)
        lstm_input = torch.cat([b_prev, e, u_t, emb_a_prev], dim=-1)  # (1, 203)

        h, c = self.lstm(lstm_input, (self._h, self._c))
        self._h, self._c = h, c

        i_t = torch.sigmoid(self.gate_i(h))     # (1, K) 输入门
        g_t = torch.tanh(self.gate_g(h))         # (1, K) 候选方向
        gate = i_t * g_t                          # (1, K) gate_τ ∈ (-1, 1)

        # DRBPN 逐类不确定性调制：
        # Var(b_k) = α_k·(S-α_k) / (S²·(S+1))  — Dirichlet 边际方差
        # 高证据类方差小→κ小→不易被新证据扰动；低证据类方差大→κ大→开放吸收
        S_prev = self._alpha_accum.sum()
        var_k = (self._alpha_accum * (S_prev - self._alpha_accum)
                 / (S_prev * S_prev * (S_prev + 1) + 1e-6))
        sigma_k = var_k.sqrt()                                     # (K,) 逐类不确定性
        kappa = sigma_k / (u + 1e-6)                               # (K,) 逐类 Kalman 增益

        self._alpha_accum = self._alpha_accum.to(e.device)
        self._alpha_accum = self._alpha_accum + kappa * e.squeeze(0) * gate.squeeze(0)
        self._alpha_accum = torch.clamp(self._alpha_accum, min=1e-6)  # Dirichlet 参数需 > 0

        S = self._alpha_accum.sum()
        b_t = self._alpha_accum / S
        return b_t.unsqueeze(0)

    @property
    def alpha_accum(self):
        """当前 Dirichlet 累加参数（供 FinalClassifier 路径 A 使用）。"""
        return self._alpha_accum.clone()

    @property
    def current_u(self):
        """当前不确定度 u_t = K / S_t。"""
        S = self._alpha_accum.sum()
        return self.k_evidence / S.clamp(min=1)

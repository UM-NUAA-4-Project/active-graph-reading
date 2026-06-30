"""SimpleDirichletAccumulator: 简单 Dirichlet 累积信念更新。

α_t = 1 + Σ_{τ=0}^{t} e_τ  (uniform prior + 累积证据)
b_t = (α_t - 1) / S_t       (信念质量)
u_t = K / S_t               (不确定度)
"""

import torch
import torch.nn as nn


class SimpleDirichletAccumulator(nn.Module):
    """简单 Dirichlet 累积器（无参数，纯统计累加）。

    替代原 LSTMBeliefUpdater 的复杂 gate 机制。
    每步直接累加证据 e_t，从 Dirichlet 参数计算信念和不确定度。

    Parameters
    ----------
    k_evidence : int  证据维度（类别数），默认 37
    """

    def __init__(self, k_evidence=37):
        super().__init__()
        self.k_evidence = k_evidence
        self._alpha = None

    def reset(self):
        """每图开始调用：重置为 uniform prior Dir(1, ..., 1)。"""
        self._alpha = torch.ones(self.k_evidence)

    def update(self, e: torch.Tensor):
        """累加证据并返回当前信念和不确定度。

        Parameters
        ----------
        e : Tensor (K,)  当前步证据向量（非负）

        Returns
        -------
        b : Tensor (K,)  更新后的信念质量
        u : Tensor (,)   更新后的不确定度
        """
        self._alpha = self._alpha.to(e.device) + e.detach()
        S = self._alpha.sum()
        b = self._alpha / S.clamp(min=1e-6)          # Dirichlet 期望，和为 1
        u = self.k_evidence / S.clamp(min=1e-6)
        return b, u

    @property
    def current_b(self) -> torch.Tensor:
        """当前信念 b = α/S（Dirichlet 期望，和为 1）。"""
        S = self._alpha.sum()
        return self._alpha / S.clamp(min=1e-6)

    @property
    def current_u(self) -> torch.Tensor:
        """当前不确定度 u = K/S。"""
        return self.k_evidence / self._alpha.sum().clamp(min=1e-6)

    @property
    def alpha(self) -> torch.Tensor:
        """当前 Dirichlet 参数 α。"""
        return self._alpha

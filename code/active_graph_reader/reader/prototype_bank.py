"""PrototypeBank: 可学习高斯混合原型库 + 闭式查询。

每步独立查询：v_t(300) → PCA降维 → 混合高斯后验 → b_t(37)
μ 可学习（原型跟随），Σ/π/PCA 固定。

用法:
  bank = PrototypeBank(mu_init, sigma2, class_id, pi, pca_W)
  b_t = bank.query(v_t)   # v_t: (batch, 300) → b_t: (batch, 37)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """可学习高斯混合原型库。

    Parameters
    ----------
    mu_init : Tensor (K_total, d)
        簇质心初始值。
    sigma2 : Tensor (K_total, d)
        对角方差（Ledoit-Wolf shrinkage 后），固定。
    class_id : Tensor (K_total,)
        每个原型对应的类别标签 (0..36)。
    pi : Tensor (37,)
        类先验 P(c=k)，固定。
    pca_W : Tensor (300, d)
        PCA 变换矩阵，固定。
    """

    def __init__(self, mu_init, sigma2, class_id, pi, pca_W):
        super().__init__()
        self.mu = nn.Parameter(mu_init.clone())
        self.register_buffer('sigma2', sigma2)
        self.register_buffer('class_id', class_id)
        self.register_buffer('pi', pi)
        self.register_buffer('pca_W', pca_W)

        self._K_total = mu_init.shape[0]
        self._d = mu_init.shape[1]
        self._n_classes = pi.shape[0]

        # 预计算每类的 log(1/M_k) 混合权重
        M_k = torch.bincount(class_id, minlength=self._n_classes).float()
        M_k = M_k.clamp(min=1)
        self.register_buffer('_log_w', -M_k.log())  # log(1/M_k) per class

        # 预计算 log_det_sigma2 的常量部分
        self.register_buffer('_log_det', self.sigma2.log().sum(dim=-1))  # (K_total,)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, v_t):
        """闭式高斯查询。

        Parameters
        ----------
        v_t : Tensor (batch, 300)
            子图嵌入（来自 g_φ 编码器）。

        Returns
        -------
        b_t : Tensor (batch, 37)
            后验信念 P(c=k | v_t)。
        """
        log_post = self.query_logits(v_t)                  # (batch, 37)
        b_t = F.softmax(log_post, dim=-1)                  # (batch, 37)
        return b_t

    def query_logits(self, v_t):
        """闭式高斯查询，返回 softmax 前的 log-posterior（logits）。

        用于训练时 F.cross_entropy(log_post, labels)，避免对已 softmax 的
        belief 再做 log_softmax 导致梯度消失。

        Parameters
        ----------
        v_t : Tensor (batch, 300)
            子图嵌入。

        Returns
        -------
        log_post : Tensor (batch, 37)
            log P(c=k | v_t)（未归一化，即 logits）。
        """
        # PCA
        vtilde = v_t @ self.pca_W                     # (batch, d)

        # 对每个原型计算 log N(vtilde; mu, sigma2)
        diff = vtilde.unsqueeze(1) - self.mu.unsqueeze(0)  # (batch, K, d)
        mahal = (diff.pow(2) / self.sigma2).sum(dim=-1)    # (batch, K)
        log_N = -0.5 * (self._d * math.log(2 * math.pi)
                        + self._log_det + mahal)            # (batch, K)

        # 混合权重 log(1/M_k)：log_p = logsumexp_m [log(1/M_k) + log N_m]
        log_w_k = self._log_w[self.class_id]                # (K,)
        log_N_weighted = log_N + log_w_k                    # (batch, K)

        # 按类别聚合 logsumexp
        log_p = scatter_logsumexp(
            log_N_weighted, self.class_id, self._n_classes
        )                                                   # (batch, 37)

        # 后验 = softmax(log prior + log likelihood)
        log_post = log_p + self.pi.log()                   # (batch, 37)
        return log_post

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K_total(self):
        return self._K_total

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def d(self):
        return self._d

    def extra_repr(self):
        return (f"K_total={self._K_total}, d={self._d}, "
                f"n_classes={self._n_classes}")


# -----------------------------------------------------------------------
# scatter_logsumexp — 按类别聚合 logsumexp
# -----------------------------------------------------------------------

def scatter_logsumexp(src, index, n_classes):
    """安全的 scatter logsumexp：对每个类别内的值做 logsumexp。

    Parameters
    ----------
    src : Tensor (..., K_total)
        每个原型的 log-prob 值。
    index : Tensor (K_total,)
        每个原型的类别标签 (0..n_classes-1)。
    n_classes : int

    Returns
    -------
    out : Tensor (..., n_classes)
    """
    # 策略：逐类做 logsumexp。K_total ~ 数百到数千，循环 37 次可忽略。
    *batch_dims, K = src.shape
    out = torch.full((*batch_dims, n_classes), float('-inf'),
                     device=src.device, dtype=src.dtype)
    for k in range(n_classes):
        mask = (index == k)
        if mask.any():
            vals = src[..., mask]                       # (..., M_k)
            out[..., k] = torch.logsumexp(vals, dim=-1)
    return out

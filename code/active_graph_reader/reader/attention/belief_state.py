"""Belief State module — cross-step memory for the selector reader.

BeliefTokens: (m, d) learnable initial belief, expanded per-batch.
BeliefUpdate:  TransformerLayer(SelfAttn→CrossAttn→FFN) absorbing new frontier info.
(No classification head — classification is delegated to an external classifier.)
"""

import torch
import torch.nn as nn


class BeliefUpdate(nn.Module):
    """One Transformer layer: SelfAttn → CrossAttn(new_info) → FFN, all with residual."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        """Update belief tokens with new structural information.

        Args:
            h:        (B, m, d)  current belief tokens
            new_info: (B, 1, d)  encoded new frontier edges from this reveal

        Returns:
            h_new: (B, m, d)  updated belief tokens
        """
        attn_out, _ = self.self_attn(h, h, h)
        h = self.norm_self(h + attn_out)

        cross_out, _ = self.cross_attn(h, new_info, new_info)
        h = self.norm_cross(h + cross_out)

        h = self.norm_ffn(h + self.ffn(h))

        return h


class BeliefState(nn.Module):
    """Belief state: initial tokens + update mechanism (no classification head).

    All m tokens participate in node selection (as FrontierPointer KV).
    """

    def __init__(self, d_model: int = 128, m: int = 4,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.m = m

        self.tokens = nn.Parameter(torch.randn(m, d_model) * 0.02)
        self.update_fn = BeliefUpdate(d_model, n_heads, dropout)

    def init_belief(self) -> torch.Tensor:
        """Return (1, m, d) initial belief for a single state."""
        return self.tokens.unsqueeze(0).clone()  # (1, m, d)

    def init_beliefs(self, n: int) -> torch.Tensor:
        """Return (n, m, d) initial beliefs for n states."""
        return self.tokens.unsqueeze(0).expand(n, -1, -1).clone()  # (n, m, d)

    def update_belief(self, h: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        """Absorb new frontier information.

        Args:
            h:        (1, m, d) or (B, m, d)  current belief
            new_info: (1, 1, d) or (B, 1, d)  encoded new edges
        Returns:
            h_new: same shape as h  updated belief
        """
        return self.update_fn(h, new_info)

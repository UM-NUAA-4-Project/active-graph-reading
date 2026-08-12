"""Belief State module — cross-step memory for the Transformer Reader.

BeliefTokens: (m, d) learnable initial belief, expanded per-batch.
BeliefUpdate:  TransformerLayer(SelfAttn→CrossAttn→FFN) absorbing new frontier info.
ClassificationHead: Flatten(h) → MLP → K-class logits.
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
        # Step A: internal alignment — "what do we already know?"
        attn_out, _ = self.self_attn(h, h, h)
        h = self.norm_self(h + attn_out)

        # Step B: absorb new info — each token extracts what it cares about
        cross_out, _ = self.cross_attn(h, new_info, new_info)
        h = self.norm_cross(h + cross_out)

        # Step C: nonlinear integration
        h = self.norm_ffn(h + self.ffn(h))

        return h


class ClassificationHead(nn.Module):
    """Flatten all belief tokens → MLP → class logits."""

    def __init__(self, d_model: int = 128, m: int = 4, n_classes: int = 37,
                 dropout: float = 0.1):
        super().__init__()
        self.m = m
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(m * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, m, d) → (B, K).  B=1 for single-state case."""
        B = h.shape[0]
        flat = h.reshape(B, -1)          # (B, m*d)
        return self.mlp(flat)            # (B, K)


class BeliefState(nn.Module):
    """Full belief state: initial tokens + update mechanism + classification head.

    All m tokens participate in both node selection (as FrontierPointer KV)
    and classification (flattened → MLP).

    Usage in training loop:
        bs = BeliefState(d_model=128, m=4)
        beliefs = [bs.init_belief() for _ in range(n_states)]
        ...
        h_new = bs.update_belief(h_old, new_info)
        logits = bs.classify(h_new)
    """

    def __init__(self, d_model: int = 128, m: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, n_classes: int = 37):
        super().__init__()
        self.d_model = d_model
        self.m = m

        self.tokens = nn.Parameter(torch.randn(m, d_model) * 0.02)
        self.update_fn = BeliefUpdate(d_model, n_heads, dropout)
        self.classifier = ClassificationHead(d_model, m, n_classes, dropout)

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

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        """(B, m, d) → (B, K) logits using all tokens."""
        return self.classifier(h)

"""(N_f, d_in) + belief (m, d) → (N_f,) scores."""

import torch
import torch.nn as nn


class FrontierPointer(nn.Module):
    """Frontier self-attn → belief cross-attn → per-node scores."""

    def __init__(self, d_in: int, d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(d_model)

        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model//4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model//4, 1),
        )

    def forward(self, x: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        """(N_f, d_in) + (m, d) belief → (N_f,) scores."""
        x = self.input_proj(x)                     # (N_f, d_model)
        x = x.unsqueeze(0)                         # (1, N_f, d_model)

        # step 1: frontier nodes pool information among themselves
        self_out, _ = self.self_attn(x, x, x)
        x = self.norm_self(x + self_out)           # (1, N_f, d_model)

        # step 2: each enriched frontier node queries belief tokens
        belief = belief.unsqueeze(0)               # (1, m, d)
        cross_out, _ = self.cross_attn(x, belief, belief)
        x = self.norm_cross(x + cross_out)         # (1, N_f, d_model)

        x = x.squeeze(0)                           # (N_f, d_model)
        return self.score_proj(x).squeeze(-1)      # (N_f,)

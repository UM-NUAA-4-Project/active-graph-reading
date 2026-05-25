"""
Bottleneck adapter for GNN fine-tuning
=======================================

Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019.
Adapted for GNNs: a small bottleneck MLP inserted after each frozen GIN layer.

  Adapter(x) = Linear_up(ReLU(Linear_down(x)))
  h_out       = GINLayer(h_in) + Adapter(GINLayer(h_in))

The residual connection preserves the frozen layer's output while the adapter
learns a lightweight domain correction (full-graph → partial subgraph).
"""

import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter: down-project → ReLU → up-project with residual.

    Parameters
    ----------
    hidden_dim : int
        Input/output dimension.
    bottleneck_dim : int
        Bottleneck dimension (typically hidden_dim // bottleneck_ratio).
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = nn.ReLU()

        # Near-zero init so adapter starts close to identity
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

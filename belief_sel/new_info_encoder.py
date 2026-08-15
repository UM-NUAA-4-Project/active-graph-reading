"""Full frontier subgraph GNN encoder for belief update information.

Subgraph = all revealed nodes + all frontier nodes (FrontierGraphBuilder output).
Two GINELayer rounds with residual pooling:
  Layer 1 → pool₁ = [h₁[v]; mean(h₁[frontier])] → res_proj → residual
  Layer 2 → pool₂ = [h₂[v]; mean(h₂[frontier])] → output_proj
  output = out + residual

This captures both local (v's neighbourhood) and global (full subgraph topology).
"""

import torch
import torch.nn as nn
import dgl

from active_graph_reader.gnn.gin_encoder import GINELayer


class NewInfoEncoder(nn.Module):
    """2-layer GIN on full frontier subgraph with residual pooling.

    Parameters
    ----------
    edge_feat_dim : int  边特征维度（PPA: 7）
    hidden_dim : int     隐层维度
    dropout : float
    """

    def __init__(self, edge_feat_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = GINELayer(hidden_dim, edge_feat_dim)
        self.conv2 = GINELayer(hidden_dim, edge_feat_dim)

        self.res_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, g: dgl.DGLGraph, h: torch.Tensor, edge_feats: torch.Tensor,
        v_idx: int, frontier_mask: torch.Tensor,
    ) -> torch.Tensor:
        """(g, h, ef) → (d,).  Single-graph convenience wrapper."""
        return self.forward_batch(
            g, h, edge_feats,
            v_indices=[v_idx],
            frontier_masks=[frontier_mask],
            offsets=[0, g.num_nodes()],
        ).squeeze(0)

    def forward_batch(
        self, batch_g: dgl.DGLGraph, h: torch.Tensor, edge_feats: torch.Tensor,
        v_indices: list, frontier_masks: list,
        offsets: list = None,
    ) -> torch.Tensor:
        """Batched full frontier subgraphs → (N, d).

        Args:
            batch_g:        DGL batched graph (all subgraphs batched together)
            h:              (ΣN_i, d) node embeddings for all subgraphs
            edge_feats:     (ΣE_i, edge_feat_dim)
            v_indices:      [v_idx_0, v_idx_1, ...] global indices of revealed nodes
            frontier_masks: [(ΣN_i,) bool, ...] one mask per subgraph (True where frontier)
            offsets:        [0, N0, N0+N1, ...] cumulative node counts

        Returns:
            (N, d) pooled embeddings
        """
        # Shared GIN layers — one invocation for all subgraphs
        h1 = self.conv1(batch_g, h, edge_feats)
        h2 = self.conv2(batch_g, h1, edge_feats)

        # Per-subgraph pooling
        results = []
        for i, (v_i, f_mask) in enumerate(zip(v_indices, frontier_masks)):
            pool1_i = torch.cat([h1[v_i], h1[f_mask].mean(0)])  # (2d,)
            residual = self.res_proj(pool1_i)                     # (d,)

            pool2_i = torch.cat([h2[v_i], h2[f_mask].mean(0)])  # (2d,)
            output = self.output_proj(pool2_i)                    # (d,)

            results.append(output + residual)

        return torch.stack(results)  # (N, d)

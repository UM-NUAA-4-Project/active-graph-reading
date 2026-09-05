"""GNN+ 组件①边特征整合（edge feature integration）。

论文（arXiv 2502.09263, Eq.6）把边向量线性投影后**加进消息聚合内部**——
GINELayer 的 `h + edge_emb` 即此模式，边编码器本身作为可替换积木：
- 'linear'：单层 Linear（OGB GINE 同款，默认档，state_dict 键与旧 checkpoint
  完全一致：`convs.i.edge_encoder.{weight,bias}`）；
- 'mlp'：两层瓶颈 MLP（仅新配置产生新键 `convs.i.edge_encoder.{0,2}.*`）。
"""

import torch.nn as nn

EDGE_ENCODER_KINDS = ('linear', 'mlp')


def make_edge_encoder(edge_feat_dim: int, hidden_dim: int, kind: str = 'linear') -> nn.Module:
    if kind == 'linear':
        return nn.Linear(edge_feat_dim, hidden_dim)
    if kind == 'mlp':
        return nn.Sequential(
            nn.Linear(edge_feat_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
    raise ValueError(f"未知边编码器 '{kind}'，可选: {EDGE_ENCODER_KINDS}")

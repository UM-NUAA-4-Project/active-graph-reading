"""classifier 的 backbone 层：gin（GINEEncoder）、gat（GATEncoder）。

新 backbone 接入约定：
1. 实现 encoder 类：构造 kwargs 从 BACKBONE_KWARGS 声明中选取（末尾追加
   带默认值），提供 out_dim 属性（读出输出维，供分类头定输入维）；
2. 在 BACKBONES 注册，并在 BACKBONE_KWARGS 声明支持的 plus/ 积木；
3. 逐层循环按 GNN+ 管线（plus/__init__ docstring 的 Eq.11）组织，
   积木关闭时必须与 backbone 原行为逐位一致。
"""

from .gat import GATEncoder, GATLayer
from .gin import GINEEncoder, GINELayer

# backbone 名称 → encoder 类；新 backbone 在此注册
BACKBONES = {
    'gin': GINEEncoder,
    'gat': GATEncoder,
}

# backbone 名 → 支持的构造 kwargs 白名单。积木（plus/ 组件）未声明即
# 不支持——GraphClassifier 对"开启了未支持积木"的配置大声报错，
# 不做静默忽略。
BACKBONE_KWARGS = {
    'gin': frozenset({'edge_feat_dim', 'hidden_dims', 'dropouts',
                      'virtual_node', 'residual', 'node_feat_dim',
                      'gnn_norm', 'readout', 'edge_encoder_kind',
                      'ffn', 'ffn_dropout', 'pos_enc', 'pos_enc_ksteps'}),
    'gat': frozenset({'edge_feat_dim', 'hidden_dims', 'dropouts',
                      'virtual_node', 'residual', 'node_feat_dim',
                      'gnn_norm', 'readout', 'edge_encoder_kind', 'n_heads'}),
}


def get_backbone(name: str):
    if name not in BACKBONES:
        raise ValueError(f"未知 backbone '{name}'，可用: {sorted(BACKBONES)}")
    return BACKBONES[name]


def backbone_init_kwargs(backbone: str, config) -> dict:
    """ClassifierConfig → 该 backbone 白名单内的构造 kwargs。"""
    full = dict(
        edge_feat_dim=config.edge_feat_dim,
        hidden_dims=config.hidden_dims,
        dropouts=config.dropouts,
        virtual_node=config.virtual_node,
        residual=config.residual,
        node_feat_dim=config.node_feat_dim,
        gnn_norm=config.gnn_norm,
        readout=config.readout,
        edge_encoder_kind=config.edge_encoder,
        ffn=config.ffn,
        ffn_dropout=config.ffn_dropout,
        pos_enc=config.pos_enc,
        pos_enc_ksteps=config.pos_enc_ksteps,
    )
    if backbone == 'gat':
        full['n_heads'] = config.n_heads
    supported = BACKBONE_KWARGS[backbone]
    return {k: v for k, v in full.items() if k in supported}


__all__ = ['GINEEncoder', 'GINELayer', 'GATEncoder', 'GATLayer',
           'BACKBONES', 'BACKBONE_KWARGS', 'get_backbone',
           'backbone_init_kwargs']

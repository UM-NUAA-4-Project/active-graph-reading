"""GNN+ 积木包：ICML 2025 统一增强框架的六组件 + 读出工具。

论文："Can Classic GNNs Be Strong Baselines for Graph-level Tasks?"
（ICML 2025, arXiv 2502.09263）。逐层管线（Eq.11）：
  MP(含边特征整合) → 归一化 → σ → Dropout → 残差 → FFN(收尾)
积木由 ClassifierConfig 字段参数化插拔，全部默认关闭（= 各 backbone
原行为，对旧 checkpoint 键集零影响）。

组件 ↔ 模块映射：
  ① 边特征整合  plus/edge.py    make_edge_encoder  → config.edge_encoder
  ② 归一化      plus/norm.py    make_norm          → config.gnn_norm
  ③ Dropout     （不设模块：encoder 循环内阶段）   → config.dropouts
  ④ 残差连接    （不设模块：encoder 循环内阶段）   → config.residual
  ⑤ FFN        plus/ffn.py     FFN（自带残差+BN）  → config.ffn / ffn_dropout
  ⑥ 位置编码    plus/pos_enc.py RWSE               → config.pos_enc / pos_enc_ksteps
  （读出）      plus/readout.py make_readout       → config.readout

依赖约束：本包只允许 import torch/dgl，禁止 import config/layers
（依赖方向 plus ← layers ← graph_classifier ← factory，防循环）。
新增积木：新建模块 + 工厂函数 + ClassifierConfig 字段（非默认才序列化）
+ BACKBONE_KWARGS 白名单声明 + tests/test_plus_blocks.py 覆盖。
"""

from .edge import EDGE_ENCODER_KINDS, make_edge_encoder
from .ffn import FFN, FFN_HIDDEN_MULT
from .norm import NORM_TYPES, make_norm
from .pos_enc import random_walk_structural_encoding
from .readout import READOUT_KINDS, READOUT_OUT_MULT, make_readout

__all__ = [
    'EDGE_ENCODER_KINDS', 'make_edge_encoder',
    'FFN', 'FFN_HIDDEN_MULT',
    'NORM_TYPES', 'make_norm',
    'random_walk_structural_encoding',
    'READOUT_KINDS', 'READOUT_OUT_MULT', 'make_readout',
]

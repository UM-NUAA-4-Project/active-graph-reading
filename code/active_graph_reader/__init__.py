"""
Active Graph Reader — v1 协议实现
=================================

模块组成：
- core    — BaseReader 抽象接口与日志
- data    — 数据加载（OGBG-PPA）
- engine  — 核心引擎（GraphState, reveal, RevealLoop）
- gnn     — GNN 骨干（GINEEncoder, GraphClassifier, SubgraphBuilder, SubgraphEncoder）
- reader  — Reader 实现（LearnableReader, AttentionReader, PrototypeReader, PrototypeBank, fixed strategies）
"""

from .reader import (
    BaseReader,
    FixedStrategyReader,
    LearnableReader,
    AttentionReader,
    SubgraphEncoder,
    build_g_phi_from_fclass,
    FrontierNodesEncoder,
    reveal_one,
    reveal_batch,
)

__all__ = [
    'BaseReader',
    'FixedStrategyReader',
    'LearnableReader',
    'AttentionReader',
    'SubgraphEncoder',
    'build_g_phi_from_fclass',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
]

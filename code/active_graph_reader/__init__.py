"""
Active Graph Reader — v1 协议实现
=================================

模块组成：
- core    — BaseReader 抽象接口与日志
- engine  — 核心引擎（GraphState, reveal, RevealLoop）
- gnn     — 子图构建与前沿编码（GIN backbone 已迁至独立的 classifier 包）
- reader  — Reader 实现（LearnableReader, AttentionReader, fixed strategies）
- config  — ReaderConfig（逐层结构配置）+ build_reader 工厂 + checkpoint 推断

分类器（GraphClassifier/GINEEncoder）位于 code/classifier/，本包不再包含。
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
from .config import ReaderConfig, build_reader

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
    'ReaderConfig',
    'build_reader',
]

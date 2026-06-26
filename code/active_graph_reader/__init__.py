"""
Active Graph Reader — v1 协议实现
=================================

基于 base-protocol-v1.md 定义的 v1 协议，构建模块化、可扩展的主动图阅读框架。

模块组成：
- core        —— 共享接口与基础类（BaseReader, 日志）
- config      —— 配置管理（dataclass）
- data        —— 数据加载与预处理（OGBG-PPA）
- engine      —— 核心引擎（图状态、揭示动作、揭示循环、reveal 工具）
- gnn         —— GNN 骨干网络（GIN encoder/classifier/subgraph builders/frontier encoder）
- reader      —— Reader 实现（fixed strategies / learnable reader）
"""

from .reader import (
    BaseReader,
    FixedStrategyReader,
    LearnableReader,
    FrontierNodesEncoder,
    reveal_one,
    reveal_batch,
)

__all__ = [
    'BaseReader',
    'FixedStrategyReader',
    'LearnableReader',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
]

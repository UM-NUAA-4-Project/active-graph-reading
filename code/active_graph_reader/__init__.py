"""
Active Graph Reader — v1 协议实现
=================================

基于 base-protocol-v1.md 定义的 v1 协议，构建模块化、可扩展的主动图阅读框架。

模块组成：
- config      —— 配置管理（dataclass）
- dataloader  —— 数据加载与预处理（OGBG-PPA + ReaderDataset）
- engine      —— 核心引擎（图状态、揭示动作、揭示循环）
- strategies  —— 可插拔前沿边揭示策略（random / bfs / max-frontier-degree）
- cost        —— 成本模型抽象（v0 单位成本 + M4+ 预留扩展点）
- encoder     —— GNN 编码器/分类器/子图构造器
- reader      —— Reader 封装（BaseReader / FixedStrategyReader / LearnableReader）
- utils       —— 工具函数（日志等）
"""

from .reader import (
    BaseReader,
    FixedStrategyReader,
    LearnableReader,
    FrontierNodesEncoder,
    reveal_one,
    reveal_batch,
    evaluate,
    create_dataloader,
)

__all__ = [
    'BaseReader',
    'FixedStrategyReader',
    'LearnableReader',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
    'evaluate',
    'create_dataloader',
]

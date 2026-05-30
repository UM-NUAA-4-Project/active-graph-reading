"""
编码器模块
==========

子模块：
- subgraph_builder  —— GraphState → DGLGraph 转换
- gin_encoder       —— GINEConv × n 层编码器
- classifier         —— GINEEncoder + MLP 分类头
"""

from .subgraph_builder import SubgraphBuilder
from .gin_encoder import GINEEncoder
from .classifier import GraphClassifier

__all__ = [
    'SubgraphBuilder',
    'GINEEncoder',
    'GraphClassifier',
]

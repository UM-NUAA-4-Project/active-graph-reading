"""
随机选择策略
============

从当前所有前沿边中均匀随机选择一条进行揭示。
完全无状态——每次 select 时直接从 graph_state 的前沿边集合中采样。

对应协议: base-protocol-v0.md 第 4.1 节 P1（Uniform Random）。
"""

import random
from typing import Optional, List, Tuple

from .base import RevealStrategy
from ..engine.graph_state import GraphState 

class RandomStrategy(RevealStrategy):
    """均匀随机选择前沿边，返回其未揭示端点。

    这是最简单的基准策略，作为其他策略的性能下界参照。
    选择概率与前沿边的数量成正比：如果一个未揭示节点有多个
    已揭示邻居，它被选中的概率更高（通过多条前沿边体现）。
    """

    @property
    def name(self) -> str:
        return 'random'

    def select(self, graph_state: 'GraphState') -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None

        # 从前沿边中随机选一条
        edge = random.choice(list(frontier))
        # edge = (revealed, unrevealed)，返回未揭示端点
        return edge[1]

    # RandomStrategy 是无状态的，不需要重写 on_new_frontier 和 reset。

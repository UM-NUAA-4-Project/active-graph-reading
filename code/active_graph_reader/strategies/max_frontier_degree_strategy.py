"""
最大前沿度策略
==============

在所有前沿边中，选择其「前沿节点」（即未揭示端点）在已揭示子图内的
度数最大的一条。该度数仅统计该节点与已揭示子图相连的边数，而非原始
完整图中的度数。

对应协议: base-protocol-v0.md 第 4.1 节 P3（Max Known-Frontier Degree）。

这是完全无状态的策略——每次 select 时从 graph_state 获取最新的
frontier_degree 数据。
"""

from typing import Optional, List, Tuple

from .base import RevealStrategy
from ..engine.graph_state import GraphState 

class MaxFrontierDegreeStrategy(RevealStrategy):
    """选择前沿节点局部度最大的前沿边。

    灵感来自贪心图遍历：优先揭示与已观测子图「连接最紧密」的节点。
    由于真实图度是隐藏的，局部度是当前已知的最优启发式指标。

    当多个前沿节点具有相同的最大局部度时，选择遍历到的第一个
    （取决于 frontier_edges 集合的迭代顺序，非确定性但对等）。
    """

    @property
    def name(self) -> str:
        return 'max-frontier-degree'

    def select(self, graph_state: 'GraphState') -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None

        best_node = None
        best_degree = -1

        for edge in frontier:
            unrevealed = edge[1]
            # get_frontier_degree 返回该节点与 V_revealed 相连的边数
            deg = graph_state.get_frontier_degree(unrevealed)
            if deg > best_degree:
                best_degree = deg
                best_node = unrevealed

        return best_node

    # MaxFrontierDegreeStrategy 是无状态的，所有信息来自 GraphState 查询。

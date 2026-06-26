"""最大前沿度策略：选择前沿节点局部度（与已揭示子图的连接数）最大的前沿边。"""

from typing import Optional

from ...core.base_reader import BaseReader
from ...engine.graph_state import GraphState


class MaxFrontierDegreeStrategy(BaseReader):
    """选择 frontier_degree 最大的前沿节点。无状态，所有信息来自 GraphState 查询。"""

    @property
    def name(self) -> str:
        return 'max-frontier-degree'

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None

        best_node = None
        best_degree = -1

        for edge in frontier:
            unrevealed = edge[1]
            deg = graph_state.get_frontier_degree(unrevealed)
            if deg > best_degree:
                best_degree = deg
                best_node = unrevealed

        return best_node

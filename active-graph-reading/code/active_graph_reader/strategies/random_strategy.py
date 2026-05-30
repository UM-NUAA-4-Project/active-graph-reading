"""随机选择策略：从当前所有前沿边中均匀随机选择一条进行揭示。"""

import random
from typing import Optional

from ..reader.base_reader import BaseReader
from ..engine.graph_state import GraphState


class RandomStrategy(BaseReader):
    """均匀随机选择前沿边，返回其未揭示端点。"""

    @property
    def name(self) -> str:
        return 'random'

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        edge = random.choice(list(frontier))
        return edge[1]

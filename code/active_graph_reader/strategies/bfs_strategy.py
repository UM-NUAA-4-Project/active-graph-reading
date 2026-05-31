"""广度优先策略：按前沿边的发现顺序（FIFO）选择下一个要揭示的节点。"""

import collections
from typing import Optional, Deque, List, Tuple

from ..reader.base_reader import BaseReader
from ..engine.graph_state import GraphState


class BfsStrategy(BaseReader):
    """按 BFS 顺序选择前沿边，保证层序遍历的确定性。"""

    def __init__(self):
        super().__init__()
        self._queue: Deque[Tuple[int, int]] = collections.deque()

    @property
    def name(self) -> str:
        return 'bfs'

    def select(self, graph_state: GraphState) -> Optional[int]:
        while self._queue:
            edge = self._queue.popleft()
            unrevealed = edge[1]
            if unrevealed not in graph_state.V_revealed:
                return unrevealed
        return None

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        for edge in edges:
            self._queue.append(edge)

    def reset(self) -> None:
        self._queue.clear()

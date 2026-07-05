"""
广度优先策略
============

按前沿边的发现顺序（FIFO 队列）选择下一个要揭示的节点。
实现标准图 BFS 遍历——从种子节点开始逐层向外扩张。

对应协议: base-protocol-v0.md 第 4.1 节 P2（BFS Order）。

内部状态：
- _queue: collections.deque，按发现顺序存储前沿边。
- 当多个已揭示节点都连接到同一未揭示节点时，该节点以最早被发现的前沿边
  入队，保证 BFS 层序的确定性。
"""

import collections
from typing import Optional, Deque, List, Tuple

from .base import RevealStrategy
from ..engine.graph_state import GraphState 

class BfsStrategy(RevealStrategy):
    """按 BFS 顺序选择前沿边。

    Attributes
    ----------
    _queue : Deque[Tuple[int, int]]
        按发现顺序存储前沿边 (revealed, unrevealed) 的 FIFO 队列。
    """

    def __init__(self):
        self._queue: Deque[Tuple[int, int]] = collections.deque()

    @property
    def name(self) -> str:
        return 'bfs'

    # ---- 核心选择逻辑 -------------------------------------------------------

    def select(self, graph_state: 'GraphState') -> Optional[int]:
        """从队列头部取出一条前沿边，返回其未揭示端点。

        自动跳过已揭示的端点（因多路径发现而提前被揭示的情况），
        继续出队直到找到有效的未揭示节点或队列耗尽。
        """
        while self._queue:
            edge = self._queue.popleft()
            unrevealed = edge[1]

            # 检查该端点是否仍未被揭示
            if unrevealed not in graph_state.V_revealed:
                return unrevealed
            # 端点已揭示 → 该前沿边失效，继续下一个

        return None  # 队列耗尽

    # ---- 状态通知 ----------------------------------------------------------

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        """将新发现的前沿边追加到 BFS 队列尾部。

        这保证了先进先出的语义：种子节点的前沿边最先入队，
        随后是第 1 层节点的前沿边，依此类推。
        """
        for edge in edges:
            self._queue.append(edge)

    def reset(self) -> None:
        """清空 BFS 队列，准备新一轮揭示。"""
        self._queue.clear()

"""
揭示策略抽象基类
================

定义前沿边揭示策略的统一接口。所有策略必须实现:
- select: 从当前图状态中选择下一个要揭示的节点。
- name: 返回策略名称。

可选钩子方法:
- on_new_frontier: 前沿边扩展通知（BFS 等有状态策略需要）。
- reset: 重置内部状态（用于多轮揭示或多种子平均）。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from ..engine.graph_state import GraphState  

class RevealStrategy(ABC):
    """前沿边揭示策略抽象基类。

    子类需实现 select 方法和 name 属性。
    默认实现是无状态的——每次 select 仅依赖当前 GraphState。

    对于需要内部状态的策略（如 BFS），应重写:
    - on_new_frontier: 接收新发现的前沿边通知。
    - reset: 清除内部状态。
    """

    @abstractmethod
    def select(self, graph_state: 'GraphState') -> Optional[int]:
        """从当前前沿中选择下一个要揭示的节点。

        Parameters
        ----------
        graph_state : GraphState
            当前图状态，提供所有查询接口:
            - graph_state.frontier_edges  → 前沿边集合
            - graph_state.get_frontier_degree(node) → 前沿节点的局部度
            - graph_state.is_frontier_empty → 前沿是否为空

        Returns
        -------
        int or None
            下一个要揭示的节点 ID。若前沿为空，返回 None。
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（如 'random', 'bfs'）。"""
        ...

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        """前沿边扩展通知（可选钩子）。

        在 reveal_node 执行完毕后、揭示循环进入下一步之前调用。
        默认实现为空（无状态策略不需要）。

        Parameters
        ----------
        edges : List[Tuple[int, int]]
            本次揭示新发现的前沿边列表，每项为 (revealed, unrevealed)。
        """
        pass

    def reset(self) -> None:
        """重置策略内部状态（可选钩子）。

        在开始新一轮揭示循环前调用（例如切换种子或图的场景）。
        默认实现为空。
        """
        pass

"""BaseReader 抽象接口：所有 Reader（固定策略/可学习）的统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ..engine.graph_state import GraphState


class BaseReader(ABC):
    """Reader 抽象基类。

    子类须实现:
    - select: 从当前图状态中选择下一个要揭示的节点。
    - name: 返回 reader 名称。

    可选钩子:
    - on_new_frontier: 前沿边扩展通知（BFS 等有状态 reader 需要）。
    - reset: 重置内部状态。
    - train/eval: 切换训练/评估模式。
    """

    def __init__(self):
        super().__init__()
        self._training = False

    @abstractmethod
    def select(self, graph_state: GraphState) -> Optional[int]:
        """从当前前沿中选择下一个要揭示的节点。返回 None 若前沿为空。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Reader 名称（如 'random', 'bfs', 'learnable'）。"""
        ...

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        pass

    def reset(self) -> None:
        pass

    def train(self, mode: bool = True):
        self._training = mode
        return self

    def eval(self):
        self._training = False
        return self

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool):
        self._training = value

"""
成本模型抽象基类
================

定义揭示动作成本计算的统一接口。v0 使用 UnitCostModel（单位成本），
M4+ 可替换为 AtomicCostModel（组合原子成本）。
"""

from abc import ABC, abstractmethod
from typing import Any
from ..engine.graph_state import GraphState

class CostModel(ABC):
    """成本模型抽象基类。

    子类需实现:
    - compute: 计算单次揭示动作的成本。
    - name: 返回成本模型名称（用于注册表）。
    """

    @abstractmethod
    def compute(self, node: int, graph_state: 'GraphState') -> float:
        """计算揭示节点 node 的成本。

        Parameters
        ----------
        node : int
            待揭示节点 ID。
        graph_state : GraphState
            当前图状态（可用于查询节点属性、边的数量等）。

        Returns
        -------
        float
            揭示成本。v0 固定返回 1.0。
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """成本模型名称（如 'unit', 'atomic'）。"""
        ...

    def reset(self) -> None:
        """重置成本模型内部状态（如有）。

        默认实现为空，有状态子类重写。
        """
        pass

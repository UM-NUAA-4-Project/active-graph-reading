"""
单位成本模型（v0 默认）
=======================

每个 reveal_node(v) 动作成本恒为 1，与节点属性、图结构无关。
对应 base-protocol-v0.md 第 3.3 节的单位成本定义。
"""

from .base import CostModel
from ..engine.graph_state import GraphState

class UnitCostModel(CostModel):
    """单位成本模型：每个 reveal_node(v) 成本 = 1。

    这是 v0 协议的唯一成本模型。M4+ 阶段将引入 AtomicCostModel
    以支持原子成本组合（c_n, c_e, c_nf, c_ef）。

    Examples
    --------
    >>> model = UnitCostModel()
    >>> model.compute(node=3, graph_state=None)
    1.0
    """

    @property
    def name(self) -> str:
        return 'unit'

    def compute(self, node: int, graph_state: 'GraphState') -> float:
        """返回固定成本 1.0。

        参数 node 和 graph_state 在 v0 中不使用，但保留接口以兼容
        M4+ 的成本模型（可能需要根据节点度数或边数动态计算成本）。
        """
        return 1.0

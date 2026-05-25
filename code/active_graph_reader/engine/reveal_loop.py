"""
揭示循环模块
============

实现协议中的核心决策循环，对应 base-protocol-v0.md 第 1.2 节：

    while B_current > 0:
        a_t = π(G_{t-1}, B_current)
        if a_t = ∅: break
        B_current ← B_current - C(a_t)
        G_t = ψ(G_{t-1}, a_t)

RevealLoop 是策略、成本模型、揭示动作之间的协调者，
三者均通过接口注入，无直接耦合。
"""

import logging
from typing import Optional

from .graph_state import GraphState
from .reveal_action import reveal_node
from ..strategies.base import RevealStrategy
from ..cost.base import CostModel

logger = logging.getLogger(__name__)


class RevealLoop:
    """揭示协议的主循环协调器。

    职责：
    - 维护预算状态。
    - 调用策略选择下一个节点。
    - 调用成本模型计算实际成本。
    - 执行揭示动作，更新图状态。
    - 通知策略前沿扩展事件。
    - 处理终止条件（前沿为空 / 预算耗尽）。

    策略和成本模型在构造时注入，运行期间不变。

    Examples
    --------
    >>> from active_graph_reader.strategies import get_strategy
    >>> from active_graph_reader.cost import get_cost_model
    >>>
    >>> state = GraphState(dgl_graph, seed_node=42)
    >>> strategy = get_strategy('bfs')
    >>> cost_model = get_cost_model('unit')
    >>>
    >>> loop = RevealLoop(state, strategy, cost_model)
    >>> stats = loop.run(budget=50)
    """

    def __init__(
        self,
        graph_state: GraphState,
        strategy: RevealStrategy,
        cost_model: CostModel,
    ):
        """初始化揭示循环。

        Parameters
        ----------
        graph_state : GraphState
            已初始化的图状态（含 G_0）。
        strategy : RevealStrategy
            前沿边揭示策略。
        cost_model : CostModel
            成本模型。
        """
        self._state = graph_state
        self._strategy = strategy
        self._cost_model = cost_model

        # 通知策略初始前沿边（G_0 的邻居，按确定性顺序）
        # GraphState._init_g0 已保证邻居按节点 ID 排序，
        # 故 BFS 策略的起始顺序是可复现的。
        initial_frontier = graph_state._initial_frontier
        if initial_frontier:
            strategy.on_new_frontier(initial_frontier)

    # ===================================================================
    # 运行揭示循环
    # ===================================================================

    def run(
        self,
        budget: Optional[int] = None,
        max_ratio: Optional[float] = None,
    ) -> dict:
        """执行完整的揭示循环。

        Parameters
        ----------
        budget : int or None
            最大揭示节点数（不含种子）。None = 揭示所有可达节点。
        max_ratio : float or None
            最大揭示节点比例。与 budget 同时设置时取较小上限。

        Returns
        -------
        dict
            揭示统计信息（见 GraphState.stats()）。

        Raises
        ------
        ValueError
            如果 budget < 0 或 max_ratio 不在 (0, 1] 范围。
        """
        # ---- 计算预算上限 --------------------------------------------------
        actual_budget = self._resolve_budget(budget, max_ratio)

        logger.info(
            "揭示循环开始: 策略=%s, 成本模型=%s, 预算=%d, |V|=%d",
            self._strategy.name, self._cost_model.name,
            actual_budget, self._state.n_nodes_total,
        )

        # ---- 主循环 --------------------------------------------------------
        step = 0
        while self._state.get_budget_used() < actual_budget:
            # 终止条件 1: 前沿为空
            if self._state.is_frontier_empty:
                logger.info("前沿为空，揭示循环终止（step=%d）", step)
                break

            # 策略选择下一个节点
            next_node = self._strategy.select(self._state)
            if next_node is None:
                logger.info("策略返回 None，揭示循环终止（step=%d）", step)
                break

            # 成本计算
            cost = self._cost_model.compute(next_node, self._state)
            remaining = actual_budget - self._state.get_budget_used()
            if cost > remaining:
                logger.info(
                    "预算不足: 需要 %.1f, 剩余 %d（step=%d）",
                    cost, remaining, step,
                )
                break

            # 执行揭示
            step += 1
            new_frontier = reveal_node(self._state, next_node)

            # 通知策略（BFS 等有状态策略需要）
            if new_frontier:
                self._strategy.on_new_frontier(new_frontier)

            if step % 100 == 0 or (
                step <= 10 and self._state.get_budget_used() <= 10
            ):
                logger.debug(
                    "step=%d: reveal %d, |V_revealed|=%d, "
                    "|frontier|=%d, budget_used=%d",
                    step, next_node,
                    len(self._state.V_revealed),
                    len(self._state.frontier_edges),
                    self._state.get_budget_used(),
                )

        # ---- 返回统计 -------------------------------------------------------
        stats = self._state.stats()
        stats['strategy'] = self._strategy.name
        stats['cost_model'] = self._cost_model.name
        stats['budget_limit'] = actual_budget

        logger.info(
            "揭示循环结束: 揭示 %d/%d 节点 (%.1f%%), "
            "前沿剩余=%d, 边特征=%d",
            stats['n_revealed'], stats['n_total'],
            stats['reveal_ratio'] * 100,
            stats['n_frontier_remaining'],
            stats['n_revealed_edge_feats'],
        )

        return stats

    # ===================================================================
    # 内部方法
    # ===================================================================

    def _resolve_budget(
        self,
        budget: Optional[int],
        max_ratio: Optional[float],
    ) -> int:
        """解析预算上限，处理 budget 和 max_ratio 的组合逻辑。"""
        # 默认：揭示所有可达节点
        resolved = self._state.n_nodes_total

        if budget is not None:
            if budget < 0:
                raise ValueError(f"budget 不能为负数: {budget}")
            resolved = min(resolved, budget)

        if max_ratio is not None:
            if not (0 < max_ratio <= 1):
                raise ValueError(
                    f"max_ratio 必须在 (0, 1] 内: {max_ratio}")
            ratio_budget = int(max_ratio * self._state.n_nodes_total)
            resolved = min(resolved, ratio_budget)

        return max(resolved, 1)  # 至少允许揭示 1 个节点

    # ===================================================================
    # 状态查询
    # ===================================================================

    @property
    def state(self) -> GraphState:
        """当前图状态（只读）。"""
        return self._state

    @property
    def strategy(self) -> RevealStrategy:
        """当前策略。"""
        return self._strategy

    @property
    def cost_model(self) -> CostModel:
        """当前成本模型。"""
        return self._cost_model

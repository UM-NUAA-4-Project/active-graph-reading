"""揭示循环模块：策略与揭示动作之间的协调者。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .graph_state import GraphState
from .reveal_action import reveal_node

if TYPE_CHECKING:
    from ..core.base_reader import BaseReader

logger = logging.getLogger(__name__)


class RevealLoop:
    """揭示协议的主循环协调器。每步揭示固定消耗 1 预算。"""

    def __init__(
        self,
        graph_state: GraphState,
        strategy: BaseReader,
    ):
        self._state = graph_state
        self._strategy = strategy

        strategy.reset()

        initial_frontier = graph_state._initial_frontier
        if initial_frontier:
            strategy.on_new_frontier(initial_frontier)

    def run(
        self,
        budget: Optional[int] = None,
        max_ratio: Optional[float] = None,
    ) -> dict:
        """执行完整的揭示循环。budget=None/max_ratio=None 表示揭示所有可达节点。"""
        actual_budget = self._resolve_budget(budget, max_ratio)
        self._state._budget_limit = actual_budget

        logger.info(
            "揭示循环开始: 策略=%s, 预算=%d, |V|=%d",
            self._strategy.name,
            actual_budget, self._state.n_nodes_total,
        )

        step = 0
        while self._state.get_budget_used() < actual_budget:
            if self._state.is_frontier_empty:
                logger.info("前沿为空，揭示循环终止（step=%d）", step)
                break

            next_node = self._strategy.select(self._state)
            if next_node is None:
                logger.info("策略返回 None，揭示循环终止（step=%d）", step)
                break

            remaining = actual_budget - self._state.get_budget_used()
            if remaining < 1:
                logger.info("预算不足: 剩余 %d（step=%d）", remaining, step)
                break

            step += 1
            new_frontier = reveal_node(self._state, next_node)

            if new_frontier:
                self._strategy.on_new_frontier(new_frontier)

            if step % 100 == 0 or (step <= 10 and self._state.get_budget_used() <= 10):
                logger.debug(
                    "step=%d: reveal %d, |V_revealed|=%d, |frontier|=%d, budget_used=%d",
                    step, next_node, len(self._state.V_revealed),
                    len(self._state.frontier_edges), self._state.get_budget_used(),
                )

        stats = self._state.stats()
        stats['strategy'] = self._strategy.name
        stats['budget_limit'] = actual_budget

        logger.info(
            "揭示循环结束: 揭示 %d/%d 节点 (%.1f%%), 前沿剩余=%d, 边特征=%d",
            stats['n_revealed'], stats['n_total'], stats['reveal_ratio'] * 100,
            stats['n_frontier_remaining'], stats['n_revealed_edge_feats'],
        )

        return stats

    def _resolve_budget(self, budget: Optional[int], max_ratio: Optional[float]) -> int:
        resolved = self._state.n_nodes_total

        if budget is not None:
            if budget < 0:
                raise ValueError(f"budget 不能为负数: {budget}")
            resolved = min(resolved, budget)

        if max_ratio is not None:
            if not (0 < max_ratio <= 1):
                raise ValueError(f"max_ratio 必须在 (0, 1] 内: {max_ratio}")
            ratio_budget = int(max_ratio * self._state.n_nodes_total)
            resolved = min(resolved, ratio_budget)

        return max(resolved, 0)

    @property
    def state(self) -> GraphState:
        return self._state

    @property
    def strategy(self) -> BaseReader:
        return self._strategy

"""随机选择策略：从当前所有前沿边中均匀随机选择一条进行揭示。"""

import random
from typing import Optional

from ...core.base_reader import BaseReader
from ...engine.graph_state import GraphState


class RandomStrategy(BaseReader):
    """均匀随机选择前沿边，返回其未揭示端点。

    确定性（原则 4）：RNG 由 (seed_node, n_nodes_total) 播种——与
    random-walk 基线同款公式（protocol-adaptation.md），同一 (graph,
    seed_i) 轨迹可复现；此前走全局 random（eval 全程未播种），两次运行
    结果不同。候选按 (u, v) 升序后抽取，选择与集合迭代序无关。
    """

    def __init__(self):
        super().__init__()
        self._rng: Optional[random.Random] = None

    @property
    def name(self) -> str:
        return 'random'

    def reset(self) -> None:
        self._rng = None

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._rng is None:
            self._rng = random.Random(
                graph_state.seed_node * 1_000_003 + graph_state.n_nodes_total)
        edge = self._rng.choice(sorted(frontier))
        return edge[1]

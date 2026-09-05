"""Random-walk 固定策略（AGR 协议适配版）。

对应 GraphSAGE 一族的"邻域随机采样"读取方式：智能体从当前节点沿随机边
游走，首次到达未揭示节点时将其揭示（重访已揭示节点不消耗预算）；每步揭示
后从新节点继续游走。与现有 random（frontier 均匀采样，广度型）相对，
random-walk 是局部相关游走（深度型）。

协议适配决策：
- 每步只返回当前 frontier 中的未揭示端点（reveal_action 仅接受 frontier 节点）；
- 只消费游走所在节点的邻接信息（爬虫站在当前节点上可见其邻居），不使用
  任何全图全局结构；
- 重访不计预算：预算语义 = 揭示节点数（不含 seed），与协议一致；
- 游走上限（∝ n_nodes）防连通分量内反复重访死循环；超限后退化为 frontier
  均匀随机（与 random 策略同分布）并视作瞬移；
- 确定性：RNG 由 (seed_node, n_nodes_total) 算术播种（不用 tuple hash，
  其进程间随机化），同一 (graph, seed_i) 轨迹可复现。
"""

import random
from typing import Optional

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


class RandomWalkStrategy(BaseReader):
    """随机游走揭示：沿随机边游走，首次到达的未揭示节点即下一个揭示对象。"""

    def __init__(self, max_steps_factor: int = 4):
        super().__init__()
        self._cur: Optional[int] = None
        self._rng: Optional[random.Random] = None
        self._max_steps_factor = max_steps_factor

    @property
    def name(self) -> str:
        return 'random-walk'

    def reset(self) -> None:
        self._cur = None
        self._rng = None

    def _neighbors(self, state: GraphState, node: int):
        """当前节点的邻居（爬虫局部可见）；忽略自环（无信息）。"""
        g = state.g
        nbrs = set(g.successors(node).tolist()) | set(g.predecessors(node).tolist())
        nbrs.discard(node)
        return sorted(nbrs)

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None

        if self._cur is None:
            # rollout 起点：种子节点；RNG 确定性播种
            self._cur = graph_state.seed_node
            self._rng = random.Random(
                graph_state.seed_node * 1_000_003 + graph_state.n_nodes_total)

        cap = self._max_steps_factor * graph_state.n_nodes_total + 16
        cur = self._cur
        for _ in range(cap):
            nbrs = self._neighbors(graph_state, cur)
            if not nbrs:
                break
            nxt = self._rng.choice(nbrs)
            if nxt not in graph_state.V_revealed:
                self._cur = nxt   # 揭示后从新节点继续游走
                return nxt
            cur = nxt

        # 游走超限/死胡同：退化 frontier 均匀随机（瞬移，预算语义不变）
        edge = self._rng.choice(sorted(frontier))
        node = edge[1]
        self._cur = node
        return node

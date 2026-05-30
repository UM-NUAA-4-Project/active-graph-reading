"""揭示动作模块：实现协议核心动作 reveal_node(v)。"""

import logging
from typing import List, Tuple

from .graph_state import GraphState

logger = logging.getLogger(__name__)


def reveal_node(state: GraphState, v: int) -> List[Tuple[int, int]]:
    """执行 reveal_node(v) 动作，修改图状态（就地）。

    ① v 成为已揭示节点
    ② v 与已揭示节点之间的边特征变为已知
    ③ v 的关联边列表/邻居 ID 加入前沿

    Returns: 新发现的前沿边列表 [(revealed, unrevealed), ...]
    """
    if v in state.V_revealed:
        raise ValueError(f"节点 {v} 已经揭示，不能重复揭示")

    edges_to_remove: List[Tuple[int, int]] = list(
        state._frontier_by_unrevealed.get(v, ())
    )

    for (ru, _rv) in edges_to_remove:
        if state.has_edge_feat:
            key = (min(ru, v), max(ru, v))
            if key not in state.revealed_edge_keys:
                state.revealed_edge_keys.add(key)
                eid = state._edge_key_to_id.get(key)
                if eid is not None:
                    state._revealed_edge_ids.append(eid)

    for edge in edges_to_remove:
        state.frontier_edges.discard(edge)

    state._frontier_by_unrevealed.pop(v, None)

    state.V_revealed.add(v)
    state.reveal_history.append(v)
    state._increment_budget()

    state._frontier_degree.pop(v, None)

    # 增量排序：v 从 frontier 移除，加入 revealed
    state._remove_frontier_node(v)
    state._add_to_revealed(v)

    new_frontier: List[Tuple[int, int]] = []
    neighbors = state._get_neighbors(v)

    for nbr in neighbors:
        if nbr == v:
            continue

        if nbr in state.V_revealed:
            if state.has_edge_feat:
                key = (min(v, nbr), max(v, nbr))
                if key not in state.revealed_edge_keys:
                    state.revealed_edge_keys.add(key)
                    eid = state._edge_key_to_id.get(key)
                    if eid is not None:
                        state._revealed_edge_ids.append(eid)
        else:
            edge = (v, nbr)
            is_new = edge not in state.frontier_edges
            state.frontier_edges.add(edge)
            state._frontier_degree[nbr] += 1
            is_new_node = nbr not in state._frontier_by_unrevealed
            state._frontier_by_unrevealed[nbr].add(edge)
            if is_new:
                new_frontier.append(edge)
            if is_new_node:
                state._add_frontier_node(nbr)

    logger.debug(
        "reveal_node(%d): 新前沿=%d, |V_revealed|=%d, |frontier|=%d",
        v, len(new_frontier), len(state.V_revealed), len(state.frontier_edges),
    )

    return new_frontier

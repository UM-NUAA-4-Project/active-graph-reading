"""
揭示动作模块
============

实现协议核心动作 reveal_node(v)，对应 base-protocol-v0.md 第 3 节。

协议定义：
- ① v 成为已揭示节点
- ② v 与已揭示节点之间的边特征变为已知
- ③ v 的关联边列表 / 邻居 ID 加入前沿

前置条件：v 必须是某个已揭示节点的邻居（在 frontier 中）。
"""

import logging
from typing import List, Tuple

from .graph_state import GraphState

logger = logging.getLogger(__name__)


def reveal_node(state: GraphState, v: int) -> List[Tuple[int, int]]:
    """执行 reveal_node(v) 动作，修改图状态。

    单位成本 = 1（成本由 RevealLoop 中的 CostModel 计算，
    此处只负责状态转移）。

    Parameters
    ----------
    state : GraphState
        当前图状态（就地修改）。
    v : int
        待揭示的节点 ID。必须是某个前沿边的未揭示端点。

    Returns
    -------
    List[Tuple[int, int]]
        新发现的前沿边列表，每项为 (revealed, unrevealed)。
        供策略的 on_new_frontier 钩子使用。

    Raises
    ------
    ValueError
        如果 v 已经在已揭示集合中。
    """
    # ---- 前置检查 ----------------------------------------------------------
    if v in state.V_revealed:
        raise ValueError(f"节点 {v} 已经揭示，不能重复揭示")

    # ---- ② 收集所有指向 v 的前沿边，获取边特征 ------------------------------
    # TODO(perf): 当前 O(|frontier|) 扫描；大规模图可维护
    # _frontier_by_unrevealed: Dict[int, Set[edge]] 反向索引，
    # 实现 O(deg_revealed) 查找。PPA 规模下暂不必要。
    edges_to_remove: List[Tuple[int, int]] = []

    for (ru, rv) in state.frontier_edges:
        if rv == v:
            edges_to_remove.append((ru, rv))

    # 记录通过前沿边连接到 v 的已揭示节点
    for (ru, _rv) in edges_to_remove:
        key = (min(ru, v), max(ru, v))
        if key not in state.revealed_edge_feats and state.has_edge_feat:
            state.revealed_edge_feats[key] = state._get_edge_feat(ru, v)

    # ---- 移除所有指向 v 的前沿边 -------------------------------------------
    for edge in edges_to_remove:
        state.frontier_edges.discard(edge)

    # ---- ① v 成为已揭示节点 ------------------------------------------------
    state.V_revealed.add(v)
    state.reveal_history.append(v)
    state._increment_budget()

    # ---- 清理 v 的前沿度 ---------------------------------------------------
    state._frontier_degree.pop(v, None)

    # ---- ③ 遍历 v 的所有邻居，扩展前沿 --------------------------------------
    new_frontier: List[Tuple[int, int]] = []
    neighbors = state._get_neighbors(v)

    for nbr in neighbors:
        if nbr == v:
            continue  # 自环跳过

        if nbr in state.V_revealed:
            # 防御性代码：对无向图，此分支在 v0 协议下为死代码——
            # nbr 揭示时已将其与 v 的边加入 frontier_edges，
            # 随后在 step ② 中已处理。仅在未来支持有向图或
            # 揭示粒度拆分（例：逐边揭示而非逐节点揭示）时可能激活。
            key = (min(v, nbr), max(v, nbr))
            if key not in state.revealed_edge_feats and state.has_edge_feat:
                state.revealed_edge_feats[key] = state._get_edge_feat(v, nbr)
        else:
            # nbr 未揭示 → 添加前沿边
            edge = (v, nbr)
            is_new = edge not in state.frontier_edges
            state.frontier_edges.add(edge)
            state._frontier_degree[nbr] += 1
            if is_new:
                new_frontier.append(edge)

    logger.debug(
        "reveal_node(%d): 新前沿=%d, |V_revealed|=%d, |frontier|=%d",
        v, len(new_frontier),
        len(state.V_revealed), len(state.frontier_edges),
    )

    return new_frontier

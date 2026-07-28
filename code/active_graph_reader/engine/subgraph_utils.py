"""子图工具：按 reveal 序列构建子图，支持多 checkpoint 快照。"""

from typing import List, Sequence, Tuple

import torch
import dgl

from .graph_state import GraphState
from .reveal_action import reveal_node
from ..gnn.subgraph_builder import SubgraphBuilder


def build_subgraph_from_reveal(
    full_g: dgl.DGLGraph,
    seed: int,
    reveal_order: Sequence[int],
    budget: int,
) -> Tuple[dgl.DGLGraph, List[int]]:
    """按 reveal_order 揭示前 budget 个节点，构建最终子图（已加自环）。

    Returns
    -------
    sub_g : dgl.DGLGraph
    original_ids : 子图节点对应的原图 ID（升序）
    """
    state = GraphState(full_g, seed_node=seed)
    for i in range(min(budget, len(reveal_order))):
        v = reveal_order[i]
        if v in state.V_revealed:
            continue
        reveal_node(state, v)
    sub_g, original_ids = SubgraphBuilder().build(state)
    return sub_g, original_ids


def reveal_with_checkpoints(
    full_g: dgl.DGLGraph,
    seed: int,
    reveal_order: Sequence[int],
    budget_checkpoints: Sequence[int],
) -> List[Tuple[int, dgl.DGLGraph, List[int]]]:
    """一次性逐步 reveal，在多个 budget checkpoint 处快照子图。

    Returns
    -------
    list of (budget, sub_g_with_self_loop, original_ids)
    """
    state = GraphState(full_g, seed_node=seed)
    checkpoints_sorted = sorted(set(budget_checkpoints))
    results: List[Tuple[int, dgl.DGLGraph, List[int]]] = []
    cp_idx = 0

    for i, v in enumerate(reveal_order):
        while (cp_idx < len(checkpoints_sorted)
               and state._budget_used >= checkpoints_sorted[cp_idx]):
            sub_g, oids = SubgraphBuilder().build(state)
            results.append((checkpoints_sorted[cp_idx], sub_g, oids))
            cp_idx += 1
        if cp_idx >= len(checkpoints_sorted):
            break
        if v in state.V_revealed:
            continue
        reveal_node(state, v)

    while cp_idx < len(checkpoints_sorted):
        if state._budget_used >= checkpoints_sorted[cp_idx]:
            sub_g, oids = SubgraphBuilder().build(state)
            results.append((checkpoints_sorted[cp_idx], sub_g, oids))
            cp_idx += 1
        else:
            break
    return results

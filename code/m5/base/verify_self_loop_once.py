"""验证自环只加一次：GraphState → SubgraphBuilder / FrontierGraphBuilder 输出的子图中
每个节点恰好 1 条自环边，且无重复（原始图自环被 reveal 机制过滤，构建时补一次）。"""

import sys
from pathlib import Path

import dgl
import torch

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder


def count_self_loops(g):
    src, dst = g.edges()
    n_sl = int((src == dst).sum().item())
    # 检查是否每条自环重复（同 key 出现两次）
    keys = [(int(u), int(v)) for u, v in zip(src, dst) if u == v]
    return n_sl, len(keys) == len(set(keys))


def main():
    torch.manual_seed(0)
    # 构造一个带自环的随机图（模拟训练时的 add_self_loop 后状态）
    g0 = dgl.rand_graph(20, 40)
    g0 = dgl.add_self_loop(g0)  # 模拟训练时的 add_self_loop
    g0.edata['feat'] = torch.randn(g0.num_edges(), 7)

    n_sl_orig, _ = count_self_loops(g0)
    print(f"原始图(已加自环): nodes={g0.num_nodes()} edges={g0.num_edges()} "
          f"self_loops={n_sl_orig} (rand_graph 可能含重复有向边, 不影响验证)")

    # 模拟训练流程：reveal 8 个节点
    state = GraphState(g0, seed_node=3)
    frontier = list(state.get_frontier_nodes())
    revealed = [3]
    for _ in range(8):
        nbrs = state._get_neighbors(frontier[0]) if frontier else []
        v = frontier[0] if frontier else 5
        if v in state.V_revealed:
            v = (v + 1) % g0.num_nodes()
        reveal_node(state, v)
        frontier = [n for n in state.get_frontier_nodes() if n not in state.V_revealed]
        revealed.append(v)

    print(f"revealed={len(state.V_revealed)} 个节点, "
          f"revealed_edge_keys={len(state.revealed_edge_keys)}, "
          f"frontier_edges={len(state.frontier_edges)}")
    # revealed/frontier 边集合不应含自环键
    assert all(u != v for u, v in state.revealed_edge_keys), "revealed_edge_keys 含自环!"
    assert all(u != v for u, v in state.frontier_edges), "frontier_edges 含自环!"
    print("  reveal 机制: revealed_edge_keys / frontier_edges 均无自环 OK")

    # SubgraphBuilder（classifier 子图）
    builder = SubgraphBuilder()
    sub_g, _ = builder.build(state)
    sl, no_dup = count_self_loops(sub_g)
    print(f"SubgraphBuilder: nodes={sub_g.num_nodes()} edges={sub_g.num_edges()} "
          f"self_loops={sl} 无重复={no_dup}")
    assert sl == sub_g.num_nodes() and no_dup, "SubgraphBuilder 自环数量异常!"

    # FrontierGraphBuilder（reader 扩展子图）
    fb = FrontierGraphBuilder()
    fsg = fb.build(state)
    sl2, no_dup2 = count_self_loops(fsg.g)
    print(f"FrontierGraphBuilder: nodes={fsg.g.num_nodes()} edges={fsg.g.num_edges()} "
          f"self_loops={sl2} 无重复={no_dup2}  n_revealed={fsg.n_revealed}")
    assert sl2 == fsg.g.num_nodes() and no_dup2, "FrontierGraphBuilder 自环数量异常!"

    # 零边路径（1 个节点）
    state1 = GraphState(g0, seed_node=7)
    sub1, _ = builder.build(state1)
    sl1, nd1 = count_self_loops(sub1)
    print(f"零边路径 SubgraphBuilder: self_loops={sl1}/{sub1.num_nodes()} 无重复={nd1}")
    assert sl1 == sub1.num_nodes() and nd1

    print("\n全部通过：自环在子图路径中恰好加一次。")


if __name__ == '__main__':
    main()

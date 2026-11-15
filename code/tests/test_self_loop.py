"""自环验证：全链路恰好加一次自环（AGENTS.md 2026-07-28 bug 防回归）。

不变量：图中每个节点恰好 1 条自环；即
    remove_self_loop(g).num_edges() == g.num_edges() - g.num_nodes()
且删除后不再残留自环。

覆盖：
  1. 各注册数据集加载器（真实数据，可用性探测后跳过）
  2. 协议子图 builder：SubgraphBuilder / FrontierGraphBuilder（从已加载图出发）
  3. 训练路径端到端：lockstep rollout 后的最终 state → 分类器子图
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from datasets import get_dataset  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import reveal_node  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from active_graph_reader.gnn.frontier.graph_builder import (  # noqa: E402
    FrontierGraphBuilder)
from tools.seeds import deterministic_seed  # noqa: E402

DATASETS = {
    'ppa': 'D:\\data\\ogbg_ppa',
    'cifar10': 'D:\\data\\superpixels',
    'collab': 'D:\\data\\COLLAB',
}


def check_exactly_one_self_loop(g, tag=''):
    n = g.num_nodes()
    no_self = g.remove_self_loop()
    added = g.num_edges() - no_self.num_edges()
    assert added == n, (f'{tag}: 自环数 {added} != 节点数 {n} '
                        f'（重复添加或缺失）')
    u, v = no_self.edges()
    assert not bool((u == v).any()), f'{tag}: remove 后仍残留自环'
    return no_self


def _raw_ppa_edge_count(root, ogb_idx):
    """绕过加载器直读 processed 文件，得到未加自环的原始边数。"""
    import dgl
    proc = Path(root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed'
    graphs, _ = dgl.load_graphs(str(proc), idx_list=[ogb_idx])
    g = graphs[0]
    u, v = g.edges()
    return g.num_edges(), int((u == v).sum())


def test_dataset_loaders():
    for name, root in DATASETS.items():
        try:
            ds = get_dataset(name, root=root, splits=('train',), max_graphs=3)
            graphs = ds.train_graphs
        except Exception as e:  # noqa: BLE001
            print(f'  [{name}] SKIP（数据不可用: {type(e).__name__}: {e}）')
            continue
        assert graphs, f'{name}: 未加载到图'
        for i, g in enumerate(graphs):
            check_exactly_one_self_loop(g, f'{name}#train[{i}]')
        print(f'  [{name}] {len(graphs)} 图：每节点恰好 1 自环 OK '
              f'(n={graphs[0].num_nodes()}, E={graphs[0].num_edges()})')

        # PPA 额外对照原始文件：loader 后边数 == 原始边数 + n_nodes
        if name == 'ppa':
            ogb_idx = ds.train_indices[0]
            raw_e, raw_self = _raw_ppa_edge_count(root, ogb_idx)
            g = graphs[0]
            assert raw_self == 0, 'PPA 原始 processed 文件不应含自环'
            assert g.num_edges() - g.num_nodes() == raw_e, \
                (g.num_edges(), raw_e, g.num_nodes())
            print(f'  [ppa] 原始 {raw_e} 边 → loader 后 {g.num_edges()} '
                  f'(= raw + {g.num_nodes()} 自环) OK')


def test_protocol_builders():
    ds = get_dataset('ppa', root=DATASETS['ppa'], splits=('train',), max_graphs=2)
    g = ds.train_graphs[0]
    subgraph_builder = SubgraphBuilder()
    frontier_builder = FrontierGraphBuilder()

    for seed_i in range(3):
        n = g.num_nodes()
        state = GraphState(g, seed_node=deterministic_seed(0, seed_i, n))
        state._budget_limit = 10
        # 揭示几个节点（沿前沿随机）
        import random
        rng = random.Random(seed_i)
        for _ in range(10):
            if state.is_frontier_empty:
                break
            reveal_node(state, rng.choice(state.get_frontier_nodes()))

        # 分类器子图：revealed↔revealed 双向 + 自环
        sub_g, _ = subgraph_builder.build(state)
        no_self = check_exactly_one_self_loop(sub_g, f'SubgraphBuilder seed{seed_i}')
        assert no_self.num_edges() % 2 == 0, 'revealed 边应双向对称'

        # 前沿子图：revealed↔revealed 双向 + revealed→frontier 单向 + 自环
        fsub = frontier_builder.build(state)
        check_exactly_one_self_loop(fsub.g, f'FrontierGraphBuilder seed{seed_i}')
    print('  SubgraphBuilder / FrontierGraphBuilder：恰好一次自环 OK')


def test_training_path():
    """lockstep rollout（BC/eval 同款路径）后的最终 state → 分类器子图。"""
    ds = get_dataset('cifar10', root=DATASETS['cifar10'], splits=('train',),
                     max_graphs=2)
    builder = SubgraphBuilder()
    import random
    rng = random.Random(0)
    for gi, g in enumerate(ds.train_graphs):
        n = g.num_nodes()
        for seed_i in range(2):
            state = GraphState(g, seed_node=deterministic_seed(gi, seed_i, n))
            state._budget_limit = 8
            while (not state.is_frontier_empty
                   and state.get_budget_used() < state._budget_limit):
                reveal_node(state, rng.choice(state.get_frontier_nodes()))
            sub_g, _ = builder.build(state)
            check_exactly_one_self_loop(sub_g, f'cifar10 rollout g{gi}s{seed_i}')
    print('  cifar10 rollout → 分类器子图：恰好一次自环 OK')


def run_all():
    test_dataset_loaders()
    test_protocol_builders()
    test_training_path()
    print('test_self_loop: ALL PASSED')


if __name__ == '__main__':
    run_all()

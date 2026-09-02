"""统一训练循环单测：BC（base/attention）+ RL（base/attention，含奖励类型）。"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgl
import torch
import torch.nn as nn

from active_graph_reader import ReaderConfig, build_reader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from train_reader.core import bc_train_epoch, rl_train_epoch
from trajectories.core.trajectory import Trajectory


def _make_graphs(n_graphs=6, n=40, n_edges=200):
    random.seed(0)
    torch.manual_seed(0)
    graphs, labels = [], []
    for i in range(n_graphs):
        src = torch.randint(0, n, (n_edges,))
        dst = torch.randint(0, n, (n_edges,))
        g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=n)
        g.edata['feat'] = torch.randn(2 * n_edges, 7)
        graphs.append(dgl.add_self_loop(g))
        labels.append(torch.tensor(i % 3))
    return graphs, labels


def _make_valid_trajs(graphs, n_steps=5):
    """真实 random rollout 生成合法轨迹（teacher 节点保证前沿可达）。"""
    rng = random.Random(1)
    trajs = []
    for gi, g in enumerate(graphs):
        seed = rng.randrange(g.num_nodes())
        state = GraphState(g, seed_node=seed)
        state._budget_limit = n_steps
        order = []
        while len(order) < n_steps and not state.is_frontier_empty:
            v = rng.choice(state.get_frontier_nodes())
            order.append(v)
            reveal_node(state, v)
        if order:
            trajs.append(Trajectory(graph_idx=gi, seed_node=seed,
                                    reveal_ratio=0.1, reveal_order=order,
                                    final_pred=gi % 3, is_correct=True,
                                    teacher_type='gradient', ce=0.5))
    return trajs


class _StubClassifier(nn.Module):
    """可微分类器桩（mean pooling + linear）。"""

    def __init__(self, d=32, n_classes=3):
        super().__init__()
        self.lin = nn.Linear(d, n_classes)

    def forward(self, g, nids):
        import dgl.function as fn
        with g.local_scope():
            g.ndata['h'] = torch.zeros(g.num_nodes(), 32)
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'H'))
            return self.lin(g.ndata['H'])


def test_bc_loops():
    graphs, labels = _make_graphs()
    trajs = _make_valid_trajs(graphs)
    graph_cache = {i: (graphs[i], int(labels[i])) for i in range(len(graphs))}
    for kind in ('base', 'attention'):
        torch.manual_seed(0)
        reader = build_reader(ReaderConfig.from_legacy(kind, hidden_dim=32,
                                                       n_layers=2, dropout=0.1))
        opt = torch.optim.Adam(reader.parameters(), lr=1e-3)
        m1 = bc_train_epoch(reader, opt, graph_cache, trajs, 6, 'cpu')
        m2 = bc_train_epoch(reader, opt, graph_cache, trajs, 6, 'cpu')
        assert m1['loss'] > 0.01 and m2['loss'] > 0.01, (kind, m1, m2)
        assert abs(m2['loss'] - m1['loss']) > 1e-6, f'{kind} loss 应在训练中变化'
        print(f'  BC {kind}: losses=({m1["loss"]:.4f}, {m2["loss"]:.4f}) OK')


def test_rl_loops():
    graphs, labels = _make_graphs()
    labels_t = torch.stack(labels)
    builder = SubgraphBuilder()
    for kind in ('base', 'attention'):
        for reward in ('binary', 'asymmetric'):
            torch.manual_seed(0)
            reader = build_reader(ReaderConfig.from_legacy(
                kind, hidden_dim=32, n_layers=2, dropout=0.1, stochastic=True,
                temperature=0.5))
            opt = torch.optim.Adam(reader.parameters(), lr=1e-3)
            m = rl_train_epoch(reader, _StubClassifier(), builder, opt,
                               graphs, labels_t, 2, 2, 'cpu', reward, 1.1,
                               (0.05, 0.1), 0.1)
            assert set(m) == {'loss', 'avg_R', 'acc'}
            assert all(isinstance(v, float) for v in m.values())
            print(f'  RL {kind}/{reward}: loss={m["loss"]:.4f} acc={m["acc"]:.3f} OK')


def run_all():
    test_bc_loops()
    test_rl_loops()
    print('test_unified_loops: ALL PASSED')


if __name__ == '__main__':
    run_all()

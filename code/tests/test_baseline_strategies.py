"""baseline 策略单测：random-walk / SAGPool 的协议合规性、确定性、预算约束。"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'code'))
sys.path.insert(0, str(_REPO_ROOT))

import random  # noqa: E402

import torch  # noqa: E402
import dgl  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from baseline.randwalk import RandomWalkStrategy  # noqa: E402
from baseline.sagpool import (SAGPoolModel, SagpoolStrategy,  # noqa: E402
                              build_visible_graph)
from train_reader.core import rollout_strategy  # noqa: E402


def _make_graph(n: int = 20, seed: int = 0):
    """随机无向小图（双向边 + 自环），带节点/边特征。"""
    rng = random.Random(seed)
    src, dst = [], []
    for u in range(n):
        for _ in range(4):
            v = rng.randrange(n)
            if v != u:
                src += [u, v]
                dst += [v, u]
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    g.ndata['feat'] = torch.randn(n, 5)
    g.edata['feat'] = torch.randn(g.num_edges(), 3)
    return dgl.add_self_loop(g)


def _rollout(agent, g, seed_node=7, budget=5):
    state = GraphState(g, seed_node=seed_node)
    state._budget_limit = budget
    rollout_strategy(agent, state)
    return state


def test_random_walk_protocol_and_determinism():
    g = _make_graph(seed=0)
    state = _rollout(RandomWalkStrategy(), g, budget=5)
    assert state.get_budget_used() == 5
    assert len(state.reveal_history) == 6          # seed + 5 次揭示
    # 每个新揭示节点都与先前揭示区相邻（reveal_action 仅接受 frontier 节点）
    revealed = {state.seed_node}
    for v in state.reveal_history[1:]:
        assert v not in revealed
        revealed.add(v)
    # 确定性：同 (graph, seed) 两条轨迹一致
    state2 = _rollout(RandomWalkStrategy(), g, budget=5)
    assert state2.reveal_history == state.reveal_history
    print('  random-walk protocol + determinism OK')


def test_random_walk_budget_overflow():
    """预算远大于连通分量时：frontier 耗尽即停，不超预算。"""
    g = _make_graph(seed=1)
    state = _rollout(RandomWalkStrategy(), g, budget=10_000)
    assert state.is_frontier_empty
    assert state.get_budget_used() <= 10_000
    print('  random-walk budget overflow OK')


def test_sagpool_select_and_visible_graph():
    torch.manual_seed(0)
    model = SAGPoolModel(node_feat_dim=5, edge_feat_dim=3, n_classes=3,
                         hidden=16, n_layers=2, pool_ratio=0.5).eval()
    agent = SagpoolStrategy()
    agent._model = model
    agent._device = torch.device('cpu')

    g = _make_graph(seed=2)
    state = GraphState(g, seed_node=7)
    v = agent.select(state)
    frontier_nodes = {e[1] for e in state.frontier_edges}
    assert v in frontier_nodes

    # 可见子图：ids = revealed 段（升序）+ frontier 段（升序），映射完整
    g_vis, ids = build_visible_graph(state)
    assert sorted(ids) == sorted(state.V_revealed | frontier_nodes)
    assert ids[:len(state.V_revealed)] == sorted(state.V_revealed)
    for i, old in enumerate(ids):
        if old in frontier_nodes:
            assert g_vis.ndata['feat'][i].abs().sum().item() == 0.0
        else:
            assert torch.equal(g_vis.ndata['feat'][i], g.ndata['feat'][old])

    # 完整 rollout：预算耗尽即停
    state._budget_limit = 5
    rollout_strategy(agent, state)
    assert state.get_budget_used() == 5
    print('  sagpool select + visible graph OK')


def test_sagpool_select_determinism():
    torch.manual_seed(0)
    model = SAGPoolModel(node_feat_dim=5, edge_feat_dim=3, n_classes=3,
                         hidden=16, n_layers=2, pool_ratio=0.5).eval()
    g = _make_graph(seed=3)
    hists = []
    for _ in range(2):
        agent = SagpoolStrategy()
        agent._model = model
        agent._device = torch.device('cpu')
        state = _rollout(agent, g, budget=5)
        hists.append(state.reveal_history)
    assert hists[0] == hists[1]
    print('  sagpool determinism OK')


def test_registry_integration():
    from baseline.strategies import register_baseline_strategies
    register_baseline_strategies()
    from active_graph_reader.reader.fixed import create_fixed_reader
    assert create_fixed_reader('random-walk').name == 'random-walk'
    print('  registry integration OK')


class _MeanClf(torch.nn.Module):
    """mock 分类器：图节点特征均值 → 线性 logits（支持 dgl.batch 输入）。"""

    def __init__(self, node_feat_dim=5, n_classes=3):
        super().__init__()
        self.lin = torch.nn.Linear(node_feat_dim, n_classes)

    def forward(self, g, nids):
        x = g.ndata['feat']
        segs = []
        off = 0
        for n in g.batch_num_nodes().tolist():
            segs.append(x[off:off + n].mean(dim=0))
            off += n
        return self.lin(torch.stack(segs))


def test_subgraphx_protocol_and_determinism():
    from baseline.subgraphx import SubgraphXStrategy
    g = _make_graph(seed=4)
    clf = _MeanClf().eval()                          # 共享同一分类器（权重一致）
    hists = []
    for _ in range(2):
        agent = SubgraphXStrategy(n_rollout=3, expand_atoms=4, max_evals=64)
        agent.configure(clf, torch.device('cpu'))
        state = _rollout(agent, g, budget=5)
        hists.append(state.reveal_history)
        assert state.get_budget_used() == 5
        assert len(state.reveal_history) == 6        # seed + 5 次揭示
        # 每个新揭示节点不重复（reveal_node 会拒绝重复，能跑完即合规）
        assert len(set(state.reveal_history)) == 6
    assert hists[0] == hists[1]                      # 确定性（无随机源）
    print('  subgraphx protocol + determinism OK')


def test_subgraphx_small_component():
    """分量耗尽早于预算：frontier 空即停，不超预算。"""
    from baseline.subgraphx import SubgraphXStrategy
    g = _make_graph(seed=5)
    agent = SubgraphXStrategy(n_rollout=2, expand_atoms=3, max_evals=32)
    agent.configure(_MeanClf().eval(), torch.device('cpu'))
    state = _rollout(agent, g, budget=10_000)
    assert state.is_frontier_empty
    assert state.get_budget_used() <= 10_000
    print('  subgraphx small component OK')


def run_all():
    test_random_walk_protocol_and_determinism()
    test_random_walk_budget_overflow()
    test_sagpool_select_and_visible_graph()
    test_sagpool_select_determinism()
    test_registry_integration()
    test_subgraphx_protocol_and_determinism()
    test_subgraphx_small_component()
    print('test_baseline_strategies: ALL PASSED')


if __name__ == '__main__':
    run_all()

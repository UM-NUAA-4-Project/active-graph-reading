"""EvidenceReader 单测。

覆盖：选点路径与 base 逐字节一致（权重对齐后 scores/frontier 相等）、
证据流形状与状态更新、协议合规（证据预测只依赖已揭示嵌入——用垃圾数据
注入前沿行验证）、config 往返/权重推断/checkpoint 重建、BC 循环冒烟。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgl
import torch

from active_graph_reader import ReaderConfig, build_reader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.reader.evidence import EvidenceReader
from train_reader.core import bc_train_epoch, load_reader_from_checkpoint

EFD = 7   # edge feat dim
D = 32    # hidden


def _graph(seed: int) -> dgl.DGLGraph:
    torch.manual_seed(seed)
    n = 30
    src = torch.randint(0, n, (120,))
    dst = torch.randint(0, n, (120,))
    g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=n)
    g.edata['feat'] = torch.randn(g.num_edges(), EFD)
    return dgl.add_self_loop(g)


def _states():
    torch.manual_seed(0)
    states = []
    for i, seed_node in enumerate((0, 5, 11)):
        g = _graph(i)
        st = GraphState(g, seed_node=seed_node)
        st._budget_limit = 6
        reveal_node(st, 3)          # 制造非初始状态（frontier 变化）
        states.append(st)
    return states


def _readers():
    torch.manual_seed(0)
    base = build_reader(ReaderConfig.from_legacy(
        'base', edge_feat_dim=EFD, hidden_dim=D, n_layers=2, dropout=0.0))
    torch.manual_seed(1)
    ev = build_reader(ReaderConfig.from_legacy(
        'evidence', edge_feat_dim=EFD, hidden_dim=D, n_layers=2, dropout=0.0,
        n_classes=3))
    missing, unexpected = ev.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, f"base 键多余: {unexpected}"
    assert missing and all(k.startswith(('evidence_',)) for k in missing), missing
    base.eval()
    ev.eval()
    return base, ev


def test_policy_identical_to_base():
    base, ev = _readers()
    states = _states()
    r_base = base.forward_states(states, temperature=1.0)
    hc = ev.init_evidence(len(states), 'cpu')
    last = [s.seed_node for s in states]
    r_ev, nh, logits = ev.forward_states_evidence(states, 1.0, hc, last)
    assert len(logits) == len(states)
    for (s1, p1, f1), (s2, p2, f2) in zip(r_base, r_ev):
        assert f1 == f2
        assert torch.allclose(s1, s2, atol=1e-6), "选点打分与 base 不一致"
        assert torch.allclose(p1, p2, atol=1e-6)
    print('  选点路径与 base 逐字节一致 OK')


def test_evidence_stream_and_protocol():
    base, ev = _readers()
    states = _states()
    hc = ev.init_evidence(len(states), 'cpu')
    last = [s.seed_node for s in states]

    # 手工复算证据流：Δ 取已揭示节点的嵌入，前沿行换成垃圾后预测必须不变
    subgraphs = [ev.frontier_builder.build(s) for s in states]
    all_embeds = ev.frontier_encoder.forward_all_batch(subgraphs)
    deltas = []
    for k, sg in enumerate(subgraphs):
        revealed_sorted = states[k].get_revealed_sorted()
        pos = revealed_sorted.index(last[k])
        deltas.append(all_embeds[k][pos])
    hn, _ = ev.evidence_lstm(
        torch.stack(deltas).unsqueeze(0),
        (torch.stack([h.squeeze(0) for h, _ in hc]).unsqueeze(0),
         torch.stack([c.squeeze(0) for _, c in hc]).unsqueeze(0)))
    h_t = hn[0]
    manual = []
    for k, sg in enumerate(subgraphs):
        z = ev.evidence_pool(h_t[k], all_embeds[k][:sg.n_revealed])
        manual.append(ev.evidence_head(torch.cat([z, h_t[k]])))

    # 前沿行注入垃圾 → 证据预测必须不变（协议合规：裁判只看已揭示内容）
    poisoned = []
    for k, sg in enumerate(subgraphs):
        h_all = all_embeds[k].clone()
        h_all[sg.n_revealed:] = torch.randn_like(h_all[sg.n_revealed:]) * 100
        poisoned.append(h_all)
    orig_forward = ev.frontier_encoder.forward_all_batch
    ev.frontier_encoder.forward_all_batch = lambda subs: poisoned
    try:
        _, nh2, logits2 = ev.forward_states_evidence(states, 1.0, hc, last)
    finally:
        ev.frontier_encoder.forward_all_batch = orig_forward

    for a, b in zip(manual, logits2):
        assert torch.allclose(a, b, atol=1e-5), "前沿嵌入泄漏进证据预测"
    for lg in logits2:
        assert lg.shape == (3,)
    print('  证据流：形状/手工复算/前沿注入不变性 OK')


def test_hc_reset_independence():
    ev = build_reader(ReaderConfig.from_legacy(
        'evidence', edge_feat_dim=EFD, hidden_dim=D, n_layers=2,
        dropout=0.0, n_classes=3))
    hc_a = ev.init_evidence(2, 'cpu')
    hc_b = ev.init_evidence(2, 'cpu')
    for (h1, c1), (h2, c2) in zip(hc_a, hc_b):
        assert torch.equal(h1, h2) and torch.equal(c1, c2)
    hc_a[0][0].fill_(1.0)
    assert torch.count_nonzero(hc_b[0][0]) == 0
    print('  证据状态按 state 独立、reset 语义 OK')


def test_config_roundtrip_infer():
    cfg = ReaderConfig.from_legacy('evidence', edge_feat_dim=EFD,
                                   hidden_dim=D, n_layers=2, dropout=0.1,
                                   n_classes=3)
    d = cfg.to_dict()
    assert d['n_classes'] == 3 and d['kind'] == 'evidence'
    cfg2 = ReaderConfig.from_dict(d)
    assert cfg2.kind == 'evidence' and cfg2.n_classes == 3

    try:
        ReaderConfig.from_legacy('evidence', n_classes=0)
        raise AssertionError('n_classes=0 应触发 ValueError')
    except ValueError:
        pass

    reader = build_reader(cfg)
    assert isinstance(reader, EvidenceReader)

    sd = reader.state_dict()
    inferred = __import__('active_graph_reader.config',
                          fromlist=['infer_reader_config']).infer_reader_config(sd)
    assert inferred.kind == 'evidence' and inferred.n_classes == 3

    ckpt = {'reader_state_dict': sd, 'config': cfg.to_dict()}
    reloaded = load_reader_from_checkpoint(ckpt, 'cpu')
    assert isinstance(reloaded, EvidenceReader)
    print('  config 往返/权重推断/checkpoint 重建 OK')


def test_bc_smoke():
    torch.manual_seed(0)
    graphs = [_graph(i) for i in range(4)]
    graph_cache = {i: (graphs[i], i % 3) for i in range(4)}
    from trajectories.core.trajectory import Trajectory
    trajs = []
    for i in range(4):
        for seed in (0, 7):
            trajs.append(Trajectory(
                graph_idx=i, seed_node=seed, reveal_ratio=0.1,
                reveal_order=[3, 9, 14], final_pred=-1, is_correct=True,
                teacher_type='gradient'))

    reader = build_reader(ReaderConfig.from_legacy(
        'evidence', edge_feat_dim=EFD, hidden_dim=D, n_layers=2,
        dropout=0.0, n_classes=3))
    optimizer = torch.optim.Adam(reader.parameters(), lr=1e-3)
    metrics = bc_train_epoch(reader, optimizer, graph_cache, trajs,
                             batch_size=4, device='cpu',
                             temperature=1.0, evidence_lambda=1.0)
    assert metrics['loss'] == metrics['loss'] and metrics['loss'] > 0
    assert 'ev_acc' in metrics
    # λ=0：证据损失关闭，ev_acc 不出现在返回值
    metrics0 = bc_train_epoch(reader, optimizer, graph_cache, trajs,
                              batch_size=4, device='cpu',
                              temperature=1.0, evidence_lambda=0.0)
    assert 'ev_acc' not in metrics0 and metrics0['loss'] == metrics0['loss']
    print('  BC 循环冒烟（λ=1 与 λ=0）OK')


def test_rl_smoke():
    """RL 循环 evidence 分支：teleport/last_revealed 维护 + L_ev 项。"""
    import dgl.function as fn
    from train_reader.core import rl_train_epoch
    from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder

    class _Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(32, 3)

        def forward(self, g, nids):
            with g.local_scope():
                g.ndata['h'] = torch.zeros(g.num_nodes(), 32)
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'H'))
                return self.lin(g.ndata['H'])

    torch.manual_seed(0)
    graphs = [_graph(i) for i in range(3)]
    labels = torch.tensor([0, 1, 2])
    reader = build_reader(ReaderConfig.from_legacy(
        'evidence', edge_feat_dim=EFD, hidden_dim=D, n_layers=2,
        dropout=0.0, n_classes=3))
    optimizer = torch.optim.Adam(reader.parameters(), lr=1e-4)
    metrics = rl_train_epoch(
        reader, _Clf(), SubgraphBuilder(), optimizer,
        graphs, labels, batch_size=3, n_seeds=1, device='cpu',
        reward_type='binary', ln_k=1.1, budget_options=(0.1,),
        neg_scale=1.0, evidence_lambda=1.0)
    assert metrics['loss'] == metrics['loss']
    assert 'ev_acc' in metrics
    print(f"  RL 循环冒烟 OK（loss={metrics['loss']:.4f}, "
          f"ev_acc={metrics['ev_acc']:.3f}）")


def test_eval_smoke():
    """评估循环 evidence 分支：终判 = 证据头（acc_*），参照 = 冻结分类器（acc_clf_*）。"""
    import dgl.function as fn
    from train_reader.core import evaluate_reader
    from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder

    class _Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(32, 3)

        def forward(self, g, nids):
            with g.local_scope():
                g.ndata['h'] = torch.zeros(g.num_nodes(), 32)
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'H'))
                return self.lin(g.ndata['H'])

    torch.manual_seed(0)
    graphs = [_graph(i) for i in range(3)]
    labels = torch.tensor([0, 1, 2])
    reader = build_reader(ReaderConfig.from_legacy(
        'evidence', edge_feat_dim=EFD, hidden_dim=D, n_layers=2,
        dropout=0.0, n_classes=3))
    results = evaluate_reader(reader, _Clf(), SubgraphBuilder(),
                              graphs, labels, budget_options=(0.1,),
                              device='cpu', n_seeds=2)
    assert 'acc_10%' in results and 'acc_overall' in results
    assert 'acc_clf_10%' in results and 'acc_clf_overall' in results
    n_per_budget = 3 * 2  # graphs × seeds
    assert len(results['outcomes'][0.1]) == n_per_budget
    assert len(results['clf_outcomes'][0.1]) == n_per_budget
    # base 路径不受影响：无 acc_clf_* 键
    base, _ = _readers()
    r_base = evaluate_reader(base, _Clf(), SubgraphBuilder(),
                             graphs, labels, budget_options=(0.1,),
                             device='cpu', n_seeds=1)
    assert 'acc_clf_10%' not in r_base
    print(f"  评估冒烟 OK（ev_acc={results['acc_overall']:.3f}, "
          f"clf_ref={results['acc_clf_overall']:.3f}）")


if __name__ == '__main__':
    test_policy_identical_to_base()
    test_evidence_stream_and_protocol()
    test_hc_reset_independence()
    test_config_roundtrip_infer()
    test_eval_smoke()
    test_bc_smoke()
    test_rl_smoke()
    print('EVIDENCE READER TESTS PASSED')

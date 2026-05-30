"""
Smoke test — 综合冒烟测试
=========================

验证 Reader 协议实现的正确性。默认运行合成图测试（无需 OGBG-PPA），
通过 --real-data 可追加真实数据集测试。

Usage:
  python m3/smoke_test.py                          # 仅合成图测试
  python m3/smoke_test.py --real-data              # 合成图 + OGBG-PPA 测试
  python m3/smoke_test.py --real-data --root [/mnt/d/ogb-master/dataset] # 指定 OGB 数据集路径
"""

import argparse
import sys
import random
from pathlib import Path

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import torch
import dgl

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_loop import RevealLoop
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.strategies.registry import get_strategy
from active_graph_reader.cost.registry import get_cost_model
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from active_graph_reader.encoder.gin_encoder import GINEEncoder
from active_graph_reader.encoder.classifier import GraphClassifier

SEPARATOR = "=" * 60


# ============================================================================
# 合成图
# ============================================================================

def make_synthetic_graph(n_nodes=20, edge_feat_dim=7):
    """Create a synthetic undirected graph with edge features.

    Generates a random connected graph with 7-dim edge features.
    """
    src = list(range(n_nodes))
    dst = list(range(1, n_nodes)) + [0]  # cycle

    random.seed(42)
    for _ in range(n_nodes):
        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)
        if u != v and abs(u - v) > 1 and (u - v) % n_nodes not in (1, n_nodes - 1):
            src.append(u)
            dst.append(v)

    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)

    n_edges = g.num_edges()
    feat = torch.zeros(n_edges, edge_feat_dim)
    feat[torch.arange(n_edges), torch.randint(0, edge_feat_dim, (n_edges,))] = 1.0
    g.edata['feat'] = feat

    return g


# ============================================================================
# Synthetic tests (always run)
# ============================================================================

def test_g0_frontier_edge_feat():
    """[Synthetic] G_0 init + frontier edge feature visibility."""
    g = make_synthetic_graph(20, edge_feat_dim=7)
    state = GraphState(g, seed_node=0)

    assert len(state.V_revealed) == 1
    assert state.get_budget_used() == 0

    count = 0
    for u, v in state.frontier_edges:
        feat = state.get_frontier_edge_feat(u, v)
        assert feat.shape == (7,)
        count += 1
    assert count > 0

    print(f"  G_0: V_revealed={len(state.V_revealed)}, "
          f"frontier_edges={len(state.frontier_edges)}, budget=0")
    print(f"  G_0 frontier edge features: {count} edges readable [OK]")
    print(f"  budget_used after reading features: {state.get_budget_used()} "
          f"(should be 0) [OK]")


def test_reveal_protocol():
    """[Synthetic] reveal_node protocol correctness."""
    g = make_synthetic_graph(20, edge_feat_dim=7)
    state = GraphState(g, seed_node=0)

    first_edge = next(iter(state.frontier_edges))
    target = first_edge[1]
    new_frontier = reveal_node(state, target)

    assert target in state.V_revealed
    assert state.get_budget_used() == 1

    if new_frontier:
        u2, v2 = new_frontier[0]
        feat = state.get_frontier_edge_feat(u2, v2)
        assert feat.shape == (7,)

    print(f"  reveal_node({target}): budget=1, new_frontier={len(new_frontier)} [OK]")


def test_all_strategies():
    """[Synthetic] All three strategies run to completion."""
    g = make_synthetic_graph(20, edge_feat_dim=7)

    for strat_name in ['random', 'bfs', 'max-frontier-degree']:
        state = GraphState(g, seed_node=0)
        strategy = get_strategy(strat_name)
        cost_model = get_cost_model('unit')
        loop = RevealLoop(state, strategy, cost_model)
        stats = loop.run(budget=5)
        assert stats['n_revealed'] >= 1
        assert stats['budget_used'] <= 5
        print(f"  {strat_name}: revealed={stats['n_revealed']}, "
              f"budget_used={stats['budget_used']} [OK]")


def test_encoder_classifier():
    """[Synthetic] SubgraphBuilder -> GINEEncoder -> GraphClassifier."""
    g = make_synthetic_graph(20, edge_feat_dim=7)

    state = GraphState(g, seed_node=0)
    loop = RevealLoop(state, get_strategy('bfs'), get_cost_model('unit'))
    loop.run(budget=8)

    builder = SubgraphBuilder()
    sub_g, _ = builder.build(state)
    print(f"  Subgraph: |V|={sub_g.num_nodes()}, |E|={sub_g.num_edges()}")

    encoder = GINEEncoder(edge_feat_dim=7, hidden_dim=32, n_layers=2, dropout=0.0)
    encoder.eval()
    batch_g = dgl.batch([sub_g])
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long)
    with torch.no_grad():
        g_emb = encoder(batch_g, node_ids)
    assert g_emb.shape == (1, 32)
    print(f"  Encoder output shape: {g_emb.shape} [OK]")

    model = GraphClassifier(edge_feat_dim=7, n_classes=5, hidden_dim=32,
                            n_layers=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        logits = model(batch_g, node_ids)
    assert logits.shape == (1, 5)
    print(f"  Classifier output shape: {logits.shape} [OK]")


def test_multiple_graphs_batch():
    """[Synthetic] Multi-graph batch inference."""
    g = make_synthetic_graph(20, edge_feat_dim=7)
    builder = SubgraphBuilder()

    subgraphs = []
    for seed, budget in [(0, 3), (5, 5), (10, 7)]:
        state = GraphState(g, seed_node=seed)
        loop = RevealLoop(state, get_strategy('bfs'), get_cost_model('unit'))
        loop.run(budget=budget)
        sub_g, _ = builder.build(state)
        subgraphs.append(sub_g)

    batch_g = dgl.batch(subgraphs)
    model = GraphClassifier(edge_feat_dim=7, n_classes=5, hidden_dim=32,
                            n_layers=2, dropout=0.0)
    model.eval()
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long)
    with torch.no_grad():
        logits = model(batch_g, node_ids)
    assert logits.shape == (3, 5)
    print(f"  Batch (3 graphs) logits shape: {logits.shape} [OK]")


def test_frontier_edge_feat_guards():
    """[Synthetic] get_frontier_edge_feat error guards."""
    g = make_synthetic_graph(20, edge_feat_dim=7)
    state = GraphState(g, seed_node=0)

    # Both endpoints unrevealed -> ValueError
    unrevealed = [n for n in range(g.num_nodes())
                  if n not in state.V_revealed
                  and not any(e[1] == n for e in state.frontier_edges)]
    if len(unrevealed) >= 2:
        try:
            state.get_frontier_edge_feat(unrevealed[0], unrevealed[1])
            assert False, "should raise ValueError"
        except ValueError:
            print(f"  Guard (both unrevealed): ValueError raised [OK]")

    # No edge features -> ValueError
    g_no = dgl.graph(([0, 1], [1, 0]))
    g_no = dgl.add_self_loop(g_no)
    state_no = GraphState(g_no, seed_node=0)
    try:
        state_no.get_frontier_edge_feat(0, 1)
        assert False, "should raise ValueError"
    except ValueError:
        print(f"  Guard (no edge feat): ValueError raised [OK]")


# ============================================================================
# Real-data tests (OGBG-PPA required, only with --real-data)
# ============================================================================

def realdata_test_g0_frontier_edge_feat(g, edge_feat_dim, seed=42):
    """[Real] G_0 frontier edge feature visible, budget=0."""
    state = GraphState(g, seed_node=seed)
    edges = list(state.frontier_edges)
    assert len(edges) > 0, "G_0 should have frontier edges"
    u, v = edges[0]
    feat = state.get_frontier_edge_feat(u, v)
    assert feat.shape == (edge_feat_dim,), f"feat shape: {feat.shape}"
    assert state.get_budget_used() == 0
    print(f"  G_0 frontier edge ({u}, {v}) feat shape: {feat.shape}")
    print(f"  budget_used (should be 0): {state.get_budget_used()} [OK]")


def realdata_test_reader_encoder_classifier(g, edge_feat_dim, n_classes, seed=42):
    """[Real] BFS Reader -> SubgraphBuilder -> GINEEncoder -> GraphClassifier."""
    # Reader
    state = GraphState(g, seed_node=seed)
    loop = RevealLoop(state, get_strategy('bfs'), get_cost_model('unit'))
    stats = loop.run(budget=10)
    print(f"  revealed: {stats['n_revealed']}/{stats['n_total']} "
          f"({stats['reveal_ratio']:.2%})")

    # Subgraph
    builder = SubgraphBuilder()
    sub_g, _ = builder.build(state)
    print(f"  subgraph: |V|={sub_g.num_nodes()}, |E|={sub_g.num_edges()}")

    # Encoder
    encoder = GINEEncoder(edge_feat_dim=edge_feat_dim, hidden_dim=32,
                          n_layers=2, dropout=0.0)
    encoder.eval()
    batch_g = dgl.batch([sub_g])
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long)
    with torch.no_grad():
        g_emb = encoder(batch_g, node_ids)
    assert g_emb.shape == (1, 32)
    print(f"  encoder output: {g_emb.shape} [OK]")

    # Classifier
    model = GraphClassifier(edge_feat_dim=edge_feat_dim, n_classes=n_classes,
                            hidden_dim=32, n_layers=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        logits = model(batch_g, node_ids)
    assert logits.shape == (1, n_classes)
    print(f"  classifier output: {logits.shape} [OK]")

    return builder, model


def realdata_test_multi_budget_batch(g, edge_feat_dim, n_classes, builder, model):
    """[Real] Multi-budget multi-subgraph batch."""
    state1 = GraphState(g, seed_node=42)
    loop1 = RevealLoop(state1, get_strategy('bfs'), get_cost_model('unit'))
    loop1.run(budget=5)
    sub_g1, _ = builder.build(state1)

    state2 = GraphState(g, seed_node=42)
    loop2 = RevealLoop(state2, get_strategy('bfs'), get_cost_model('unit'))
    loop2.run(budget=20)
    sub_g2, _ = builder.build(state2)

    batch_g = dgl.batch([sub_g1, sub_g2])
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long)
    with torch.no_grad():
        logits = model(batch_g, node_ids)
    assert logits.shape == (2, n_classes)
    print(f"  Batch (2 graphs) logits: {logits.shape} [OK]")


def realdata_test_budget_zero_subgraph(g, edge_feat_dim, n_classes, builder, model):
    """[Real] Budget=0: subgraph with no revealed edges."""
    state = GraphState(g, seed_node=42)
    sub_g, _ = builder.build(state)
    print(f"  budget=0 subgraph: |V|={sub_g.num_nodes()}, |E|={sub_g.num_edges()}")

    batch_g = dgl.batch([sub_g])
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long)
    with torch.no_grad():
        logits = model(batch_g, node_ids)
    assert logits.shape == (1, n_classes)
    print(f"  Logits shape: {logits.shape} [OK]")


def run_realdata_tests(root: str):
    """Run all OGBG-PPA tests."""
    from ogb.graphproppred import DglGraphPropPredDataset

    print("\n[Real-1] Loading OGBG-PPA (graph 0)...")
    ds = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
    n_classes = int(ds.num_classes)
    g, label = ds[0]
    g = dgl.add_self_loop(g)
    edge_feat_dim = g.edata['feat'].shape[1] if 'feat' in g.edata else 0
    label = label.item()
    print(f"  |V|={g.num_nodes()}, |E|={g.num_edges()}, "
          f"edge_feat_dim={edge_feat_dim}, label={label}")

    print("\n[Real-2] G_0 frontier edge feature visibility...")
    realdata_test_g0_frontier_edge_feat(g, edge_feat_dim)

    print("\n[Real-3] Reader -> SubgraphBuilder -> Encoder -> Classifier...")
    builder, model = realdata_test_reader_encoder_classifier(
        g, edge_feat_dim, n_classes)

    print("\n[Real-4] Multi-budget batch...")
    realdata_test_multi_budget_batch(g, edge_feat_dim, n_classes, builder, model)

    print("\n[Real-5] Budget=0 subgraph...")
    realdata_test_budget_zero_subgraph(g, edge_feat_dim, n_classes, builder, model)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Smoke test — synthetic (always) + optional real-data')
    parser.add_argument('--real-data', action='store_true',
                        help='Also run OGBG-PPA real-data tests')
    parser.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                        help='OGB dataset root directory')
    args = parser.parse_args()

    test_count = 0
    fail_count = 0

    def run_test(name, fn, *fn_args):
        nonlocal test_count, fail_count
        test_count += 1
        try:
            fn(*fn_args)
        except Exception as e:
            fail_count += 1
            print(f"  FAIL: {e}")

    # ---- Synthetic tests ----------------------------------------------------
    print(SEPARATOR)
    print("Smoke Test — Synthetic Graphs")
    print(SEPARATOR)
    print()

    print("[1] G_0 init + frontier edge feature visibility...")
    run_test("G_0", test_g0_frontier_edge_feat)

    print("\n[2] reveal_node protocol correctness...")
    run_test("reveal", test_reveal_protocol)

    print("\n[3] All three strategies (random / bfs / max-frontier-degree)...")
    run_test("strategies", test_all_strategies)

    print("\n[4] SubgraphBuilder -> GINEEncoder -> GraphClassifier...")
    run_test("encoder-classifier", test_encoder_classifier)

    print("\n[5] Multi-graph batch inference...")
    run_test("batch", test_multiple_graphs_batch)

    print("\n[6] get_frontier_edge_feat error guards...")
    run_test("guards", test_frontier_edge_feat_guards)

    synthetic_ok = (fail_count == 0)

    # ---- Real-data tests ----------------------------------------------------
    if args.real_data:
        print()
        print(SEPARATOR)
        print("Smoke Test — OGBG-PPA Real Data")
        print(SEPARATOR)
        print(f"  root: {args.root}")
        run_test("real-data", run_realdata_tests, args.root)

    # ---- Summary ------------------------------------------------------------
    print()
    print(SEPARATOR)
    if fail_count == 0:
        print(f"All {test_count} tests passed.")
    else:
        print(f"{fail_count}/{test_count} tests FAILED.")
    print(SEPARATOR)

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

"""Smoke test for 2026-08-04 bug fixes (Bug 1 / Bug 2 / Issue 3).

Self-contained: synthetic DGL graphs + mock reader/classifier.
No PPA dataset or trained checkpoint required.

Bug 1  diagnose_checkpoint.run_batched_trajectories 预算超支
Bug 2  双重 dgl.add_self_loop（SubgraphBuilder 已加，外层重复）
Issue 3  eval._evaluate_learnable_batched 达 max budget 后不停止（性能）

Run:
  /mnt/d/anoconda/envs/gnn/python.exe code/m5/base/analysis/smoke_test_fixes.py
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import torch
import dgl

_HERE = Path(__file__).resolve()
_ANALYSIS = _HERE.parent          # code/m5/base/analysis
_M5 = _ANALYSIS.parent.parent     # code/m5
_CODE = _M5.parent               # code

for p in (str(_CODE), str(_M5), str(_ANALYSIS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DIAG = _load_module('diag_smoke', _ANALYSIS / 'diagnose_checkpoint.py')
_EVAL = _load_module('eval_smoke', _ANALYSIS / 'eval.py')


# ---------------------------------------------------------------------------
# synthetic graphs + mocks
# ---------------------------------------------------------------------------

def make_graph(n_nodes, degree=4):
    """Connected directed graph (ring+forward), 7-dim edge feat, no self-loops.

    eval.py adds self-loops itself (line 149); SubgraphBuilder adds self-loops
    to the subgraph (subgraph_builder.py:86). We mirror that contract.
    """
    src, dst = [], []
    for i in range(n_nodes):
        for d in range(1, degree + 1):
            src.append(i)
            dst.append((i + d) % n_nodes)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n_nodes)
    g.edata['feat'] = torch.randn(g.num_edges(), 7)
    return g


class MockReader:
    """Uniform-over-frontier reader. Bypasses FrontierEncoder GNN entirely."""
    temperature = 0.2
    frontier_encoder = True   # so evaluate_strategy_split would pick learnable path

    def reset(self):
        pass

    def on_new_frontier(self, frontier):
        pass

    def forward_states(self, states, temperature=1.0):
        out = []
        for s in states:
            fn = s.get_frontier_nodes()
            if not fn:
                out.append((torch.empty(0), torch.empty(0), []))
                continue
            n = len(fn)
            probs = torch.full((n,), 1.0 / n)
            out.append((torch.zeros(n), probs, fn))
        self.state_steps = getattr(self, 'state_steps', 0) + len(states)
        return out


class MockClassifier:
    """Returns random logits + asserts single self-loop per node (Bug 2 guard)."""

    def __call__(self, batch_g, nids):
        s, d = batch_g.edges()
        sl = int((s == d).sum())
        n = batch_g.num_nodes()
        assert sl == n, (
            f"Bug2 regression: batched subgraph has {sl} self-loops for {n} "
            f"nodes (expected exactly {n} = 1/node; double add_self_loop?)"
        )
        return torch.randn(batch_g.batch_size, 37, device=batch_g.device)


# ---------------------------------------------------------------------------
# Test 1 — Bug 2: SubgraphBuilder produces exactly one self-loop per node
# ---------------------------------------------------------------------------

def test_subgraph_self_loop():
    g = make_graph(20)
    state = GraphState(g, seed_node=0)
    fn = state.get_frontier_nodes()
    for _ in range(6):
        if not fn:
            break
        reveal_node(state, fn[0])
        fn = state.get_frontier_nodes()

    sub_g, _ = SubgraphBuilder().build(state)
    s, d = sub_g.edges()
    n = sub_g.num_nodes()
    sl = int((s == d).sum())
    assert sl == n, f"Bug2: SubgraphBuilder self-loops {sl} != nodes {n}"
    for i in range(n):
        cnt = int(((s == d) & (s == i)).sum())
        assert cnt == 1, f"Bug2: node {i} has {cnt} self-loops (expected 1)"
    print(f"  [Bug2] SubgraphBuilder: {n} nodes, {sl} self-loops (1/node)  OK")


# ---------------------------------------------------------------------------
# Test 2 — Bug 1: diagnose run_batched_trajectories stops at each state's budget
# ---------------------------------------------------------------------------

def test_diagnose_budget_no_overrun():
    # mixed-size batch: small (B=5) and large (B=20) at 10% budget
    g_small = make_graph(50)
    g_large = make_graph(200)
    states_info = []
    for g, B, tag in [(g_small, 5, 's'), (g_large, 20, 'l')]:
        for _ in range(2):  # 2 samples per graph
            st = GraphState(g, seed_node=0)
            st._budget_limit = B
            states_info.append({
                'state': st, 'graph_gi': tag,
                'norm_entropies': [], 'revealed': {0},
            })

    _DIAG.run_batched_trajectories(
        MockReader(), states_info, 0.2, torch.device('cpu'),
    )

    for si in states_info:
        st = si['state']
        bu = st.get_budget_used()
        lim = st._budget_limit
        ne = len(si['norm_entropies'])
        assert bu == lim, (
            f"Bug1: budget_used={bu} != limit={lim} (overran by {bu - lim})"
        )
        assert ne == lim, (
            f"Bug1: norm_entropies len={ne} != budget {lim}"
        )
    print(f"  [Bug1] {len(states_info)} states (B=5 & B=20) all stopped at "
          f"own budget, no overrun  OK")


# ---------------------------------------------------------------------------
# Test 3 — Bug 2 (eval batching) + Issue 3 (max-budget stop) via real
#           _evaluate_learnable_batched with mixed-size graphs
# ---------------------------------------------------------------------------

def test_eval_learnable_batched():
    # small (50 nodes, max budget 7) + large (200 nodes, max budget 30)
    graphs = [make_graph(50), make_graph(200)]
    indices = [10, 20]
    labels = torch.zeros(201, dtype=torch.long)
    labels[10] = 0
    labels[20] = 1

    results = {'best': defaultdict(list), 'worst': defaultdict(list)}
    per_class = {
        'best': defaultdict(lambda: defaultdict(list)),
        'worst': defaultdict(lambda: defaultdict(list)),
    }

    reader = MockReader()
    classifier = MockClassifier()
    budgets_rel_sorted = sorted(_EVAL.BUDGETS_REL)  # [0.05, 0.10, 0.15]

    agg, _ = _EVAL._evaluate_learnable_batched(
        reader, graphs, labels, indices, classifier, torch.device('cpu'),
        'smoke', budgets_rel_sorted, results, per_class, chunk_size=2,
    )

    # (a) all budgets present for both seed types
    for kind in ('best', 'worst'):
        for b in budgets_rel_sorted:
            assert b in agg[kind] and len(agg[kind][b]) > 0, (
                f"missing results: {kind}/{b}"
            )
    # 2 graphs × 2 seeds → each (kind,budget) has 2 entries
    for kind in ('best', 'worst'):
        for b in budgets_rel_sorted:
            assert len(agg[kind][b]) == 2, (
                f"expected 2 entries for {kind}/{b}, got {len(agg[kind][b])}"
            )

    # (b) Bug 2: MockClassifier already asserted single self-loop per node
    #     on every batched subgraph (would raise if outer add_self_loop regressed)

    # (c) Issue 3: small graphs (max budget 7) must stop at 7, not run full
    #     max_steps=30. Total state-steps ≈ 2*7 + 2*30 = 74 with fix;
    #     ≈ 4*30 = 120 without fix. Threshold 90 cleanly separates.
    steps = reader.state_steps
    assert steps <= 90, (
        f"Issue3 regression: reader forwarded {steps} state-steps "
        f"(expected ~74 with max-budget stop, ~120 without)"
    )
    print(f"  [Bug2+Issue3] eval learnable: results complete for all 3 budgets "
          f"× 2 seeds; self-loop guard passed; state-steps={steps} (≤90)  OK")


# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Smoke test: 2026-08-04 bug fixes (Bug 1 / Bug 2 / Issue 3)")
    print("=" * 60)
    tests = [
        ("Bug2  SubgraphBuilder self-loop", test_subgraph_self_loop),
        ("Bug1  diagnose budget no-overrun", test_diagnose_budget_no_overrun),
        ("Bug2+Issue3  eval learnable batched", test_eval_learnable_batched),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    if failed:
        print(f"RESULT: {failed}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    print(f"RESULT: all {len(tests)} tests passed")
    print("=" * 60)


if __name__ == '__main__':
    main()

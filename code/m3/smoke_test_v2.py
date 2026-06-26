"""
Smoke test v2 — 验证优化后的 reader pipeline
============================================

使用合成图测试（不加载 OGBG-PPA）：
1. SubgraphBuilder 向量化正确性（与旧逻辑对比）
2. reveal_one() 协议正确性
3. 增量 reveal + 批量分类评测 pipeline
4. DataLoader 集成
5. 端到端速度 sanity check（64 图 × 3 strategy × 3 budget）
"""

import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_loop import RevealLoop
from active_graph_reader.reader.fixed.registry import get_strategy
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.reader import FixedStrategyReader, reveal_one

SEP = "=" * 60


# ==========================================================================
# 合成图生成
# ==========================================================================

def make_synthetic_graph(n_nodes=30, edge_feat_dim=7, seed=42):
    """Create a random connected undirected graph with edge features."""
    rng = random.Random(seed)
    src = list(range(n_nodes))
    dst = list(range(1, n_nodes)) + [0]  # cycle ensures connectivity

    for _ in range(n_nodes * 2):
        u = rng.randint(0, n_nodes - 1)
        v = rng.randint(0, n_nodes - 1)
        if u != v and abs(u - v) > 1:
            src.append(u)
            dst.append(v)

    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)

    n_edges = g.num_edges()
    feat = torch.randn(n_edges, edge_feat_dim)
    g.edata['feat'] = feat

    return g


def make_synthetic_dataset(n_graphs=64, n_nodes_range=(20, 80), edge_feat_dim=7):
    """Generate a synthetic dataset mimicking DglGraphPropPredDataset interface."""
    class SyntheticDataset:
        def __init__(self):
            rng = random.Random(42)
            self.graphs = []
            self.labels = []
            for i in range(n_graphs):
                n_nodes = rng.randint(*n_nodes_range)
                g = make_synthetic_graph(n_nodes, edge_feat_dim, seed=i)
                self.graphs.append(g)
                self.labels.append(i % 5)

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return self.graphs[idx], self.labels[idx]

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    return SyntheticDataset()


# ==========================================================================
# 本地评测 helper（替代已删除的 reader.evaluate）
# ==========================================================================

@torch.no_grad()
def _evaluate(
    reader,
    dataset,
    indices: List[int],
    classifier,
    budgets_rel: List[float],
    seeds_per_graph: int,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[float, List[float]]:
    """增量 reveal + 批量 GPU 推理评测。"""
    budgets_rel_sorted = sorted(budgets_rel)
    results: Dict[float, List[float]] = defaultdict(list)

    builder = SubgraphBuilder()

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]

        subgraphs: List[dgl.DGLGraph] = []
        meta: List[Tuple[int, float, int]] = []

        for batch_i, idx in enumerate(batch_indices):
            g_raw, label = dataset[int(idx)]
            if hasattr(label, 'item'):
                label = label.item()
            n_nodes = g_raw.num_nodes()
            budgets_abs = [max(1, int(r * n_nodes)) for r in budgets_rel_sorted]

            for seed_i in range(seeds_per_graph):
                seed_node = (idx * 31 + seed_i * 17 + 7) % max(n_nodes, 1)

                state = GraphState(g_raw, seed_node=seed_node)
                loop = RevealLoop(state, reader)

                for abs_b, rel_b in zip(budgets_abs, budgets_rel_sorted):
                    loop.run(budget=abs_b)
                    sub_g, _ = builder.build(state)
                    subgraphs.append(sub_g)
                    meta.append((batch_i, rel_b, label))

        batch_g = dgl.batch(subgraphs).to(device)
        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, node_ids)
        preds = logits.argmax(dim=1).cpu().tolist()

        graph_budget_correct: Dict[Tuple[int, float], List[float]] = defaultdict(list)
        for flat_i, (batch_i, rel_b, label) in enumerate(meta):
            graph_budget_correct[(batch_i, rel_b)].append(
                1.0 if preds[flat_i] == label else 0.0
            )

        for (_batch_i, rel_b), corrects in graph_budget_correct.items():
            results[rel_b].append(np.mean(corrects))

    return dict(results)


# ==========================================================================
# Test 1: SubgraphBuilder vectorized correctness
# ==========================================================================

def test_subgraph_builder_vectorized():
    """Compare new SubgraphBuilder against reference (old logic) implementation."""
    print("\n[Test 1] SubgraphBuilder vectorized correctness...")

    class ReferenceBuilder:
        def build(self, state):
            original_ids = sorted(state.V_revealed)
            n_nodes = len(original_ids)
            if n_nodes == 0:
                raise ValueError("empty")
            old_to_new = {old: i for i, old in enumerate(original_ids)}

            if not state.revealed_edge_keys:
                sub_g = dgl.graph(([], []), num_nodes=n_nodes)
                return sub_g, original_ids

            src_list, dst_list, feat_list = [], [], []
            for (u, v) in state.revealed_edge_keys:
                if u in old_to_new and v in old_to_new:
                    eid = state._edge_key_to_id[(u, v)]
                    feat = state.edge_feat[eid]
                    src_list.append(old_to_new[u])
                    dst_list.append(old_to_new[v])
                    feat_list.append(feat)
                    src_list.append(old_to_new[v])
                    dst_list.append(old_to_new[u])
                    feat_list.append(feat)

            src_t = torch.tensor(src_list, dtype=torch.long)
            dst_t = torch.tensor(dst_list, dtype=torch.long)
            sub_g = dgl.graph((src_t, dst_t), num_nodes=n_nodes)
            if feat_list:
                sub_g.edata['feat'] = torch.stack(feat_list, dim=0)
            return sub_g, original_ids

    ref_builder = ReferenceBuilder()
    new_builder = SubgraphBuilder()

    for i in tqdm(range(100), desc="  comparing", leave=False):
        g = make_synthetic_graph(n_nodes=random.randint(15, 50), seed=i)
        seed = random.randint(0, g.num_nodes() - 1)
        budget = random.randint(3, 15)

        state_new = GraphState(g, seed_node=seed)
        loop_new = RevealLoop(state_new, get_strategy('bfs'))
        loop_new.run(budget=budget)

        state_ref = GraphState(g, seed_node=seed)
        loop_ref = RevealLoop(state_ref, get_strategy('bfs'))
        loop_ref.run(budget=budget)

        sub_new, ids_new = new_builder.build(state_new)
        sub_ref, ids_ref = ref_builder.build(state_ref)

        assert ids_new == ids_ref, f"Graph {i}: original_ids mismatch"
        assert sub_new.num_nodes() == sub_ref.num_nodes(), f"Graph {i}: nodes mismatch"
        assert sub_new.num_edges() == sub_ref.num_edges(), f"Graph {i}: edges mismatch"
        if 'feat' in sub_new.edata:
            assert torch.allclose(sub_new.edata['feat'], sub_ref.edata['feat']), \
                f"Graph {i}: edge feat mismatch"

    print("  100 graphs compared: all match [OK]")


# ==========================================================================
# Test 2: reveal_one()
# ==========================================================================

def test_reader_reveal_one():
    """Basic protocol check via ActiveGraphReader."""
    print("\n[Test 2] reveal_one()...")

    g = make_synthetic_graph(30, edge_feat_dim=7)
    reader = FixedStrategyReader('bfs')

    sub_g, ids = reveal_one(reader, g, budget=10, seed_node=0)

    assert sub_g.num_nodes() >= 1
    assert len(ids) == sub_g.num_nodes()
    assert 'feat' in sub_g.edata or sub_g.num_edges() == 0
    print(f"  budget=10: |V|={sub_g.num_nodes()}, |E|={sub_g.num_edges()} [OK]")

    # Budget=0: only seed revealed, no edges
    sub_g0, ids0 = reveal_one(reader, g, budget=0, seed_node=5)
    assert sub_g0.num_nodes() == 1
    assert sub_g0.num_edges() == 0
    print(f"  budget=0: |V|={sub_g0.num_nodes()}, |E|={sub_g0.num_edges()} [OK]")


# ==========================================================================
# Test 3: 增量 reveal + 批量分类评测 pipeline
# ==========================================================================

def test_reader_evaluate():
    """End-to-end reveal + classify pipeline with a random classifier."""
    print("\n[Test 3] Reveal + classify pipeline...")

    dataset = make_synthetic_dataset(n_graphs=16, n_nodes_range=(20, 50))
    indices = list(range(len(dataset)))

    model = GraphClassifier(edge_feat_dim=7, n_classes=5, hidden_dim=32,
                            n_layers=2, dropout=0.0)
    model.eval()

    for strat in ['random', 'bfs', 'max-frontier-degree']:
        reader = FixedStrategyReader(strat)
        results = _evaluate(
            reader=reader,
            dataset=dataset,
            indices=indices,
            classifier=model,
            budgets_rel=[0.05, 0.10, 0.20, 0.50],
            seeds_per_graph=2,
            device=torch.device('cpu'),
            batch_size=4,
        )
        for b in sorted(results.keys()):
            accs = results[b]
            assert len(accs) == len(indices), \
                f"{strat} budget={b}: expected {len(indices)} graphs, got {len(accs)}"
        print(f"  {strat}: {len(results)} budget levels, "
              f"{sum(len(v) for v in results.values())} graph-acc entries [OK]")


# ==========================================================================
# Test 4: DataLoader 集成
# ==========================================================================

def test_reader_dataloader():
    """DataLoader with collate_fn for graph batching."""
    print("\n[Test 4] DataLoader integration...")

    dataset = make_synthetic_dataset(n_graphs=32, n_nodes_range=(15, 40))

    from torch.utils.data import DataLoader, Dataset

    class SimpleGraphDataset(Dataset):
        def __init__(self, syn_dataset):
            self.ds = syn_dataset

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            g, label = self.ds[idx]
            if hasattr(label, 'item'):
                label = label.item()
            return g, label

    def collate(batch):
        graphs, labels = zip(*batch)
        batched_g = dgl.batch(list(graphs))
        label_t = torch.tensor(labels, dtype=torch.long)
        return batched_g, label_t

    ds = SimpleGraphDataset(dataset)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)

    batch_count = 0
    graph_count = 0
    for batched_g, labels in loader:
        batch_count += 1
        graph_count += labels.size(0)
        assert batched_g.num_nodes() > 0
        assert labels.dim() == 1
        assert labels.size(0) <= 8

    assert batch_count > 0
    print(f"  {batch_count} batches, {graph_count} subgraphs [OK]")


# ==========================================================================
# Test 5: Speed sanity check
# ==========================================================================

def test_speed():
    """End-to-end timing: 64 graphs × 3 strategies × 3 budgets."""
    print("\n[Test 5] Speed sanity check (64 graphs × 3 strats × 3 budgets)...")

    dataset = make_synthetic_dataset(n_graphs=64, n_nodes_range=(20, 60))
    indices = list(range(len(dataset)))

    model = GraphClassifier(edge_feat_dim=7, n_classes=5, hidden_dim=32,
                            n_layers=2, dropout=0.0)
    model.eval()

    budgets_rel = [0.05, 0.10, 0.20]
    strategies = ['random', 'bfs', 'max-frontier-degree']

    total_start = time.time()
    for strat in strategies:
        reader = FixedStrategyReader(strat)
        t0 = time.time()
        _evaluate(
            reader=reader,
            dataset=dataset, indices=indices, classifier=model,
            budgets_rel=budgets_rel, seeds_per_graph=2,
            device=torch.device('cpu'), batch_size=8,
        )
        elapsed = time.time() - t0
        print(f"  {strat}: {elapsed:.2f}s "
              f"({len(indices)} graphs × {len(budgets_rel)} budgets × 2 seeds)")

    total = time.time() - total_start
    print(f"  Total: {total:.2f}s")


# ==========================================================================
# Main
# ==========================================================================

def main():
    test_count = 0
    fail_count = 0

    def run_test(name, fn):
        nonlocal test_count, fail_count
        test_count += 1
        try:
            fn()
        except Exception as e:
            fail_count += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(SEP)
    print("Smoke Test v2 — Optimized Reader Pipeline")
    print(SEP)

    run_test("SubgraphBuilder", test_subgraph_builder_vectorized)
    run_test("reveal_one", test_reader_reveal_one)
    run_test("reveal+classify pipeline", test_reader_evaluate)
    run_test("DataLoader", test_reader_dataloader)
    run_test("Speed", test_speed)

    print()
    print(SEP)
    if fail_count == 0:
        print(f"All {test_count} tests passed.")
    else:
        print(f"{fail_count}/{test_count} tests FAILED.")
    print(SEP)

    return fail_count == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

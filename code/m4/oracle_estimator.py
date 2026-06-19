"""Oracle Type II 性能上界估计 + BFS/random 基线对比。

在训练集上，由 gradient / structural-degree teacher 分别选出最优 seed，
从每个 seed 出发运行 4 种 reader（gradient, structural-degree, BFS, random），
预算花光后分类器评估准确率。

产出: oracle_report.json / oracle_accuracy.png / oracle_variance.png

Usage:
  python m4/oracle_estimator.py \
    --pretrained m3/best_model_ft.pt \
    --root D:\\ogb-master\\dataset \
    --device cuda:0 \
    --k-per-class 10 \
    --budgets-rel 0.05 0.10 0.15 \
    --output-dir m4/
"""

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from ogb.graphproppred import DglGraphPropPredDataset

from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.reader import FixedStrategyReader
from active_graph_reader.engine.reveal import run_reveal

from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_stratified(dataset, indices: List[int], k_per_class: int,
                      n_classes: int = 37, seed: int = 42) -> List[int]:
    rng = random.Random(seed)
    class_buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        _, label = dataset[idx]
        lbl = int(label.item()) if hasattr(label, 'item') else int(label)
        class_buckets[lbl].append(idx)

    selected = []
    for lbl in range(n_classes):
        bucket = class_buckets.get(lbl, [])
        if not bucket:
            continue
        n_pick = min(k_per_class, len(bucket))
        selected.extend(rng.sample(bucket, n_pick))
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Full-graph accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_graph(classifier, dataset, indices, device, batch_size=64) -> float:
    correct, total = 0, 0
    for i in tqdm(range(0, len(indices), batch_size), desc="full-graph eval", leave=False):
        batch_indices = indices[i:i + batch_size]
        graphs, labels = [], []
        for idx in batch_indices:
            g, label = dataset[idx]
            graphs.append(g)
            labels.append(int(label.item()) if hasattr(label, 'item') else int(label))
        batch_g = dgl.batch(graphs).to(device)
        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, node_ids)
        preds = logits.argmax(dim=1).cpu().tolist()
        for pred, label in zip(preds, labels):
            total += 1
            if pred == label:
                correct += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Core: Oracle Type II estimation
# ---------------------------------------------------------------------------

def estimate_oracle(
    dataset,
    train_indices: List[int],
    classifier: torch.nn.Module,
    budgets_rel: List[float],
    device: torch.device,
    graph_batch_size: int,
) -> Dict[str, Dict[str, Dict[float, List[float]]]]:
    """Run Oracle Type II estimation.

    Returns: results[seed_source][reader][budget_rel] = [0/1, ...]
    """
    grad_teacher = GradientTeacher()
    struct_teacher = StructuralTeacher(method='degree')
    builder = SubgraphBuilder()

    bfs_reader = FixedStrategyReader('bfs')
    random_reader = FixedStrategyReader('random')

    seed_sources = ['gradient', 'structural-degree']
    reader_names = ['gradient', 'structural-degree', 'bfs', 'random']
    br_sorted = sorted(budgets_rel)

    results: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        src: {rdr: defaultdict(list) for rdr in reader_names}
        for src in seed_sources
    }

    n_total = len(train_indices)

    for batch_start in tqdm(range(0, n_total, graph_batch_size), desc="oracle estimation"):
        batch_indices = train_indices[batch_start:batch_start + graph_batch_size]

        # ---- Step 1: per-graph scores + seeds ----
        batch_data = {}
        for bi, idx in enumerate(batch_indices):
            g_raw, label = dataset[int(idx)]
            if hasattr(label, 'item'):
                label = label.item()
            n_nodes = g_raw.num_nodes()
            budgets_abs = [max(1, int(r * n_nodes)) for r in br_sorted]

            # Precompute edge_key_to_id
            edge_key_to_id = {}
            src_edge = g_raw.edges()[0]
            dst_edge = g_raw.edges()[1]
            for eid in range(g_raw.num_edges()):
                u, v = int(src_edge[eid]), int(dst_edge[eid])
                key = (min(u, v), max(u, v))
                if key not in edge_key_to_id:
                    edge_key_to_id[key] = eid

            grad_scores = grad_teacher.score_nodes(g_raw, classifier, label)
            struct_scores = struct_teacher.score_nodes(g_raw)
            grad_seed = int(grad_scores.argmax().item())
            struct_seed = int(struct_scores.argmax().item())

            batch_data[bi] = {
                'graph': g_raw,
                'label': label,
                'grad_scores': grad_scores,
                'struct_scores': struct_scores,
                'grad_seed': grad_seed,
                'struct_seed': struct_seed,
                'budgets_abs': budgets_abs,
                'edge_key_to_id': edge_key_to_id,
            }

        # ---- Step 2: run all reveals, collect (state, meta) ----
        all_states = []
        all_meta = []  # (batch_i, seed_src, reader, budget_rel, label)

        for bi in range(len(batch_indices)):
            bd = batch_data[bi]
            g_raw = bd['graph']
            label = bd['label']
            budgets_abs = bd['budgets_abs']
            grad_scores = bd['grad_scores']
            struct_scores = bd['struct_scores']
            ek2id = bd['edge_key_to_id']

            for seed_src in seed_sources:
                seed = bd['grad_seed'] if seed_src == 'gradient' else bd['struct_seed']

                for br, abs_b in zip(br_sorted, budgets_abs):
                    # Gradient reader
                    _, st = grad_teacher.generate_trajectory(
                        g_raw, grad_scores, seed, abs_b, edge_key_to_id=ek2id,
                    )
                    all_states.append(st)
                    all_meta.append((bi, seed_src, 'gradient', br, label))

                    # Structural-degree reader
                    _, st = struct_teacher.generate_trajectory(
                        g_raw, struct_scores, seed, abs_b, edge_key_to_id=ek2id,
                    )
                    all_states.append(st)
                    all_meta.append((bi, seed_src, 'structural-degree', br, label))

                    # BFS reader
                    bfs_reader.reset()
                    st = run_reveal(bfs_reader, g_raw, abs_b, seed_node=seed)
                    all_states.append(st)
                    all_meta.append((bi, seed_src, 'bfs', br, label))

                    # Random reader
                    random_reader.reset()
                    st = run_reveal(random_reader, g_raw, abs_b, seed_node=seed)
                    all_states.append(st)
                    all_meta.append((bi, seed_src, 'random', br, label))

        # ---- Step 3: batch build + classify ----
        batched_g, _ = builder.build_batch(all_states)
        batched_g = batched_g.to(device)
        node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long, device=device)

        with torch.no_grad():
            logits = classifier(batched_g, node_ids)
            preds = logits.argmax(dim=1).cpu().tolist()

        # ---- Step 4: aggregate ----
        for flat_i, (_bi, seed_src, rdr, br, label) in enumerate(all_meta):
            correct = 1.0 if preds[flat_i] == label else 0.0
            results[seed_src][rdr][br].append(correct)

        # Cleanup
        del all_states, all_meta, batched_g, batch_data
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting: grouped bar charts
# ---------------------------------------------------------------------------

def plot_oracle_charts(
    results: Dict,
    budgets_rel: List[float],
    full_graph_acc: float,
    output_dir: Path,
):
    """Generate accuracy and variance grouped bar charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    seed_sources = ['gradient', 'structural-degree']
    reader_names = ['gradient', 'structural-degree', 'bfs', 'random']
    reader_labels = {
        'gradient': 'Gradient',
        'structural-degree': 'Struct-Deg',
        'bfs': 'BFS',
        'random': 'Random',
    }
    colors = {
        'gradient': '#2ecc71',
        'structural-degree': '#3498db',
        'bfs': '#f39c12',
        'random': '#e74c3c',
    }
    br_sorted = sorted(budgets_rel)
    n_budgets = len(br_sorted)
    n_readers = len(reader_names)
    bar_width = 0.18
    x = np.arange(n_budgets)

    # ---- Figure 1: Accuracy ----
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig1.suptitle('Oracle Type II — Accuracy (Training Set)', fontsize=14, fontweight='bold')

    for sub_i, seed_src in enumerate(seed_sources):
        ax = axes1[sub_i]
        title = 'Seed: Gradient Best' if seed_src == 'gradient' else 'Seed: Structural-Deg Best'
        ax.set_title(title, fontsize=13)

        for r_i, rdr in enumerate(reader_names):
            means = [np.mean(results[seed_src][rdr].get(br, [0.0])) for br in br_sorted]
            offset = (r_i - n_readers / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, means, bar_width,
                          color=colors[rdr], label=reader_labels[rdr],
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.axhline(y=full_graph_acc, color='black', linestyle='--', linewidth=1.5,
                   label=f'Full-graph ({full_graph_acc:.3f})')
        ax.set_xlabel('Budget (ratio of |V|)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b:.0%}' for b in br_sorted])
        if sub_i == 0:
            ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(full_graph_acc * 1.15, 0.15))

    fig1.tight_layout()
    acc_path = output_dir / 'oracle_accuracy.png'
    fig1.savefig(acc_path, dpi=150)
    plt.close(fig1)
    print(f"Accuracy chart saved to {acc_path}")

    # ---- Figure 2: Standard deviation ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle('Oracle Type II — Standard Deviation (Training Set)', fontsize=14, fontweight='bold')

    for sub_i, seed_src in enumerate(seed_sources):
        ax = axes2[sub_i]
        title = 'Seed: Gradient Best' if seed_src == 'gradient' else 'Seed: Structural-Deg Best'
        ax.set_title(title, fontsize=13)

        for r_i, rdr in enumerate(reader_names):
            stds = []
            for br in br_sorted:
                accs = results[seed_src][rdr].get(br, [0.0])
                stds.append(np.std(accs) if len(accs) > 1 else 0.0)
            offset = (r_i - n_readers / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, stds, bar_width,
                          color=colors[rdr], label=reader_labels[rdr],
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, stds):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.set_xlabel('Budget (ratio of |V|)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b:.0%}' for b in br_sorted])
        if sub_i == 0:
            ax.set_ylabel('Standard Deviation', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    fig2.tight_layout()
    var_path = output_dir / 'oracle_variance.png'
    fig2.savefig(var_path, dpi=150)
    plt.close(fig2)
    print(f"Variance chart saved to {var_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(results, budgets_rel, full_graph_acc, config) -> Dict:
    seed_sources = ['gradient', 'structural-degree']
    reader_names = ['gradient', 'structural-degree', 'bfs', 'random']
    br_sorted = sorted(budgets_rel)

    report_results = {}
    for src in seed_sources:
        report_results[src] = {}
        for rdr in reader_names:
            report_results[src][rdr] = {}
            for br in br_sorted:
                accs = results[src][rdr].get(br, [])
                if accs:
                    report_results[src][rdr][f'{br:.2f}'] = {
                        'mean': round(float(np.mean(accs)), 4),
                        'std': round(float(np.std(accs)), 4),
                        'min': round(float(np.min(accs)), 4),
                        'max': round(float(np.max(accs)), 4),
                        'n': len(accs),
                    }

    headroom = {}
    for src in seed_sources:
        headroom[src] = {}
        for br in br_sorted:
            oracle_mean = np.mean(results[src]['gradient'].get(br, [0]))
            bfs_mean = np.mean(results[src]['bfs'].get(br, [0]))
            rand_mean = np.mean(results[src]['random'].get(br, [0]))
            headroom[src][f'{br:.2f}'] = {
                'oracle_vs_bfs': round(oracle_mean - bfs_mean, 4),
                'oracle_vs_random': round(oracle_mean - rand_mean, 4),
                'bfs_vs_random': round(bfs_mean - rand_mean, 4),
            }

    return {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'full_graph_accuracy': round(full_graph_acc, 4),
        'results': report_results,
        'headroom': headroom,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Oracle Type II — Performance Upper Bound Estimation')
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to frozen classifier checkpoint')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--budgets-rel', type=float, nargs='+',
                   default=[0.05, 0.10, 0.15],
                   help='Budget as fraction of |V|')
    p.add_argument('--k-per-class', type=int, default=10,
                   help='Graphs per class (37 classes × k)')
    p.add_argument('--max-graphs', type=int, default=0,
                   help='Direct graph count limit (overrides --k-per-class)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: script dir)')
    p.add_argument('--graph-batch-size', type=int, default=16,
                   help='Graphs per mini-batch')
    p.add_argument('--full-graph-acc', type=float, default=None,
                   help='Pre-computed full-graph accuracy (skip eval)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.k_per_class <= 0 and args.max_graphs <= 0:
        raise ValueError("--k-per-class or --max-graphs required")

    use_cuda = torch.cuda.is_available() and 'cuda' in args.device
    device = torch.device(args.device if use_cuda else 'cpu')
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Oracle Type II — Performance Upper Bound Estimation")
    print("=" * 60)
    print(f"Budgets (rel): {args.budgets_rel}")
    print(f"Device:        {device}")
    print()

    # ---- 1. Dataset ----
    print("[1/5] Loading OGBG-PPA dataset...")
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split_idx = dataset.get_idx_split()
    train_indices_all = split_idx['train'].tolist()
    test_indices = split_idx['test'].tolist()
    n_classes = int(dataset.num_classes)

    if args.max_graphs > 0:
        sampled = random.sample(train_indices_all, min(args.max_graphs, len(train_indices_all)))
    else:
        sampled = sample_stratified(dataset, train_indices_all, args.k_per_class, n_classes)
    print(f"  Sampled {len(sampled)} training graphs")

    # ---- 2. Classifier ----
    print("\n[2/5] Loading frozen classifier...")
    classifier = GraphClassifier(edge_feat_dim=7, n_classes=n_classes).to(device)
    ckpt = classifier.load_pretrained(args.pretrained, device=str(device))
    classifier.eval()
    # NOTE: 不能设置 requires_grad=False，GradientTeacher 需要反向传播
    print(f"  Loaded from {args.pretrained}  (val_acc={ckpt.get('val_acc', 'N/A')})")

    # ---- 3. Full-graph accuracy ----
    if args.full_graph_acc is not None:
        full_graph_acc = args.full_graph_acc
        print(f"\n[3/5] Full-graph accuracy: {full_graph_acc:.4f} (provided)")
    else:
        print(f"\n[3/5] Full-graph accuracy on test set ({len(test_indices)} graphs)...")
        full_graph_acc = evaluate_full_graph(classifier, dataset, test_indices, device)
        print(f"  Full-graph test-acc = {full_graph_acc:.4f}")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 4. Oracle estimation ----
    print(f"\n[4/5] Oracle Type II estimation ({len(sampled)} graphs × 2 seeds × 4 readers × {len(args.budgets_rel)} budgets)...")
    results = estimate_oracle(
        dataset=dataset,
        train_indices=sampled,
        classifier=classifier,
        budgets_rel=args.budgets_rel,
        device=device,
        graph_batch_size=args.graph_batch_size,
    )

    # Summary table
    print("\n  Results:")
    header = f"  {'Seed Source':<20} {'Reader':<20}"
    for br in sorted(args.budgets_rel):
        header += f"  B={br:.0%}   "
    print(header)
    print("  " + "-" * len(header))
    for src in ['gradient', 'structural-degree']:
        for rdr in ['gradient', 'structural-degree', 'bfs', 'random']:
            row = f"  {src:<20} {rdr:<20}"
            for br in sorted(args.budgets_rel):
                accs = results[src][rdr].get(br, [])
                m = np.mean(accs) if accs else 0.0
                row += f"  {m:.4f}"
            print(row)

    # Headroom
    print("\n  Headroom (oracle — baseline):")
    for src in ['gradient', 'structural-degree']:
        print(f"    Seed: {src}")
        for br in sorted(args.budgets_rel):
            o = np.mean(results[src]['gradient'].get(br, [0]))
            bf = np.mean(results[src]['bfs'].get(br, [0]))
            rn = np.mean(results[src]['random'].get(br, [0]))
            print(f"      B={br:.0%}: oracle={o:.4f}, bfs={bf:.4f} "
                  f"(Δ={o-bf:+.4f}), random={rn:.4f} (Δ={o-rn:+.4f})")

    # ---- 5. Report + plots ----
    print(f"\n[5/5] Saving report and charts...")
    config = {
        'pretrained': args.pretrained,
        'k_per_class': args.k_per_class if args.max_graphs <= 0 else None,
        'max_graphs': args.max_graphs if args.max_graphs > 0 else None,
        'n_graphs': len(sampled),
        'budgets_rel': args.budgets_rel,
        'seed_sources': ['gradient', 'structural-degree'],
        'readers': ['gradient', 'structural-degree', 'bfs', 'random'],
    }
    report = build_report(results, args.budgets_rel, full_graph_acc, config)

    report_path = output_dir / 'oracle_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report:   {report_path}")

    plot_oracle_charts(results, args.budgets_rel, full_graph_acc, output_dir)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()

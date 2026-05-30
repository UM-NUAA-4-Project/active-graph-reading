"""M3 Benchmark: multi-seed strategy evaluation + budget-accuracy curves.

Completes the three remaining M3 roadmap items:
  1. Average each policy over multiple random seeds per graph
  2. Plot budget-performance curves vs. full-graph upper bound
  3. Kill-criterion check: BFS @ ≤10% reveal vs 90% full-graph accuracy

Usage:                                                                                                    
  # 正式 quick benchmark：图、seed、7budget + 预训练模型                                                                                                            
  python m3/benchmark_m3.py --k-per-class []  --seeds-per-graph []  --pretrained [] --budgets-rel 0.01 0.02 0.05 0.1 0.2 0.5--device cuda:0                                                                                                                                                                                                                                                              
"""

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from ogb.graphproppred import DglGraphPropPredDataset

from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.reader import FixedStrategyReader, evaluate

# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_stratified(dataset, indices: List[int], k_per_class: int,
                      n_classes: int = 37, seed: int = 42) -> List[int]:
    """从每类随机选 k 张图，返回选中的图索引列表。

    Parameters
    ----------
    dataset : DglGraphPropPredDataset
    indices : 候选图索引列表（如验证集全部索引）。
    k_per_class : 每类选取数量。
    n_classes : 类别总数。
    seed : 随机种子（保证可复现）。

    Returns
    -------
    List[int]
        选中的图索引，按类别分组后打乱。
    """
    rng = random.Random(seed)
    class_buckets: Dict[int, List[int]] = defaultdict(list)

    for idx in indices:
        _, label = dataset[idx]
        lbl = int(label.item()) if hasattr(label, 'item') else int(label)
        class_buckets[lbl].append(idx)

    selected = []
    for lbl in range(n_classes):
        bucket = class_buckets.get(lbl, [])
        if len(bucket) == 0:
            continue
        n_pick = min(k_per_class, len(bucket))
        selected.extend(rng.sample(bucket, n_pick))

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='M3 Benchmark: strategy evaluation')
    p.add_argument('--pretrained', type=str, default=None,
                   help='Path to best_model.pt (required for accuracy; skip with --quick for protocol check)')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='Device (cuda:0)')
    p.add_argument('--strategies', type=str, nargs='+',
                   default=['random', 'bfs', 'max-frontier-degree'])
    p.add_argument('--budgets-rel', type=float, nargs='+',
                   default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
                   help='Budget as fraction of |V|')
    p.add_argument('--seeds-per-graph', type=int, default=5,
                   help='Random seeds per graph for averaging')
    p.add_argument('--k-per-class', type=int, default=10,
                   help='从每个类别随机选 k 张验证图（37 类 × k = 总图数）')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: script dir)')
    # Quick / smoke modes
    p.add_argument('--quick', action='store_true')
    p.add_argument('--protocol-only', action='store_true',
                   help='Verify protocol runs without accuracy evaluation'
                        ' (no classifier needed)')
    # Full-graph upper bound (optional: pre-computed)
    p.add_argument('--full-graph-acc', type=float, default=None,
                   help='Full-graph accuracy (if known). Auto-evaluated if omitted.')
    # Batch sizes
    p.add_argument('--eval-batch-size', type=int, default=64,
                   help='Graphs per batch in full-graph evaluation')
    p.add_argument('--graph-batch-size', type=int, default=8,
                   help='Graphs per batch in strategy evaluation')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Full-graph accuracy (simple, no OGB Evaluator, no self-loop dependency)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_graph(
    classifier: GraphClassifier,
    dataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """在完整图上评估分类准确率（批处理）。"""
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
            if pred == label:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Single strategy evaluation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Kill-criterion check
# ---------------------------------------------------------------------------

def check_kill_criterion(
    bfs_results: Dict[float, List[float]],
    full_graph_acc: float,
    threshold_rel_budget: float = 0.10,
    threshold_acc_ratio: float = 0.90,
) -> dict:
    """Check if BFS reaches ≥90% of full-graph accuracy at ≤10% reveal.

    Returns verdict dict.
    """
    # Find BFS accuracy at (or below) the threshold budget
    relevant = [b for b in bfs_results.keys() if b <= threshold_rel_budget + 1e-6]
    if not relevant:
        return {
            'triggered': False,
            'verdict': 'INCONCLUSIVE',
            'reason': f'No BFS data at ≤{threshold_rel_budget:.0%} budget',
        }

    # Use the largest budget ≤ threshold
    check_budget = max(relevant)
    bfs_mean = np.mean(bfs_results[check_budget])
    ratio = bfs_mean / max(full_graph_acc, 1e-6)

    triggered = ratio >= threshold_acc_ratio
    return {
        'triggered': bool(triggered),
        'verdict': 'KILL' if triggered else 'PASS',
        'bfs_accuracy': round(float(bfs_mean), 4),
        'full_graph_accuracy': round(full_graph_acc, 4),
        'bfs_reveal_ratio': check_budget,
        'accuracy_ratio': round(float(ratio), 4),
        'threshold_ratio': threshold_acc_ratio,
        'detail': (
            f'BFS@{check_budget:.0%} reveal: {bfs_mean:.4f} vs '
            f'full-graph: {full_graph_acc:.4f} '
            f'({ratio:.1%} of upper bound)'
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(
    results: Dict[str, Dict[float, List[float]]],
    full_graph_acc: float,
    output_path: str,
):
    """Plot budget-accuracy curves for all strategies."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'random': '#e74c3c', 'bfs': '#2ecc71',
              'max-frontier-degree': '#3498db'}
    markers = {'random': 'o', 'bfs': 's', 'max-frontier-degree': '^'}

    for strategy_name, strat_results in results.items():
        budgets = sorted(strat_results.keys())
        means = [np.mean(strat_results[b]) for b in budgets]
        stds = [np.std(strat_results[b]) for b in budgets]

        color = colors.get(strategy_name, '#95a5a6')
        marker = markers.get(strategy_name, 'x')
        label = {'random': 'Random', 'bfs': 'BFS',
                 'max-frontier-degree': 'Max Frontier Degree'}.get(strategy_name, strategy_name)

        ax.errorbar(
            [b * 100 for b in budgets], means, yerr=stds,
            color=color, marker=marker, label=label,
            capsize=4, linewidth=2, markersize=8,
        )

    # Full-graph upper bound (horizontal line)
    ax.axhline(y=full_graph_acc, color='black', linestyle='--', linewidth=1.5,
               label=f'Full-graph upper bound ({full_graph_acc:.3f})')

    # Kill-criterion zone
    ax.axvspan(0, 10, alpha=0.08, color='orange', label='Kill zone (≤10% reveal)')

    ax.set_xlabel('Budget (% of |V|)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('M3 Benchmark: Budget-Accuracy Curves (OGBG-PPA)', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(max(b * 100 for b in results.get('random', {}).keys()), 50) + 2)
    ax.set_ylim(0, max(full_graph_acc * 1.1, 0.7))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.quick:
        args.k_per_class = 2
        args.seeds_per_graph = 20
        args.budgets_rel = [0.05, 0.10, 0.20, 0.25,0.30,0.40,0.50]

    if args.protocol_only:
        args.pretrained = None

    has_model = args.pretrained is not None

    device = torch.device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("M3 Benchmark Evaluation")
    print("=" * 60)
    print(f"Strategies: {args.strategies}")
    print(f"Budgets (rel): {args.budgets_rel}")
    print(f"Seeds/graph: {args.seeds_per_graph}")
    print(f"k-per-class: {args.k_per_class}")
    print()

    print("[1/5] Loading dataset...")
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split_idx = dataset.get_idx_split()
    n_classes = int(dataset.num_classes)

    valid_indices = sample_stratified(
        dataset, split_idx['valid'].tolist(), args.k_per_class, n_classes)
    full_valid_indices = split_idx['valid'].tolist()
    test_indices = split_idx['test'].tolist()

    edge_feat_dim = 7
    print(f"  Strategy eval on {len(valid_indices)} validation graphs "
          f"(k={args.k_per_class} per class)")
    print(f"  Full-graph eval on {len(test_indices)} test graphs")

    # ------------------------------------------------------------------
    # 2. Load or skip classifier
    # ------------------------------------------------------------------
    classifier = None
    if has_model:
        print("\n[2/5] Loading frozen classifier...")
        classifier = GraphClassifier(
            edge_feat_dim=edge_feat_dim, n_classes=n_classes
        ).to(device)
        ckpt = classifier.load_pretrained(args.pretrained, device=str(device))
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False
        print(f"  Loaded from {args.pretrained}")
        print(f"  Pretrained val_acc = {ckpt.get('val_acc', 'N/A')}")
    else:
        print("\n[2/5] No pretrained model (--quick without --pretrained).")
        print("  Running protocol verification only — accuracy numbers "
              "are placeholder (random).")
        classifier = GraphClassifier(
            edge_feat_dim=edge_feat_dim, n_classes=n_classes
        ).to(device)
        classifier.eval()

    # ------------------------------------------------------------------
    # 3. Full-graph upper bound (only with pretrained model)
    #    Phase A: full test set → test-acc
    #    Phase B: release memory → sleep 10s → full validation set → valid-acc
    # ------------------------------------------------------------------
    full_graph_test_acc = None
    full_graph_valid_acc = None
    if has_model:
        print("\n[3/5] Full-graph upper bound...")
        if args.full_graph_acc is not None:
            full_graph_test_acc = args.full_graph_acc
            full_graph_valid_acc = args.full_graph_acc
            print(f"  Using provided value: {full_graph_test_acc:.4f}")
        else:
            # --- Phase A: Test set ---
            print(f"  [3a] Evaluating on full test set "
                  f"({len(test_indices)} graphs)...")
            full_graph_test_acc = evaluate_full_graph(
                classifier, dataset, test_indices, device,
                batch_size=args.eval_batch_size,
            )
            print(f"  Test-acc  = {full_graph_test_acc:.4f}")

            # --- Release memory ---
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Memory released, waiting 10s before validation eval...")
            time.sleep(10)

            # --- Phase B: Validation set ---
            print(f"  [3b] Evaluating on full validation set "
                  f"({len(full_valid_indices)} graphs)...")
            full_graph_valid_acc = evaluate_full_graph(
                classifier, dataset, full_valid_indices, device,
                batch_size=args.eval_batch_size,
            )
            print(f"  Valid-acc = {full_graph_valid_acc:.4f}")
    else:
        print("\n[3/5] Full-graph upper bound: SKIPPED (no pretrained model)")
        print(f"  BLOCKER: train full-graph GIN first — "
              f"python m3/train_full_graph.py --device cuda:0")

    # backward-compat alias: full_graph_acc → test-acc
    full_graph_acc = full_graph_test_acc

    # ------------------------------------------------------------------
    # Step 3 后完整清理：释放 full-graph eval 期间占用的所有内存
    # ------------------------------------------------------------------
    del test_indices, full_valid_indices
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("\n  Step 3 cleanup complete — test/valid indices released, "
          "CUDA cache cleared.")

    # ------------------------------------------------------------------
    # 4. Multi-seed strategy evaluation (protocol + accuracy)
    # ------------------------------------------------------------------
    print("\n[4/5] Running protocol + strategy evaluation...")
    all_results: Dict[str, Dict[float, List[float]]] = {}

    for strategy_name in args.strategies:
        print(f"\n  Strategy: {strategy_name}")
        reader = FixedStrategyReader(strategy_name)
        results = evaluate(
            reader=reader,
            dataset=dataset,
            indices=valid_indices,
            classifier=classifier,
            budgets_rel=args.budgets_rel,
            seeds_per_graph=args.seeds_per_graph,
            device=device,
            batch_size=args.graph_batch_size,
        )
        all_results[strategy_name] = results

        # Print summary
        for b in sorted(results.keys()):
            accs = results[b]
            if has_model:
                print(f"    Budget {b:5.0%}: mean={np.mean(accs):.4f}  "
                      f"std={np.std(accs):.4f}  "
                      f"min={np.min(accs):.4f}  max={np.max(accs):.4f}")
            else:
                print(f"    Budget {b:5.0%}: mean={np.mean(accs):.4f}  "
                      f"(placeholder — no pretrained model)")

    # ------------------------------------------------------------------
    # 5. Kill-criterion + Plot + Report
    # ------------------------------------------------------------------
    if has_model and full_graph_acc is not None:
        print("\n[5/5] Kill-criterion check...")
        kill_result = None
        if 'bfs' in all_results:
            kill_result = check_kill_criterion(all_results['bfs'], full_graph_acc)
            print(f"  Verdict: {kill_result['verdict']}")
            print(f"  {kill_result['detail']}")
            if kill_result['triggered']:
                print("  Dataset may be too easy — consider switching or hardening G_0")
        else:
            print("  BFS not in strategies — skipping kill-criterion check")

        # Plot
        plot_path = str(output_dir / 'm3_benchmark_curves.png')
        plot_curves(all_results, full_graph_acc, plot_path)
        plot_msg = f"\n  Plot:   {plot_path}"
    else:
        print("\n[5/5] Kill-criterion + Plot: SKIPPED (no pretrained model)")
        kill_result = None
        plot_msg = ""

    # Report (save regardless of model availability)
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'pretrained': args.pretrained,
            'k_per_class': args.k_per_class,
            'n_graphs': len(valid_indices),
            'seeds_per_graph': args.seeds_per_graph,
            'budgets_rel': args.budgets_rel,
            'strategies': args.strategies,
        },
        'has_model': has_model,
        'full_graph_test_accuracy': round(full_graph_test_acc, 4) if full_graph_test_acc is not None else None,
        'full_graph_valid_accuracy': round(full_graph_valid_acc, 4) if full_graph_valid_acc is not None else None,
        'full_graph_accuracy': round(full_graph_test_acc, 4) if full_graph_test_acc is not None else None,  # backward-compat
        'results': {},
        'kill_criterion': kill_result,
    }

    for strategy_name, strat_results in all_results.items():
        report['results'][strategy_name] = {}
        for b in sorted(strat_results.keys()):
            accs = strat_results[b]
            report['results'][strategy_name][f'{b:.2f}'] = {
                'mean': round(float(np.mean(accs)), 4),
                'std': round(float(np.std(accs)), 4),
                'min': round(float(np.min(accs)), 4),
                'max': round(float(np.max(accs)), 4),
            }

    report_path = output_dir / 'm3_benchmark_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Summary table
    print("\n" + "=" * 60)
    print("M3 Benchmark Summary")
    print("=" * 60)
    if has_model:
        print(f"  Full-graph test-acc:  {full_graph_test_acc:.4f}")
        print(f"  Full-graph valid-acc: {full_graph_valid_acc:.4f}")
    else:
        print(f"  Protocol verification only (no pretrained model)")
    print(f"\n  {'Strategy':<22}", end='')
    for b in args.budgets_rel:
        print(f"  B={b:.0%}  ", end='')
    print()
    print("  " + "-" * (22 + 12 * len(args.budgets_rel)))
    for strategy_name in args.strategies:
        print(f"  {strategy_name:<22}", end='')
        for b in args.budgets_rel:
            if b in all_results.get(strategy_name, {}):
                m = np.mean(all_results[strategy_name][b])
                print(f"  {m:.4f}", end='')
            else:
                print(f"    N/A  ", end='')
        print()

    if kill_result:
        print(f"\n  Kill criterion: {kill_result['verdict']}")
        print(f"  {kill_result['detail']}")

    print(f"\n  Report: {report_path}" + plot_msg)
    print("=" * 60)


if __name__ == '__main__':
    main()

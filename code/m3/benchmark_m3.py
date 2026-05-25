"""M3 Benchmark: multi-seed strategy evaluation + budget-accuracy curves.

Completes the three remaining M3 roadmap items:
  1. Average each policy over multiple random seeds per graph
  2. Plot budget-performance curves vs. full-graph upper bound
  3. Kill-criterion check: BFS @ ≤10% reveal vs 90% full-graph accuracy

Usage:
  python m3/benchmark_m3.py --pretrained best_model.pt --root /mnt/d/ogb-master/dataset
  python m3/benchmark_m3.py --pretrained best_model.pt --quick
"""

import argparse
import json
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

from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_loop import RevealLoop
from active_graph_reader.strategies.registry import get_strategy
from active_graph_reader.cost.registry import get_cost_model

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='M3 Benchmark: strategy evaluation')
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to best_model.pt')
    p.add_argument('--root', type=str, default='dataset',
                   help='OGB dataset root')
    p.add_argument('--device', type=str, default='cpu',
                   help='Device (cpu / cuda:0)')
    p.add_argument('--strategies', type=str, nargs='+',
                   default=['random', 'bfs', 'max-frontier-degree'])
    p.add_argument('--budgets-rel', type=float, nargs='+',
                   default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
                   help='Budget as fraction of |V|')
    p.add_argument('--seeds-per-graph', type=int, default=5,
                   help='Random seeds per graph for averaging')
    p.add_argument('--n-graphs', type=int, default=500,
                   help='Number of validation graphs to evaluate')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: script dir)')
    # Quick mode
    p.add_argument('--quick', action='store_true',
                   help='Quick: 100 graphs, 3 seeds, budgets=[0.05,0.10,0.20]')
    # Full-graph upper bound (optional: pre-computed)
    p.add_argument('--full-graph-acc', type=float, default=None,
                   help='Full-graph accuracy (if known). Auto-evaluated if omitted.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Full-graph upper bound
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_graph(
    classifier: GraphClassifier,
    dataset,
    indices: List[int],
    device: torch.device,
) -> float:
    """Evaluate classifier accuracy on full graphs (upper bound)."""
    evaluator = Evaluator('ogbg-ppa')

    y_true, y_pred = [], []
    for idx in tqdm(indices, desc="full-graph eval", leave=False):
        g, label = dataset[idx]
        if hasattr(label, 'item'):
            label = label.item()
        g = dgl.add_self_loop(g)
        node_ids = torch.zeros(g.num_nodes(), dtype=torch.long)

        logits = classifier(g.to(device), node_ids.to(device))
        y_true.append(label)
        y_pred.append(int(logits.argmax(dim=1).cpu().item()))

    y_true_np = np.array(y_true).reshape(-1, 1)
    y_pred_np = np.array(y_pred).reshape(-1, 1)
    return float(evaluator.eval({'y_true': y_true_np, 'y_pred': y_pred_np})['acc'])


# ---------------------------------------------------------------------------
# Single strategy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_strategy(
    strategy_name: str,
    classifier: GraphClassifier,
    dataset,
    indices: List[int],
    budgets_rel: List[float],
    seeds_per_graph: int,
    subgraph_builder: SubgraphBuilder,
    device: torch.device,
) -> Dict[float, List[float]]:
    """Run one strategy with multi-seed × multi-budget evaluation.

    Returns {budget_rel: [per-graph-mean accuracies]}, one entry per graph.
    """
    strategy = get_strategy(strategy_name)
    cost_model = get_cost_model('unit')

    # budget_rel → list of per-graph mean accuracies
    results: Dict[float, List[float]] = defaultdict(list)

    for idx in tqdm(indices, desc=strategy_name, leave=False):
        g_raw, label = dataset[idx]
        if hasattr(label, 'item'):
            label = label.item()
        g = dgl.add_self_loop(g_raw)
        n_nodes = g.num_nodes()

        for rel_budget in budgets_rel:
            budget = max(1, int(rel_budget * n_nodes))
            seed_correct = []

            for seed_i in range(seeds_per_graph):
                seed_node = (idx * 31 + seed_i * 17 + 7) % max(n_nodes, 1)

                state = GraphState(g, seed_node=seed_node)
                loop = RevealLoop(state, strategy, cost_model)
                loop.run(budget=budget)

                sub_g, orig_ids = subgraph_builder.build(state)
                node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long)
                logits = classifier(sub_g.to(device), node_ids.to(device))
                pred = int(logits.argmax(dim=1).cpu().item())
                seed_correct.append(1.0 if pred == label else 0.0)

            # Per-graph mean accuracy (averaged over seeds)
            results[rel_budget].append(np.mean(seed_correct))

    return dict(results)


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
        'triggered': triggered,
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
        args.n_graphs = 100
        args.seeds_per_graph = 3
        args.budgets_rel = [0.05, 0.10, 0.20]

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
    print(f"Graphs: {args.n_graphs}")
    print()

    print("[1/5] Loading dataset...")
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split_idx = dataset.get_idx_split()
    n_classes = int(dataset.num_classes)
    edge_feat_dim = 7
    valid_indices = split_idx['valid'].tolist()[:args.n_graphs]
    print(f"  Evaluating on {len(valid_indices)} validation graphs")

    # ------------------------------------------------------------------
    # 2. Load classifier
    # ------------------------------------------------------------------
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

    subgraph_builder = SubgraphBuilder()

    # ------------------------------------------------------------------
    # 3. Full-graph upper bound
    # ------------------------------------------------------------------
    print("\n[3/5] Full-graph upper bound...")
    if args.full_graph_acc is not None:
        full_graph_acc = args.full_graph_acc
        print(f"  Using provided value: {full_graph_acc:.4f}")
    else:
        full_graph_acc = evaluate_full_graph(
            classifier, dataset, valid_indices, device
        )
        print(f"  Evaluated: {full_graph_acc:.4f}")

    # ------------------------------------------------------------------
    # 4. Multi-seed strategy evaluation
    # ------------------------------------------------------------------
    print("\n[4/5] Multi-seed strategy evaluation...")
    all_results: Dict[str, Dict[float, List[float]]] = {}

    for strategy_name in args.strategies:
        print(f"\n  Strategy: {strategy_name}")
        results = evaluate_strategy(
            strategy_name=strategy_name,
            classifier=classifier,
            dataset=dataset,
            indices=valid_indices,
            budgets_rel=args.budgets_rel,
            seeds_per_graph=args.seeds_per_graph,
            subgraph_builder=subgraph_builder,
            device=device,
        )
        all_results[strategy_name] = results

        # Print summary
        for b in sorted(results.keys()):
            accs = results[b]
            print(f"    Budget {b:5.0%}: mean={np.mean(accs):.4f}  "
                  f"std={np.std(accs):.4f}  "
                  f"min={np.min(accs):.4f}  max={np.max(accs):.4f}")

    # ------------------------------------------------------------------
    # 5. Kill-criterion + Plot + Report
    # ------------------------------------------------------------------
    print("\n[5/5] Kill-criterion check...")

    kill_result = None
    if 'bfs' in all_results:
        kill_result = check_kill_criterion(all_results['bfs'], full_graph_acc)
        print(f"  Verdict: {kill_result['verdict']}")
        print(f"  {kill_result['detail']}")
        if kill_result['triggered']:
            print("  ⚠ Dataset may be too easy — consider switching or hardening G_0")
    else:
        print("  BFS not in strategies — skipping kill-criterion check")

    # Plot
    plot_path = str(output_dir / 'm3_benchmark_curves.png')
    plot_curves(all_results, full_graph_acc, plot_path)

    # Report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'pretrained': args.pretrained,
            'n_graphs': args.n_graphs,
            'seeds_per_graph': args.seeds_per_graph,
            'budgets_rel': args.budgets_rel,
            'strategies': args.strategies,
        },
        'full_graph_accuracy': round(full_graph_acc, 4),
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
    print(f"  Full-graph accuracy: {full_graph_acc:.4f}")
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

    print(f"\n  Report: {report_path}")
    print(f"  Plot:   {plot_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

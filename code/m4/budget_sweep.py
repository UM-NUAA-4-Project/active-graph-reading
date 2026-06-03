"""Sweep budget_ratio values and measure retry behavior.

For each budget_ratio, runs the guaranteed pipeline (generate_all_phased)
and records how many retry rounds each graph needed to reach k correct
trajectories. Produces a line chart (budget vs mean retries) + JSON data.

Usage:
  python -m m4.budget_sweep --pretrained best_model.pt --device cuda:0
  python -m m4.budget_sweep --pretrained best_model.pt --device cuda:0 \
      --graphs-file smoke_test_graphs.pt --budget-min 0.05 --budget-max 0.25
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from active_graph_reader.dataloader.loader import PPADataset
from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder

from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.teachers.info_gain_teacher import InfoGainTeacher
from m4.trajectory_generator import TrajectoryGenerator


def parse_args():
    p = argparse.ArgumentParser(description='Budget sweep — retry rounds vs budget_ratio')
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to best_model.pt')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root')
    p.add_argument('--graphs-file', type=str, default=None,
                   help='Path to sampled graphs .pt (default: smoke_test_graphs.pt in script dir)')

    # Sweep range
    p.add_argument('--budget-min', type=float, default=0.05,
                   help='Min budget_ratio (default 0.05)')
    p.add_argument('--budget-max', type=float, default=0.25,
                   help='Max budget_ratio (default 0.25)')
    p.add_argument('--budget-step', type=float, default=0.025,
                   help='Step size (default 0.025)')

    # Pipeline params
    p.add_argument('--seeds-per-graph', type=int, default=5)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--max-retries', type=int, default=5)
    p.add_argument('--seeds-increment', type=int, default=3)
    p.add_argument('--skip-t2', action='store_true',
                   help='Skip T2 (info-gain) teacher')

    # Output
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: script dir)')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip plotting, only save JSON')
    return p.parse_args()


def main():
    args = parse_args()

    graphs_file = (
        Path(args.graphs_file) if args.graphs_file
        else Path(__file__).parent / 'smoke_test_graphs.pt'
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading sampled graphs from {graphs_file}...")
    samples = torch.load(str(graphs_file), map_location='cpu', weights_only=False)
    flat_indices = []
    for entries in samples.values():
        for e in entries:
            flat_indices.append(e['graph_idx'])
    n_graphs = len(flat_indices)
    print(f"Loaded {n_graphs} graph indices")

    print(f"Loading OGBG-PPA from {args.root}...")
    dataset = PPADataset(root=args.root)

    # ------------------------------------------------------------------
    # 2. Load classifier
    # ------------------------------------------------------------------
    device = torch.device(args.device)
    print(f"Loading classifier from {args.pretrained}...")
    classifier = GraphClassifier(edge_feat_dim=7, n_classes=37).to(device)
    classifier.load_pretrained(args.pretrained, device=str(device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # 3. Teachers
    # ------------------------------------------------------------------
    teachers = [
        GradientTeacher(aggregate='mean'),
        StructuralTeacher(method='degree'),
        StructuralTeacher(method='pagerank'),
        HybridTeacher(alpha=0.3),
        HybridTeacher(alpha=0.7),
    ]
    if not args.skip_t2:
        teachers.append(InfoGainTeacher(temperature=1.0))
    print(f"Teachers: {[t.name for t in teachers]}")

    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=SubgraphBuilder(),
        dataset=dataset,
        device=str(device),
    )

    # Pre-compute GraphMeta once (shared across all budgets)
    print("\nPre-computing GraphMeta (shared across all budgets)...")
    from m4.graph_meta import GraphMetaCache
    cache = GraphMetaCache(dataset, classifier, str(device))

    teacher_names = {t.name for t in teachers}
    need_t1 = 'gradient' in teacher_names or 'hybrid' in teacher_names
    need_t3_degree = 'structural-degree' in teacher_names or 'hybrid' in teacher_names
    need_t3_pagerank = 'structural-pagerank' in teacher_names

    metas = cache.precompute(
        flat_indices,
        skip_t1=not need_t1,
        skip_t3_degree=not need_t3_degree,
        skip_t3_pagerank=not need_t3_pagerank,
    )
    print(f"Pre-computed {len(metas)} GraphMeta entries")

    # ------------------------------------------------------------------
    # 4. Sweep budgets
    # ------------------------------------------------------------------
    budget_values = []
    b = args.budget_min
    while b <= args.budget_max + 1e-9:
        budget_values.append(round(b, 4))
        b += args.budget_step

    print(f"\nSweeping {len(budget_values)} budgets: {budget_values}")
    print("=" * 60)

    sweep_results = {}  # budget_ratio -> stats

    min_correct = args.top_k

    for budget_ratio in budget_values:
        print(f"\n--- budget_ratio={budget_ratio} ---")

        # Reset generator's cache to use pre-computed metas
        generator.classifier = classifier  # ensure classifier is fresh

        filtered = generator.generate_all_phased(
            indices=flat_indices,
            budget_ratio=budget_ratio,
            seeds_per_graph=args.seeds_per_graph,
            top_k=args.top_k,
            min_correct_per_graph=min_correct,
            max_retries=args.max_retries,
            seeds_increment=args.seeds_increment,
            skip_t2=args.skip_t2,
            chunk_size=len(flat_indices),
            precomputed_metas=metas,
        )

        retry_stats = getattr(generator, '_last_retry_stats', {})
        n_retried = len(retry_stats)
        mean_retries = sum(retry_stats.values()) / max(n_retried, 1)

        # Per-graph correct counts
        correct_per_graph = defaultdict(int)
        for t in filtered:
            if t.is_correct:
                correct_per_graph[t.graph_idx] += 1

        n_fully_correct = sum(
            1 for idx in flat_indices
            if correct_per_graph.get(idx, 0) >= min_correct
        )

        result = {
            'budget_ratio': budget_ratio,
            'n_graphs': n_graphs,
            'n_fully_correct': n_fully_correct,
            'pct_fully_correct': round(100.0 * n_fully_correct / n_graphs, 1),
            'n_retried': n_retried,
            'pct_retried': round(100.0 * n_retried / n_graphs, 1),
            'mean_retries': round(mean_retries, 2),
            'n_trajectories': len(filtered),
            'n_correct_trajectories': sum(1 for t in filtered if t.is_correct),
        }
        sweep_results[str(budget_ratio)] = result

        print(f"  Fully correct (≥{min_correct}): {n_fully_correct}/{n_graphs} "
              f"({result['pct_fully_correct']}%)")
        print(f"  Needed retries: {n_retried}/{n_graphs} "
              f"({result['pct_retried']}%), mean retries: {mean_retries:.2f}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_path = output_dir / 'budget_sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    if not args.no_plot:
        try:
            _plot_sweep(sweep_results, output_dir, args.top_k)
        except Exception as e:
            print(f"Plotting failed: {e}")
            print("Install matplotlib to enable plotting: pip install matplotlib")


def _plot_sweep(sweep_results: dict, output_dir: Path, top_k: int):
    """Generate line chart: budget_ratio vs retry metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budgets = []
    pct_retried = []
    mean_retries = []

    for k in sorted(sweep_results.keys(), key=float):
        r = sweep_results[k]
        budgets.append(r['budget_ratio'] * 100)  # convert to %
        pct_retried.append(r['pct_retried'])
        mean_retries.append(r['mean_retries'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: % graphs needing retries
    color1 = '#e74c3c'
    ax1.plot(budgets, pct_retried, 'o-', color=color1, linewidth=2, markersize=6,
             label='% graphs retried')
    ax1.fill_between(budgets, 0, pct_retried, alpha=0.1, color=color1)
    ax1.set_xlabel('Budget ratio (% of |V|)')
    ax1.set_ylabel('% graphs needing retries')
    ax1.set_title(f'Graphs needing retries to reach ≥{top_k} correct\n'
                  f'(lower is better)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(pct_retried) * 1.1 + 1)

    # Right: mean retry rounds
    color2 = '#2980b9'
    ax2.plot(budgets, mean_retries, 's-', color=color2, linewidth=2, markersize=6,
             label='mean retries')
    ax2.fill_between(budgets, 0, mean_retries, alpha=0.1, color=color2)
    ax2.set_xlabel('Budget ratio (% of |V|)')
    ax2.set_ylabel('Mean retry rounds (deficient graphs only)')
    ax2.set_title(f'Average retry rounds per deficient graph\n'
                  f'(lower is better)')
    ax2.grid(True, alpha=0.3)
    if max(mean_retries) > 0:
        ax2.set_ylim(0, max(mean_retries) * 1.1 + 0.5)

    plt.tight_layout()
    plot_path = output_dir / 'budget_sweep_plot.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()

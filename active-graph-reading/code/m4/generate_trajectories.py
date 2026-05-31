"""M4 main script: generate, score, filter, and analyze teacher trajectories.

End-to-end pipeline:
  1. Load frozen classifier (clean full-graph GIN, no adapters)
  2. Initialize teachers (T1 gradient, T2 info-gain, T3-degree, T3-pagerank)
  3. Generate trajectories on train/valid sets
  4. Score and filter
  5. Oracle upper-bound estimation
  6. Feasibility analysis → M5 route recommendation

Usage:
  python -m m4.generate_trajectories --pretrained best_model.pt --device cuda:0
  python -m m4.generate_trajectories --pretrained best_model.pt --quick --device cpu
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import dgl
import numpy as np
import torch

# Ensure myworkplace/code/ is on sys.path
_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from active_graph_reader.dataloader.loader import PPADataset
from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_loop import RevealLoop
from active_graph_reader.strategies.registry import get_strategy
from active_graph_reader.cost.registry import get_cost_model

from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.teachers.info_gain_teacher import InfoGainTeacher
from m4.trajectory_generator import TrajectoryGenerator
from m4.trajectory_scorer import TrajectoryScorer
from m4.trajectory_filter import TrajectoryFilter
from m4.oracle_estimator import OracleEstimator
from m4.feasibility_analyzer import FeasibilityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='M4: Teacher trajectory generation and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to best_model.pt from full-graph training')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root directory')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='Device for computation')

    # Data scope
    p.add_argument('--train-graphs', type=int, default=0,
                   help='Number of training graphs (0 = skip train, for quick mode)')
    p.add_argument('--valid-graphs', type=int, default=500,
                   help='Number of validation graphs')
    p.add_argument('--test-graphs', type=int, default=0,
                   help='Number of test graphs (0 = skip)')

    # Generation params
    p.add_argument('--budgets', type=int, nargs='+', default=[5, 10, 15, 20],
                   help='Budget values (legacy mode; use --budget-ratio for phased mode)')
    p.add_argument('--budget-ratio', type=float, default=None,
                   help='Budget as fraction of |V| (enables phased pipeline, e.g. 0.15)')
    p.add_argument('--seeds-per-graph', type=int, default=5,
                   help='Random seeds per graph')
    p.add_argument('--skip-t2', action='store_true',
                   help='Skip T2 (info-gain) teacher — expensive')
    p.add_argument('--top-k', type=int, default=3,
                   help='Max trajectories per graph after filtering')
    p.add_argument('--chunk-size', type=int, default=500,
                   help='Graphs per chunk for GPU batch eval (phased mode only)')

    # Output
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: m4_output/)')
    p.add_argument('--save-interval', type=int, default=500,
                   help='Save checkpoint every N trajectories')
    p.add_argument('--cache-dir', type=str, default=None,
                   help='Directory for GraphMeta cache persistence')

    # Modes
    p.add_argument('--quick', action='store_true',
                   help='Quick mode: 200 valid graphs, budgets=[5,10], 1 seed, skip T2')
    p.add_argument('--phased', action='store_true', default=None,
                   help='Use optimized phased pipeline (auto-enabled with --budget-ratio)')
    p.add_argument('--seed', type=int, default=42,
                   help='Global random seed')

    return p.parse_args()


# ============================================================================
# Baseline accuracy computation
# ============================================================================

@torch.no_grad()
def compute_baseline_acc(
    strategy_name: str,
    dataset: PPADataset,
    indices: List[int],
    classifier,
    subgraph_builder: SubgraphBuilder,
    budgets: List[int],
    seeds_per_graph: int = 3,
    seed_offset: int = 0,
) -> Dict[int, float]:
    """Compute baseline accuracy for a fixed strategy (random, BFS, etc.).

    Returns {budget: accuracy}.
    """
    strategy = get_strategy(strategy_name)
    cost_model = get_cost_model('unit')

    results = defaultdict(list)

    for idx in indices:
        g_raw, label = dataset[idx]
        g = dgl.add_self_loop(g_raw)
        n_nodes = g.num_nodes()
        rng = random.Random(idx * 31 + seed_offset * 17)

        for _ in range(seeds_per_graph):
            seed_node = rng.randint(0, max(n_nodes - 1, 0))

            for budget in budgets:
                state = GraphState(g, seed_node=seed_node)
                loop = RevealLoop(state, strategy, cost_model)
                loop.run(budget=budget)

                sub_g, orig_ids = subgraph_builder.build(state)
                cls_device = next(classifier.parameters()).device
                sub_g = sub_g.to(cls_device)
                node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=cls_device)
                logits = classifier(sub_g, node_ids)
                pred = int(logits.argmax(dim=1).item())
                results[budget].append(1.0 if pred == label else 0.0)

    return {b: np.mean(accs) for b, accs in sorted(results.items())}


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.quick:
        args.valid_graphs = min(args.valid_graphs, 200)
        args.budgets = [5, 10]
        args.seeds_per_graph = 1
        args.skip_t2 = True
        args.train_graphs = 0

    # Detect phased mode: explicit --phased flag or --budget-ratio provided
    use_phased = args.phased or (args.budget_ratio is not None)
    if args.phased is None:
        use_phased = args.budget_ratio is not None
    if use_phased and args.budget_ratio is None:
        args.budget_ratio = 0.15  # default ratio for phased mode

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / 'data' / 'trajectories'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading OGBG-PPA dataset...")
    dataset = PPADataset(root=args.root)
    n_classes = int(dataset.num_classes)
    edge_feat_dim = 7  # PPA edge feature dimension

    valid_indices = dataset.valid_indices[:args.valid_graphs]
    train_indices = dataset.train_indices[:args.train_graphs] if args.train_graphs else []
    test_indices = dataset.test_indices[:args.test_graphs] if args.test_graphs else []

    # ------------------------------------------------------------------
    # 2. Load frozen classifier
    # ------------------------------------------------------------------
    logger.info("Loading frozen classifier from %s", args.pretrained)
    classifier = GraphClassifier(
        edge_feat_dim=edge_feat_dim,
        n_classes=n_classes,
    ).to(device)
    classifier.load_pretrained(args.pretrained, device=str(device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    logger.info("Classifier frozen (%d params)", sum(p.numel() for p in classifier.parameters()))

    subgraph_builder = SubgraphBuilder()

    # ------------------------------------------------------------------
    # 3. Compute baselines
    # ------------------------------------------------------------------
    logger.info("Computing baseline accuracies on %d validation graphs...", len(valid_indices))
    all_indices = list(valid_indices) + list(train_indices) + list(test_indices)
    eval_indices = valid_indices if valid_indices else all_indices[:100]

    random_acc = compute_baseline_acc(
        'random', dataset, eval_indices, classifier, subgraph_builder,
        args.budgets, seeds_per_graph=args.seeds_per_graph, seed_offset=0,
    )
    bfs_acc = compute_baseline_acc(
        'bfs', dataset, eval_indices, classifier, subgraph_builder,
        args.budgets, seeds_per_graph=args.seeds_per_graph, seed_offset=0,
    )
    logger.info("Random baseline: %s", {b: f'{v:.4f}' for b, v in random_acc.items()})
    logger.info("BFS baseline:     %s", {b: f'{v:.4f}' for b, v in bfs_acc.items()})

    # ------------------------------------------------------------------
    # 4. Initialize teachers
    # ------------------------------------------------------------------
    teachers = [
        GradientTeacher(aggregate='mean'),
        StructuralTeacher(method='degree'),
        StructuralTeacher(method='pagerank'),
        HybridTeacher(alpha=0.3),   # T4: structural-leaning
        HybridTeacher(alpha=0.7),   # T4: gradient-leaning
    ]

    if not args.skip_t2:
        # T2 is expensive — only add for validation set
        teachers.append(InfoGainTeacher(temperature=1.0))
        logger.info("T2 (info-gain) teacher enabled")
    else:
        logger.info("T2 (info-gain) teacher skipped")

    logger.info("Teachers: %s", [t.name for t in teachers])

    # ------------------------------------------------------------------
    # 5. Generate trajectories
    # ------------------------------------------------------------------
    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=subgraph_builder,
        dataset=dataset,
        device=str(device),
    )

    all_trajs = []
    filtered_trajs = []

    cache_dir = args.cache_dir if args.cache_dir else str(output_dir / 'cache')

    for split_name, split_indices in [
        ('train', train_indices),
        ('valid', valid_indices),
        ('test', test_indices),
    ]:
        if not split_indices:
            continue

        if use_phased:
            logger.info(
                "[PHASED] Generating %s trajectories (%d graphs, ratio=%.2f, seeds=%d)...",
                split_name, len(split_indices), args.budget_ratio, args.seeds_per_graph,
            )
            split_filtered = generator.generate_all_phased(
                indices=list(split_indices),
                budget_ratio=args.budget_ratio,
                seeds_per_graph=args.seeds_per_graph,
                top_k=args.top_k,
                output_dir=str(output_dir),
                checkpoint_file=str(output_dir / f'checkpoint_{split_name}.json'),
                chunk_size=args.chunk_size,
                skip_t2=args.skip_t2,
                cache_dir=cache_dir,
            )
            filtered_trajs.extend(split_filtered)
            logger.info(
                "[PHASED] %s: %d filtered trajectories", split_name, len(split_filtered),
            )
        else:
            logger.info(
                "Generating %s trajectories (%d graphs)...", split_name, len(split_indices),
            )
            trajs = generator.generate_all(
                indices=split_indices,
                budgets=args.budgets,
                seeds_per_graph=args.seeds_per_graph,
                output_dir=str(output_dir / 'raw'),
                checkpoint_file=str(output_dir / f'checkpoint_{split_name}.json'),
                save_interval=args.save_interval,
                seed_offset={'train': 0, 'valid': 1000, 'test': 2000}[split_name],
            )
            all_trajs.extend(trajs)

    # ------------------------------------------------------------------
    # 6. Score & Filter (legacy mode only; phased mode already filters)
    # ------------------------------------------------------------------
    if not use_phased:
        if not all_trajs:
            logger.error("No trajectories generated. Check dataset indices and budgets.")
            sys.exit(1)

        logger.info("Total raw trajectories: %d", len(all_trajs))

        logger.info("Scoring trajectories...")
        scorer = TrajectoryScorer(w_be=0.5, w_ig=0.3, w_stability=0.2)

        graph_sizes = {}
        for traj in all_trajs:
            if traj.graph_idx not in graph_sizes:
                g_raw, _ = dataset[traj.graph_idx]
                graph_sizes[traj.graph_idx] = g_raw.num_nodes()

        scored = scorer.score_all(all_trajs, graph_sizes=graph_sizes)
        logger.info("Scored %d trajectories", len(scored))

        scored_dir = output_dir / 'scored'
        scored_dir.mkdir(parents=True, exist_ok=True)
        torch.save(scored, scored_dir / 'scored_trajectories.pt')

        logger.info("Filtering trajectories (top_k=%d)...", args.top_k)
        traj_filter = TrajectoryFilter(top_k=args.top_k, require_diversity=True)
        filtered_trajs = traj_filter.filter(scored)

        filtered_dir = output_dir / 'filtered'
        filtered_dir.mkdir(parents=True, exist_ok=True)
        torch.save(filtered_trajs, filtered_dir / 'filtered_trajectories.pt')
        logger.info("Filtered trajectories saved: %d", len(filtered_trajs))
    else:
        # In phased mode, all_trajs from legacy path is empty; use filtered
        # from phased pipeline as the canonical list for reporting
        all_trajs = filtered_trajs
        if not all_trajs:
            logger.error("No trajectories generated. Check dataset indices and budgets.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 8. Oracle upper-bound estimation
    # ------------------------------------------------------------------
    if not use_phased:
        logger.info("Estimating oracle upper bounds...")
        oracle = OracleEstimator(classifier, subgraph_builder, device=str(device))

        oracle_type2 = oracle.estimate_type2(all_trajs, teacher_name='gradient')
        logger.info("Oracle Type II (gradient): %s", {b: f'{v:.4f}' for b, v in oracle_type2.items()})

        oracle_type1_subset = eval_indices[:50]
        type1_by_budget = defaultdict(list)
        for idx in oracle_type1_subset:
            g_raw, label = dataset[idx]
            g = dgl.add_self_loop(g_raw)
            n_nodes = g.num_nodes()
            seed_node = random.randint(0, max(n_nodes - 1, 0))
            for budget in args.budgets:
                is_correct, _ = oracle.estimate_type1(g, label, seed_node, budget)
                type1_by_budget[budget].append(1.0 if is_correct else 0.0)

        oracle_type1 = {b: np.mean(accs) for b, accs in sorted(type1_by_budget.items())}
        logger.info("Oracle Type I (label-aware, %d graphs): %s",
                    len(oracle_type1_subset), {b: f'{v:.4f}' for b, v in oracle_type1.items()})

        headroom = oracle.headroom_analysis(oracle_type2, bfs_acc, random_acc)
        logger.info("Headroom (oracle vs BFS): max=%.4f, verdict=%s",
                    headroom['max_headroom_bfs'], headroom['verdict'])

        logger.info("Running feasibility analysis...")
        analyzer = FeasibilityAnalyzer(dfi_threshold_high=0.5, dfi_threshold_low=0.3)
        dfi_result = analyzer.analyze(
            oracle_acc=oracle_type2,
            bfs_acc=bfs_acc,
            scored_trajectories=scored,
            trajectories=all_trajs,
        )
    else:
        # Phased mode: simplified oracle/feasibility
        # Compute accuracy per teacher from filtered trajectories
        from collections import Counter
        teacher_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for t in filtered_trajs:
            teacher_stats[t.teacher_type]['total'] += 1
            if t.is_correct:
                teacher_stats[t.teacher_type]['correct'] += 1

        oracle_type1 = {}
        oracle_type2 = {}
        for name, stats in sorted(teacher_stats.items()):
            acc = stats['correct'] / max(stats['total'], 1)
            oracle_type2[name] = acc
            logger.info("Teacher %s: %.4f (%d/%d)", name, acc, stats['correct'], stats['total'])

        headroom = {'max_headroom_bfs': 0.0, 'max_headroom_random': 0.0, 'verdict': 'phased_mode'}
        dfi_result = {
            'dfi': 0.0,
            'recommendation': 'Phased mode — run legacy pipeline for full feasibility analysis.',
            'components': {},
        }

    # ------------------------------------------------------------------
    # 10. Save report
    # ------------------------------------------------------------------
    config_info = {
        'pretrained': args.pretrained,
        'n_train_graphs': len(train_indices),
        'n_valid_graphs': len(valid_indices),
        'n_test_graphs': len(test_indices),
        'seeds_per_graph': args.seeds_per_graph,
        'top_k': args.top_k,
        'teachers': [t.name for t in teachers],
    }
    if use_phased:
        config_info['budget_ratio'] = args.budget_ratio
        config_info['mode'] = 'phased'
    else:
        config_info['budgets'] = args.budgets
        config_info['mode'] = 'legacy'

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config_info,
        'baselines': {
            'random': {str(b): round(v, 4) for b, v in random_acc.items()},
            'bfs': {str(b): round(v, 4) for b, v in bfs_acc.items()},
        },
        'oracle': {
            'type1_label_aware': {str(b): round(v, 4) for b, v in oracle_type1.items()},
            'type2_gradient': {str(b): round(v, 4) for b, v in oracle_type2.items()},
        },
        'headroom': {
            'max_headroom_bfs': round(headroom.get('max_headroom_bfs', 0), 4),
            'max_headroom_random': round(headroom.get('max_headroom_random', 0), 4),
            'verdict': headroom.get('verdict', ''),
        },
        'feasibility': {
            'dfi': dfi_result.get('dfi', 0),
            'recommendation': dfi_result.get('recommendation', ''),
            'components': dfi_result.get('components', {}),
        },
        'statistics': {
            'n_raw_trajectories': len(all_trajs),
            'n_filtered': len(filtered_trajs),
            'pct_correct': round(
                100.0 * sum(1 for t in all_trajs if t.is_correct) / max(len(all_trajs), 1), 1
            ),
        },
    }

    report_path = output_dir / 'm4_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("M4 report saved to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("M4 Complete — Summary")
    print("=" * 60)
    print(f"  Mode: {'PHASED' if use_phased else 'LEGACY'}")
    print(f"  Raw trajectories:     {len(all_trajs)}")
    print(f"  Filtered trajectories: {len(filtered_trajs)}")
    print(f"  % Correct:             {report['statistics']['pct_correct']:.1f}%")
    if use_phased:
        print(f"  Budget ratio: {args.budget_ratio}")
        print(f"\n  Teacher Accuracy (filtered):")
        for name in sorted(oracle_type2):
            if oracle_type2[name]:
                print(f"    {name}: {oracle_type2[name]:.4f}")
    else:
        print(f"\n  Budget → Accuracy:")
        for b in args.budgets:
            r = random_acc.get(b, 0)
            bf = bfs_acc.get(b, 0)
            o2 = oracle_type2.get(b, 0)
            o1 = oracle_type1.get(b, 0)
            print(f"    B={b:2d}: random={r:.4f}  bfs={bf:.4f}  oracle-T2={o2:.4f}  oracle-T1={o1:.4f}")
    print(f"\n  DFI: {dfi_result.get('dfi', 0):.4f}" if 'dfi' in dfi_result else "")
    print(f"  Verdict: {dfi_result.get('recommendation', 'N/A')}")
    print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

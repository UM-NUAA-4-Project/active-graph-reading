"""M4 main script: generate and filter teacher trajectories for M5 BC training.

End-to-end pipeline:
  1. Load frozen classifier
  2. Initialize teachers (T1 gradient, T2 info-gain, T3 structural, T4 hybrid)
  3. Generate trajectories (legacy per-budget or optimized 3-phase pipeline)
  4. Filter top-k per graph by cross-entropy
  5. Save filtered trajectories + report

Usage:
  python -m m4.generate_trajectories --pretrained best_model.pt --device cuda:0
  python -m m4.generate_trajectories --pretrained best_model.pt --budget-ratio 0.15
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

from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.teachers.info_gain_teacher import InfoGainTeacher
from m4.trajectory_generator import TrajectoryGenerator, filter_top_k

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
    p.add_argument('--min-correct-per-graph', type=int, default=None,
                   help='Min correct trajectories per graph (default: --top-k value)')
    p.add_argument('--max-retries', type=int, default=3,
                   help='Max retry rounds for graphs below min-correct-per-graph')
    p.add_argument('--seeds-increment', type=int, default=3,
                   help='Additional seeds per graph per retry round')

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
    # 3. Initialize teachers
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
    # 4. Generate trajectories
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
                min_correct_per_graph=args.min_correct_per_graph,
                max_retries=args.max_retries,
                seeds_increment=args.seeds_increment,
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
            )
            all_trajs.extend(trajs)

    # ------------------------------------------------------------------
    # 5. Filter (unified: top-k per graph by cross_entropy)
    # ------------------------------------------------------------------
    if not use_phased:
        if not all_trajs:
            logger.error("No trajectories generated. Check dataset indices and budgets.")
            sys.exit(1)

        logger.info("Total raw trajectories: %d", len(all_trajs))
        filtered_trajs = filter_top_k(all_trajs, args.top_k)
        logger.info("Filtered trajectories: %d (top_%d per graph)", len(filtered_trajs), args.top_k)

        filtered_dir = output_dir / 'filtered'
        filtered_dir.mkdir(parents=True, exist_ok=True)
        torch.save(filtered_trajs, filtered_dir / 'filtered_trajectories.pt')
    else:
        all_trajs = filtered_trajs
        if not all_trajs:
            logger.error("No trajectories generated. Check dataset indices and budgets.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Compute per-teacher accuracy (informational only)
    # ------------------------------------------------------------------
    teacher_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for t in all_trajs:
        teacher_stats[t.teacher_type]['total'] += 1
        if t.is_correct:
            teacher_stats[t.teacher_type]['correct'] += 1

    teacher_acc = {}
    for name, stats in sorted(teacher_stats.items()):
        acc = stats['correct'] / max(stats['total'], 1)
        teacher_acc[name] = acc
        logger.info("Teacher %s: %.4f (%d/%d)", name, acc, stats['correct'], stats['total'])

    # ------------------------------------------------------------------
    # 7. Save report
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
        'teacher_accuracy': {k: round(v, 4) for k, v in teacher_acc.items()},
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
    if use_phased:
        print(f"  Budget ratio: {args.budget_ratio}")
    else:
        print(f"  Budgets: {args.budgets}")
    print(f"  Raw trajectories:     {len(all_trajs)}")
    print(f"  Filtered trajectories: {len(filtered_trajs)}")
    print(f"  % Correct:             {report['statistics']['pct_correct']:.1f}%")
    print(f"\n  Teacher Accuracy:")
    for name in sorted(teacher_acc):
        print(f"    {name}: {teacher_acc[name]:.4f}")
    print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

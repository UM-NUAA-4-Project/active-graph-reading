"""Smoke test: generate teacher trajectories on sampled graphs and validate.

Loads a set of sampled graphs (from sample_test_graphs.py), runs every
teacher on every graph, and checks trajectory invariants.

Usage:
  python -m m4.run_smoke_test --pretrained best_model.pt --device cuda:0
  python -m m4.run_smoke_test --pretrained best_model.pt --device cpu --graphs-file smoke_test_graphs.pt
"""

import argparse
import sys
import traceback
from pathlib import Path

import dgl
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
    p = argparse.ArgumentParser(description='M4 smoke test — validate teacher trajectories')
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to best_model.pt from full-graph training')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='Device for computation')
    p.add_argument('--graphs-file', type=str, default=None,
                   help='Path to sampled graphs .pt (default: smoke_test_graphs.pt in script dir)')
    p.add_argument('--budget-ratio', type=float, default=0.10,
                   help='Budget as fraction of |V| (default 0.10 for speed)')
    p.add_argument('--seeds-per-graph', type=int, default=5,
                   help='Random seeds per graph (initial round)')

    # Phased mode (tests guaranteed top-k pipeline)
    p.add_argument('--phased', action='store_true',
                   help='Use phased pipeline with retry (tests guaranteed top-k)')
    p.add_argument('--top-k', type=int, default=3,
                   help='Top-k for phased mode')
    p.add_argument('--min-correct-per-graph', type=int, default=None,
                   help='Min correct per graph (default: --top-k value)')
    p.add_argument('--max-retries', type=int, default=3,
                   help='Max retry rounds for guaranteed mode')
    p.add_argument('--seeds-increment', type=int, default=3,
                   help='Additional seeds per graph per retry round')
    p.add_argument('--skip-t2', action='store_true',
                   help='Skip T2 (info-gain) teacher')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root (phased mode only)')
    return p.parse_args()


def validate_trajectory(traj, n_nodes, budget, label):
    """Check trajectory invariants. Returns list of error strings (empty = pass)."""
    errors = []

    # -- structural invariants --
    if len(traj.reveal_order) != traj.budget_used:
        errors.append(
            f"reveal_order len {len(traj.reveal_order)} != budget_used {traj.budget_used}"
        )
    if traj.budget_used > budget:
        errors.append(f"budget_used {traj.budget_used} > budget {budget}")
    if traj.label != label:
        errors.append(f"label {traj.label} != expected {label}")
    if not (0 <= traj.seed_node < n_nodes):
        errors.append(f"seed_node {traj.seed_node} out of range [0, {n_nodes})")

    # -- node validity --
    invalid = [v for v in traj.reveal_order if not (0 <= v < n_nodes)]
    if invalid:
        errors.append(f"invalid node IDs in reveal_order: {invalid[:5]}...")
    if len(set(traj.reveal_order)) != len(traj.reveal_order):
        errors.append("duplicate nodes in reveal_order")

    # -- per-step arrays --
    if len(traj.confidence_per_step) != traj.budget_used + 1:
        errors.append(
            f"confidence_per_step len {len(traj.confidence_per_step)} != "
            f"budget_used+1 ({traj.budget_used + 1})"
        )
    if len(traj.entropy_per_step) != traj.budget_used + 1:
        errors.append(
            f"entropy_per_step len {len(traj.entropy_per_step)} != "
            f"budget_used+1 ({traj.budget_used + 1})"
        )

    # -- final prediction --
    if not (0 <= traj.final_pred < 37):
        errors.append(f"final_pred {traj.final_pred} out of range [0, 37)")
    if traj.cross_entropy == float('inf') or traj.cross_entropy != traj.cross_entropy:
        errors.append(f"cross_entropy is inf/NaN: {traj.cross_entropy}")
    if traj.cross_entropy < 0:
        errors.append(f"cross_entropy negative: {traj.cross_entropy:.6f}")
    if not isinstance(traj.is_correct, bool):
        errors.append(f"is_correct not bool: {type(traj.is_correct)}")

    # -- metadata --
    if not traj.teacher_type:
        errors.append("teacher_type is empty")

    return errors


def validate_phased_output(trajs, indices, top_k, min_correct):
    """Check that every graph has exactly top_k trajectories, count correct per graph."""
    from collections import defaultdict, Counter

    per_graph = defaultdict(list)
    for t in trajs:
        per_graph[t.graph_idx].append(t)

    errors = []
    n_graphs = len(set(indices))
    n_covered = len(per_graph)
    n_correct_graphs = 0
    missing = []

    for idx in indices:
        entries = per_graph.get(idx, [])
        if len(entries) != top_k:
            missing.append((idx, len(entries)))
        n_correct = sum(1 for t in entries if t.is_correct)
        if n_correct >= min_correct:
            n_correct_graphs += 1

    if missing:
        errors.append(f"{len(missing)}/{n_graphs} graphs don't have exactly {top_k} trajectories")

    return {
        'n_graphs': n_graphs,
        'n_covered': n_covered,
        'n_correct_graphs': n_correct_graphs,
        'n_total': len(trajs),
        'missing': missing[:10],
        'errors': errors,
    }


def main():
    args = parse_args()

    graphs_file = (
        Path(args.graphs_file) if args.graphs_file
        else Path(__file__).parent / 'smoke_test_graphs.pt'
    )

    # ------------------------------------------------------------------
    # 1. Load sampled graphs
    # ------------------------------------------------------------------
    print(f"Loading sampled graphs from {graphs_file}...")
    samples = torch.load(str(graphs_file), map_location='cpu', weights_only=False)
    # samples: {split_name: [{graph_idx, graph, label}, ...]}

    total_graphs = sum(len(v) for v in samples.values())
    print(f"Loaded {total_graphs} graphs across {list(samples.keys())}")

    # ------------------------------------------------------------------
    # 2. Load frozen classifier
    # ------------------------------------------------------------------
    device = torch.device(args.device)
    print(f"Loading frozen classifier from {args.pretrained}...")
    classifier = GraphClassifier(
        edge_feat_dim=7,
        n_classes=37,
    ).to(device)
    classifier.load_pretrained(args.pretrained, device=str(device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"Classifier frozen ({sum(p.numel() for p in classifier.parameters())} params)")

    # ------------------------------------------------------------------
    # 3. If phased mode: run full pipeline with retry, then exit
    # ------------------------------------------------------------------
    if args.phased:
        min_correct = args.min_correct_per_graph if args.min_correct_per_graph else args.top_k

        # Collect graph indices from samples
        flat_indices = []
        for entries in samples.values():
            for e in entries:
                flat_indices.append(e['graph_idx'])

        print(f"Phased mode: {len(flat_indices)} graphs, top_k={args.top_k}, "
              f"min_correct={min_correct}, budget_ratio={args.budget_ratio}, "
              f"max_retries={args.max_retries}")

        # Load full dataset (needed by generate_all_phased)
        dataset = PPADataset(root=args.root)

        # Teachers for the phased pipeline
        teachers = [
            GradientTeacher(aggregate='mean'),
            StructuralTeacher(method='degree'),
            StructuralTeacher(method='pagerank'),
            HybridTeacher(alpha=0.3),
            HybridTeacher(alpha=0.7),
        ]
        if not args.skip_t2:
            teachers.append(InfoGainTeacher(temperature=1.0))

        generator = TrajectoryGenerator(
            teachers=teachers,
            classifier=classifier,
            subgraph_builder=SubgraphBuilder(),
            dataset=dataset,
            device=str(device),
        )

        filtered = generator.generate_all_phased(
            indices=flat_indices,
            budget_ratio=args.budget_ratio,
            seeds_per_graph=args.seeds_per_graph,
            top_k=args.top_k,
            min_correct_per_graph=min_correct,
            max_retries=args.max_retries,
            seeds_increment=args.seeds_increment,
            skip_t2=args.skip_t2,
            chunk_size=len(flat_indices),
        )

        result = validate_phased_output(filtered, flat_indices, args.top_k, min_correct)

        print("\n" + "=" * 60)
        print("Phased Smoke Test Results")
        print("=" * 60)
        print(f"  Graphs:        {result['n_graphs']}")
        print(f"  Covered:       {result['n_covered']}")
        print(f"  Correct ≥{min_correct}: {result['n_correct_graphs']}/{result['n_graphs']} "
              f"({100.0 * result['n_correct_graphs'] / max(result['n_graphs'], 1):.1f}%)")
        print(f"  Trajectories:  {result['n_total']}")

        # Per-teacher breakdown
        from collections import defaultdict
        by_teacher = defaultdict(lambda: {'total': 0, 'correct': 0})
        for t in filtered:
            by_teacher[t.teacher_type]['total'] += 1
            if t.is_correct:
                by_teacher[t.teacher_type]['correct'] += 1
        print("\n  Per teacher:")
        for name in sorted(by_teacher):
            stats = by_teacher[name]
            acc = 100.0 * stats['correct'] / max(stats['total'], 1)
            print(f"    {name}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

        if result['missing']:
            print(f"\n  Graphs != {args.top_k} trajectories:")
            for idx, n in result['missing']:
                print(f"    graph {idx}: {n} trajectories")

        if result['errors']:
            print(f"\n  Errors: {result['errors']}")

        print("=" * 60)
        if result['n_correct_graphs'] == result['n_graphs']:
            print("\nPHASED SMOKE TEST PASSED")
            sys.exit(0)
        else:
            print(f"\nPHASED SMOKE TEST FAILED: "
                  f"{result['n_graphs'] - result['n_correct_graphs']} graphs below threshold")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 3. (Per-teacher mode) Initialize teachers
    # ------------------------------------------------------------------
    teachers = [
        GradientTeacher(aggregate='mean'),
        StructuralTeacher(method='degree'),
        StructuralTeacher(method='pagerank'),
        HybridTeacher(alpha=0.3),
        HybridTeacher(alpha=0.7),
        InfoGainTeacher(temperature=1.0),
    ]
    print(f"Teachers: {[t.name for t in teachers]}")

    subgraph_builder = SubgraphBuilder()

    # ------------------------------------------------------------------
    # 4. Generate + validate
    # ------------------------------------------------------------------
    results = []
    n_done = 0

    for split_name, entries in samples.items():
        for entry in entries:
            g_raw = entry['graph']
            g = dgl.add_self_loop(g_raw)
            n_nodes = g.num_nodes()
            label = entry['label']
            graph_idx = entry['graph_idx']

            if n_nodes == 0:
                print(f"  SKIP {split_name}/class_{label}/idx_{graph_idx}: empty graph")
                continue

            budget = max(1, int(n_nodes * args.budget_ratio))
            seed_node = (graph_idx * 31 + 7) % n_nodes

            for teacher in teachers:
                n_done += 1
                try:
                    traj = teacher.generate_trajectory(
                        graph=g,
                        classifier=classifier,
                        graph_idx=graph_idx,
                        label=label,
                        seed_node=seed_node,
                        budget=budget,
                        subgraph_builder=subgraph_builder,
                    )
                    errs = validate_trajectory(traj, n_nodes, budget, label)
                    results.append({
                        'split': split_name,
                        'class': label,
                        'graph_idx': graph_idx,
                        'n_nodes': n_nodes,
                        'budget': budget,
                        'teacher': teacher.name,
                        'passed': len(errs) == 0,
                        'errors': errs,
                        'is_correct': traj.is_correct,
                        'ce': traj.cross_entropy,
                    })
                except Exception as e:
                    results.append({
                        'split': split_name,
                        'class': label,
                        'graph_idx': graph_idx,
                        'n_nodes': n_nodes,
                        'budget': budget,
                        'teacher': teacher.name,
                        'passed': False,
                        'errors': [f"exception: {e}\n{traceback.format_exc()}"],
                        'is_correct': None,
                        'ce': None,
                    })

                # Progress
                if n_done % 50 == 0:
                    passed_so_far = sum(1 for r in results if r['passed'])
                    print(f"  [{n_done}] {passed_so_far}/{len(results)} passed")

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    passed = sum(1 for r in results if r['passed'])
    failed = [r for r in results if not r['passed']]

    print("\n" + "=" * 60)
    print("Smoke Test Results")
    print("=" * 60)
    print(f"  Total:  {len(results)} trajectories")
    print(f"  Passed: {passed}")
    print(f"  Failed: {len(failed)}")

    # By teacher
    from collections import defaultdict
    by_teacher = defaultdict(lambda: {'passed': 0, 'total': 0})
    for r in results:
        by_teacher[r['teacher']]['total'] += 1
        if r['passed']:
            by_teacher[r['teacher']]['passed'] += 1

    print("\n  Per teacher:")
    for name in sorted(by_teacher):
        stats = by_teacher[name]
        print(f"    {name}: {stats['passed']}/{stats['total']}")

    # Correct %
    correct = sum(1 for r in results if r['is_correct'])
    total_with_pred = sum(1 for r in results if r['is_correct'] is not None)
    if total_with_pred:
        print(f"\n  Correct predictions: {correct}/{total_with_pred} "
              f"({100.0 * correct / total_with_pred:.1f}%)")

    if failed:
        print(f"\n  First 10 failures:")
        for r in failed[:10]:
            print(f"    [{r['teacher']}] {r['split']}/class_{r['class']}/"
                  f"idx_{r['graph_idx']} (|V|={r['n_nodes']}): {r['errors']}")

    print("=" * 60)
    

    if failed:
        print("\nSMOKE TEST FAILED")
        sys.exit(1)
    else:
        print("\nSMOKE TEST PASSED")
        sys.exit(0)



if __name__ == '__main__':
    main()

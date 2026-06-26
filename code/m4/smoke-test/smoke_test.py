"""M4 冒烟测试 — 加载 PPA 样本，验证 teacher 轨迹生成 pipeline。

Usage:
  # Step 1: 采样
  python code/m4/smoke-test/sample_ppa_smoke.py --root D:\\ogb-master\\dataset
  # Step 2: 冒烟测试
  python code/m4/smoke-test/smoke_test.py --load code\small_checkpoints\best_model.pt
  --reveal-ratio 0.1 --n-seeds 5 --device cuda:0
"""

import argparse
import sys
import logging
from pathlib import Path

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import torch

from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.teachers.info_gain_teacher import InfoGainTeacher
from m4.trajectory_generator import TrajectoryGenerator

logging.basicConfig(level=logging.WARNING)

SEPARATOR = "=" * 60


def parse_args():
    p = argparse.ArgumentParser(description='M4 Teacher Trajectory Smoke Test')
    p.add_argument('--load', type=str, required=True,
                   help='Path to frozen classifier (best_model.pt)')
    p.add_argument('--sample', type=str, default=None,
                   help='Path to smoke sample (default: m4/smoke_sample.pt)')
    p.add_argument('--device', type=str, default='cpu',
                   help='Device (cpu / cuda:0)') 
    p.add_argument('--reveal-ratio', type=float, default=0.05,
                   help='Reveal ratio (default: 0.05)')
    p.add_argument('--n-seeds', type=int, default=3,
                   help='Number of top-scored seeds per teacher per graph')
    p.add_argument('--graph-batch-size', type=int, default=32,
                   help='Mini-batch size for cross-graph validation')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    sample_path = Path(args.sample) if args.sample \
        else _script_dir / 'm4' / 'smoke_sample.pt'
    if not sample_path.exists():
        print(f"Sample file not found: {sample_path}")
        print("Run 'python code/m4/sample_ppa_smoke.py --root <root>' first.")
        sys.exit(1)

    # 加载分类器
    print(f"Loading classifier from {args.load}...")
    ckpt = torch.load(args.load, map_location=device)
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    classifier.load_state_dict(ckpt['model_state_dict'])
    val_acc = ckpt.get('val_acc', 0.0)
    print(f"  val_acc={val_acc:.4f}")
    classifier.eval()

    # 加载样本
    print(f"Loading smoke sample from {sample_path}...")
    sample = torch.load(sample_path, map_location='cpu')
    graphs = sample['graphs']
    labels = sample['labels']
    indices = sample['indices']
    print(f"  {len(graphs)} graphs, {len(set(labels))} classes")
    subgraph_builder = SubgraphBuilder()
    teachers = [
        GradientTeacher(),
        StructuralTeacher(method='degree'),
        StructuralTeacher(method='pagerank'),
        HybridTeacher(alpha=0.5),
        InfoGainTeacher(classifier=classifier, subgraph_builder=subgraph_builder,
                        device=str(device)),
    ]
    print(f"Teachers: {[t.name for t in teachers]}")
    print(f"Reveal ratio: {args.reveal_ratio}")

    

    # -------------------------------------------------------------------
    # Test 1: Gradient node scores 有区分度
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 1: Gradient teacher node score diversity")
    print(SEPARATOR)
    gt = GradientTeacher()
    g0, label0 = graphs[0], labels[0]
    scores = gt.score_nodes(g0, classifier, label0)

    score_std = float(scores.std())
    score_nonzero = int((scores > 0).sum().item())
    print(f"  graph 0 (|V|={g0.num_nodes()}): std={score_std:.6f}, "
          f"nonzero={score_nonzero}/{g0.num_nodes()}")
    assert score_std > 0, "Gradient scores should have variance!"
    print("  [OK] Gradient scores have diversity")

    # -------------------------------------------------------------------
    # Test 2: Structural teachers produce valid scores
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 2: Structural teacher score range")
    print(SEPARATOR)
    for st in [StructuralTeacher(method='degree'), StructuralTeacher(method='pagerank')]:
        scores = st.score_nodes(g0)
        print(f"  {st.name}: min={scores.min():.4f}, max={scores.max():.4f}, "
              f"nonzero={(scores>0).sum().item()}/{g0.num_nodes()}")
        assert scores.max() > 0, f"{st.name} should have non-zero max!"
    print("  [OK] Structural scores valid")

    # -------------------------------------------------------------------
    # Test 2b: InfoGain single-step IG computation
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 2b: InfoGain teacher single-step IG")
    print(SEPARATOR)
    ig_teacher = InfoGainTeacher(
        classifier=classifier,
        subgraph_builder=subgraph_builder,
        device=str(device),
    )
    from active_graph_reader.engine.graph_state import GraphState
    g0_state = GraphState(g0, seed_node=0)
    frontier_nodes = g0_state.get_frontier_nodes()
    if frontier_nodes:
        # Build current subgraph and compute entropy
        cur_g, _ = subgraph_builder.build(g0_state)
        H_cur = ig_teacher._entropy(cur_g)
        # Build candidate subgraph for first frontier node
        cg = ig_teacher._build_candidate_subgraph(g0_state, frontier_nodes[0])
        H_after = ig_teacher._entropy(cg)
        ig_val = H_cur - H_after
        print(f"  G_0: H_current={H_cur:.4f}, H_after(reveal {frontier_nodes[0]})={H_after:.4f}, "
              f"IG={ig_val:.4f}")
        print(f"  frontier nodes: {len(frontier_nodes)}")
    print("  [OK] InfoGain single-step IG computation works")

    # -------------------------------------------------------------------
    # Test 3: Trajectory generation pipeline
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 3: TrajectoryGenerator pipeline (reveal_ratio={:.3f})".format(args.reveal_ratio))
    print(SEPARATOR)
    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=subgraph_builder,
        device=str(device),
        graph_batch_size=args.graph_batch_size,
    )
    print(f"Graph batch size: {args.graph_batch_size}")

    # 用 list-based adapter 适配 generate_all 的新接口（无额外内存开销）
    class _ListDataset:
        def __getitem__(self, idx):
            return graphs[idx], labels[idx]

    result = generator.generate_all(
        dataset=_ListDataset(),
        train_idx=list(range(len(graphs))),
        reveal_ratio=args.reveal_ratio,
        n_seeds=args.n_seeds,
    )

    total_correct = 0
    total_trajs = 0
    for tname, per_graph in result.items():
        count = 0
        correct_count = 0
        for trajs in per_graph:
            count += len(trajs)
            for t in trajs:
                if t.is_correct:
                    correct_count += 1
        total_trajs += count
        total_correct += correct_count
        print(f"  {tname}: {count} trajectories, {correct_count} correct")

    print(f"  Total: {total_trajs} trajectories across {len(graphs)} graphs")
    assert total_correct == total_trajs, "_generate_for_graph should only return correct trajectories!"
    print("  [OK] All trajectories is_correct=True")

    # -------------------------------------------------------------------
    # Per-graph summary
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Per-graph trajectory counts")
    print(SEPARATOR)
    n_graphs = len(graphs)
    teacher_names = list(result.keys())
    per_graph_counts = [0] * n_graphs
    for tname in teacher_names:
        for i, trajs in enumerate(result[tname]):
            per_graph_counts[i] += len(trajs)

    print(f"  {'graph_idx':>10s}  {'#trajs':>7s}")
    for i, count in enumerate(per_graph_counts):
        print(f"  {indices[i]:>10d}  {count:>7d}")
    empty_graphs = sum(1 for c in per_graph_counts if c == 0)
    print(f"  Graphs with 0 trajectories: {empty_graphs}/{n_graphs}")

    # -------------------------------------------------------------------
    # Teacher failure stats — 每个 teacher 在多少张图上找到 0 条轨迹
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Teacher failure counts (graphs with 0 trajectories)")
    print(SEPARATOR)
    print(f"  {'teacher':>20s}  {'failures':>9s}")
    for tname in teacher_names:
        failures = sum(1 for trajs in result[tname] if len(trajs) == 0)
        print(f"  {tname:>20s}  {failures:>9d}")

    print(f"\n{SEPARATOR}")
    print("M4 Smoke Test — ALL TESTS PASSED")
    print(SEPARATOR)


if __name__ == '__main__':
    main()

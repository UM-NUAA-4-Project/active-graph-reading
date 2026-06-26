"""M4 补充轨迹生成脚本 — 按类别标签过滤训练集生成轨迹。

用于为尾部类别补充轨迹数据。接口与 generate_trajectories.py 一致，
但不分片，按 label 独立保存。

用法:
  python code/m4/supplement_trajectories.py \
      --root D:\ogb-master\dataset \
      --load code/m3/best_model.pt \
      --labels 0 5 10 11 13 14 16 19 20 27 28 30 32 \
      --reveal-ratios 0.05 0.10 0.15 \
      --output-dir code/m4/data/stage2_trimmed
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import torch
import dgl
from tqdm import tqdm

from ogb.graphproppred import DglGraphPropPredDataset

from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.trajectory_generator import TrajectoryGenerator
from m4.teachers.info_gain_teacher import InfoGainTeacher

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='M4 Supplement Trajectory Generation')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root')
    p.add_argument('--load', type=str, required=True,
                   help='Path to frozen classifier (best_model.pt)')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='Device (cuda:0 / cpu)')
    p.add_argument('--labels', type=int, nargs='+', required=True,
                   help='Target class labels to supplement (e.g. --labels 13 28 30)')
    p.add_argument('--reveal-ratios', type=float, nargs='+',
                   default=[0.05, 0.10, 0.15],
                   help='Reveal ratios (default: 0.05 0.10 0.15)')
    p.add_argument('--n-seeds', type=int, default=5,
                   help='Number of top-scored seeds per teacher per graph')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: m4/data/stage2_trimmed)')
    p.add_argument('--teachers', type=str, nargs='+',
                   default=['gradient', 'structural-degree', 'structural-pagerank',
                            'hybrid-0.3', 'hybrid-0.7'],
                   help='Teachers to use')
    p.add_argument('--graph-batch-size', type=int, default=32,
                   help='Mini-batch size for cross-graph validation (default: 32)')
    return p.parse_args()


def build_teacher(name: str, classifier=None, subgraph_builder=None, device='cpu'):
    if name == 'gradient':
        return GradientTeacher()
    elif name == 'structural-degree':
        return StructuralTeacher(method='degree')
    elif name == 'structural-pagerank':
        return StructuralTeacher(method='pagerank')
    elif name.startswith('hybrid-'):
        alpha = float(name.split('-')[1])
        return HybridTeacher(alpha=alpha)
    elif name == 'info-gain':
        return InfoGainTeacher(classifier=classifier, subgraph_builder=subgraph_builder,
                               device=device)
    else:
        raise ValueError(f"Unknown teacher: {name}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir / 'data' / 'stage2_trimmed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载冻结分类器
    logger.info("Loading frozen classifier from %s...", args.load)
    ckpt = torch.load(args.load, map_location=device)
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    classifier.load_state_dict(ckpt['model_state_dict'])
    val_acc = ckpt.get('val_acc', 0.0)
    logger.info("Loaded classifier (val_acc=%.4f)", val_acc)
    classifier.eval()

    subgraph_builder = SubgraphBuilder()

    teachers = [build_teacher(name, classifier=classifier, subgraph_builder=subgraph_builder,
                              device=str(device)) for name in args.teachers]
    logger.info("Teachers: %s", [t.name for t in teachers])

    # 加载训练集 → 按 label 分组索引
    logger.info("Loading OGBG-PPA train split (root=%s)...", args.root)
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split = dataset.get_idx_split()
    train_idx = split['train'].tolist()
    logger.info("Train set: %d graphs", len(train_idx))

    target_set = set(args.labels)

    # 第一遍扫描：按 label 分组 OGB 索引
    logger.info("Scanning training labels...")
    label_to_ogb_indices = defaultdict(list)
    for i in tqdm(train_idx, desc="Scan labels"):
        lbl = dataset[int(i)][1]
        label = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
        if label in target_set:
            label_to_ogb_indices[label].append(int(i))

    # 统计各类别训练图数量
    logger.info("Training graphs per target label:")
    for label in sorted(args.labels):
        count = len(label_to_ogb_indices.get(label, []))
        logger.info("  class %2d: %d graphs", label, count)

    # 第二遍：只加载目标 label 的图，加自环
    target_graphs_by_label = {}
    for label in sorted(args.labels):
        ogb_indices = label_to_ogb_indices.get(label, [])
        if not ogb_indices:
            logger.warning("No training graphs for class %d, skipping", label)
            continue

        logger.info("Loading %d graphs for class %d...", len(ogb_indices), label)
        graphs = []
        for i in tqdm(ogb_indices, desc=f"Load class {label}"):
            g, lbl = dataset[int(i)]
            graphs.append((dgl.add_self_loop(g), lbl))
        target_graphs_by_label[label] = graphs

    del dataset
    del label_to_ogb_indices

    # 为每个目标 label 生成轨迹
    for label in sorted(args.labels):
        label_dataset = target_graphs_by_label.get(label, [])
        if not label_dataset:
            continue

        local_indices = list(range(len(label_dataset)))
        logger.info("=== Processing class %d: %d graphs ===", label, len(label_dataset))

        # 每种 ratio 复用同一个 generator
        generator = TrajectoryGenerator(
            teachers=teachers,
            classifier=classifier,
            subgraph_builder=subgraph_builder,
            device=str(device),
            graph_batch_size=args.graph_batch_size,
        )

        for ratio in args.reveal_ratios:
            ratio_str = f"{ratio:.3f}".replace('.', '_')
            logger.info("  Ratio = %.3f (%d graphs)", ratio, len(local_indices))

            result = generator.generate_all(
                dataset=label_dataset,
                train_idx=local_indices,
                reveal_ratio=ratio,
                n_seeds=args.n_seeds,
            )

            total_trajs = 0
            for teacher_name, per_graph in result.items():
                count = sum(len(trajs) for trajs in per_graph)
                total_trajs += count
                logger.info("    %s: %d trajectories", teacher_name, count)
            for teacher in teachers:
                failures = sum(1 for trajs in result[teacher.name] if len(trajs) == 0)
                logger.info("    %20s  non-empty: %d/%d",
                            teacher.name, len(local_indices) - failures, len(local_indices))

            output_path = output_dir / f'trajectories_r{ratio_str}_of_{label}.pt'
            torch.save({
                'trajectories': result,
                'config': {
                    'reveal_ratio': ratio,
                    'n_seeds': args.n_seeds,
                    'n_graphs': len(local_indices),
                    'teacher_names': [t.name for t in teachers],
                    'target_label': label,
                },
            }, output_path)
            logger.info("    Saved %d trajectories to %s", total_trajs, output_path)

        # 释放该 label 的数据
        del label_dataset
        del generator
        target_graphs_by_label[label] = None  # 帮助 GC

    logger.info("=== Supplement generation complete ===")
    total_files = len(args.labels) * len(args.reveal_ratios)
    logger.info("Generated up to %d files in %s", total_files, output_dir)


if __name__ == '__main__':
    main()

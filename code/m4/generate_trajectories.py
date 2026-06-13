"""M4 Teacher 轨迹生成 CLI 入口。

使用方式：
  python code/m4/generate_trajectories.py --root /mnt/data --load code/m3/best_model_ft.pt 
  --device cuda:0 --n-seeds 50 --k-shards 10 --shard-idx 1
"""

import argparse
import logging
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import torch
import dgl
from tqdm import tqdm

from ogb.graphproppred import DglGraphPropPredDataset

from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.trajectory_generator import TrajectoryGenerator
from m4.teachers.info_gain_teacher import InfoGainTeacher
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='M4 Teacher Trajectory Generation')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root')
    p.add_argument('--load', type=str, required=True,
                   help='Path to frozen classifier (best_model.pt)')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='Device (cuda:0 / cpu)')
    p.add_argument('--reveal-ratios', type=float, nargs='+',
                   default=[0.10, 0.15, 0.20, 0.25],
                   help='Reveal ratios (e.g. 0.01 0.05 0.10)')
    p.add_argument('--n-seeds', type=int, default=5,
                   help='Number of top-scored seeds per teacher per graph')
    p.add_argument('--max-graphs', type=int, default=0,
                   help='Limit number of training graphs (0 = all)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: m4/output)')
    p.add_argument('--teachers', type=str, nargs='+',
                   default=['gradient', 'structural-degree', 'structural-pagerank',
                            'hybrid-0.3', 'hybrid-0.7','info-gain'],
                   help='Teachers to use')
    p.add_argument('--k-shards', type=int, default=1,
                   help='Split training set into k shards (default: 1 = all at once)')
    p.add_argument('--shard-idx', type=int, nargs='+', default=None,
                   help='Process only this/these shards (1-indexed). '
                        'Accepts multiple: --shard-idx 1 3 5')
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


def split_into_shards(train_idx, k):
    """将 train_idx 均分为 k 片。"""
    n = len(train_idx)
    shard_size = (n + k - 1) // k  # ceil division
    shards = []
    for i in range(k):
        start = i * shard_size
        end = min(start + shard_size, n)
        if start >= n:
            break
        shards.append(train_idx[start:end])
    return shards


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir / 'output'
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

    # 加载训练集 → 提取子集 → 释放原始数据 → 加自环
    logger.info("Loading OGBG-PPA train split (root=%s)...", args.root)
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split = dataset.get_idx_split()
    train_idx = split['train'].tolist()
    if args.max_graphs > 0:
        train_idx = train_idx[:args.max_graphs]
        logger.info("Limited to %d training graphs", args.max_graphs)

    logger.info("Extracting training subset (%d graphs)...", len(train_idx))
    train_dataset = [dataset[int(i)] for i in tqdm(train_idx, desc="Extract train")]
    del dataset

    for i in tqdm(range(len(train_dataset)), desc="self-loops"):
        train_dataset[i] = (dgl.add_self_loop(train_dataset[i][0]), train_dataset[i][1])

    subgraph_builder = SubgraphBuilder()

    teachers = [build_teacher(name, classifier=classifier, subgraph_builder=subgraph_builder,
                              device=str(device)) for name in args.teachers]
    logger.info("Teachers: %s", [t.name for t in teachers])
    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=subgraph_builder,
        device=str(device),
        graph_batch_size=args.graph_batch_size,
    )

    shard_list = split_into_shards(train_idx, args.k_shards)
    k_total = len(shard_list)
    logger.info("Training set split into %d shards (~%d graphs each)", k_total, len(shard_list[0]))

    if args.shard_idx is not None:
        for si in args.shard_idx:
            if not (1 <= si <= k_total):
                raise ValueError(f"--shard-idx={si} out of range [1, {k_total}]")
        shards = {si: shard_list[si - 1] for si in sorted(set(args.shard_idx))}
        logger.info("Processing %d shard(s): %s (%d graphs total)",
                     len(shards), sorted(shards.keys()),
                     sum(len(s) for s in shards.values()))
    else:
        shards = {i + 1: s for i, s in enumerate(shard_list)}

    for ratio in args.reveal_ratios:
        ratio_str = f"{ratio:.3f}".replace('.', '_')
        logger.info("=== Reveal ratio = %.3f ===", ratio)

        for shard_i, shard_idx in sorted(shards.items()):
            logger.info("--- Shard %d/%d (%d graphs) ---", shard_i, k_total, len(shard_idx))
            result = generator.generate_all(
                dataset=train_dataset,
                train_idx=shard_idx,
                reveal_ratio=ratio,
                n_seeds=args.n_seeds,
            )

            total_trajs = 0
            for teacher_name, per_graph in result.items():
                count = sum(len(trajs) for trajs in per_graph)
                total_trajs += count
                logger.info("  %s: %d trajectories", teacher_name, count)
            for teacher in teachers:
                failures = sum(1 for trajs in result[teacher.name] if len(trajs) == 0)
                logger.info("  %20s  failures: %d", teacher.name, failures)

            output_path = output_dir / f'trajectories_r{ratio_str}_shard{shard_i:02d}_of_{k_total:02d}.pt'
            torch.save({
                'trajectories': result,
                'config': {
                    'reveal_ratio': ratio,
                    'n_seeds': args.n_seeds,
                    'n_graphs': len(shard_idx),
                    'shard': shard_i,
                    'k_shards': k_total,
                    'teacher_names': [t.name for t in teachers],
                },
            }, output_path)
            logger.info("Saved %d trajectories to %s", total_trajs, output_path)


if __name__ == '__main__':
    main()

"""M4 Teacher 轨迹生成 CLI 入口。

使用方式：
  python generate_trajectories.py --root D:\\ogb-master\\dataset --load ../m3/best_model.pt
  python generate_trajectories.py --root D:\\ogb-master\\dataset --load ../m3/best_model.pt \\
      --max-graphs 10 --reveal-ratios 0.01 0.05 --device cuda:0
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
                   default=[0.01, 0.02, 0.05, 0.10],
                   help='Reveal ratios (e.g. 0.01 0.05 0.10)')
    p.add_argument('--k-top', type=int, default=3,
                   help='Top-k trajectories per graph per teacher')
    p.add_argument('--max-attempts', type=int, default=100,
                   help='Max seed attempts per graph')
    p.add_argument('--max-graphs', type=int, default=0,
                   help='Limit number of training graphs (0 = all)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: m4/output)')
    p.add_argument('--teachers', type=str, nargs='+',
                   default=['gradient', 'structural-degree', 'structural-pagerank',
                            'hybrid-0.3', 'hybrid-0.7'],
                   help='Teachers to use')
    return p.parse_args()


def build_teacher(name: str):
    if name == 'gradient':
        return GradientTeacher()
    elif name == 'structural-degree':
        return StructuralTeacher(method='degree')
    elif name == 'structural-pagerank':
        return StructuralTeacher(method='pagerank')
    elif name.startswith('hybrid-'):
        alpha = float(name.split('-')[1])
        return HybridTeacher(alpha=alpha)
    else:
        raise ValueError(f"Unknown teacher: {name}")


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

    # 加载训练集
    logger.info("Loading OGBG-PPA train split (root=%s)...", args.root)
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split = dataset.get_idx_split()
    train_idx = split['train'].tolist()

    if args.max_graphs > 0:
        train_idx = train_idx[:args.max_graphs]
        logger.info("Limited to %d training graphs", args.max_graphs)

    graphs: list = []
    labels: list = []
    for i in tqdm(train_idx, desc="Loading graphs"):
        g, lbl = dataset[int(i)]
        g = dgl.add_self_loop(g)
        graphs.append(g)
        labels.append(int(lbl.item()))

    teachers = [build_teacher(name) for name in args.teachers]
    logger.info("Teachers: %s", [t.name for t in teachers])

    subgraph_builder = SubgraphBuilder()
    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=subgraph_builder,
        device=str(device),
    )

    for ratio in args.reveal_ratios:
        logger.info("=== Reveal ratio = %.3f ===", ratio)
        result = generator.generate_all(
            graphs=graphs,
            labels=labels,
            indices=train_idx,
            reveal_ratio=ratio,
            k_top=args.k_top,
            max_attempts=args.max_attempts,
        )

        total_trajs = 0
        for teacher_name, per_graph in result.items():
            count = sum(len(trajs) for trajs in per_graph)
            total_trajs += count
            logger.info("  %s: %d trajectories", teacher_name, count)

        ratio_str = f"{ratio:.3f}".replace('.', '_')
        output_path = output_dir / f'trajectories_r{ratio_str}.pt'
        torch.save({
            'trajectories': result,
            'config': {
                'reveal_ratio': ratio,
                'k_top': args.k_top,
                'max_attempts': args.max_attempts,
                'n_graphs': len(graphs),
                'teacher_names': [t.name for t in teachers],
            },
        }, output_path)
        logger.info("Saved %d trajectories to %s", total_trajs, output_path)


if __name__ == '__main__':
    main()

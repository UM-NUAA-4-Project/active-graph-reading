"""M7 教师轨迹生成：在任意 m7 registry 数据集上生成教师阅读轨迹。

与 m4/generate_trajectories.py（PPA 绑定：OGB CSV + dgl.load_graphs）不同，
本脚本数据完全来自 m7 datasets registry（get_dataset → ds.train_graphs 已在内存，
自环已由加载器处理）。教师与轨迹生成逻辑复用 m4（TrajectoryGenerator /
GradientTeacher / StructuralTeacher），正确性过滤发生在 m4 generate_all 内部
（只保留 is_correct=True，见 trajectory_generator.py:133）。

graph_idx 语义：ds.train_graphs 中的本地位置（0..n-1），与 train_bc.py / train_rl.py 一致。

用法:
  python code/m7/generate_trajectories.py --dataset cifar10 --root D:\\superpixels \\
      --classifier m7/checkpoint/classifier/finetune/best_cifar10_classifier.pt \\
      --teachers gradient structural-degree --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
_project_root = _repo_root.parent                   # 仓库根
sys.path.insert(0, str(_repo_root))
# 移除 code/m7 自身：dgl→torchdata 的 `import datasets` 不得解析到 code/m7/datasets
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import torch

from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from m4.trajectory_generator import TrajectoryGenerator
from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m7.datasets.registry import get_dataset


class _GraphList:
    """把 train_graphs + train_labels 包装成 m4 generate_all 需要的 dataset[i] -> (g, lbl)。"""

    def __init__(self, graphs, labels):
        self._graphs = graphs
        self._labels = labels

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, i):
        return self._graphs[i], self._labels[i]


def ratio_key(r: float) -> str:
    """0.05 -> '0_050'（对齐 m5 BUDGET_RATIOS 命名）。"""
    return f'{r:.3f}'.replace('.', '_')


def build_classifier(cls_ckpt: dict, spec, device) -> GraphClassifier:
    """按 checkpoint 自身 config 构建上界 GIN（同 train_rl.py 模式）。"""
    _cfg = cls_ckpt.get('config', {})
    clf = GraphClassifier(
        edge_feat_dim=_cfg.get('edge_feat_dim', spec.edge_feat_dim),
        n_classes=_cfg.get('n_classes', spec.n_classes),
        hidden_dim=_cfg.get('hidden_dim', 300),
        n_layers=_cfg.get('n_layers', 5),
        dropout=_cfg.get('dropout', 0.5),
    ).to(device)
    clf.load_state_dict(cls_ckpt['model_state_dict'], strict=True)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf


def main():
    parser = argparse.ArgumentParser(description='M7 教师轨迹生成（m7 registry 数据集）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--classifier', required=True,
                        help='上界 GIN checkpoint（相对 code/ 目录，或绝对路径）')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='预算档位（默认 spec.budget_options）')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='每图每教师选 top-n 个 seed')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='限制训练图数（0=全部；冒烟测试传小值）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--graph-batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='m7/data/trajectories',
                        help='输出目录（相对 code/ 或绝对路径）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None)
    spec = ds.spec
    ratios = args.budget_ratios or list(spec.budget_options)

    train_graphs = ds.train_graphs[:args.max_graphs] if args.max_graphs > 0 else ds.train_graphs
    train_labels = ds.train_labels[:len(train_graphs)]
    n_graphs = len(train_graphs)
    print(f"Graphs: {n_graphs} (of {len(ds.train_graphs)})")

    clf_path = Path(args.classifier)
    if not clf_path.is_absolute():
        clf_path = _repo_root / args.classifier
    cls_ckpt = torch.load(str(clf_path), map_location=device, weights_only=False)
    classifier = build_classifier(cls_ckpt, spec, device)
    print(f"Classifier: {clf_path} (edge_feat_dim={spec.edge_feat_dim}, "
          f"n_classes={spec.n_classes})")

    teachers = []
    for name in args.teachers:
        if name == 'gradient':
            teachers.append(GradientTeacher())
        elif name == 'structural-degree':
            teachers.append(StructuralTeacher(method='degree'))
        else:
            raise ValueError(f"未知教师 '{name}'，可用: gradient, structural-degree")
    print(f"Teachers: {[t.name for t in teachers]}")

    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=SubgraphBuilder(),
        device=str(device),
        graph_batch_size=args.graph_batch_size,
    )

    dataset = _GraphList(train_graphs, train_labels)
    train_idx = list(range(n_graphs))  # graph_idx = 本地位置

    output_dir = _repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ratio in ratios:
        key = ratio_key(ratio)
        print(f"\n=== reveal_ratio={ratio} ===")
        result = generator.generate_all(dataset, train_idx, ratio, n_seeds=args.n_seeds)
        n_correct = {t: sum(len(l) for l in result[t]) for t in args.teachers}
        summary[key] = n_correct
        for t, cnt in n_correct.items():
            print(f"  {t}: {cnt} correct trajectories")
        save_path = output_dir / f'trajectories_r{key}.pt'
        torch.save({'trajectories': result, 'config': {
            'dataset': args.dataset,
            'budget_ratio': ratio,
            'teachers': args.teachers,
            'n_seeds': args.n_seeds,
            'n_graphs': n_graphs,
        }}, str(save_path))
        print(f"  Saved: {save_path}")

    print("\n=== Yield summary (teacher x ratio) ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

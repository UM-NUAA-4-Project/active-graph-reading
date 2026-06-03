"""从 PPA 训练集每类随机抽 1 张图，保存为 M4 smoke test 样本。

Usage:
  python code/m4/sample_ppa_smoke.py --root D:\\ogb-master\\dataset
"""

import argparse
import sys
import random
from pathlib import Path
from collections import defaultdict

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import torch
import dgl
from tqdm import tqdm
from ogb.graphproppred import DglGraphPropPredDataset


def parse_args():
    p = argparse.ArgumentParser(description='Sample 1 graph per class from PPA train set')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root')
    p.add_argument('--output', type=str, default=None,
                   help='Output path (default: m4/smoke_sample.pt)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # 输出路径
    output_path = Path(args.output) if args.output \
        else _script_dir / 'm4' / 'smoke_sample.pt'

    # 加载 PPA 训练集
    print(f"Loading OGBG-PPA train split (root={args.root})...")
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split = dataset.get_idx_split()
    train_idx = split['train'].tolist()

    # 按类分组
    class_to_indices = defaultdict(list)
    for idx in tqdm(train_idx, desc="Grouping by class"):
        label = int(dataset.labels[int(idx)].item())
        class_to_indices[label].append(idx)

    n_classes = len(class_to_indices)
    print(f"Found {n_classes} classes, {len(train_idx)} total training graphs")

    # 每类随机抽 1 张
    sampled_indices = []
    for label in sorted(class_to_indices.keys()):
        idx_list = class_to_indices[label]
        chosen = random.choice(idx_list)
        sampled_indices.append(chosen)

    print(f"Sampled {len(sampled_indices)} graphs (1 per class)")

    # 加载图（加自环）
    graphs = []
    labels = []
    for idx in tqdm(sampled_indices, desc="Loading graphs"):
        g, lbl = dataset[int(idx)]
        g = dgl.add_self_loop(g)
        graphs.append(g)
        labels.append(int(lbl.item()))

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'graphs': graphs,
        'labels': labels,
        'indices': sampled_indices,
        'n_classes': n_classes,
        'seed': args.seed,
    }, output_path)
    print(f"Saved {len(graphs)} graphs to {output_path}")


if __name__ == '__main__':
    main()

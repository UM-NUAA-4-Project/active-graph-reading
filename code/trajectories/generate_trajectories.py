"""教师轨迹生成（任意注册数据集）。

自 m4/generate_trajectories.py（PPA 绑定）与 m7/generate_trajectories.py
（registry 驱动）合并泛化：
- 数据来自 datasets/registry（--dataset 切换）；
- 教师经 trajectories.teachers.build_teacher 工厂（gradient / structural-* /
  hybrid-a* / info-gain）；
- --labels 只生成指定类别的轨迹（多台服务器按类别分摊）；
- --k-shards / --shard-idx 分片：输出 trajectories_r{key}_shard{i}of{K}.pt，
  已存在的分片自动跳过 → 中断后续跑（重跑同一命令即可）；
- graph_idx 语义 = ds.train_graphs 本地位置（与 train_bc/train_rl 一致）。

用法:
  python code/trajectories/generate_trajectories.py --dataset cifar10 \
      --root D:\\superpixels --classifier result/cifar10/classifier/best_cifar10_classifier.pt \
      --teachers gradient structural-degree --device cuda:0
  # 只生成类别 0 3 的轨迹 + 分片
  ... --labels 0 3 --k-shards 4 --shard-idx 0 1 2 3
"""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from trajectories.core.generator import TrajectoryGenerator  # noqa: E402
from trajectories.core.rollout import build_classifier  # noqa: E402
from trajectories.teachers import build_teacher  # noqa: E402


class _GraphList:
    """把 train_graphs + train_labels 包装成 generate_all 需要的 dataset[i] -> (g, lbl)。"""

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


def main():
    parser = argparse.ArgumentParser(description='教师轨迹生成（registry 数据集）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--classifier', required=True,
                        help='上界分类器 checkpoint（相对 code/，或绝对路径）')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='预算档位（默认 spec.budget_options）')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='每图每教师选 top-n 个 seed')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                        help='只生成指定类别的轨迹（默认全部类别；'
                             '便于多台服务器按类别分摊）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='限制训练图数（0=全部；冒烟测试传小值）')
    parser.add_argument('--k-shards', type=int, default=1,
                        help='把图列表切成 K 片分批生成（断点续跑单位）')
    parser.add_argument('--shard-idx', type=int, nargs='+', default=None,
                        help='本次运行的分片号（默认 0..K-1 全部；已存在的'
                             '分片文件自动跳过实现断点续跑）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--graph-batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 data/{dataset}/trajectories）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None, splits=['train'],
                     max_graphs=args.max_graphs)
    spec = ds.spec
    ratios = args.budget_ratios or list(spec.budget_options)

    # 类别过滤（--labels）→ 保持原索引（graph_idx 语义不变）
    train_labels = ds.train_labels
    if args.labels:
        keep = [i for i in range(len(train_labels))
                if int(train_labels[i]) in set(args.labels)]
        print(f"Label filter {args.labels}: {len(keep)}/{len(train_labels)} graphs")
    else:
        keep = list(range(len(train_labels)))

    output_dir = (Path(args.output_dir) if args.output_dir
                  else CODE_DIR / 'data' / spec.name / 'trajectories')
    if not output_dir.is_absolute():
        output_dir = CODE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 分片 + 断点续跑：已完成分片跳过 ----
    k = max(1, args.k_shards)
    shard_ids = args.shard_idx if args.shard_idx is not None else list(range(k))
    shards = []            # (shard_id, [graph_idx...], output_suffix)
    for sid in shard_ids:
        part = keep[sid::k] if k > 1 else keep
        suffix = f'_shard{sid}of{k}' if k > 1 else ''
        missing = {r: output_dir / f'trajectories_r{ratio_key(r)}{suffix}.pt'
                   for r in ratios}
        if all(p.exists() for p in missing.values()):
            print(f"Shard {sid}: 全部档位已存在，跳过（断点续跑）")
            continue
        shards.append((sid, part, suffix))
    if not shards:
        print("所有分片均已完成，无需生成。")
        return

    # ---- 分类器（gradient 教师需要 autograd 图 → requires_grad=True）----
    clf_path = Path(args.classifier)
    if not clf_path.is_absolute():
        clf_path = CODE_DIR / args.classifier
    cls_ckpt = torch.load(str(clf_path), map_location=device, weights_only=False)
    classifier = build_classifier(cls_ckpt, device, requires_grad=True)
    print(f"Classifier: {clf_path} (edge_feat_dim={spec.edge_feat_dim}, "
          f"n_classes={spec.n_classes})")

    builder = SubgraphBuilder()
    teachers = [build_teacher(name, classifier=classifier, subgraph_builder=builder,
                              device=str(device)) for name in args.teachers]
    print(f"Teachers: {[t.name for t in teachers]}")

    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=builder,
        device=str(device),
        graph_batch_size=args.graph_batch_size,
    )

    summary = {}
    for sid, part, suffix in shards:
        dataset = _GraphList([ds.train_graphs[i] for i in part],
                             [train_labels[i] for i in part])
        part_idx = list(range(len(part)))     # 本地位置
        # trajectory.graph_idx = 全局本地位置（keep 中的原值）
        ogb_to_local = {g: l for l, g in enumerate(part)}
        for ratio in ratios:
            key = ratio_key(ratio)
            out_path = output_dir / f'trajectories_r{key}{suffix}.pt'
            if out_path.exists():
                print(f"  shard{sid} r={ratio}: 已存在，跳过")
                continue
            print(f"\n=== shard{sid} reveal_ratio={ratio} ({len(part)} graphs) ===")
            result = generator.generate_all(dataset, part_idx, ratio,
                                            n_seeds=args.n_seeds,
                                            ogb_to_local=ogb_to_local)
            n_correct = {t: sum(len(l) for l in result[t])
                         for t in result}
            summary[f'shard{sid}_{key}'] = n_correct
            for t, cnt in n_correct.items():
                print(f"  {t}: {cnt} correct trajectories")
            torch.save({'trajectories': result, 'config': {
                'dataset': args.dataset,
                'budget_ratio': ratio,
                'teachers': list(result.keys()),
                'n_seeds': args.n_seeds,
                'n_graphs': len(part),
                'labels': args.labels,
                'shard': {'idx': sid, 'k': k} if k > 1 else None,
            }}, str(out_path))
            print(f"  Saved: {out_path}")

    print("\n=== Yield summary (shard x ratio) ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

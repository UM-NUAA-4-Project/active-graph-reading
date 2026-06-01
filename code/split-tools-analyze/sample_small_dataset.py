#!/usr/bin/env python3
"""
从 OGBG-PPA 官方训练集中分层抽样生成小数据集
================================================

仅从 OGB train split（species split 的训练部分）抽样，遵守 OGB 的物种隔离
约定。在 OGB 训练集内逐类别分层：每个类别独立随机抽取 sample_frac 比例，
再按 split_ratios 划分 train/val/test。

用法:
  python split-tools-analyze/sample_small_dataset.py
  python split-tools-analyze/sample_small_dataset.py --sample 0.05 --split 0.7 0.15 0.15
  python split-tools-analyze/sample_small_dataset.py --out_dir dataset/ogbg_ppa/small

输出: {out_dir}/
  train.bin / val.bin / test.bin / meta.json
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.data.utils import save_graphs


def stratified_sample_and_split(labels, sample_frac, split_ratios, seed=42):
    """逐类别分层抽样 + 划分 train/val/test。

    对每个类别独立操作:
      1. 随机抽取 sample_frac 比例的样本
      2. 按 split_ratios 划分为 train/val/test
      3. 合并所有类别的结果

    保证: 每个类别在 train/val/test 中的比例完全一致,
          且每个子集必定包含所有类别 (只要该类别有足够样本)。

    Parameters
    ----------
    labels : np.ndarray  全量标签, shape=(n,).
    sample_frac : float  采样比例 (如 0.10)。
    split_ratios : list[float]  train/val/test 比例, sum=1。

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        全量数据集中的索引。
    """
    labels = np.asarray(labels).ravel()
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    assert abs(sum(split_ratios) - 1.0) < 1e-6

    train_parts, val_parts, test_parts = [], [], []
    class_stats = []

    for cls in unique_classes:
        # 该类别的全部索引
        cls_indices = np.where(labels == cls)[0]
        n_cls = len(cls_indices)

        # 随机打乱
        rng.shuffle(cls_indices)

        # 抽取 sample_frac
        n_sample = max(1, int(round(n_cls * sample_frac)))
        sampled = cls_indices[:n_sample]

        # 按 split_ratios 划分 (保证 train 和 val 至少各 1 个)
        n_train = max(1, int(round(n_sample * split_ratios[0])))
        n_val = max(1, int(round(n_sample * split_ratios[1])))
        n_test = n_sample - n_train - n_val
        if n_test < 1:
            # 样本太少时优先保证 train, 其次 val
            if n_sample >= 3:
                n_train = n_sample - 2
                n_val = 1
                n_test = 1
            elif n_sample == 2:
                n_train, n_val, n_test = 1, 1, 0
            else:
                n_train, n_val, n_test = 1, 0, 0

        train_parts.append(sampled[:n_train])
        if n_val > 0:
            val_parts.append(sampled[n_train:n_train + n_val])
        if n_test > 0:
            test_parts.append(sampled[n_train + n_val:])

        class_stats.append((int(cls), n_cls, n_sample, n_train, n_val, n_test))

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    # 验证
    assert len(train_idx) > 0, "train 集为空"
    assert len(val_idx) > 0, "val 集为空"
    assert len(test_idx) > 0, "test 集为空"
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(np.unique(all_idx)) == len(all_idx), "索引重叠!"

    return train_idx, val_idx, test_idx, class_stats


def save_subset(dataset, indices, path, desc):
    graphs = [dataset.graphs[i] for i in indices]
    labels = dataset.labels[indices]
    print(f"  保存 {desc}: {len(graphs):,} 图 -> {path}")
    save_graphs(path, graphs, labels={"labels": labels})
    size_mb = os.path.getsize(path) / 1e6
    print(f"    文件大小: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="从 OGBG-PPA 逐类别分层抽样生成小数据集"
    )
    parser.add_argument("--sample", type=float, default=0.10,
                        help="采样比例 (默认: 0.10 = 10%%)")
    parser.add_argument("--split", type=float, nargs=3, default=[0.60, 0.20, 0.20],
                        help="train/val/test 比例 (默认: 0.60 0.20 0.20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="dataset/ogbg_ppa/small",
                        help="输出目录 (默认: dataset/ogbg_ppa/small)")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有文件")
    args = parser.parse_args()

    ratios = [r / sum(args.split) for r in args.split]

    meta_path = os.path.join(args.out_dir, "meta.json")
    if os.path.exists(meta_path) and not args.force:
        with open(meta_path) as f:
            existing = json.load(f)
        print(f"小数据集已存在 ({args.out_dir}/)")
        print(f"  图形数: {existing.get('n_sampled')}")
        print(f"  train={existing['train_size']}  val={existing['val_size']}  test={existing['test_size']}")
        print(f"  {existing.get('created')}")
        print(f"  如需重新生成, 使用 --force")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  OGBG-PPA 逐类别分层抽样 → 小数据集")
    print(f"  采样: {args.sample:.0%}  |  train/val/test: {'/'.join(f'{r:.0%}' for r in ratios)}")
    print(f"  种子: {args.seed}")
    print("=" * 70)

    # ---- 加载全量 ----
    print(f"\n[1/3] 加载全量数据集 ...")
    dataset = DglGraphPropPredDataset(name="ogbg-ppa", root=r"D:\ogb-master\dataset")
    n_total = len(dataset)
    n_classes = int(dataset.num_classes)
    all_labels = dataset.labels.numpy().ravel()
    split = dataset.get_idx_split()

    # 打印类别分布概况
    _, full_counts = np.unique(all_labels, return_counts=True)
    print(f"  全量: {n_total:,} 图, {n_classes} 类")
    print(f"  每类样本数: min={full_counts.min()}, median={np.median(full_counts):.0f}, max={full_counts.max()}")

    # ---- 逐类别分层抽样 + 划分（仅从 OGB 训练集） ----
    print(f"\n[2/3] 逐类别分层抽样 {args.sample:.0%} + 划分")
    print(f"  [仅从 OGB train split ({len(split['train']):,} 图) 中抽样，遵守 species split]")
    ogb_train_idx = split['train'].numpy()
    ogb_train_labels = all_labels[ogb_train_idx]

    sampled_train_idx, val_idx, test_idx, class_stats = stratified_sample_and_split(
        ogb_train_labels, args.sample, ratios, seed=args.seed
    )
    # sampled_train_idx/val_idx/test_idx 是相对索引（在 ogb_train_labels 中的位置），
    # 映射回全量数据集的绝对索引
    train_idx = ogb_train_idx[sampled_train_idx]
    val_idx = ogb_train_idx[val_idx]
    test_idx = ogb_train_idx[test_idx]

    # 打印每个类别的抽样情况
    print(f"  {'class':>6s}  {'full':>7s}  {'sample':>7s}  {'train':>7s}  {'val':>7s}  {'test':>7s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for cls, n_full, n_sample, n_train, n_val, n_test in class_stats:
        flag = " !" if n_val == 0 or n_test == 0 else ""
        print(f"  {cls:>6d}  {n_full:>7,d}  {n_sample:>7,d}  {n_train:>7,d}  {n_val:>7,d}  {n_test:>7,d}{flag}")

    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]
    test_labels = all_labels[test_idx]

    print(f"\n  train: {n_train:,} 图, {len(np.unique(train_labels))} 类")
    print(f"  val:   {n_val:,} 图, {len(np.unique(val_labels))} 类")
    print(f"  test:  {n_test:,} 图, {len(np.unique(test_labels))} 类")
    print(f"  总计:  {n_train + n_val + n_test:,} 图")

    # ---- 保存 ----
    print(f"\n[3/3] 保存到 {args.out_dir}/ ...")
    save_subset(dataset, train_idx, os.path.join(args.out_dir, "train.bin"), "train")
    save_subset(dataset, val_idx, os.path.join(args.out_dir, "val.bin"), "val")
    save_subset(dataset, test_idx, os.path.join(args.out_dir, "test.bin"), "test")

    meta = {
        "n_total": int(n_total),
        "n_sampled": int(n_train + n_val + n_test),
        "sample_frac": args.sample,
        "split_ratios": ratios,
        "train_size": int(n_train),
        "val_size": int(n_val),
        "test_size": int(n_test),
        "num_classes": n_classes,
        "seed": args.seed,
        "source": "OGB train split only (species split respected)",
        "ogb_train_size": int(len(ogb_train_idx)),
        "created": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  meta.json 已保存")

    total_mb = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir) if f.endswith('.bin')
    ) / 1e6

    print(f"\n{'=' * 70}")
    print(f"  完成: {n_train+n_val+n_test:,} 图 ({total_mb:.0f} MB)")
    print(f"  train {n_train:,} | val {n_val:,} | test {n_test:,}")
    print(f"{'=' * 70}")

    del dataset
    gc.collect()


if __name__ == "__main__":
    main()

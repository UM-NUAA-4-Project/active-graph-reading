#!/usr/bin/env python3
"""
OGBG-PPA 数据预分割与持久化脚本
================================

一次性运行: 将 PPA 数据集按类别分层抽样划分为 k 个不相交训练子集,
并将验证集/测试集单独保存为 DGL 二进制文件。

支持任意份数的分割，训练脚本可按需加载指定子集，无需全量数据驻留内存。

内存变化:
  全量加载:  ~32 GB (158,100 图)
  按需加载:  ~5 GB (1 个训练子集) + ~9 GB (val) ≈ 14 GB (峰值)

用法:
  # 5 种预定义划分
  python split_and_save_ppa.py --preset 2fold    # 0.5, 0.5
  python split_and_save_ppa.py --preset 3fold    # 0.3, 0.3, 0.4 (默认)
  python split_and_save_ppa.py --preset 4fold    # 0.25, 0.25, 0.25, 0.25
  python split_and_save_ppa.py --preset 5fold    # 0.2, 0.2, 0.2, 0.2, 0.2
  python split_and_save_ppa.py --preset 10fold   # 10 × 0.1

  # 自定义比例
  python split_and_save_ppa.py --split_ratios 0.4 0.3 0.3

  # 指定输出目录
  python split_and_save_ppa.py --preset 3fold --out_dir dataset/ogbg_ppa/splits_3fold

  # 强制覆盖已有文件
  python split_and_save_ppa.py --preset 3fold --force

输出目录: {out_dir}/
  subset_0.bin   — 训练子集 0
  subset_1.bin   — 训练子集 1
  ...
  val.bin        — 验证集
  test.bin       — 测试集
  split_meta.json — 分割元信息 (类别数、比例、种子等)
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
import torch
from dgl.data.utils import save_graphs
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ogb.graphproppred import DglGraphPropPredDataset


# ============================================================================
# 预设划分
# ============================================================================

PRESETS = {
    '2fold':  [0.5, 0.5],
    '3fold':  [0.3, 0.3, 0.4],
    '4fold':  [0.25, 0.25, 0.25, 0.25],
    '5fold':  [0.2, 0.2, 0.2, 0.2, 0.2],
    '10fold': [0.1] * 10,
}


# ============================================================================
# 通用 k 份分层划分
# ============================================================================

def stratified_split_k(labels, ratios, seed=42):
    """将样本按类别比例分层划分为 k 个不相交子集。

    使用迭代 StratifiedShuffleSplit：每次从剩余样本中拆分出一份，
    保证每份的类别分布与原始分布一致。

    Parameters
    ----------
    labels : np.ndarray
        标签数组，shape=(n,)。
    ratios : list[float]
        k 个比例，应归一化到 sum=1。
    seed : int
        随机种子。

    Returns
    -------
    list[np.ndarray]
        k 个索引数组（指向原始 labels 的位置索引）。
    """
    labels = np.asarray(labels).ravel()
    n = len(labels)
    k = len(ratios)
    assert k >= 2, f"至少需要 2 份，当前 k={k}"
    assert abs(sum(ratios) - 1.0) < 1e-6, f"比例和应为 1.0，当前 {sum(ratios):.6f}"

    subsets = []
    remaining_idx = np.arange(n)
    remaining_labels = labels

    for i in range(k - 1):
        # 当前份占剩余样本的比例
        remaining_ratio_sum = sum(ratios[i:])
        split_frac = ratios[i] / remaining_ratio_sum

        # split_frac 是"分出去"的比例 → 作为 test_size
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=split_frac, random_state=seed + i
        )
        # sss.split() 返回 (train=keep, test=split_off)
        idx_keep_local, idx_split_local = next(
            sss.split(np.zeros(len(remaining_idx)), remaining_labels)
        )

        subsets.append(remaining_idx[idx_split_local])

        remaining_idx = remaining_idx[idx_keep_local]
        remaining_labels = remaining_labels[idx_keep_local]

    # 最后一份就是剩余的全部
    subsets.append(remaining_idx)

    # 验证
    all_idx = np.concatenate(subsets)
    assert len(all_idx) == n, f"子集并集大小 {len(all_idx)} != {n}"
    assert len(set(all_idx)) == n, "子集存在重叠!"

    return subsets


def save_subset(graphs, labels, path, desc):
    """保存一组图+标签到 DGL 二进制文件。"""
    print(f"  保存 {desc}: {len(graphs):,} 图 -> {path}")
    save_graphs(path, graphs, labels={"labels": labels})
    size_gb = os.path.getsize(path) / 1e9
    print(f"    文件大小: {size_gb:.2f} GB")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OGBG-PPA 数据预分割 (支持 k 份分层抽样)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="预设: " + ", ".join(PRESETS.keys()),
    )

    # 两种指定比例的方式（互斥）
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--split_ratios", type=float, nargs='+',
        default=[0.3, 0.3, 0.4],
        help="训练子集的比例 (空格分隔, 自动归一化, 默认: 0.3 0.3 0.4)",
    )
    group.add_argument(
        "--preset", type=str, choices=list(PRESETS.keys()),
        help=f"预定义比例: {', '.join(f'{k}={PRESETS[k]}' for k in PRESETS)}",
    )

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--out_dir", type=str, default="dataset/ogbg_ppa/splits",
        help="输出目录 (默认: dataset/ogbg_ppa/splits)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制覆盖已有分割文件",
    )
    args = parser.parse_args()

    # ---- 解析比例 ----
    if args.preset:
        ratios_raw = PRESETS[args.preset]
    else:
        ratios_raw = args.split_ratios

    # 归一化
    total = sum(ratios_raw)
    ratios = [r / total for r in ratios_raw]
    k = len(ratios)

    print(f"  预设: {args.preset or '自定义'}")
    print(f"  原始比例: {ratios_raw}")
    print(f"  归一化:   {[f'{r:.4f}' for r in ratios]}")
    print(f"  份数 k:   {k}")

    # ---- 检查是否已有分割文件 ----
    meta_path = os.path.join(args.out_dir, "split_meta.json")
    if os.path.exists(meta_path) and not args.force:
        with open(meta_path) as f:
            existing = json.load(f)
        print(f"\n分割文件已存在 ({args.out_dir}/)")
        print(f"  已有比例: {existing.get('ratios')}")
        print(f"  创建时间: {existing.get('created')}")
        print(f"  如需重新分割, 请使用 --force")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载全量数据 (约 32 GB, 需要充足内存) ----
    print("=" * 70)
    print("  OGBG-PPA 数据预分割")
    print(f"  k={k}, 比例: {[f'{r:.2%}' for r in ratios]}")
    print(f"  种子: {args.seed}")
    print("=" * 70)
    print(f"\n[1/3] 加载全量数据集 (约 32 GB, 请确保内存充足) ...")

    dataset = DglGraphPropPredDataset(name="ogbg-ppa", root='D:\ogb-master\dataset')
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]
    num_classes = int(dataset.num_classes)

    print(f"  图数: {len(dataset):,}  |  类别: {num_classes}")
    print(f"  训练: {len(train_idx):,}  |  验证: {len(val_idx):,}  "
          f"|  测试: {len(test_idx):,}")

    # ---- 分层划分 ----
    print(f"\n[2/3] 分层划分训练集 (k={k}) ...")
    train_labels = dataset.labels[train_idx].numpy().ravel()
    subset_indices = stratified_split_k(train_labels, ratios=ratios, seed=args.seed)

    train_idx_np = train_idx.numpy()
    subset_sizes = []
    for i, local_idx in enumerate(subset_indices):
        global_idx = train_idx_np[local_idx]
        subset_labels = train_labels[local_idx]
        unique, counts = np.unique(subset_labels, return_counts=True)
        subset_sizes.append(len(global_idx))
        print(f"  子集 {i}: {len(global_idx):,} 图, {len(unique)} 类, "
              f"每类 [{counts.min()}, {counts.max()}]")

    # ---- 保存各子集 ----
    print(f"\n[3/3] 保存分割文件 ...")

    # 保存 k 个训练子集
    for i, local_idx in enumerate(subset_indices):
        global_idx = train_idx_np[local_idx]
        subset_graphs = [dataset.graphs[j] for j in global_idx]
        subset_labels = dataset.labels[global_idx]
        path = os.path.join(args.out_dir, f"subset_{i}.bin")
        save_subset(subset_graphs, subset_labels, path, f"训练子集 {i}")
        del subset_graphs, subset_labels
        gc.collect()

    # 保存验证集
    val_graphs = [dataset.graphs[j] for j in val_idx.numpy()]
    val_labels = dataset.labels[val_idx]
    save_subset(val_graphs, val_labels, os.path.join(args.out_dir, "val.bin"), "验证集")
    del val_graphs, val_labels
    gc.collect()

    # 保存测试集
    test_graphs = [dataset.graphs[j] for j in test_idx.numpy()]
    test_labels = dataset.labels[test_idx]
    save_subset(test_graphs, test_labels, os.path.join(args.out_dir, "test.bin"), "测试集")
    del test_graphs, test_labels
    gc.collect()

    # 保存元信息
    meta = {
        "k": k,
        "num_classes": num_classes,
        "ratios": ratios,
        "ratios_raw": ratios_raw,
        "preset": args.preset,
        "seed": args.seed,
        "train_total": int(len(train_idx)),
        "subset_sizes": subset_sizes,
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "created": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  保存元信息 -> {meta_path}")

    # ---- 最终报告 ----
    print(f"\n{'=' * 70}")
    print(f"  分割完成!  (k={k}, {args.out_dir})")
    print(f"{'=' * 70}")
    total_disk = 0
    for fname in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, fname)
        size_gb = os.path.getsize(fpath) / 1e9
        total_disk += size_gb
        print(f"  {fname:30s}  {size_gb:6.2f} GB")
    print(f"  {'合计':30s}  {total_disk:6.2f} GB")

    # ---- 释放全量数据集 ----
    del dataset
    gc.collect()
    print(f"\n  全量数据集已释放。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OGBG-PPA 数据预分割与持久化脚本
================================

一次性运行: 将 PPA 数据集按类别分层抽样划分为 3 个不相交训练子集,
并将验证集/测试集单独保存为 DGL 二进制文件。

训练脚本可按需加载指定子集, 无需全量数据驻留内存。

内存变化:
  全量加载:  ~32 GB (158,100 图)
  按需加载:  ~5 GB (1 个训练子集) + ~9 GB (val) ≈ 14 GB (峰值)

用法:
  # 默认 3:3:4 分层抽样
  python split_and_save_ppa.py

  # 自定义比例
  python split_and_save_ppa.py --split_ratios 0.3 0.3 0.4

  # 强制覆盖已有文件
  python split_and_save_ppa.py --force

输出目录: dataset/ogbg_ppa/splits/
  subset_0.bin   — 训练子集 0 (30%)
  subset_1.bin   — 训练子集 1 (30%)
  subset_2.bin   — 训练子集 2 (40%)
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


def stratified_split(labels, ratios=(0.3, 0.3, 0.4), seed=42):
    """将样本按类别比例分层划分为 3 个不相交子集。"""
    labels = np.asarray(labels).ravel()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratios[0], random_state=seed)
    idx_0, idx_rest = next(sss1.split(np.zeros(len(labels)), labels))

    rest_ratio = ratios[1] / (ratios[1] + ratios[2])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - rest_ratio, random_state=seed + 1)
    labels_rest = labels[idx_rest]
    idx_rest_local_1, idx_rest_local_2 = next(sss2.split(np.zeros(len(idx_rest)), labels_rest))

    idx_1 = idx_rest[idx_rest_local_1]
    idx_2 = idx_rest[idx_rest_local_2]

    return [idx_0, idx_1, idx_2]


def save_subset(graphs, labels, path, desc):
    """保存一组图+标签到 DGL 二进制文件。"""
    print(f"  保存 {desc}: {len(graphs):,} 图 → {path}")
    save_graphs(path, graphs, labels={"labels": labels})
    size_gb = os.path.getsize(path) / 1e9
    print(f"    文件大小: {size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="OGBG-PPA 数据预分割")
    parser.add_argument("--split_ratios", type=float, nargs=3, default=[0.3, 0.3, 0.4],
                        help="三个训练子集的比例 (默认: 0.3 0.3 0.4)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out_dir", type=str, default="dataset/ogbg_ppa/splits",
                        help="输出目录")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有分割文件")
    args = parser.parse_args()

    ratios = [r / sum(args.split_ratios) for r in args.split_ratios]

    # ---- 检查是否已有分割文件 ----
    meta_path = os.path.join(args.out_dir, "split_meta.json")
    if os.path.exists(meta_path) and not args.force:
        with open(meta_path) as f:
            existing = json.load(f)
        print(f"分割文件已存在 ({args.out_dir}/)")
        print(f"  已有比例: {existing.get('ratios')}")
        print(f"  创建时间: {existing.get('created')}")
        print(f"  如需重新分割, 请使用 --force")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载全量数据 (约 32 GB, 需要充足内存) ----
    print("=" * 70)
    print("  OGBG-PPA 数据预分割")
    print(f"  子集比例: {ratios[0]:.1%} : {ratios[1]:.1%} : {ratios[2]:.1%}")
    print(f"  种子: {args.seed}")
    print("=" * 70)
    print(f"\n[1/3] 加载全量数据集 (约 32 GB, 请确保内存充足) ...")

    dataset = DglGraphPropPredDataset(name="ogbg-ppa")
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]
    num_classes = int(dataset.num_classes)

    print(f"  图数: {len(dataset):,}  |  类别: {num_classes}")
    print(f"  训练: {len(train_idx):,}  |  验证: {len(val_idx):,}  |  测试: {len(test_idx):,}")

    # ---- 分层划分 ----
    print(f"\n[2/3] 分层划分训练集 ...")
    train_labels = dataset.labels[train_idx].numpy().ravel()
    subset_indices = stratified_split(train_labels, ratios=ratios, seed=args.seed)

    train_idx_np = train_idx.numpy()
    subset_sizes = []
    for i, local_idx in enumerate(subset_indices):
        global_idx = train_idx_np[local_idx]
        subset_labels = train_labels[local_idx]
        unique, counts = np.unique(subset_labels, return_counts=True)
        subset_sizes.append(len(global_idx))
        print(f"  子集 {i}: {len(global_idx):,} 图, {len(unique)} 类, "
              f"每类 [{counts.min()}, {counts.max()}]")

    # 完整性校验
    all_indices = np.concatenate([train_idx_np[si] for si in subset_indices])
    assert len(all_indices) == len(train_idx), "子集并集 != 全训练集"
    assert len(set(all_indices)) == len(all_indices), "子集存在重叠!"

    # ---- 保存各子集 ----
    print(f"\n[3/3] 保存分割文件 ...")

    # 保存 3 个训练子集
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
        "num_classes": num_classes,
        "ratios": ratios,
        "seed": args.seed,
        "train_total": len(train_idx),
        "subset_sizes": subset_sizes,
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "created": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  保存元信息 → {meta_path}")

    # ---- 最终报告 ----
    print(f"\n{'=' * 70}")
    print(f"  分割完成!")
    print(f"{'=' * 70}")
    total_disk = 0
    for fname in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, fname)
        size_gb = os.path.getsize(fpath) / 1e9
        total_disk += size_gb
        print(f"  {fname:25s}  {size_gb:6.2f} GB")
    print(f"  {'合计':25s}  {total_disk:6.2f} GB")
    print(f"\n  训练时按需加载, 峰值内存约:")
    print(f"    训练阶段: 1 个子集 (~{subset_sizes[0]*32//158100} GB) + val (~9 GB) ≈ 14 GB")
    print(f"    评估阶段: val (~9 GB) 或 test (~7 GB) ≈ 9 GB")
    print(f"    对比全量: 32 GB → 节省约 55%")

    # ---- 释放全量数据集 ----
    del dataset
    gc.collect()
    print(f"\n  全量数据集已释放。")


if __name__ == "__main__":
    main()

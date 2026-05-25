#!/usr/bin/env python3
"""
OGBG-PPA 数据集分析脚本
=======================

分析 OGB Graph Property Prediction benchmark 中的 PPA (Protein-Protein Association) 数据集。

PPA 数据集背景:
  - 来自 1,581 个物种的蛋白质关联网络
  - 任务是预测蛋白质关联网络属于哪个物种（37 类多分类问题）
  - 图: 每个图代表一个物种的 PPI 网络（蛋白质关联网络）
  - 节点: 蛋白质
  - 边: 蛋白质之间的关联关系
  - 边特征: 7 维 one-hot 编码（表示 7 种不同的关联类型）

依赖:
  - ogb
  - torch
  - numpy
  - matplotlib (可选，用于绘图)
"""

import sys
import os
import numpy as np
import torch

# 将项目根目录加入 path，确保能导入 ogb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

# ============================================================
# 0. 配置
# ============================================================
DATASET_NAME = "ogbg-ppa"
ROOT_DIR = "dataset"  # 数据存储根目录
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print(f"  OGBG-PPA 数据集分析")
    print("=" * 70)

    # ============================================================
    # 1. 加载数据集
    # ============================================================
    print("\n[1/7] 正在加载 OGBG-PPA 数据集...")
    print("       提示: 首次运行会自动下载数据集,请稍候。")
    dataset = PygGraphPropPredDataset(name = 'ogbg-ppa') 
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"]   # numpy array
    valid_idx = split_idx["valid"]   # numpy array
    test_idx = split_idx["test"]     # numpy array

    num_graphs = len(dataset)

    print(f"       数据集加载完毕，共 {num_graphs} 个图。")
    print(f"       元信息: task_type={dataset.task_type}, "
          f"num_classes={dataset.num_classes}, "
          f"num_tasks={dataset.num_tasks}, "
          f"eval_metric={dataset.eval_metric}")

    # ============================================================
    # 2. 遍历数据集，收集统计信息
    # ============================================================
    print("\n[2/7] 遍历所有图，收集节点数/边数/特征信息...")

    node_counts = np.zeros(num_graphs, dtype=np.int32)
    edge_counts = np.zeros(num_graphs, dtype=np.int32)  # 记录原始边数（去重前）

    all_node_feat_none = True   # 是否所有图的 node_feat 都是 None
    all_node_feat_empty = True  # 是否所有图的 node_feat 都是空张量
    node_feat_dim = 0           # 节点特征维度（首个非空图）

    all_edge_feat_none = True   # 是否所有图的 edge_feat 都是 None
    edge_feat_dim = 0           # 边特征维度（首个非空图）

    for i in range(num_graphs):
        data = dataset[i]  # graph 是 dict, label 是 numpy array

        # ---------- 节点数 ----------
        n_nodes = data.num_nodes
        node_counts[i] = n_nodes
        # ---------- 边数（原始边数 = edge_index.shape[1] / 2, 因为 add_inverse_edge=True） ----------
        n_edges_raw = data.edge_index.shape[1] // 2
        edge_counts[i] = n_edges_raw

        # ---------- 节点特征 ----------
        node_feat = data.x
        if node_feat is not None:
            all_node_feat_none = False
            if isinstance(node_feat, torch.Tensor) and node_feat.numel() > 0:
                all_node_feat_empty = False
                if node_feat_dim == 0:
                    node_feat_dim = node_feat.shape[1]

        # ---------- 边特征 ----------
        edge_feat = data.edge_attr
        if edge_feat is not None:
            all_edge_feat_none = False
            if edge_feat_dim == 0 and isinstance(edge_feat, torch.Tensor) and edge_feat.numel() > 0:
                edge_feat_dim = edge_feat.shape[1]

    print(f"       遍历完成。共处理 {num_graphs} 个图。")

    # ============================================================
    # 3. 节点数分布统计
    # ============================================================
    print("\n[3/7] 节点数分布统计")
    print("-" * 50)
    print(f"  最小值     : {node_counts.min():>8,}")
    print(f"  最大值     : {node_counts.max():>8,}")
    print(f"  均值       : {node_counts.mean():>12,.2f}")
    print(f"  中位数     : {np.median(node_counts):>12,.0f}")
    print(f"  标准差     : {node_counts.std():>12,.2f}")

    # 分位数
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q_values = np.quantile(node_counts, quantiles)
    print(f"\n  分位数:")
    for q, v in zip(quantiles, q_values):
        print(f"    {q:>5.0%}  : {v:>10,.0f}")

    # ============================================================
    # 4. 边数分布统计
    # ============================================================
    print("\n[4/7] 边数分布统计（原始边数，不含反向边）")
    print("-" * 50)
    print(f"  最小值     : {edge_counts.min():>8,}")
    print(f"  最大值     : {edge_counts.max():>8,}")
    print(f"  均值       : {edge_counts.mean():>12,.2f}")
    print(f"  中位数     : {np.median(edge_counts):>12,.0f}")
    print(f"  标准差     : {edge_counts.std():>12,.2f}")

    eq_values = np.quantile(edge_counts, quantiles)
    print(f"\n  分位数:")
    for q, v in zip(quantiles, eq_values):
        print(f"    {q:>5.0%}  : {v:>10,.0f}")

    # ============================================================
    # 5. 节点特征检查
    # ============================================================
    print("\n[5/7] 节点特征 (node_feat) 检查")
    print("-" * 50)
    if all_node_feat_none:
        print("  [结论] 节点特征缺失: 所有图的 node_feat 均为 None")
        print("  [判定] 节点特征: 缺失")
    elif all_node_feat_empty:
        print("  [结论] 节点特征为空: 所有图的 node_feat 均为空张量 (size=0)")
        print(f"  [判定] 节点特征: 缺失（空张量）")
    else:
        print(f"  [结论] 节点特征存在: 维度 = {node_feat_dim}")
        print(f"  [判定] 节点特征: 存在")
        # 采样几个图看看节点特征的具体情况
        sample_feat = dataset[0].x
        print(f"  [示例] 第 1 个图的 node_feat dtype={sample_feat.dtype}, "
              f"min={sample_feat.min():.4f}, max={sample_feat.max():.4f}")

    # ============================================================
    # 6. 边特征检查
    # ============================================================
    print("\n[6/7] 边特征 (edge_feat) 检查")
    print("-" * 50)
    if all_edge_feat_none:
        print("  [结论] 边特征缺失: 所有图的 edge_feat 均为 None")
        print("  [判定] 边特征: 缺失")
    else:
        print(f"  [结论] 边特征存在: 维度 = {edge_feat_dim}")
        # 采样几个图看看边特征的具体情况
        sample_ef = dataset[0].edge_attr
        print(f"  [示例] 第 1 个图的 edge_feat shape={sample_ef.shape}, "
              f"dtype={sample_ef.dtype}")
        print(f"         值分布: min={sample_ef.min():.4f}, max={sample_ef.max():.4f}")
        # 判断是否是 one-hot 编码
        row_sums = sample_ef.sum(axis=1)
        unique_sums = np.unique(row_sums)
        if len(unique_sums) <= 3:  # one-hot sum = 1, 或全零+全1
            print(f"  [推测] 边特征每行之和的 unique 值: {unique_sums}, 疑似 one-hot / 指示特征")
        else:
            print(f"  [推测] 边特征每行之和的 unique 值较多: {len(unique_sums)} 种")

    # ============================================================
    # 7. 标签分析
    # ============================================================
    print("\n[7/7] 标签 (label) 分析")
    print("-" * 50)

    labels = dataset.y  # numpy array, shape (num_graphs, 1) for single-task multiclass
    print(f"  标签 shape  : {labels.shape}")
    print(f"  标签 dtype  : {labels.dtype}")
    print(f"  任务类型    : {dataset.task_type}")
    print(f"  类别数量    : {dataset.num_classes}")
    print(f"  评估指标    : {dataset.eval_metric}")

    # 如果是多类别单标签, label 值范围应该是 [0, num_classes-1]
    unique_labels = np.unique(labels)
    print(f"  唯一标签值  : {unique_labels[:20]}{'...' if len(unique_labels) > 20 else ''} "
          f"(共 {len(unique_labels)} 个)")

    # 类别分布
    label_counts = {}
    for label in labels.ravel():
        lbl = int(label)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    sorted_labels = sorted(label_counts.items())

    print(f"\n  各类别样本数 (按类别编号排序):")
    for lbl, cnt in sorted_labels:
        pct = cnt / num_graphs * 100
        bar = "█" * int(pct / 2)  # 简易条形图
        print(f"    Class {lbl:>3d}: {cnt:>6d} ({pct:5.2f}%) {bar}")

    # 不平衡度统计
    counts_array = np.array([c for _, c in sorted_labels])
    print(f"\n  类别不平衡统计:")
    print(f"    最多样本类别: Class {np.argmax(counts_array)} ({counts_array.max()} 图)")
    print(f"    最少样本类别: Class {np.argmin(counts_array)} ({counts_array.min()} 图)")
    print(f"    imbalance ratio (max/min): {counts_array.max() / counts_array.min():.2f}")

    # ============================================================
    # 8. 数据集划分统计
    # ============================================================
    print("\n" + "=" * 70)
    print("  数据集划分统计 (Species Split)")
    print("=" * 70)
    print(f"  训练集 (train): {len(train_idx):>8,d} 个图  ({len(train_idx)/num_graphs*100:5.1f}%)")
    print(f"  验证集 (valid): {len(valid_idx):>8,d} 个图  ({len(valid_idx)/num_graphs*100:5.1f}%)")
    print(f"  测试集 (test):  {len(test_idx):>8,d} 个图  ({len(test_idx)/num_graphs*100:5.1f}%)")
    print(f"  总计:           {num_graphs:>8,d} 个图")
    print(f"  划分方式:       species split (按物种划分，不同物种的图分到不同集合)")

    # ============================================================
    # 9. 结论
    # ============================================================
    print("\n" + "=" * 70)
    print("  核心结论")
    print("=" * 70)

    # 结论 1: 边特征
    if all_edge_feat_none:
        edge_conclusion = f"OGBG-PPA 不具有边特征"
    else:
        edge_conclusion = f"OGBG-PPA 具有边特征: 是, 维度 = {edge_feat_dim}"
    print(f"\n  结论 1: {edge_conclusion}")

    # 结论 2: 节点特征
    if all_node_feat_none or all_node_feat_empty:
        node_conclusion = "节点特征缺失: 是 (所有图的 node_feat 均为 None 或空张量)"
    else:
        node_conclusion = f"节点特征缺失: 否 (节点特征存在, 维度 = {node_feat_dim})"
    print(f"  结论 2: {node_conclusion}")

    # ============================================================
    # 10. 汇总报告摘要
    # ============================================================
    summary = f"""
{'='*70}
  OGBG-PPA 数据集分析报告  |  Summary
{'='*70}

数据集基本信息:
  - 名称           : ogbg-ppa (Protein-Protein Association)
  - 总图数         : {num_graphs:,}
  - 任务类型        : {dataset.task_type}
  - 类别数          : {dataset.num_classes}
  - 评估指标        : {dataset.eval_metric}

图结构统计:
  - 节点数           : min={node_counts.min():,}, max={node_counts.max():,},
                       mean={node_counts.mean():,.1f}, median={np.median(node_counts):,.0f}
  - 边数（原始）     : min={edge_counts.min():,}, max={edge_counts.max():,},
                       mean={edge_counts.mean():,.1f}, median={np.median(edge_counts):,.0f}

特征信息:
  - 节点特征        : {'缺失' if (all_node_feat_none or all_node_feat_empty) else f'存在 (维度={node_feat_dim})'}
  - 边特征          : {'缺失' if all_edge_feat_none else f'存在 (维度={edge_feat_dim})'}

数据集划分 (Species Split):
  - 训练集          : {len(train_idx):,} 图 ({len(train_idx)/num_graphs*100:.1f}%)
  - 验证集          : {len(valid_idx):,} 图 ({len(valid_idx)/num_graphs*100:.1f}%)
  - 测试集          : {len(test_idx):,} 图 ({len(test_idx)/num_graphs*100:.1f}%)

类别不平衡:
  - max/min ratio   : {counts_array.max() / counts_array.min():.2f}

核心结论:
  1. {edge_conclusion}
  2. {node_conclusion}

{'='*70}
"""
    print(summary)

    # 写入报告文件
    report_path = os.path.join(OUTPUT_DIR, "ppa_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[报告] 已保存至: {os.path.abspath(report_path)}")

    # ============================================================
    # 11. 生成可视化图表 (matplotlib)
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无 GUI 后端
        import matplotlib.pyplot as plt

        print("\n[绘图] 正在生成可视化图表...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("OGBG-PPA Dataset Analysis", fontsize=16, fontweight="bold")

        # (a) 节点数直方图
        ax1 = axes[0, 0]
        ax1.hist(node_counts, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
        ax1.axvline(node_counts.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={node_counts.mean():.0f}")
        ax1.axvline(np.median(node_counts), color="orange", linestyle="--", linewidth=1.5, label=f"Median={np.median(node_counts):.0f}")
        ax1.set_title("Node Count Distribution")
        ax1.set_xlabel("Number of Nodes")
        ax1.set_ylabel("Frequency")
        ax1.legend(fontsize=8)

        # (b) 边数直方图
        ax2 = axes[0, 1]
        ax2.hist(edge_counts, bins=50, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax2.axvline(edge_counts.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={edge_counts.mean():.0f}")
        ax2.axvline(np.median(edge_counts), color="orange", linestyle="--", linewidth=1.5, label=f"Median={np.median(edge_counts):.0f}")
        ax2.set_title("Edge Count Distribution (raw, excl. inverse)")
        ax2.set_xlabel("Number of Edges")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)

        # (c) 节点数 vs 边数 散点图
        ax3 = axes[0, 2]
        ax3.scatter(node_counts, edge_counts, s=3, alpha=0.4, color="#673AB7")
        ax3.set_xlabel("Number of Nodes")
        ax3.set_ylabel("Number of Edges")
        ax3.set_title("Nodes vs Edges per Graph")

        # (d) 类别分布
        ax4 = axes[1, 0]
        lbls = [l for l, _ in sorted_labels]
        cnts_arr = np.array([c for _, c in sorted_labels])
        ax4.bar(lbls, cnts_arr, color="#FF9800", edgecolor="white", alpha=0.85)
        ax4.set_title("Class Distribution (37 species)")
        ax4.set_xlabel("Class ID")
        ax4.set_ylabel("Number of Graphs")
        ax4.axhline(y=num_graphs/37, color="red", linestyle="--", linewidth=1, label=f"Uniform ({num_graphs/37:.0f})")
        ax4.legend(fontsize=8)

        # (e) 划分饼图
        ax5 = axes[1, 1]
        sizes = [len(train_idx), len(valid_idx), len(test_idx)]
        labels_pie = ["Train", "Valid", "Test"]
        colors_pie = ["#2196F3", "#FF9800", "#4CAF50"]
        ax5.pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%",
                startangle=90, explode=(0.02, 0.02, 0.02))
        ax5.set_title("Dataset Split (Species)")

        # (f) 节点数 CDF
        ax6 = axes[1, 2]
        sorted_nodes = np.sort(node_counts)
        cdf = np.arange(1, len(sorted_nodes) + 1) / len(sorted_nodes)
        ax6.plot(sorted_nodes, cdf, color="#E91E63", linewidth=1.5)
        ax6.set_xlabel("Number of Nodes")
        ax6.set_ylabel("CDF")
        ax6.set_title("Node Count CDF")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "ppa_analysis_plots.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[绘图] 图表已保存至: {os.path.abspath(fig_path)}")
    except ImportError:
        print("[绘图] matplotlib 未安装，跳过图表生成。")
        print("       安装: pip install matplotlib")

    print("\n" + "=" * 70)
    print("  分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

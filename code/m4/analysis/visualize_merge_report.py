"""可视化 merge_report.json — 预算标签分布。

用法:
  python code/m4/analysis/visualize_merge_report.py
    --report code/m4/data/stage6_merged/merge_report.json
    --output-dir code/m4/analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N_CLASSES = 37
BUDGET_KEYS = ["0_050", "0_100", "0_150", "0_200"]
BUDGET_LABELS = {"0_050": "0.05", "0_100": "0.10", "0_150": "0.15", "0_200": "0.20"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=str,
                   default='code/m4/data/stage6_merged/merge_report.json')
    p.add_argument('--output-dir', type=str, default='code/m4/analysis')
    return p.parse_args()


def plot_per_budget(report: dict, output_path: Path):
    """图1: 四个预算下的标签分布（4 子图）。"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    labels = list(range(N_CLASSES))
    colors = ['#5B9BD5' if l in (1,2,3,4,6,12,17,18,21,23,24,33,36) else '#ED7D31'
              for l in labels]  # blue=regular, orange=scarce

    for i, bk in enumerate(BUDGET_KEYS):
        ax = axes[i]
        dist = report['per_budget'][bk]['label_distribution']
        counts = [dist.get(str(l), 0) for l in labels]

        bars = ax.bar(labels, counts, color=colors, edgecolor='white', width=0.8)
        ax.set_title(f'Budget = {BUDGET_LABELS[bk]}', fontsize=12)
        ax.set_xlabel('Class')
        ax.set_ylabel('Trajectories')
        ax.set_xticks(labels[::2])
        ax.set_xticklabels(labels[::2], fontsize=6)

        total = sum(counts)
        mean_c = np.mean([c for c in counts if c > 0])
        ax.axhline(y=mean_c, color='gray', linestyle='--', linewidth=0.7)
        ax.text(0.98, 0.92, f'n={total:,}\nμ={mean_c:.0f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Per-Budget Label Distribution', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_combined(report: dict, output_path: Path):
    """图2: 3-budget (0.05+0.10+0.15) vs 4-budget (全) 的标签分布对比。"""
    fig, ax = plt.subplots(figsize=(18, 6))

    labels = list(range(N_CLASSES))
    x = np.arange(len(labels))
    width = 0.35

    # 3-budget: 0.05 + 0.10 + 0.15
    budget_3 = ["0_050", "0_100", "0_150"]
    counts_3 = np.zeros(N_CLASSES, dtype=int)
    for bk in budget_3:
        dist = report['per_budget'][bk]['label_distribution']
        for l in range(N_CLASSES):
            counts_3[l] += dist.get(str(l), 0)

    # 4-budget: all
    counts_4 = np.zeros(N_CLASSES, dtype=int)
    for bk in BUDGET_KEYS:
        dist = report['per_budget'][bk]['label_distribution']
        for l in range(N_CLASSES):
            counts_4[l] += dist.get(str(l), 0)

    bars1 = ax.bar(x - width/2, counts_3, width, label=f'0.05+0.10+0.15 (n={counts_3.sum():,})',
                   color='#5B9BD5', edgecolor='white')
    bars2 = ax.bar(x + width/2, counts_4, width, label=f'0.05+0.10+0.15+0.20 (n={counts_4.sum():,})',
                   color='#ED7D31', edgecolor='white')

    ax.set_xlabel('Class')
    ax.set_ylabel('Trajectories')
    ax.set_title('Combined Label Distribution: 3 Budgets vs 4 Budgets', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=10)

    # 标注差值
    diffs = counts_4 - counts_3
    for i in range(N_CLASSES):
        if diffs[i] > 0:
            ax.text(i, counts_4[i] + max(counts_4)*0.01, f'+{diffs[i]}',
                    ha='center', fontsize=5, color='#ED7D31')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.report) as f:
        report = json.load(f)

    plot_per_budget(report, output_dir / 'stage6_per_budget_dist.png')
    plot_combined(report, output_dir / 'stage6_combined_dist.png')

    print("Done.")


if __name__ == '__main__':
    main()

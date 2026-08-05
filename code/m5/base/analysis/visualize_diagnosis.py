"""diagnosis.json 可视化脚本 — 生成 4 张诊断图。

图片:
  1. difficulty_distribution.png  左: 难度饼状图, 右: 难度计数柱状图
  2. entropy_curve.png            每步决策熵折线图
  3. class_diagnosis.png          逐类成功率柱状图
  4. training_curves.png          三子图: train loss / avg reward / accuracy

用法:
  python code/m5/base/analysis/visualize_diagnosis.py \
    --diagnosis code/m5/base/rl/asymmetric/ex2/diagnosis/diagnosis.json \
    --training-log code/m5/base/rl/asymmetric/ex2/training_log.json \
    --output-dir code/m5/base/rl/asymmetric/ex2/diagnosis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIFF_COLORS = {'easy': '#7CB342', 'middle': '#FFB300', 'hard': '#E53935'}
SERIES = [
    ('train_loss', '#D32F2F', 'Train Loss'),
    ('train_avg_R', '#388E3C', 'Train Avg Reward'),
    ('train_acc', '#1976D2', 'Train Accuracy'),
]


def parse_args():
    p = argparse.ArgumentParser(description='diagnosis.json 可视化')
    p.add_argument('--diagnosis', type=str,
                   default='code/m5/base/rl/asymmetric/ex2/diagnosis/diagnosis.json')
    p.add_argument('--training-log', type=str,
                   default='code/m5/base/rl/asymmetric/ex2/training_log.json')
    p.add_argument('--output-dir', type=str,
                   default='code/m5/base/rl/asymmetric/ex2/diagnosis')
    return p.parse_args()


def load_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_difficulty_distribution(d: dict, output: Path):
    """图1: 左饼图 + 右柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    counts = [d['easy_count'], d['middle_count'], d['hard_count']]
    labels = ['Easy', 'Middle', 'Hard']
    colors = [DIFF_COLORS['easy'], DIFF_COLORS['middle'], DIFF_COLORS['hard']]
    total = sum(counts)

    # 左: 饼图
    ax = axes[0]
    wedges, _, _ = ax.pie(counts, colors=colors, startangle=90,
                          counterclock=False, autopct=lambda p: f'{p:.1f}%',
                          pctdistance=0.78,
                          wedgeprops=dict(width=0.38, edgecolor='white'))
    ax.text(0, 0, f'Budget 10%\nn={total:,}',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.legend(wedges, [f'{l} ({c:,})' for l, c in zip(labels, counts)],
              loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=9)
    ax.set_title('Difficulty Distribution', fontsize=12, fontweight='bold')

    # 右: 柱状图
    ax = axes[1]
    bars = ax.bar(labels, counts, color=colors, width=0.55, edgecolor='white')
    ax.set_ylabel('Number of Graphs')
    ax.set_ylim(0, max(counts) * 1.18)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.008,
                f'{c:,}\n({c / max(total, 1) * 100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    ax.set_title('Difficulty Counts', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Diagnosis: Easy / Middle / Hard', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_entropy_curve(d: dict, output: Path):
    """图2: 每步决策熵折线图。"""
    ent = d['norm_entropy_by_step']
    steps = np.arange(1, len(ent) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, ent, marker='o', markersize=3, color='#1E88E5',
            linewidth=1.8, label='Normalized entropy')
    ax.axhline(y=np.mean(ent), color='gray', linestyle='--', linewidth=0.9)
    ax.text(len(ent) * 0.98, np.mean(ent), f'mean={np.mean(ent):.3f}',
            ha='right', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel('Reveal Step')
    ax.set_ylabel('Normalized Entropy  H(π) / log(|frontier|)')
    ax.set_title('Decision Entropy over Reveal Steps (Budget 10%)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(1, len(ent) + 1, max(1, len(ent) // 10)))
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_class_diagnosis(d: dict, output: Path, n_graphs: int):
    """图3: 逐类成功率柱状图。"""
    per_class = d['per_class_acc']
    n_classes = max(int(k) for k in per_class.keys()) + 1
    classes = list(range(n_classes))
    accs = [per_class.get(str(c), 0.0) for c in classes]

    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    colors = ['#5C6BC0' for _ in classes]
    ax.bar(classes, accs, color=colors, edgecolor='white', width=0.8)

    ax.axhline(y=mean_acc, color='#2E7D32', linestyle='--', linewidth=1.2)
    ax.text(0.99, mean_acc, f'mean={mean_acc:.3f}',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            fontsize=8, color='#2E7D32')

    ax.set_xlabel('Class')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(classes)
    ax.set_title(f'Pre-Class Success Rate-Diagnosis (budget 10%, n={n_graphs:,} graphs)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_training_curves(log_entries, output: Path):
    """图4: 红/绿/蓝三子图 — train loss / avg reward(带注释) / accuracy。"""
    epochs = [e['epoch'] for e in log_entries]
    series = [('train_loss', SERIES[0]), ('train_avg_R', SERIES[1]),
              ('train_acc', SERIES[2])]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for ax, (key, (_, color, label)) in zip(axes, series):
        vals = [e[key] for e in log_entries]
        ax.plot(epochs, vals, marker='o', markersize=4, color=color,
                linewidth=1.8, label=label)
        ax.set_xlabel('Epoch')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if key == 'train_loss':
            ax.set_ylabel('Loss')
            ax.annotate(f'final={vals[-1]:.2f}',
                        xy=(epochs[-1], vals[-1]), xytext=(0, -14),
                        textcoords='offset points', ha='center', fontsize=8,
                        color='#C62828')
        elif key == 'train_avg_R':
            ax.set_ylabel('Avg Reward')
            ax.axhline(y=0.0, color='black', linewidth=0.8)
            for x, v in zip(epochs, vals):
                ax.annotate(f'{v:.2f}', xy=(x, v), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=7,
                            color='#2E7D32')
        else:
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1.0)
            ax.annotate(f'final={vals[-1]:.3f}',
                        xy=(epochs[-1], vals[-1]), xytext=(0, -14),
                        textcoords='offset points', ha='center', fontsize=8,
                        color='#0D47A1')

    fig.suptitle('Training Curves (RL asymmetric)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnosis = load_json(Path(args.diagnosis))
    by_budget = diagnosis['by_budget']
    for budget_key, d in by_budget.items():
        plot_difficulty_distribution(d, output_dir / 'difficulty_distribution.png')
        plot_entropy_curve(d, output_dir / 'entropy_curve.png')
        plot_class_diagnosis(d, output_dir / 'class_diagnosis.png', d['n_graphs'])

    training_log = load_json(Path(args.training_log))
    plot_training_curves(training_log, output_dir / 'training_curves.png')

    print("Done.")


if __name__ == '__main__':
    main()

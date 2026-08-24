"""独立可视化脚本：读取 eval 结果 JSON，绘制预算准确率图与逐类别图。

接受一个或多个 eval JSON（m5 eval.py / m6 eval_heldout.py / m7 eval.py 输出，
结构 {split: {strategy: {seed: {budget: {mean,std,n}}}}, per_class_{split}: {...}}），
风格对齐 code/m5/base/analysis/plot_three_stage.py，不绘制误差棒：
  - {name}_plot.png            主图：列 = split，策略分组柱（只画跨 seed 平均
                                mean）+ 柱顶数值 + full-graph 虚线
  - per_class_{split}_{strategy}_{seed}.png   每策略 × seed 一张逐类别图

输出目录默认 JSON 所在目录；主图名默认 eval_plot.png（--name 覆盖）。

用法:
  python code/plot_eval_results.py --json code/m5/base/analysis/eavl_fixed/eval_fixed.json
  python code/plot_eval_results.py --json m6/attention/bc/evaluate/eval_heldout_valid.json \
      --name bc_valid --output-dir m6/attention/bc/evaluate
  python code/plot_eval_results.py --json r1.json r2.json --no-per-class
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SPLIT_NAMES = ('train', 'valid', 'test')
STRATEGY_COLORS = {
    'random': '#999999',
    'bfs': '#55A868',
    'learnable': '#4C72B0',
    'bc': '#4C72B0',
    'rl': '#C44E52',
}


def _load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _splits(data: dict):
    return [k for k in _SPLIT_NAMES if k in data]


def _strategies(data: dict, split: str):
    return [s for s in data[split] if isinstance(data[split][s], dict)]


def _seed_types(data: dict, split: str, strategy: str):
    seed_dict = data[split][strategy]
    seeds = [k for k in seed_dict if k.startswith('s')]
    if 'mean' in seed_dict:
        seeds.append('mean')
    return seeds


def _ref_acc(data: dict, split: str):
    refs = data.get('config', {}).get('full_graph_ref', {}) or {}
    return refs.get(split)


def _budgets(data: dict, split: str, strategy: str, seed: str):
    return sorted(float(b) for b in data[split][strategy][seed])


def plot_budget_accuracy(data: dict, splits, strategies, seed_types,
                         out_path: Path):
    """主图：列 = split，策略分组柱，只画跨 seed 平均 'mean'。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budgets = _budgets(data, splits[0], strategies[0], seed_types[0])
    budget_labels = [f'{b:.0%}' for b in budgets]
    x = np.arange(len(budgets))
    n_strategies = len(strategies)
    width = 0.8 / n_strategies if n_strategies > 0 else 0.8
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_strategies) \
        if n_strategies > 0 else [0]

    n_cols = len(splits)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 3.4),
                             sharex='all', sharey='all')
    axes = [axes] if n_cols == 1 else list(axes)

    for col, split in enumerate(splits):
        ax = axes[col]
        for si, strategy in enumerate(strategies):
            seed_stats = data[split][strategy].get('mean', {})
            means = [seed_stats.get(f'{b:.2f}', {}).get('mean', 0.0)
                     for b in budgets]
            color = STRATEGY_COLORS.get(strategy, '#999999')
            bars = ax.bar(x + offsets[si], means, width,
                          label=strategy, color=color)
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=6)

        ax.set_title(f'{split.capitalize()} — mean of seeds', fontsize=10)
        ax.set_xlabel('Budget Ratio')
        if col == 0:
            ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(budget_labels)
        ax.grid(axis='y', alpha=0.3)

        ref = _ref_acc(data, split)
        if ref is not None:
            ref_frac = ref / 100 if ref > 1 else ref
            ref_pct = ref if ref > 1 else ref * 100
            ax.axhline(y=ref_frac, color='red', linestyle='--', linewidth=1.2,
                       label=f'Full-graph ({ref_pct:.2f}%)')
        ax.legend(fontsize=7)

    fig.suptitle(' vs '.join(strategies), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def plot_per_class(data: dict, split: str, strategies, seed_types,
                   output_dir: Path):
    """逐类别图：每策略一张，画跨 seed 平均 'mean'（不逐 seed 画）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budgets = _budgets(data, split, strategies[0], seed_types[0])
    budget_labels = [f'{b:.0%}' for b in budgets]
    colors = ['#4C72B0', '#55A868', '#C44E52']

    for strategy in strategies:
        per_class = data.get(f'per_class_{split}', {}).get(strategy, {})
        class_dict = per_class.get('mean', {})
        if not class_dict:
            continue
        classes = sorted(class_dict.keys(), key=lambda x: int(x))
        n_classes = len(classes)
        fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.35), 6))

        x = np.arange(n_classes)
        n_budgets = len(budgets)
        width = 0.8 / n_budgets
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_budgets)

        for bi, rel_b in enumerate(budgets):
            b_key = f'{rel_b:.2f}'
            means = [class_dict[cls][b_key]['mean'] for cls in classes]
            ax.bar(x + offsets[bi], means, width,
                   label=budget_labels[bi],
                   color=colors[bi % len(colors)])

        ax.set_title(f'{split.capitalize()} — {strategy} '
                     f'(per-class, mean of seeds)')
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=90, fontsize=5)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        out = output_dir / f'per_class_{split}_{strategy}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Per-class plot saved to {out}")


def main():
    parser = argparse.ArgumentParser(description='可视化 eval 结果 JSON')
    parser.add_argument('--json', nargs='+', required=True,
                        help='一个或多个 eval 结果 JSON（m5/m6/m7 eval 输出）')
    parser.add_argument('--name', default='',
                        help='主图名前缀（默认 eval_plot.png）')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 JSON 所在目录）')
    parser.add_argument('--no-per-class', action='store_true',
                        help='跳过逐类别图')
    args = parser.parse_args()

    for json_arg in args.json:
        path = Path(json_arg)
        data = _load_json(path)
        out_dir = Path(args.output_dir) if args.output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        name = args.name or 'eval_plot'

        splits = _splits(data)
        strategies = _strategies(data, splits[0]) if splits else []
        if not strategies:
            print(f"  (skip) {path}: no split/strategy data")
            continue
        seed_types = _seed_types(data, splits[0], strategies[0])
        plot_budget_accuracy(data, splits, strategies, seed_types,
                             out_dir / f'{name}.png')
        if not args.no_per_class:
            for split in splits:
                plot_per_class(data, split, strategies, seed_types, out_dir)


if __name__ == '__main__':
    main()

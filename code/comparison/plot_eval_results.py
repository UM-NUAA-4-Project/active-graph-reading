"""独立可视化脚本：读取 eval 结果 JSON，绘制预算准确率图与逐类别图。

接受一个或多个 eval JSON（m5 eval.py / m6 eval_heldout.py / m7 eval.py 输出，
结构 {split: {strategy: {seed: {budget: {mean,std,n}}}}, per_class_{split}: {...}}）。
2026-09-08 起：多 JSON 默认合并为一张跨文件对比图（策略并集，便于把
基线 fixed JSON 与 reader JSON 画在一起）；重名策略自动加 @文件名 后缀。
--no-merge 恢复旧行为（逐份独立出图）。基线策略配色表覆盖 random/bfs/
max-frontier-degree/random-walk/sagpool/learnable/bc/rl，未注册策略从
备用色环取色（不再一律灰色）。不绘制误差棒：
  - {name}_plot.png            主图：列 = split，策略分组柱（只画跨 seed 平均
                                mean）+ 柱顶数值 + full-graph 虚线
  - per_class_{split}_{strategy}_{seed}.png   每策略 × seed 一张逐类别图

输出目录默认 JSON 所在目录（多 JSON 合并时取第一份所在目录）；
主图名默认 eval_plot.png（--name 覆盖）。

用法:
  # 完整基线 JSON 单独出图（5 条基线并列）
  python code/comparison/plot_eval_results.py --json result/cifar10/fixed/eval_fixed.json
  # 基线 + BC + RL 合并为一张图
  python code/comparison/plot_eval_results.py \
      --json result/cifar10/fixed/eval_fixed.json \
             result/cifar10/experiments/ev1/evidence/bc/eval_learnable.json \
             result/cifar10/experiments/ev/evidence/rl/eval_learnable.json
"""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np


def _resolve(p: str) -> Path:
    """相对路径按 code/ 解析（与 plot_three_stage.resolve 一致）。"""
    path = Path(p)
    if path.is_absolute():
        return path
    from tools.paths import CODE_DIR
    return CODE_DIR / p

_SPLIT_NAMES = ('train', 'valid', 'test')
STRATEGY_COLORS = {
    'random': '#999999',
    'bfs': '#55A868',
    'max-frontier-degree': '#8172B2',
    'random-walk': '#CCB974',
    'sagpool': '#64B5CD',
    'learnable': '#4C72B0',
    'bc': '#4C72B0',
    'rl': '#C44E52',
}
# 未注册策略的备用色环（按出现顺序取色，保证同图内互异）
_FALLBACK_PALETTE = ['#937860', '#DA8BC3', '#E24A33', '#348ABD',
                     '#A1D99B', '#FBC15E', '#6B6ECF', '#8C8C8C']


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


def merge_evals(loaded):
    """多份 eval JSON 按策略并集合并为一份绘图用数据。

    loaded: [(文件主干名, data)]；split 表与 per_class_* 深合并（整策略块
    为单位，绘图数据不做数值冲突检查，重名即改名）；重名策略自动加
    '@文件主干' 后缀并返回告警。full_graph_ref 逐 split 取先到的非空值。
    返回 (merged, notes)。
    """
    notes = []
    occupied = set()
    merged = {'config': {}}
    for stem, data in loaded:
        # 本文件所需的重命名（基于全局已占用策略名）
        local = {}
        for top, block in data.items():
            if top == 'config' or not isinstance(block, dict):
                continue
            for strat, cells in block.items():
                if not isinstance(cells, dict):
                    continue
                if strat in local:
                    continue
                if strat in occupied:
                    new_name = f'{strat}@{stem}'
                    notes.append(f"策略重名 '{strat}'（来自 {stem}）→ "
                                 f"图中显示为 '{new_name}'")
                    local[strat] = new_name
                else:
                    local[strat] = strat
                occupied.add(local[strat])
        for top, block in data.items():
            if top == 'config':
                continue
            if not isinstance(block, dict):
                continue
            dst = merged.setdefault(top, {})
            for strat, cells in block.items():
                if not isinstance(cells, dict):
                    continue
                dst[local[strat]] = cells
        refs = (data.get('config') or {}).get('full_graph_ref') or {}
        merged_refs = merged['config'].setdefault('full_graph_ref', {})
        for sp, v in refs.items():
            if merged_refs.get(sp) is None and v is not None:
                merged_refs[sp] = v
    return merged, notes


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

    def color_for(si, strategy):
        if strategy in STRATEGY_COLORS:
            return STRATEGY_COLORS[strategy]
        return _FALLBACK_PALETTE[si % len(_FALLBACK_PALETTE)]

    n_cols = len(splits)
    # 策略多时加宽，避免柱与柱顶数值互相挤压
    fig_w = max(6, 1.5 * n_strategies) * n_cols
    rotate = n_strategies > 4          # 数值标签竖排，防重叠
    val_fs = 6 if n_strategies <= 4 else 4.8

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 3.4),
                             sharex='all', sharey='all')
    axes = [axes] if n_cols == 1 else list(axes)

    for col, split in enumerate(splits):
        ax = axes[col]
        for si, strategy in enumerate(strategies):
            seed_stats = data[split][strategy].get('mean', {})
            means = [seed_stats.get(f'{b:.2f}', {}).get('mean', 0.0)
                     for b in budgets]
            bars = ax.bar(x + offsets[si], means, width,
                          label=strategy, color=color_for(si, strategy))
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{m:.3f}', ha='center',
                        va='bottom', fontsize=val_fs,
                        rotation=90 if rotate else 0)

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

    fig.suptitle(' vs '.join(strategies) if n_strategies <= 4
                 else f'Comparison of {n_strategies} strategies', fontsize=12)
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


def _render(data: dict, out_dir: Path, name: str, no_per_class: bool):
    """按统一格式数据出一套图（主图 + 逐类别图）。"""
    splits = _splits(data)
    strategies = _strategies(data, splits[0]) if splits else []
    if not strategies:
        print("  (skip) no split/strategy data")
        return
    seed_types = _seed_types(data, splits[0], strategies[0])
    plot_budget_accuracy(data, splits, strategies, seed_types,
                         out_dir / f'{name}.png')
    if not no_per_class:
        for split in splits:
            plot_per_class(data, split, strategies, seed_types, out_dir)


def main():
    parser = argparse.ArgumentParser(description='可视化 eval 结果 JSON')
    parser.add_argument('--json', nargs='+', required=True,
                        help='一个或多个 eval 结果 JSON（m5/m6/m7 eval 输出；'
                             '多个时默认合并为一张对比图）')
    parser.add_argument('--name', default='',
                        help='主图名前缀（默认 eval_plot.png）')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认第一份 JSON 所在目录）')
    parser.add_argument('--no-per-class', action='store_true',
                        help='跳过逐类别图')
    parser.add_argument('--no-merge', action='store_true',
                        help='多 JSON 时仍逐份独立出图（默认合并）')
    args = parser.parse_args()

    loaded = [(_resolve(p)) for p in args.json]
    if len(loaded) > 1 and not args.no_merge:
        pairs = [(path.stem, _load_json(path)) for path in loaded]
        data, notes = merge_evals(pairs)
        for n in notes:
            print(f"  [merge] {n}")
        out_dir = (_resolve(args.output_dir) if args.output_dir
                   else loaded[0].parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = args.name or 'eval_plot'
        print(f"Merged {len(loaded)} JSONs -> {out_dir / (name + '.png')}")
        _render(data, out_dir, name, args.no_per_class)
        return

    for path in loaded:
        data = _load_json(path)
        out_dir = _resolve(args.output_dir) if args.output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        name = args.name or 'eval_plot'
        print(f"Plotting {path}")
        _render(data, out_dir, name, args.no_per_class)


if __name__ == '__main__':
    main()

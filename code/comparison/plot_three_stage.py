"""三阶段（Random/BFS vs BC vs RL）预算准确率对比图（泛化版）。

自 code/m5/base/analysis/plot_three_stage.py 去硬编码改写：
- 输入 JSON 路径、划分、预算档、参考上界全部 CLI 传入；
- 各 JSON 缺失的 split 自动跳过（如 m7 eval 只有 test）；
- 图样式与旧版一致（分组柱 + 柱顶数值 + full-graph 虚线 + 面板编号），
  输出 SVG/PDF/PNG + source_data.csv。

用法:
  python code/comparison/plot_three_stage.py \
      --fixed-json result/ppa/fixed/eval_heldout_test_random.json \
      --bc-json result/ppa/experiments/bc_baseline/base/bc/eval_heldout_test_reader.json \
      --rl-json result/ppa/experiments/rl_asymmetric_ex1/base/rl/eval_heldout_test_reader.json \
      --splits valid test --ref-valid 0.66 --ref-test 0.656 \
      --output-dir result/ppa/comparison
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# 旧版出版样式（保持一致）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False

COLORS = {'Random': '#999999', 'BFS': '#55A868', 'BC': '#4C72B0', 'RL': '#C44E52'}

# 供 qa_three_stage 引用的默认结构与轴范围
YMIN, YMAX = 0.10, 0.70


def resolve(p):
    path = Path(p)
    if path.is_absolute():
        return path
    from tools.paths import CODE_DIR
    return CODE_DIR / p


def parse_args():
    parser = argparse.ArgumentParser(description='三阶段对比图（泛化 CLI）')
    parser.add_argument('--fixed-json', required=True,
                        help='固定策略 eval JSON（含 random/bfs 键）')
    parser.add_argument('--bc-json', default='', help='BC eval JSON')
    parser.add_argument('--rl-json', default='', help='RL eval JSON')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test'],
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--budgets', type=float, nargs='+',
                        default=[0.05, 0.10, 0.15])
    parser.add_argument('--ref-valid', type=float, default=None,
                        help='valid 集全图上界（JSON config 无 full_graph_ref 时必填）')
    parser.add_argument('--ref-test', type=float, default=None,
                        help='test 集全图上界')
    parser.add_argument('--name', default='three_stage_compare')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def build_stages(args):
    stages = [('Random', Path(args.fixed_json), 'random', COLORS['Random']),
              ('BFS', Path(args.fixed_json), 'bfs', COLORS['BFS'])]
    if args.bc_json:
        stages.append(('BC', Path(args.bc_json), 'learnable', COLORS['BC']))
    if args.rl_json:
        stages.append(('RL', Path(args.rl_json), 'learnable', COLORS['RL']))
    return stages


def load_stats(stages, splits, budgets, ref_valid, ref_test):
    """stage -> split -> seed -> budget -> (mean, sem)；各 JSON 缺失 split 跳过。"""
    stats = {}
    for name, path, strategy, _color in stages:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        stats[name] = {'path': path, 'strategy': strategy}
        for split in splits:
            if split not in data or strategy not in data[split]:
                continue
            stats[name][split] = {}
            for seed in ('mean',):
                cells = data[split][strategy][seed]
                stats[name][split][seed] = {
                    b: (cells[f'{b:.2f}']['mean'],
                        cells[f'{b:.2f}']['std'] / np.sqrt(max(cells[f'{b:.2f}']['n'], 1)))
                    for b in budgets
                }
        if 'config' in data and 'full_graph_ref' in data['config']:
            refs = data['config']['full_graph_ref'] or {}
            if ref_valid is None and refs.get('valid') is not None:
                ref_valid = refs['valid'] / 100 if refs['valid'] > 1 else refs['valid']
            if ref_test is None and refs.get('test') is not None:
                ref_test = refs['test'] / 100 if refs['test'] > 1 else refs['test']
    return stats, ref_valid, ref_test


def common_splits(stats, splits):
    """所有 stage 都具备的 split 子集。"""
    out = []
    for split in splits:
        if all(split in stats[name] for name, _p, _s, _c in _stage_tuples(stats)):
            out.append(split)
    return out


def _stage_tuples(stats):
    return [(name, stats[name]['path'], stats[name]['strategy'], COLORS.get(name, '#999999'))
            for name in stats]


def write_source_data(out_dir, stats, splits, budgets):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'source_data.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'seed', 'budget', 'method', 'mean', 'sem'])
        for split in splits:
            for b in budgets:
                for name, _p, _s, _c in _stage_tuples(stats):
                    mean, sem = stats[name][split]['mean'][b]
                    w.writerow([split, 'mean', f'{b:.2f}', name,
                                f'{mean:.6f}', f'{sem:.6f}'])


def build_figure(stats, splits, budgets, ref_valid=None, ref_test=None):
    stages = _stage_tuples(stats)
    n_stages = len(stages)
    w = 0.8 / n_stages
    x = np.arange(len(budgets))
    budget_labels = [f'{b * 100:.0f}%' for b in budgets]

    n_cols = len(splits)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.0), sharey='all')
    axes = np.atleast_2d(axes)

    for ci, split in enumerate(splits):
        ax = axes[0][ci]
        for si, (name, _p, _s, color) in enumerate(stages):
            cells = stats[name][split]['mean']
            means = [cells[b][0] for b in budgets]
            offset = (si - (n_stages - 1) / 2) * w
            ax.bar(x + offset, means, w,
                   color=color, edgecolor='white', linewidth=0.4)
            for bx, m in zip(x + offset, means):
                ax.text(bx, m + 0.012, f'{m:.2f}', ha='center', va='bottom',
                        fontsize=5.2, color='#4D4D4D')

        ax.set_xticks(x)
        ax.set_xticklabels(budget_labels)
        ax.set_ylim(YMIN, YMAX)
        ax.grid(axis='y', color='#E0E0E0', linewidth=0.5, zorder=0)

        ref = ref_valid if split == 'valid' else ref_test
        if ref is not None:
            ax.axhline(ref, color='#4D4D4D', linestyle='--',
                       linewidth=0.8, zorder=3)
            ax.text(1.0, ref + 0.008, f'Full graph {ref:.3f}',
                    transform=blended_transform_factory(ax.transAxes, ax.transData),
                    ha='right', va='bottom', fontsize=6, color='#4D4D4D', zorder=4)

        if ci == 0:
            ax.set_ylabel('Accuracy')
        ax.set_xlabel('Budget ratio')
        ax.set_title(f'{split.capitalize()} — mean of seeds', fontsize=7, pad=4)
        ax.text(-0.09, 1.04, 'ab'[ci], transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='bottom', ha='left')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _n, _p, _s, c in stages]
    labels = [n for n, _p, _s, _c in stages]
    fig.legend(handles, labels, ncol=len(handles), loc='upper center',
               bbox_to_anchor=(0.5, 0.995), fontsize=7,
               handlelength=1.2, handleheight=1.0, columnspacing=1.2)

    note = 'Mean over deterministic seeds'
    if ref_valid is not None or ref_test is not None:
        refs = []
        if ref_valid is not None:
            refs.append(f'{ref_valid:.3f} valid')
        if ref_test is not None:
            refs.append(f'{ref_test:.3f} test')
        note += f'; dashed lines: full-graph accuracy ({ " / ".join(refs) })'
    fig.text(0.5, 0.012, note, ha='center', fontsize=6, color='#4D4D4D')

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes


def main():
    args = parse_args()
    stages = build_stages(args)
    stats, ref_valid, ref_test = load_stats(stages, args.splits, args.budgets,
                                            args.ref_valid, args.ref_test)
    splits = common_splits(stats, args.splits)
    if not splits:
        raise SystemExit(f"各 JSON 无共同 split：{args.splits}")
    out_dir = resolve(args.output_dir)
    write_source_data(out_dir, stats, splits, args.budgets)
    fig, _axes = build_figure(stats, splits, args.budgets, ref_valid, ref_test)
    base = out_dir / args.name
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {out_dir}')


if __name__ == '__main__':
    main()

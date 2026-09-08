"""基线 vs BC vs RL 预算准确率对比图（泛化版，原"三阶段图"）。

自 code/m5/base/analysis/plot_three_stage.py 去硬编码改写：
- 输入 JSON 路径、划分、预算档、参考上界全部 CLI 传入；
- 各 JSON 缺失的 split 自动跳过（如 m7 eval 只有 test）；
- 图样式与旧版一致（分组柱 + 柱顶数值 + full-graph 虚线）；
- 2026-09-08 起每个 split 输出一张独立图（{name}_{split}.svg/pdf/png，
  图内自带图例，无面板编号），另写合并的 source_data.csv；
- 2026-09-08 起基线不再限于 Random/BFS：默认画 fixed JSON 中实际存在的
  全部基线（random/bfs/max-frontier-degree/random-walk/sagpool，按此规范
  序排列），--fixed-strategies 可手动限定子集。

用法:
  # 2026-09-05 起统一格式：fixed JSON 含各基线键，reader JSON 含 learnable 键；
  # 上界虚线优先取各 JSON config.full_graph_ref
  python code/comparison/plot_three_stage.py \
      --fixed-json result/cifar10/fixed/eval_fixed.json \
      --bc-json result/cifar10/experiments/ev1/evidence/bc/eval_learnable.json \
      --rl-json result/cifar10/experiments/ev/evidence/rl/eval_learnable.json \
      --output-dir result/cifar10/comparison
  # 产物：three_stage_compare_valid.{svg,pdf,png} + three_stage_compare_test.{...}
  # 只要旧两条基线：--fixed-strategies random bfs
  # 仍兼容旧单 split JSON 与 CLI 显式上界：
  #   --splits valid test --ref-valid 0.66 --ref-test 0.656
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

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

COLORS = {'Random': '#999999', 'BFS': '#55A868',
          'Max-Frontier-Degree': '#8172B2', 'Random-Walk': '#CCB974',
          'SAGPool': '#64B5CD', 'BC': '#4C72B0', 'RL': '#C44E52'}

# fixed JSON 的基线键 → 图例名（顺序即图中排列顺序，与配色一一对应）
BASELINES = [('random', 'Random'),
             ('bfs', 'BFS'),
             ('max-frontier-degree', 'Max-Frontier-Degree'),
             ('random-walk', 'Random-Walk'),
             ('sagpool', 'SAGPool')]

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
                        help='固定策略 eval JSON（默认画其中实际存在的全部基线）')
    parser.add_argument('--fixed-strategies', nargs='+', default=None,
                        choices=[k for k, _n in BASELINES],
                        help='限定绘制的基线（默认全部 5 条中 JSON 里有的）')
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


def _fixed_json_strategies(path):
    """fixed JSON 里实际存在的策略键（任一 split 出现即算，保持出现序）。"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    found = []
    for key, block in data.items():
        if key in ('train', 'valid', 'test') and isinstance(block, dict):
            for s in block:
                if isinstance(block[s], dict) and s not in found:
                    found.append(s)
    return found


def build_stages(args):
    """基线（JSON 里实际有的）+ 可选 BC/RL → [(图例名, 路径, 策略键, 颜色)]。"""
    fixed_path = Path(resolve(args.fixed_json))
    present = _fixed_json_strategies(fixed_path)
    wanted = set(args.fixed_strategies or [k for k, _n in BASELINES])
    stages = []
    for key, display in BASELINES:
        if key in wanted and key in present:
            stages.append((display, fixed_path, key, COLORS[display]))
    if wanted and not stages:
        raise SystemExit(f'fixed JSON {fixed_path} 中没有任何所请求的基线：'
                         f'现有 {present}，请求 {sorted(wanted)}')
    if args.bc_json:
        stages.append(('BC', Path(resolve(args.bc_json)), 'learnable', COLORS['BC']))
    if args.rl_json:
        stages.append(('RL', Path(resolve(args.rl_json)), 'learnable', COLORS['RL']))
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


def build_split_figure(stats, split, budgets, ref=None):
    """单 split 独立图：分组柱 + 柱顶数值 + full-graph 虚线 + 顶部图例。"""
    stages = _stage_tuples(stats)
    n_stages = len(stages)
    w = 0.8 / n_stages
    x = np.arange(len(budgets))
    budget_labels = [f'{b * 100:.0f}%' for b in budgets]

    # 策略多时加宽图幅、数值标签竖排，避免挤压
    fig, ax = plt.subplots(figsize=(max(3.6, 0.62 * n_stages), 3.0))
    rotate = n_stages > 4
    val_fs = 5.2 if n_stages <= 4 else 4.6
    for si, (name, _p, _s, color) in enumerate(stages):
        cells = stats[name][split]['mean']
        means = [cells[b][0] for b in budgets]
        offset = (si - (n_stages - 1) / 2) * w
        ax.bar(x + offset, means, w,
               color=color, edgecolor='white', linewidth=0.4)
        for bx, m in zip(x + offset, means):
            ax.text(bx, m + 0.012, f'{m:.2f}', ha='center', va='bottom',
                    fontsize=val_fs, color='#4D4D4D',
                    rotation=90 if rotate else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.set_ylim(YMIN, YMAX)
    ax.grid(axis='y', color='#E0E0E0', linewidth=0.5, zorder=0)

    if ref is not None:
        ax.axhline(ref, color='#4D4D4D', linestyle='--',
                   linewidth=0.8, zorder=3)
        ax.text(1.0, ref + 0.008, f'Full graph {ref:.3f}',
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                ha='right', va='bottom', fontsize=6, color='#4D4D4D', zorder=4)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Budget ratio')
    ax.set_title(f'{split.capitalize()} — mean of seeds', fontsize=7, pad=4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _n, _p, _s, c in stages]
    labels = [n for n, _p, _s, _c in stages]
    fig.legend(handles, labels, ncol=min(n_stages, 4), loc='upper center',
               bbox_to_anchor=(0.5, 0.995),
               fontsize=7 if n_stages <= 5 else 6,
               handlelength=1.2, handleheight=1.0, columnspacing=1.2)

    note = 'Mean over deterministic seeds'
    if ref is not None:
        note += f'; dashed line: full-graph accuracy ({ref:.3f})'
    fig.text(0.5, 0.012, note, ha='center', fontsize=6, color='#4D4D4D')

    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    return fig, ax


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
    for split in splits:
        ref = ref_valid if split == 'valid' else ref_test
        fig, _ax = build_split_figure(stats, split, args.budgets, ref)
        base = out_dir / f'{args.name}_{split}'
        fig.savefig(f'{base}.svg', bbox_inches='tight')
        fig.savefig(f'{base}.pdf', bbox_inches='tight')
        fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Figure saved: {base}.png')


if __name__ == '__main__':
    main()

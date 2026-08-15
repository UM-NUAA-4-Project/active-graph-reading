"""Fixed (Random/BFS) vs BC vs RL 三阶段验证准确率对比图。

读取三份 eval JSON，绘制 2x2 分组柱状图：
  行 = seed 起点 (best / worst)，列 = 划分 (valid / test)
  x 轴 = 预算比例 (5% / 10% / 15%)，每组 4 柱 = Random / BFS / BC / RL
输出 SVG/PDF/PNG + 源数据 CSV 到 ./three_stage_compare/
"""
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'three_stage_compare'

FIXED_JSON = HERE / 'eavl_fixed' / 'eval_fixed.json'
BC_JSON = HERE / 'eval_results_BC' / 'eval_learnable.json'
RL_JSON = HERE.parent / 'rl' / 'asymmetric' / 'ex2' / 'analysis' / 'eval_learnable.json'

BUDGETS = [0.05, 0.10, 0.15]
BUDGET_LABELS = ['5%', '10%', '15%']
SPLITS = ['valid', 'test']
SEEDS = ['best', 'worst']

# 测试集全图上界（eval JSON 的 config 中未存，2026-08-15 补充）
REF_TEST_DEFAULT = 0.6560

STAGES = [
    ('Random', FIXED_JSON, 'random', '#999999'),
    ('BFS', FIXED_JSON, 'bfs', '#55A868'),
    ('BC', BC_JSON, 'learnable', '#4C72B0'),
    ('RL', RL_JSON, 'learnable', '#C44E52'),
]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False


def load_stats():
    """stage -> split -> seed -> budget -> (mean, sem)"""
    stats = {}
    ref_valid = None
    for name, path, strategy, _color in STAGES:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        stats[name] = {}
        for split in SPLITS:
            stats[name][split] = {}
            for seed in SEEDS:
                cells = data[split][strategy][seed]
                stats[name][split][seed] = {
                    b: (cells[f'{b:.2f}']['mean'],
                        cells[f'{b:.2f}']['std'] / np.sqrt(cells[f'{b:.2f}']['n']))
                    for b in BUDGETS
                }
        if ref_valid is None and 'config' in data and 'full_graph_ref' in data['config']:
            ref_valid = data['config']['full_graph_ref'].get('valid')
            ref_test = data['config']['full_graph_ref'].get('test')
    if ref_test is None:
        ref_test = REF_TEST_DEFAULT
    return stats, ref_valid, ref_test


def write_source_data(stats):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'source_data.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'seed', 'budget', 'method', 'mean', 'sem'])
        for split in SPLITS:
            for seed in SEEDS:
                for b in BUDGETS:
                    for name, _p, _s, _c in STAGES:
                        mean, sem = stats[name][split][seed][b]
                        w.writerow([split, seed, f'{b:.2f}', name,
                                    f'{mean:.6f}', f'{sem:.6f}'])


def build_figure(stats, ref_valid=None, ref_test=None):
    n_stages = len(STAGES)
    w = 0.8 / n_stages
    x = np.arange(len(BUDGETS))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey='all')

    panel_titles = {
        (0, 0): 'Validation — best seed',
        (0, 1): 'Test — best seed',
        (1, 0): 'Validation — worst seed',
        (1, 1): 'Test — worst seed',
    }

    for ri, seed in enumerate(SEEDS):
        for ci, split in enumerate(SPLITS):
            ax = axes[ri][ci]
            for si, (name, _p, _s, color) in enumerate(STAGES):
                cells = stats[name][split][seed]
                means = [cells[b][0] for b in BUDGETS]
                sems = [cells[b][1] for b in BUDGETS]
                offset = (si - (n_stages - 1) / 2) * w
                ax.bar(x + offset, means, w, yerr=sems,
                       color=color, edgecolor='white', linewidth=0.4,
                       error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'capsize': 2})
                for bx, m in zip(x + offset, means):
                    ax.text(bx, m + 0.012, f'{m:.2f}', ha='center', va='bottom',
                            fontsize=5.2, color='#4D4D4D')

            ax.set_xticks(x)
            ax.set_xticklabels(BUDGET_LABELS)
            ax.set_ylim(0.10, 0.70)
            ax.set_yticks(np.arange(0.1, 0.71, 0.1))
            ax.yaxis.set_major_formatter(lambda v, _: f'{v:.1f}')
            ax.grid(axis='y', color='#E0E0E0', linewidth=0.5, zorder=0)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_zorder(2)

            ref = ref_valid if split == 'valid' else ref_test
            if ref is not None:
                ax.axhline(ref, color='#4D4D4D', linestyle='--',
                           linewidth=0.8, zorder=3)
                ax.text(1.0, ref + 0.008, f'Full graph {ref:.3f}',
                        transform=blended_transform_factory(ax.transAxes, ax.transData),
                        ha='right', va='bottom', fontsize=6, color='#4D4D4D',
                        zorder=4)

            trans = blended_transform_factory(ax.transAxes, ax.transData)
            for dy in ((-0.004, 0.004), (0.004, -0.004)):
                ax.plot([-0.014, 0.014], [0.10 + dy[0], 0.10 + dy[1]],
                        color='black', linewidth=0.8, transform=trans,
                        clip_on=False, zorder=5)

            if ci == 0:
                ax.set_ylabel('Accuracy')
            if ri == 1:
                ax.set_xlabel('Budget ratio')
            ax.set_title(panel_titles[(ri, ci)], fontsize=7, pad=4)

            letter = 'abcd'[ri * 2 + ci]
            ax.text(-0.09, 1.04, letter, transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='bottom', ha='left')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _n, _p, _s, c in STAGES]
    labels = [n for n, _p, _s, _c in STAGES]
    fig.legend(handles, labels, ncol=len(handles), loc='upper center',
               bbox_to_anchor=(0.5, 0.995), fontsize=7,
               handlelength=1.2, handleheight=1.0, columnspacing=1.2)

    note = 'Error bars: s.e.m. across graphs (n = 45,100 validation / 34,800 test)'
    if ref_valid is not None and ref_test is not None:
        note += (f'; dashed lines: full-graph accuracy '
                 f'({ref_valid:.3f} validation / {ref_test:.3f} test)')
    fig.text(0.5, 0.012, note, ha='center', fontsize=6, color='#4D4D4D')

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes


def main():
    stats, ref_valid, ref_test = load_stats()
    write_source_data(stats)
    fig, _axes = build_figure(stats, ref_valid, ref_test)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'three_stage_compare'
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {OUT_DIR}')


if __name__ == '__main__':
    main()

"""可视化 eval_learnable.json 数据."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- 加载数据 ----
json_path = Path(__file__).resolve().parent / 'eval_results' / 'eval_learnable.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

budgets = data['config']['budgets']
ref_valid = data['config']['full_graph_ref']['valid']

budget_labels = [f'{b:.0%}' for b in budgets]

# ---- 图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for col, split_name in enumerate(['valid', 'test']):
    ax = axes[col]
    split_data = data[split_name]['learnable']
    x = np.arange(len(budgets))
    width = 0.35

    for offset, kind, color, hatch in [
        (-width/2, 'best', '#4C72B0', ''),
        (+width/2, 'worst', '#C44E52', '//'),
    ]:
        means = [split_data[kind][f'{b:.2f}']['mean'] for b in budgets]
        stds = [split_data[kind][f'{b:.2f}']['std'] for b in budgets]
        n = split_data[kind][f'{budgets[0]:.2f}']['n']
        bars = ax.bar(
            x + offset, means, width, yerr=stds,
            label=f'{kind} seed (n={n})', capsize=4,
            color=color, edgecolor='white', linewidth=0.5,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8,
            )

    if split_name == 'valid' and ref_valid is not None:
        ref_frac = ref_valid if ref_valid <= 1 else ref_valid / 100
        ax.axhline(y=ref_frac, color='green', linestyle='--', linewidth=1.5,
                   label=f'Full-graph ({ref_frac*100:.2f}%)')

    ax.set_title(f'{split_name.upper()} — LearnableReader', fontsize=13, fontweight='bold')
    ax.set_xlabel('Budget Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.set_ylim(0, ref_valid * 1.05 if ref_valid else None)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('LearnableReader: Best vs Worst Seed (by Degree)', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()

out_path = Path(__file__).resolve().parent / 'eval_results' / 'learnable_plot.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved to {out_path}')

# ---- 打印表格 ----
print(f"\n{'Budget':>8}  {'Val-best':>10}  {'Val-worst':>10}  {'Test-best':>10}  {'Test-worst':>10}")
print('-' * 52)
for b in budgets:
    b_key = f'{b:.2f}'
    vb = data['valid']['learnable']['best'][b_key]['mean']
    vw = data['valid']['learnable']['worst'][b_key]['mean']
    tb = data['test']['learnable']['best'][b_key]['mean']
    tw = data['test']['learnable']['worst'][b_key]['mean']
    print(f'{b:>8.0%}  {vb:>10.4f}  {vw:>10.4f}  {tb:>10.4f}  {tw:>10.4f}')

if ref_valid is not None:
    ref_frac = ref_valid if ref_valid <= 1 else ref_valid / 100
    print(f'\nFull-graph ref valid: {ref_frac*100:.2f}%')

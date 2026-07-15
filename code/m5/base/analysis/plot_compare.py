import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent

# Load data
files = {
    'REINFORCE (tanh)': base / 'eval_results_tanh' / 'eval_learnable.json',
    'REINFORCE (binary)': base / 'eval_results_binary' / 'eval_learnable.json',
    'BC': base / 'eval_results_BC' / 'eval_learnable.json',
}

budgets = [0.05, 0.10, 0.15]
budget_labels = ['5%', '10%', '15%']

colors = {'REINFORCE (tanh)': '#E74C3C', 'REINFORCE (binary)': '#3498DB', 'BC': '#2ECC71'}
markers = {'REINFORCE (tanh)': 's', 'REINFORCE (binary)': 'o', 'BC': '^'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Learnable Reader Accuracy vs Budget', fontsize=14, fontweight='bold')

for split_name, ax, title in [('valid', axes[0], 'Validation'), ('test', axes[1], 'Test')]:
    for method_name, path in files.items():
        with open(path) as f:
            data = json.load(f)
        # use "best" seed strategy for overall comparison
        means = [data[split_name]['learnable']['best'][f'{b:.2f}']['mean'] * 100 for b in budgets]
        ax.plot(budget_labels, means, color=colors[method_name], marker=markers[method_name],
                linewidth=2, markersize=8, label=method_name)

    ax.set_title(title)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(8, 16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
out_path = base / 'accuracy_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.close()

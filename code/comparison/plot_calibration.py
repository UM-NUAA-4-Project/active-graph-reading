"""校准/可靠性分析：evidence 证据头 vs 冻结子图分类器（预测 5.3c / 命题 5.4）。

读取 eval.py --export-confidence 产出的 calibration.json：
  {'calibration': {split: {budget_str: {'ev': [[conf, correct], ...],
                                        'clf': [...]}}, ...}, 'config': {...}}
理论背景：模型内积分后验天然校准（命题 5.4），故证据头应比冻结分类器
更校准（ECE 更低），高预算档差距最大（预测 5.3c）。

产出（输出目录默认第一份 JSON 所在目录；多 JSON 各自独立出图）：
  - {prefix}calibration_{split}.png   可靠性图：panel=预算档，
                                      ev / clf 两条可靠性曲线 + 对角参考线
  - {prefix}calibration_ece.csv       ECE 表（15 等宽桶，ECE = Σ|acc−conf|·n/N）

用法:
  python code/comparison/plot_calibration.py --json result/collab/experiments/evidence1/evidence/bc/calibration.json
  python code/comparison/plot_calibration.py --json a/calibration.json b/calibration.json --bins 10
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

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

READOUTS = (('ev', 'Evidence head', '#4C72B0'),
            ('clf', 'Frozen classifier', '#999999'))


def ece(pairs, n_bins=15):
    """等宽置信度桶 ECE。pairs: [[conf, correct], ...] → (ece, curve, edges)。"""
    arr = np.asarray(pairs, dtype=float)
    conf, corr = arr[:, 0], arr[:, 1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    curve = []  # (桶中心, 桶内 accuracy, 桶内样本数, 桶内平均置信度)
    total = len(conf)
    e = 0.0
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        if n == 0:
            curve.append((0.5 * (edges[k] + edges[k + 1]), np.nan, 0,
                          np.nan))
            continue
        acc, avg_conf = float(corr[m].mean()), float(conf[m].mean())
        center = 0.5 * (edges[k] + edges[k + 1])
        curve.append((center, acc, n, avg_conf))
        e += n / total * abs(acc - avg_conf)
    return e, curve, edges


def resolve(p):
    path = Path(p)
    if path.is_absolute():
        return path
    from tools.paths import CODE_DIR
    return CODE_DIR / p


def process(json_path: Path, out_dir: Path, prefix: str, n_bins: int):
    data = json.loads(json_path.read_text(encoding='utf-8'))
    calib = data['calibration']
    splits = [s for s in ('train', 'valid', 'test') if s in calib]
    if not splits:
        print(f'  (skip) {json_path}: calibration 无 split 数据')
        return

    rows = []
    for split in splits:
        budgets = sorted(calib[split], key=float)
        fig, axes = plt.subplots(1, len(budgets),
                                 figsize=(3.2 * len(budgets), 3.2),
                                 sharey='all')
        axes = np.atleast_1d(axes)
        for bi, budget in enumerate(budgets):
            ax = axes[bi]
            ax.plot([0, 1], [0, 1], linestyle='--', color='#4D4D4D',
                    linewidth=0.8, label='perfect')
            for key, label, color in READOUTS:
                pairs = calib[split][budget].get(key) or []
                if not pairs:
                    continue
                e, curve, _edges = ece(pairs, n_bins)
                xs = [c[0] for c in curve if c[2] > 0]
                ys = [c[1] for c in curve if c[2] > 0]
                arr = np.asarray(pairs, dtype=float)
                ax.plot(xs, ys, marker='o', markersize=3, color=color,
                        label=f'{label} (ECE={e:.3f})')
                rows.append([split, budget, key, f'{e:.4f}', len(pairs),
                             f'{arr[:, 0].mean():.4f}', f'{arr[:, 1].mean():.4f}'])
            ax.set_title(f'B = {float(budget) * 100:.0f}%', fontsize=9)
            ax.set_xlabel('Confidence')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, loc='upper left')
        axes[0].set_ylabel('Accuracy')
        fig.suptitle(f'Reliability — {json_path.parent.parent.parent.stem} '
                     f'{split} (mean of seeds)', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = out_dir / f'{prefix}calibration_{split}.png'
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f'  Plot saved to {out_png}')

    csv_path = out_dir / f'{prefix}calibration_ece.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['split', 'budget', 'readout', 'ece', 'n',
                    'mean_conf', 'acc'])
        w.writerows(rows)
    print(f'  ECE table saved to {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='校准/可靠性分析（预测 5.3c）')
    parser.add_argument('--json', nargs='+', required=True,
                        help='一个或多个 calibration.json（eval.py '
                             '--export-confidence 产出）')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认第一份 JSON 所在目录）')
    parser.add_argument('--name', default='',
                        help='输出文件名前缀（默认空）')
    parser.add_argument('--bins', type=int, default=15,
                        help='ECE 等宽桶数（默认 15）')
    args = parser.parse_args()

    for p in args.json:
        path = resolve(p)
        out_dir = resolve(args.output_dir) if args.output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Processing {path}')
        process(path, out_dir, args.name, args.bins)


if __name__ == '__main__':
    main()

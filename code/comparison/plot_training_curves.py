"""训练曲线可视化：loss / RL 奖励 / 准确率 / 证据准确率 多面板 + 证据 bar。

接受一个或多个训练日志 JSON，自动识别类型：
  - classifier history.json（dict 含 'epochs' 列表，字段 loss/train_acc/test_acc）
  - BC  training_log.json（list，字段 epoch/train_loss/act_acc[/ev_acc/ev_acc_final]）
  - RL  training_log.json（list，字段 epoch/train_loss/train_avg_R/train_acc
        [/ev_acc/ev_acc_final]）

输出（默认第一个 JSON 所在目录，--output-dir 覆盖）：
  - {name}_curves.png   多面板曲线：loss / reward / accuracy / evidence acc
                        （只有含数据的面板会画出；多 run 一文件一线叠加）
  - {name}_ev_bar.png   证据准确率 bar（最后一轮）：每 run 两根柱——
                        ev_acc_final（终局判定，主指标）与 ev_acc（逐步平均，参照）
  - {name}_source.csv   曲线源数据

用法:
  python code/comparison/plot_training_curves.py --json \
      result/cifar10/classifier/finetune2/history.json \
      result/cifar10/experiments/1-teacher/base/bc/training_log.json \
      result/cifar10/experiments/1-teacher/base/rl/training_log.json
  # 多 run 对比（如 base vs attention vs evidence）：--labels 与 --json 对齐
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#4C72B0', '#C44E52', '#55A868', '#8172B2', '#CCB974', '#64B5CD']


def _load(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _extract(path: Path, label: str) -> dict:
    """日志文件 → {'epochs','loss','reward','acc','ev_acc','ev_acc_final'}
    （缺字段为 None/空）。"""
    data = _load(path)
    if isinstance(data, dict) and isinstance(data.get('epochs'), list):
        records = data['epochs']          # classifier history.json
    elif isinstance(data, list):
        records = data                    # BC/RL training_log.json
    else:
        raise ValueError(f"{path}: 无法识别的训练日志格式（期望 history.json 或 "
                         f"training_log.json）")

    out = {'epochs': [], 'loss': [], 'reward': [], 'acc': [],
           'acc_kind': None, 'ev_acc': [], 'ev_acc_final': []}
    for rec in records:
        if not isinstance(rec, dict) or 'epoch' not in rec:
            continue
        loss = rec.get('train_loss', rec.get('loss'))
        reward = rec.get('train_avg_R', rec.get('avg_R'))
        acc = rec.get('train_acc', rec.get('act_acc'))
        if acc is not None:
            out['acc_kind'] = ('rollout' if 'train_acc' in rec else 'teacher')
        if loss is None and reward is None and acc is None:
            continue
        out['epochs'].append(int(rec['epoch']))
        out['loss'].append(loss)
        out['reward'].append(reward)
        out['acc'].append(acc)
        out['ev_acc'].append(rec.get('ev_acc'))
        out['ev_acc_final'].append(rec.get('ev_acc_final'))
    out['label'] = label
    return out


def _default_label(path: Path) -> str:
    """默认标签：bc/rl 取 {实验名}/{kind}，classifier 取目录名。"""
    if path.parent.name in ('bc', 'rl'):
        return f'{path.parent.parent.name}/{path.parent.name}'
    return path.parent.name


def _write_csv(runs: list, out_path: Path) -> None:
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['run', 'epoch', 'loss', 'reward', 'acc',
                    'ev_acc', 'ev_acc_final'])
        for r in runs:
            for i, ep in enumerate(r['epochs']):
                fmt = lambda v: f'{v:.6f}' if v is not None else ''
                w.writerow([r['label'], ep, fmt(r['loss'][i]),
                            fmt(r['reward'][i]), fmt(r['acc'][i]),
                            fmt(r['ev_acc'][i]), fmt(r['ev_acc_final'][i])])


def _line(ax, runs, field, title, ylabel, styles=None):
    """折线面板：styles[i] = (marker, linestyle)，缺省实线圆点。"""
    for i, r in enumerate(runs):
        vals = r[field]
        if not any(v is not None for v in vals):
            continue
        marker, ls = (styles[i] if styles else ('o', '-'))
        ax.plot(r['epochs'], vals, marker=marker, markersize=3,
                linewidth=1.2, linestyle=ls,
                color=COLORS[i % len(COLORS)], label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_curves(runs: list, out_path: Path) -> None:
    """多面板：loss / reward / accuracy / evidence accuracy（按需出面板）。"""
    panels = [('loss', 'Training loss', 'Train loss', None)]
    if any(any(v is not None for v in r['reward']) for r in runs):
        panels.append(('reward', 'RL reward', 'Avg reward', None))
    if any(any(v is not None for v in r['acc']) for r in runs):
        panels.append(('acc', 'Accuracy', 'Accuracy', None))
    if any(any(v is not None for v in r['ev_acc']) for r in runs):
        # 逐步平均（实线圆点）vs 终局判定（虚线方块）
        panels.append(('ev', 'Evidence accuracy',
                       'Evidence accuracy',
                       'special'))

    n = len(panels)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 3.4 * n_rows), squeeze=False)
    axes = [ax for row in axes for ax in row]

    for ax, (field, title, ylabel, special) in zip(axes, panels):
        if special == 'special':
            _line(ax, runs, 'ev_acc', title + ' (per-step)', ylabel,
                  styles=[('o', '-')] * len(runs))
            _line(ax, runs, 'ev_acc_final', title + ' (final)', ylabel,
                  styles=[('s', '--')] * len(runs))
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=7)
        else:
            _line(ax, runs, field, title, ylabel)

    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Curves saved to {out_path}")


def plot_ev_bar(runs: list, out_path: Path) -> None:
    """证据准确率 bar（最后一轮）：终局判定为主指标，逐步平均为参照。"""
    import numpy as np

    runs = [r for r in runs
            if r['ev_acc_final'] and r['ev_acc_final'][-1] is not None]
    if not runs:
        print('(无 ev_acc_final 字段，跳过证据 bar)')
        return
    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(runs) + 2), 3.6))
    x = np.arange(len(runs))
    width = 0.36
    finals = [r['ev_acc_final'][-1] for r in runs]
    steps = [r['ev_acc'][-1] if r['ev_acc'] and r['ev_acc'][-1] is not None
             else 0.0 for r in runs]
    b1 = ax.bar(x - width / 2, finals, width, color='#C44E52',
                label='final (terminal judgment)')
    b2 = ax.bar(x + width / 2, steps, width, color='#4C72B0', alpha=0.55,
                label='per-step avg')
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in runs], fontsize=8)
    ax.set_ylabel('Evidence accuracy (last epoch)')
    ax.set_title('Evidence accuracy: terminal vs per-step', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Evidence bar saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='训练曲线可视化（loss/奖励/准确率/证据准确率 + 证据 bar）')
    parser.add_argument('--json', nargs='+', required=True,
                        help='一个或多个训练日志（classifier history.json / '
                             'BC·RL training_log.json）')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='与 --json 对齐的曲线名（默认从路径推导）')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认第一个 JSON 所在目录）')
    parser.add_argument('--name', default='training_curves',
                        help='输出文件名前缀')
    args = parser.parse_args()

    paths = [Path(p) for p in args.json]
    labels = args.labels or [_default_label(p) for p in paths]
    if len(labels) != len(paths):
        raise SystemExit(f'--labels 数量（{len(labels)}）须与 --json 一致（{len(paths)}）')

    runs = [_extract(p, lb) for p, lb in zip(paths, labels)]
    runs = [r for r in runs if r['epochs']]
    if not runs:
        raise SystemExit('所有输入都没有可绘制的记录')

    out_dir = Path(args.output_dir) if args.output_dir else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(runs, out_dir / f'{args.name}_source.csv')
    print(f"Source data saved to {out_dir / f'{args.name}_source.csv'}")

    plot_curves(runs, out_dir / f'{args.name}_curves.png')
    plot_ev_bar(runs, out_dir / f'{args.name}_ev_bar.png')


if __name__ == '__main__':
    main()

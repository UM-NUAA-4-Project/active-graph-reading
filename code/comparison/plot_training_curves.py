"""训练曲线可视化：loss 曲线（classifier / BC / RL）+ RL 奖励曲线。

接受一个或多个训练日志 JSON，自动识别类型：
  - classifier history.json（dict 含 'epochs' 列表，字段 loss/train_acc/test_acc）
  - BC  training_log.json（list，字段 epoch/train_loss）
  - RL  training_log.json（list，字段 epoch/train_loss/train_avg_R/train_acc）

输出（默认第一个 JSON 所在目录，--output-dir 覆盖）：
  - {name}_loss.png     所有输入的 train loss 曲线（一文件一线）
  - {name}_reward.png   RL 奖励曲线（只有含 train_avg_R 的输入；有 train_acc
                        时附 accuracy 面板）
  - {name}_source.csv   曲线源数据（run, epoch, loss, reward, acc）

用法:
  python code/comparison/plot_training_curves.py --json \
      result/cifar10/classifier/finetune2/history.json \
      result/cifar10/experiments/1-teacher/attention/bc/training_log.json \
      result/cifar10/experiments/1-teacher/attention/rl/training_log.json
  # 多 run 对比（如 base vs attention）：--labels 名字列表 与 --json 对齐
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
    """日志文件 → {'epochs': [], 'loss': [], 'reward': [], 'acc': []}（缺则空）。"""
    data = _load(path)
    if isinstance(data, dict) and isinstance(data.get('epochs'), list):
        records = data['epochs']          # classifier history.json
    elif isinstance(data, list):
        records = data                    # BC/RL training_log.json
    else:
        raise ValueError(f"{path}: 无法识别的训练日志格式（期望 history.json 或 "
                         f"training_log.json）")

    out = {'epochs': [], 'loss': [], 'reward': [], 'acc': []}
    for rec in records:
        if not isinstance(rec, dict) or 'epoch' not in rec:
            continue
        loss = rec.get('train_loss', rec.get('loss'))
        reward = rec.get('train_avg_R', rec.get('avg_R'))
        acc = rec.get('train_acc')
        if loss is None and reward is None:
            continue
        out['epochs'].append(int(rec['epoch']))
        out['loss'].append(loss)
        out['reward'].append(reward)
        out['acc'].append(acc)
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
        w.writerow(['run', 'epoch', 'loss', 'reward', 'acc'])
        for r in runs:
            for i, ep in enumerate(r['epochs']):
                loss = r['loss'][i]
                reward = r['reward'][i]
                acc = r['acc'][i]
                w.writerow([r['label'], ep,
                            f'{loss:.6f}' if loss is not None else '',
                            f'{reward:.6f}' if reward is not None else '',
                            f'{acc:.6f}' if acc is not None else ''])


def plot_loss(runs: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, r in enumerate(runs):
        ax.plot(r['epochs'], r['loss'], marker='o', markersize=3,
                linewidth=1.2, color=COLORS[i % len(COLORS)],
                label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    ax.set_title('Training loss')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {out_path}")


def plot_reward(runs: list, out_path: Path) -> None:
    runs = [r for r in runs if any(v is not None for v in r['reward'])]
    if not runs:
        print('(无含 reward 的 run，跳过奖励曲线)')
        return
    with_acc = any(a is not None for r in runs for a in r['acc'])
    n_panels = 2 if with_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 3.6))
    axes = [axes] if n_panels == 1 else list(axes)

    for i, r in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        epochs = r['epochs']
        if with_acc:
            reward_ax, acc_ax = axes
            reward_ax.plot(epochs, r['reward'], marker='o', markersize=3,
                           linewidth=1.2, color=color, label=r['label'])
            acc_ax.plot(epochs, r['acc'], marker='s', markersize=3,
                        linewidth=1.0, linestyle='--', color=color,
                        label=r['label'])
        else:
            axes[0].plot(epochs, r['reward'], marker='o', markersize=3,
                         linewidth=1.2, color=color, label=r['label'])

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Avg reward')
    axes[0].set_title('RL reward')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    if with_acc:
        acc_ax.set_xlabel('Epoch')
        acc_ax.set_ylabel('Rollout accuracy')
        acc_ax.set_title('RL rollout accuracy')
        acc_ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Reward curve saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='训练曲线可视化（loss + RL 奖励）')
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

    plot_loss(runs, out_dir / f'{args.name}_loss.png')
    if any(r['reward'][0] is not None for r in runs):
        plot_reward(runs, out_dir / f'{args.name}_reward.png')
    else:
        print('(无 reward 字段，跳过奖励曲线)')


if __name__ == '__main__':
    main()

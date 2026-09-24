"""临时观察图：scratch-RL vs 两阶段链的 reader 分类头奖励曲线对比（cifar10/collab）。

所有 epoch 统一按 R = ev_acc_final − 0.1·(1 − ev_acc_final) 换算
（= 1.1·ev_acc_final − 0.1，即 asymmetric 变换作用在 reader 证据头终判上；
原始奖励为 binary 的 run 重算仅统一纵轴刻度、不改变形状）。

  cifar10: 两阶段链 = default RL（classifier 判定）9 轮 + ev-R-read
           （reader 判定）6 轮，共 15 轮；scratch-RL 从零 15 轮。
  collab : 两阶段链 = evidence1 RL 10 轮（BC epoch2 起步，binary + 冻结
           分类器判定）；scratch-RL 从零 10 轮。

用法:
  python code/comparison/plot_rl_reward_scratch_compare.py [cifar10|collab]
产物: result/{ds}/comparison/rl_reward_scratch_vs_evR.{png,pdf}
"""

import json
import sys
from pathlib import Path

import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from tools.paths import CODE_DIR

BLUE_MAIN = '#0F4D92'    # 两阶段链（主结果）
NEUTRAL_MID = '#767676'  # scratch-RL（从零基线）

mpl_rc = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
}

RESULT = CODE_DIR / 'result'

DATASETS = {
    'cifar10': dict(
        title='CIFAR10', ylim=(0.10, 0.27), divider_x=9.5,
        chain=[
            (RESULT / 'cifar10/experiments/default/evidence/rl/training_log.json', 0),
            (RESULT / 'cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt', 9),
        ],
        scratch=RESULT / 'cifar10/experiments/scratch-rl/evidence/rl/checkpoint_epoch15.pt',
    ),
    'collab': dict(
        title='COLLAB', ylim=(0.31, 0.62), divider_x=None,
        chain=[
            (RESULT / 'collab/experiments/evidence1/evidence/rl/checkpoint_epoch10.pt', 0),
        ],
        scratch=RESULT / 'collab/experiments/scratch-rl/evidence/rl/training_log.json',
    ),
}


def reward(ev_acc_final: float) -> float:
    return ev_acc_final - 0.1 * (1 - ev_acc_final)


def load_log(path: Path) -> list:
    if path.suffix == '.pt':
        return torch.load(str(path), map_location='cpu',
                          weights_only=False)['training_log']
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'
    cfg = DATASETS[ds]

    plt.rcParams.update(mpl_rc)
    fig, ax = plt.subplots(figsize=(89 / 25.4, 58 / 25.4))
    fig.subplots_adjust(left=0.145, right=0.965, bottom=0.165, top=0.90)

    if cfg['divider_x'] is not None:
        ax.axvline(cfg['divider_x'], ls=(0, (3, 2)), lw=0.7,
                   color='#9A9A9A', zorder=1)

    epochs, rewards = [], []
    for path, offset in cfg['chain']:
        for e in load_log(path):
            epochs.append(e['epoch'] + offset)
            rewards.append(reward(e['ev_acc_final']))
    ax.plot(epochs, rewards, color=BLUE_MAIN, lw=1.2, marker='o', ms=2.6,
            mfc=BLUE_MAIN, mec='white', mew=0.4, zorder=3, label='ev-R-read')

    log = load_log(cfg['scratch'])
    ax.plot([e['epoch'] for e in log], [reward(e['ev_acc_final']) for e in log],
            color=NEUTRAL_MID, lw=1.2, marker='o', ms=2.6,
            mfc=NEUTRAL_MID, mec='white', mew=0.4, zorder=3,
            label='scratch-RL')

    ax.set_xlabel('RL epoch', fontsize=6.5)
    ax.set_ylabel('Mean training reward', fontsize=7)
    n = max(epochs)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(*cfg['ylim'])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.tick_params(labelsize=6)
    ax.legend(loc='lower right', fontsize=6, handlelength=1.6,
              borderaxespad=0.2)

    ax.set_title(cfg['title'], loc='left', fontsize=8, fontweight='bold',
                 pad=7)

    out_dir = RESULT / ds / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / 'rl_reward_scratch_vs_evR'
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {base}.png / .pdf')


if __name__ == '__main__':
    main()

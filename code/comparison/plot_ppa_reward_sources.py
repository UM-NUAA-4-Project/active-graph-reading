"""PPA 换源奖励链曲线（evidence reader，单面板 Nature 风格）。

PPA 换源实验为一条三阶段训练链（后一阶段 bootstrap 前一阶段 checkpoint）：
  Stage 1  ev256-classifier  3 轮  asymmetric + 冻结分类器判定
  Stage 2  ev256-soft-both   4 轮  软 both（R = p_clf(y)·p_reader(y)）
  Stage 3  ev256-both        1 轮  硬 both（双方 argmax 同对才给 1）

三阶段各自的 train_avg_R 量纲不同（判定准确率变换 / 概率乘积 / 硬联合
命中率），不可同轴比较；图内统一换算为
  R = ev_acc_final - 0.1·(1-ev_acc_final) = 1.1·ev_acc_final - 0.1
（即 reader 判定 asymmetric 的 R̄ 恒等式，与主结果 ev256 曲线及 collab
统一口径一致），只反映 reader 分类头终判质量随链式训练的变化。

图式与主结果奖励曲线家族一致：单条蓝色连续曲线，阶段边界相连，分隔虚
线 + 阶段奖励源名标注（本图主题即换源，故标注奖励源名）。

产物（写入 result/ppa/comparison/）：
  rl_reward_sources.pdf / .svg / .tiff(600dpi) / .png(600dpi)
  rl_reward_sources.alignment.json（单面板 → NOT APPLICABLE 记录）

用法:
  python code/comparison/plot_ppa_reward_sources.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

SKILL_SCRIPTS = Path.home() / '.zcode' / 'skills' / 'nature-figure' / 'scripts'

mpl_rc = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
    'legend.frameon': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
}

BLUE_MAIN = '#0F4D92'
NEUTRAL_DARK = '#4D4D4D'
DIVIDER_GRAY = '#9A9A9A'

FIG_W_IN, FIG_H_IN = 89 / 25.4, 58 / 25.4   # Nature 单栏 89mm
DPI = 600

ROOT = Path(__file__).resolve().parents[1] / 'result'
OUT_DIR = ROOT / 'ppa/comparison'


def unified_reward(entry: dict) -> float:
    """R = ev_acc_final - 0.1·(1-ev_acc_final)（统一口径，见模块 docstring）。"""
    p = entry['ev_acc_final']
    return p - 0.1 * (1 - p)


# ---- 三阶段链（epochs_offset 把各阶段 epoch 重排到连续轴上） ----
STAGES = [
    dict(log=ROOT / 'ppa/experiments/ev256-classifier/evidence/rl/training_log.json',
         offset=0, label='Classifier', label_x=2.0),
    dict(log=ROOT / 'ppa/experiments/ev256-soft-both/evidence/rl/training_log.json',
         offset=3, label='Soft-both', label_x=5.5),
    dict(log=ROOT / 'ppa/experiments/ev256-both/evidence/rl/training_log.json',
         offset=7, label='Both', label_x=8.0),
]
DIVIDER_X = (3.5, 7.5)


def load_stages() -> list:
    stages = []
    for st in STAGES:
        with open(st['log'], 'r', encoding='utf-8') as f:
            log = json.load(f)
        if not log:
            raise SystemExit(f'空训练日志：{st["log"]}')
        stages.append((st, log))
    return stages


def plot_figure(stages: list) -> Path:
    plt.rcParams.update(mpl_rc)
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.145, right=0.965, bottom=0.165, top=0.775)

    for x in DIVIDER_X:
        ax.axvline(x, ls=(0, (3, 2)), lw=0.7, color=DIVIDER_GRAY, zorder=1)
    for st, _ in stages:
        ax.text(st['label_x'], 0.95, st['label'],
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=NEUTRAL_DARK)

    prev_last = None
    for st, log in stages:
        epochs = [e['epoch'] + st['offset'] for e in log]
        rewards = [unified_reward(e) for e in log]
        line_x, line_y = epochs, rewards
        if prev_last is not None:
            # 阶段边界处曲线相连：从前一阶段最后一点接入；
            # 标记只画在实际数据点上，不重复画边界点
            line_x = [prev_last[0]] + line_x
            line_y = [prev_last[1]] + line_y
        ax.plot(line_x, line_y, color=BLUE_MAIN, lw=1.2, zorder=3)
        ax.plot(epochs, rewards, 'o', color=BLUE_MAIN, ms=2.6,
                mfc=BLUE_MAIN, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], rewards[-1])

    ax.set_xlabel('RL epoch', fontsize=6.5)
    ax.set_ylabel('Mean training reward', fontsize=7)
    ax.set_ylim(0.17, 0.50)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlim(0.5, 8.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(labelsize=6)

    ax.set_title('PPA', loc='left', fontsize=8, fontweight='bold', pad=7)

    # ---- 渲染时对齐门禁（单面板 → NOT APPLICABLE 记录留档） ----
    try:
        import sys
        sys.path.insert(0, str(SKILL_SCRIPTS))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT_DIR / 'rl_reward_sources.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    except ImportError:
        print('  [warn] audit_panel_alignment 不可用，跳过对齐记录')

    base = OUT_DIR / 'rl_reward_sources'
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.tiff', dpi=DPI, bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return base


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = plot_figure(load_stages())
    print(f'[ppa] saved: {base}.png/.pdf/.svg/.tiff')


if __name__ == '__main__':
    main()

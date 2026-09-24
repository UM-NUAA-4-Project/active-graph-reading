"""COLLAB 切换奖励函数对比图（evidence reader，单面板 Nature 风格）。

数据来源为用户手工整理的 result/collab/experiments/history.json（14 轮）：
第 1–6 轮 asymmetric + 冻结分类器判定（train_avg_R = 1.1·train_acc − 0.1，
平台 ≈0.075）；第 7 轮起切换为 reader 自带头终判，日志记录的 train_avg_R
为 reader 原始口径（= ev_acc_final），统一乘 1.1 减 0.1 折算为 asymmetric
形式后与第一阶段同轴。

图式与 cifar10 切换图（plot_cifar10_reward_switch.py）一致：左轴蓝色
训练平均奖励，右轴橙色最终判别精度 ev_acc_final，虚线（切换后曲线段）
+ 阴影（切换后背景区域 axvspan）标记奖励源更换。

产物（写入 result/collab/comparison/）：
  rl_reward_switch.pdf / .svg / .tiff(600dpi) / .png(600dpi)
  rl_reward_switch.alignment.json（单面板 → NOT APPLICABLE 记录）

用法:
  python code/comparison/plot_collab_reward_switch.py
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
ACC_COLOR = '#D55E00'   # 最终判别精度曲线（右轴）
NEUTRAL_DARK = '#4D4D4D'

FIG_W_IN, FIG_H_IN = 89 / 25.4, 58 / 25.4   # Nature 单栏 89mm
DPI = 600

ROOT = Path(__file__).resolve().parents[1] / 'result'
OUT_DIR = ROOT / 'collab/comparison'
LOG = ROOT / 'collab/experiments/history.json'

SWITCH_EPOCH = 7         # 第 7 轮起 reader 源原始口径，需折算
READER_TO_ASYMMETRIC = lambda r: r * 1.1 - 0.1

# ---- 两阶段链（offset 排到连续轴） ----
STAGES = [
    dict(offset=0, dashed=False, label='soft-both', label_x=3.5),
    dict(offset=0, dashed=True, label='Reader', label_x=9.0),  # epoch 本身连续，无需 offset
]
SWITCH_X = 6.5           # 奖励源更换边界（阶段 6 轮之后）
SHADE_SPAN = (6.5, 14.5)  # 阴影标记切换后的奖励源区域


def in_stage2(epoch: int) -> bool:
    return epoch >= SWITCH_EPOCH


def load_log() -> list:
    with open(LOG, 'r', encoding='utf-8') as f:
        log = json.load(f)
    if not log:
        raise SystemExit(f'空训练日志：{LOG}')
    return log


def plot_figure(log: list) -> Path:
    plt.rcParams.update(mpl_rc)
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.145, right=0.86, bottom=0.165, top=0.775)
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)

    # 阴影：切换后的奖励源区域（置于曲线下层）
    ax.axvspan(*SHADE_SPAN, color=BLUE_MAIN, alpha=0.07, lw=0, zorder=0)

    # 奖励源标注：各阶段区域顶部居中
    for st in STAGES:
        ax.text(st['label_x'], 0.95, st['label'],
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=NEUTRAL_DARK)

    def join_prev(prev_last, xs, ys):
        if prev_last is None:
            return xs, ys
        # 阶段边界处曲线相连：从前一阶段最后一点接入
        return [prev_last[0]] + xs, [prev_last[1]] + ys

    # 左轴：训练平均奖励（第 7 轮起折算为 asymmetric 形式）
    prev_last = None
    for st in STAGES:
        seg = [e for e in log if in_stage2(e['epoch']) == st['dashed']]
        epochs = [e['epoch'] + st['offset'] for e in seg]
        rewards = [e['train_avg_R'] if not st['dashed']
                   else READER_TO_ASYMMETRIC(e['train_avg_R']) for e in seg]
        line_x, line_y = join_prev(prev_last, epochs, rewards)
        ax.plot(line_x, line_y, color=BLUE_MAIN, lw=1.2, zorder=3,
                ls=(0, (3, 2)) if st['dashed'] else '-')
        ax.plot(epochs, rewards, 'o', color=BLUE_MAIN, ms=2.6,
                mfc=BLUE_MAIN, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], rewards[-1])

    # 右轴：最终判别精度（reader 自带头终判；切换后虚线，与奖励曲线同式）
    prev_last = None
    for st in STAGES:
        seg = [e for e in log if in_stage2(e['epoch']) == st['dashed']]
        epochs = [e['epoch'] + st['offset'] for e in seg]
        accs = [e['ev_acc_final'] for e in seg]
        line_x, line_y = join_prev(prev_last, epochs, accs)
        ax2.plot(line_x, line_y, color=ACC_COLOR, lw=1.2, zorder=3,
                 ls=(0, (3, 2)) if st['dashed'] else '-')
        ax2.plot(epochs, accs, 'o', color=ACC_COLOR, ms=2.6,
                 mfc=ACC_COLOR, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], accs[-1])

    ax.set_xlabel('RL epoch', fontsize=6.5)
    ax.set_ylabel('Mean training reward', fontsize=7)
    ax.set_ylim(0.05, 0.70)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 右轴与左轴同量程：reader 源 asymmetric 奖励(1.1a-0.1)恒低于 acc，
    # 同量程保证 ev_acc_final 曲线始终画在奖励曲线上方，避免双轴误导
    ax2.set_ylabel('Final accuracy', fontsize=7)
    ax2.set_ylim(0.05, 0.70)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    n_epochs = max(epochs)
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(labelsize=6)
    ax2.tick_params(labelsize=6)

    handles = [
        plt.Line2D([], [], color=BLUE_MAIN, lw=1.2, marker='o', ms=2.6,
                   mec='white', mew=0.4, label='Mean training reward'),
        plt.Line2D([], [], color=ACC_COLOR, lw=1.2, marker='o', ms=2.6,
                   mec='white', mew=0.4, label='Final accuracy'),
    ]
    ax2.legend(handles=handles, loc='lower right', fontsize=5.5,
               handlelength=1.6, borderaxespad=0.3)

    ax.set_title('COLLAB', loc='left', fontsize=8, fontweight='bold', pad=7)

    # ---- 渲染时对齐门禁（单面板 → NOT APPLICABLE 记录留档） ----
    try:
        import sys
        sys.path.insert(0, str(SKILL_SCRIPTS))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT_DIR / 'rl_reward_switch.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    except ImportError:
        print('  [warn] audit_panel_alignment 不可用，跳过对齐记录')

    base = OUT_DIR / 'rl_reward_switch'
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.tiff', dpi=DPI, bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return base


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = plot_figure(load_log())
    print(f'[collab] saved: {base}.png/.pdf/.svg/.tiff')


if __name__ == '__main__':
    main()

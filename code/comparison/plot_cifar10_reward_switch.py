"""cifar10 切换奖励函数对比图（evidence reader，单面板 Nature 风格）。

主结果链：default 实验 9 轮（asymmetric + 冻结分类器判定，平台 ≈0.11）
→ 从其 epoch9 checkpoint 续训 ev-R-read 6 轮（asymmetric + reader 自带头
终判，升至 0.256）。左轴为日志记录的 train_avg_R；右轴叠加最终判别
精度 ev_acc_final（reader 自带头终判准确率），体现性能随链式训练的变化。

与主结果奖励曲线（plot_rl_reward_curves.py，中性 Stage 标注）不同，本图
专为主题"切换奖励函数"而画：以虚线（切换后曲线段）+ 阴影（切换后背景
区域 axvspan）标记奖励源更换，无阶段文字标注。

产物（写入 result/cifar10/comparison/）：
  rl_reward_switch.pdf / .svg / .tiff(600dpi) / .png(600dpi)
  rl_reward_switch.alignment.json（单面板 → NOT APPLICABLE 记录）

用法:
  python code/comparison/plot_cifar10_reward_switch.py
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
OUT_DIR = ROOT / 'cifar10/comparison/change-reward-sourse'

# ---- 两阶段链（都读日志记录的 train_avg_R；offset 排到连续轴） ----
STAGES = [
    dict(log=ROOT / 'cifar10/experiments/default/evidence/rl/training_log.json',
         offset=0, dashed=False, label='Soft-both', label_x=5.0),
    dict(log=ROOT / 'cifar10/experiments/ev-R-read/evidence/rl/training_log.json',
         offset=9, dashed=True, label='Reader', label_x=12.5),
]
SWITCH_X = 9.5          # 奖励源更换边界（阶段 9 轮之后）
SHADE_SPAN = (9.5, 15.5)  # 阴影标记切换后的奖励源区域


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
    fig.subplots_adjust(left=0.145, right=0.86, bottom=0.165, top=0.775)
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)

    # 阴影：切换后的奖励源区域（置于曲线下层）
    ax.axvspan(*SHADE_SPAN, color=BLUE_MAIN, alpha=0.07, lw=0, zorder=0)

    # 奖励源标注：各阶段区域顶部居中
    for st, _ in stages:
        ax.text(st['label_x'], 0.95, st['label'],
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=NEUTRAL_DARK)

    def join_prev(prev_last, xs, ys):
        if prev_last is None:
            return xs, ys
        # 阶段边界处曲线相连：从前一阶段最后一点接入
        return [prev_last[0]] + xs, [prev_last[1]] + ys

    # 左轴：训练平均奖励
    prev_last = None
    for st, log in stages:
        epochs = [e['epoch'] + st['offset'] for e in log]
        rewards = [e['train_avg_R'] for e in log]
        line_x, line_y = join_prev(prev_last, epochs, rewards)
        ax.plot(line_x, line_y, color=BLUE_MAIN, lw=1.2, zorder=3,
                ls=(0, (3, 2)) if st['dashed'] else '-')
        ax.plot(epochs, rewards, 'o', color=BLUE_MAIN, ms=2.6,
                mfc=BLUE_MAIN, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], rewards[-1])

    # 右轴：最终判别精度（reader 自带头终判）
    prev_last = None
    for st, log in stages:
        epochs = [e['epoch'] + st['offset'] for e in log]
        accs = [e['ev_acc_final'] for e in log]
        line_x, line_y = join_prev(prev_last, epochs, accs)
        ax2.plot(line_x, line_y, color=ACC_COLOR, lw=1.2, zorder=3,
                 ls=(0, (3, 2)) if st['dashed'] else '-')
        ax2.plot(epochs, accs, 'o', color=ACC_COLOR, ms=2.6,
                 mfc=ACC_COLOR, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], accs[-1])

    ax.set_xlabel('RL epoch', fontsize=6.5)
    ax.set_ylabel('Mean training reward', fontsize=7)
    ax.set_ylim(0.08, 0.34)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 右轴与左轴同量程：reader 源 asymmetric 奖励(1.1a-0.1)恒低于 acc，
    # 同量程保证 ev_acc_final 曲线始终画在奖励曲线上方，避免双轴误导
    ax2.set_ylabel('Final accuracy', fontsize=7)
    ax2.set_ylim(0.08, 0.34)
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    n_epochs = max(e['epoch'] + st['offset'] for st, log in stages for e in log)
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

    ax.set_title('CIFAR10', loc='left', fontsize=8, fontweight='bold', pad=7)

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
    base = plot_figure(load_stages())
    print(f'[cifar10] saved: {base}.png/.pdf/.svg/.tiff')


if __name__ == '__main__':
    main()

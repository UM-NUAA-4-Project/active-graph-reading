"""cifar10 切换奖励函数对比图（evidence reader，单面板 Nature 风格）。

主结果链：default 实验 9 轮（asymmetric + soft-both 判定，平台 ≈0.11）
→ 从其 epoch9 checkpoint 续训 ev-R-read 6 轮（asymmetric + reader 自带头
终判，升至 0.256）。

2026-09-25 起图规格（跨数据集 1×1 版，供 reward-curves/ 拼图与单独引用）：
只绘制最终判别精度 ev_acc_final 曲线（奖励源量纲不同不再同轴画奖励）；
阴影（切换后区域，blue 0.07，以 cross-dataset-figures/reward-curves/
rl_reward_curves_1x4.svg 存档版为准）标记奖励源更换；切换点处把前一阶
段收敛末值以水平虚线延伸到图右端，作为对比基准。

产物（写入 result/cross-dataset-figures/reward-curves/）：
  rl_reward_switch_cifar10.pdf / .png(600dpi) / .svg
  rl_reward_switch_cifar10.alignment.json（单面板 → NOT APPLICABLE 记录）

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
NEUTRAL_DARK = '#4D4D4D'
# 阴影 = BLUE_MAIN 7% 白底预混实色（视觉同 rgba(15,77,146,0.07)）。
# 不用 CSS opacity：PyMuPDF 等渲染器对 opacity 元素会整组渲染坏。
SHADE_COLOR = '#EEF2F8'

FIG_W_IN, FIG_H_IN = 89 / 25.4, 58 / 25.4   # Nature 单栏 89mm
DPI = 600

ROOT = Path(__file__).resolve().parents[1] / 'result'
OUT_DIR = ROOT / 'cross-dataset-figures/reward-curves'

# ---- 两阶段链（都读日志 ev_acc_final；offset 排到连续轴） ----
STAGES = [
    dict(log=ROOT / 'cifar10/experiments/default/evidence/rl/training_log.json',
         offset=0, dashed=False, label='Soft-both', label_x=5.0),
    dict(log=ROOT / 'cifar10/experiments/ev-R-read/evidence/rl/training_log.json',
         offset=9, dashed=True, label='Reader', label_x=12.5),
]
SWITCH_X = 9.5          # 奖励源更换边界（阶段 9 轮之后）
SHADE_SPAN = (9.5, 15.5)  # 阴影标记切换后的奖励源区域（同 1×4 存档版）


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
    fig.subplots_adjust(left=0.135, right=0.965, bottom=0.165, top=0.775)

    # 阴影：切换后的奖励源区域（置于曲线下层）
    ax.axvspan(*SHADE_SPAN, color=SHADE_COLOR, lw=0, zorder=0)

    # 奖励源标注：各阶段区域顶部居中
    for st, _ in stages:
        ax.text(st['label_x'], 0.95, st['label'],
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=NEUTRAL_DARK)

    # 切换点水平虚线：前一阶段收敛末值延伸到图右端作对比基准
    n_epochs = max(e['epoch'] + st['offset'] for st, log in stages for e in log)
    accs_prev = [e['ev_acc_final'] for _, log in stages[:1] for e in log]
    ax.axhline(accs_prev[-1], xmin=(SWITCH_X - 0.5) / n_epochs,
               color=NEUTRAL_DARK, lw=0.8, ls=(0, (3, 2)), zorder=2)

    prev_last = None
    for st, log in stages:
        epochs = [e['epoch'] + st['offset'] for e in log]
        accs = [e['ev_acc_final'] for e in log]
        line_x, line_y = epochs, accs
        if prev_last is not None:
            # 阶段边界处曲线相连：从前一阶段最后一点接入
            line_x = [prev_last[0]] + line_x
            line_y = [prev_last[1]] + line_y
        ax.plot(line_x, line_y, color=BLUE_MAIN, lw=1.2, zorder=3,
                ls=(0, (3, 2)) if st['dashed'] else '-')
        ax.plot(epochs, accs, 'o', color=BLUE_MAIN, ms=2.6,
                mfc=BLUE_MAIN, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], accs[-1])

    ax.set_xlabel('RL epoch', fontsize=6.5)
    ax.set_ylabel('Final accuracy', fontsize=7)
    ax.set_ylim(0.20, 0.35)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(labelsize=6)

    ax.set_title('CIFAR10', loc='left', fontsize=8, fontweight='bold', pad=7)

    # ---- 渲染时对齐门禁（单面板 → NOT APPLICABLE 记录留档） ----
    try:
        import sys
        sys.path.insert(0, str(SKILL_SCRIPTS))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT_DIR / 'rl_reward_switch_cifar10.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    except ImportError:
        print('  [warn] audit_panel_alignment 不可用，跳过对齐记录')

    base = OUT_DIR / 'rl_reward_switch_cifar10'
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return base


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = plot_figure(load_stages())
    print(f'[cifar10] saved: {base}.png/.pdf/.svg')


if __name__ == '__main__':
    main()

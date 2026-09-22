"""RL 主结果实验奖励曲线（evidence reader，三数据集各一张 Nature 风格单面板）。

只画 evidence reader 主结果的 RL 训练奖励曲线（train_avg_R：每条轨迹奖励
的 epoch 均值），奖励判定源（reward source）用统一颜色语言跨图区分：
  - 蓝 = reader 自身证据头终判（reward-source reader）
  - 灰 = 冻结上界分类器判定（reward-source classifier，旧 schema 缺省）
分阶段切换奖励源的实验（cifar10：先 classifier 判定收敛，再 reader 判定
微调）画在同一条连续 epoch 轴上：曲线在阶段边界处相连（用户要求节点
连贯），颜色在分隔虚线处切换，边界点标记保留上一阶段颜色。

奖励变换（train_reader/core/common.py::compute_reward）：
  asymmetric = 判对 +1 / 判错 -0.1（故 R̄ = 1.1·判定准确率 - 0.1）
  binary     = 判对 1 / 判错 0（故 R̄ = 判定准确率）

产物（写入各自 rl 结果目录）：
  rl_reward_curve.pdf / .svg / .tiff(600dpi) / .png(600dpi)
  rl_reward_curve.alignment.json（单面板 → NOT APPLICABLE 记录）

用法:
  python code/comparison/plot_rl_reward_curves.py            # 三数据集全画
  python code/comparison/plot_rl_reward_curves.py cifar10    # 只画指定数据集
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# ---- nature-figure 样式契约（89mm 单栏，Arial，≥5pt，可编辑文本） ----
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

BLUE_MAIN = '#0F4D92'      # reader 证据头判定（主结果信号色）
NEUTRAL_MID = '#767676'    # 冻结分类器判定（中性参照色）
NEUTRAL_DARK = '#4D4D4D'
DIVIDER_GRAY = '#9A9A9A'

FIG_W_IN, FIG_H_IN = 89 / 25.4, 58 / 25.4   # Nature 单栏 89mm
DPI = 600

ROOT = Path(__file__).resolve().parents[1] / 'result'

# ---- 主结果实验定义（阶段按训练时间顺序排列；epochs_offset 用于连续轴） ----
EXPERIMENTS = [
    dict(
        key='ppa', title='PPA',
        out_dir=ROOT / 'ppa/experiments/ev256/evidence/rl',
        subtitle='Asymmetric reward (+1 / -0.1) · reader-judged',
        ylim=(0.17, 0.40), ytick=0.05, yfmt='%.2f',
        stages=[dict(log='training_log.json', source='reader')],
    ),
    dict(
        key='cifar10', title='CIFAR10',
        out_dir=ROOT / 'cifar10/experiments/ev-R-read/evidence/rl',
        subtitle='Asymmetric reward (+1 / -0.1)',
        ylim=(0.09, 0.28), ytick=0.05, yfmt='%.2f',
        # 主结果链：先 classifier 判定收敛（default 实验，9 轮），
        # 再从其 epoch9 checkpoint 续训 reader 判定（ev-R-read，3 轮）
        stages=[
            dict(log=ROOT / 'cifar10/experiments/default/evidence/rl/training_log.json',
                 source='classifier', epochs_offset=0,
                 label='Stage 1\nclassifier-judged', label_x=5.0),
            dict(log='training_log.json', source='reader', epochs_offset=9,
                 label='Stage 2\nreader-judged', label_x=10.9),
        ],
        divider_x=9.5, span=(9.5, 12.5),
    ),
    dict(
        key='collab', title='COLLAB',
        out_dir=ROOT / 'collab/experiments/evidence1/evidence/rl',
        subtitle='Binary reward (1 / 0) · classifier-judged',
        ylim=(0.174, 0.187), ytick=0.005, yfmt='%.3f',
        stages=[dict(log='training_log.json', source='classifier')],
    ),
]

SOURCE_COLOR = {'reader': BLUE_MAIN, 'classifier': NEUTRAL_MID}


def load_log(path: Path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        log = json.load(f)
    if not log:
        raise SystemExit(f'空训练日志：{path}')
    return log


def plot_figure(exp: dict) -> Path:
    plt.rcParams.update(mpl_rc)
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.155, right=0.965, bottom=0.165, top=0.775)

    stages = []
    for st in exp['stages']:
        st_path = Path(st['log'])
        if not st_path.is_absolute():
            st_path = exp['out_dir'] / st_path
        stages.append((st, load_log(st_path)))

    if exp.get('span'):
        ax.axvspan(*exp['span'], color=BLUE_MAIN, alpha=0.05, zorder=0, lw=0)
    if exp.get('divider_x'):
        ax.axvline(exp['divider_x'], ls=(0, (3, 2)), lw=0.7,
                   color=DIVIDER_GRAY, zorder=1)

    prev_last = None
    for st, log in stages:
        epochs = [e['epoch'] + st.get('epochs_offset', 0) for e in log]
        rewards = [e['train_avg_R'] for e in log]
        color = SOURCE_COLOR[st['source']]
        line_x, line_y = epochs, rewards
        if prev_last is not None:
            # 阶段边界处曲线相连：从前一阶段最后一点接入；
            # 边界点标记保留上一阶段颜色，不重复画
            line_x = [prev_last[0]] + line_x
            line_y = [prev_last[1]] + line_y
        ax.plot(line_x, line_y, color=color, lw=1.2, zorder=3)
        ax.plot(epochs, rewards, 'o', color=color, ms=2.6,
                mfc=color, mec='white', mew=0.4, ls='none', zorder=4)
        prev_last = (epochs[-1], rewards[-1])

    if exp.get('span'):
        ax.text(exp['stages'][0]['label_x'], 0.95,
                exp['stages'][0]['label'], transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=NEUTRAL_DARK,
                linespacing=1.15)
        ax.text(exp['stages'][1]['label_x'], 0.95,
                exp['stages'][1]['label'], transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=BLUE_MAIN,
                linespacing=1.15)

    ax.set_xlabel('RL epoch', fontsize=6.5)
    ax.set_ylabel('Mean training reward', fontsize=7)
    ax.set_ylim(*exp['ylim'])
    ax.yaxis.set_major_locator(MultipleLocator(exp['ytick']))
    ax.yaxis.set_major_formatter(FormatStrFormatter(exp['yfmt']))
    n_epochs = max(e['epoch'] + st.get('epochs_offset', 0)
                   for st, log in stages for e in log)
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(labelsize=6)

    ax.set_title(exp['title'], loc='left', fontsize=8, fontweight='bold',
                 pad=11)
    ax.text(0.0, 1.03, exp['subtitle'], transform=ax.transAxes,
            fontsize=6, color=NEUTRAL_DARK, ha='left', va='bottom')

    # ---- 渲染时对齐门禁（单面板 → NOT APPLICABLE 记录留档） ----
    try:
        import sys
        sys.path.insert(0, str(SKILL_SCRIPTS))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(exp['out_dir'] / 'rl_reward_curve.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    except ImportError:
        print('  [warn] audit_panel_alignment 不可用，跳过对齐记录')

    base = exp['out_dir'] / 'rl_reward_curve'
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.tiff', dpi=DPI, bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return base


def main():
    parser = argparse.ArgumentParser(description='RL 主结果奖励曲线（三数据集）')
    parser.add_argument('datasets', nargs='*',
                        help='只画指定数据集（ppa/cifar10/collab；缺省全画）')
    args = parser.parse_args()
    valid = {e['key'] for e in EXPERIMENTS}
    unknown = set(args.datasets) - valid
    if unknown:
        parser.error(f'未知数据集: {sorted(unknown)}（可选: {sorted(valid)}）')
    want = set(args.datasets) or valid
    for exp in EXPERIMENTS:
        if exp['key'] not in want:
            continue
        exp['out_dir'].mkdir(parents=True, exist_ok=True)
        base = plot_figure(exp)
        print(f'[{exp["key"]}] saved: {base}.png/.pdf/.svg/.tiff')


if __name__ == '__main__':
    main()

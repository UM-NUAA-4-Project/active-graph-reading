"""合并行为分析图（1×4 单行）：cifar10 与 COLLAB 主结果 reader 的 b/c 两面板。

图级契约
--------
核心主张：两个真实数据集上，reader 的逐步选点行为呈现同一套可解释规律
——前沿度数偏好（cifar10 早期强偏好后期交叉；COLLAB 全程度数贪心）与
信息增益真实值（cifar10 chosen 节点实际熵减早正晚负；COLLAB 全程为负）。
面板角色（行为分析图 b/c 的跨数据集拼接，编码与原图一致，单行排布）：
  a = cifar10 前沿度数：被选 vs 可选（逐步，1–10 + ≥11 桶）
  b = cifar10 chosen 节点真实信息增益（nats，0 线=揭示不带来熵减）
  c = COLLAB 前沿度数（同 a 编码）
  d = COLLAB chosen 节点真实信息增益（同 b 编码）
数据来源（2026-09-25 起信息增益面板为真实值，排名分位口径弃用）：
  cifar10 = choice_behavior_b10_g100s4.json（原 10k 全量 JSON 已删，本地
  重采 test 前 100 图 × 4 seeds，同协议 argmax rollout）；
  collab = choice_behavior_b10.json（test 501 图原 JSON，聚合同序）。
导出契约：183×52 mm 单行，Arial ≥5 pt，可编辑文字，svg/pdf/tiff/png +
面板对齐门禁（1.5 pt strict）+ PDF 碰撞审计。面板窄于 2×2 版：
x 刻度隔步标注（1,3,5,7,9,≥11），y 轴标题与说明文字用短版。

用法（仓库根运行）:
  python code/comparison/plot_choice_behavior_1x4.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SKILL_SCRIPTS = Path.home() / '.zcode' / 'skills' / 'nature-figure' / 'scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

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

TEAL = '#2E7D64'    # 被选节点（前沿度数）
GRAY = '#767676'    # 可选候选 / 中性
ORANGE = '#B65A1F'  # 信息增益
DARK = '#4D4D4D'

FIG_W_MM, FIG_H_MM = 183, 52
BOOT_REPS = 2000
N_STEP_BUCKETS = 11
TICK_STEPS = [1, 3, 5, 7, 9, 11]
TICK_LABELS = ['1', '3', '5', '7', '9', '≥11']

BASE = Path(__file__).resolve().parents[1] / 'result'
CIF_JSON = BASE / 'cifar10' / 'experiments' / 'ev-R-read' / 'evidence' / 'rl' \
    / 'behavior-analysis' / 'choice_behavior_b10_g100s4.json'
COL_JSON = BASE / 'collab' / 'experiments' / 'evidence1' / 'evidence' / 'rl' \
    / 'behavior-analysis' / 'choice_behavior_b10.json'
OUT_PREFIX = BASE / 'cross-dataset-figures' / 'behavior' / 'choice_behavior_1x4'


def boot_ci(num, den, rng, reps=BOOT_REPS):
    """按图聚类 bootstrap 的 95% CI（与 plot_choice_behavior.py 同款）。"""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    keep = den > 0
    num, den = num[keep], den[keep]
    n = len(num)
    idx = rng.integers(0, n, size=(reps, n))
    m = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def aggregate(path, name):
    """per_graph 记录 → 逐步（1–10 + ≥11 桶）chosen/available 度数与
    chosen 真实 IG 的均值 + 按图聚类 bootstrap 95% CI。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    graphs = data['per_graph']
    per_graph = []
    for g in graphs:
        ch_deg = [0.0] * N_STEP_BUCKETS
        av_deg = [0.0] * N_STEP_BUCKETS
        n_q = [0] * N_STEP_BUCKETS
        ig_val = [0.0] * N_STEP_BUCKETS
        n_ig = [0] * N_STEP_BUCKETS
        for r in g['steps']:
            if r.get('teleport'):
                continue
            b = min(r['step'], N_STEP_BUCKETS - 1)
            n_q[b] += 1
            ch_deg[b] += r['frontier_degree']
            av_deg[b] += r.get('cand_degree_mean', 0.0)
            if r.get('info_gain') is not None:
                ig_val[b] += r['info_gain']
                n_ig[b] += 1
        per_graph.append((ch_deg, av_deg, n_q, ig_val, n_ig))

    rng = np.random.default_rng(0)
    ch_line, av_line, ig_line = [], [], []
    for b in range(N_STEP_BUCKETS):
        den_b = np.array([p[2][b] for p in per_graph], dtype=float)
        ch_line.append(boot_ci(np.array([p[0][b] for p in per_graph]), den_b,
                               rng))
        av_line.append(boot_ci(np.array([p[1][b] for p in per_graph]), den_b,
                               rng))
        ig_line.append(boot_ci(np.array([p[3][b] for p in per_graph]),
                               np.array([p[4][b] for p in per_graph],
                                        dtype=float), rng))

    def unzip(lines):
        return ([x[0] for x in lines], [x[1] for x in lines],
                [x[2] for x in lines])
    ch, av, ig = unzip(ch_line), unzip(av_line), unzip(ig_line)
    n_steps = int(sum(sum(p[2]) for p in per_graph))
    print(f"{name}: graphs={len(per_graph)} steps={n_steps}")
    print('chosen degree:', [round(x, 3) for x in ch[0]])
    print('avail   degree:', [round(x, 3) for x in av[0]])
    print('chosen real IG:', [round(x, 4) for x in ig[0][1:]])
    return {'ch': ch, 'av': av, 'ig': ig}


def draw_degree(ax, d, dataset, with_legend, explainer):
    xs = np.arange(1, N_STEP_BUCKETS + 1)
    los = list(d['av'][1]) + list(d['ch'][1])
    his = list(d['av'][2]) + list(d['ch'][2])
    ax.set_ylim(0.9 * min(los), 1.06 * max(his))
    for key, color, label, lw in (('av', GRAY, 'Available candidates', 1.0),
                                  ('ch', TEAL, 'Chosen nodes', 1.4)):
        m, lo, hi = d[key]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(xs, m, color=color, lw=lw, marker='o', ms=2.2, label=label)
    ax.set_xticks(TICK_STEPS)
    ax.set_xticklabels(TICK_LABELS, fontsize=5.8)
    ax.set_xlabel('Reveal step (≥11 pooled)')
    ax.set_ylabel('Frontier degree (mean)')
    if with_legend:
        ax.legend(loc='lower right', fontsize=5.6, handlelength=1.4,
                  labelspacing=0.3)
    ax.text(0.05, 0.975, explainer, transform=ax.transAxes, fontsize=5.4,
            color=GRAY, ha='left', va='top', linespacing=1.25)
    ax.set_title(dataset, fontsize=7, pad=3, color='#1A1A1A')


def draw_ig(ax, d, dataset, annotations):
    """chosen 节点真实信息增益（nats）逐步曲线；0 线 = 揭示不带来熵减。"""
    m, lo, hi = [v[1:] for v in d['ig']]    # 桶 1（第 1 步）IG 无定义
    xs_ig = np.arange(2, N_STEP_BUCKETS + 1)
    ax.fill_between(xs_ig, lo, hi, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(xs_ig, m, color=ORANGE, lw=1.4, marker='o', ms=2.2)
    ax.axhline(0.0, color=DARK, lw=0.9, ls=(0, (3, 2)))
    if annotations == 'cifar10':
        ax.text(0.97, 0.010, '0 = no entropy reduction',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=5.6, color=DARK)
        ax.annotate('early: positive', xy=(2, hi[0] + 0.004),
                    fontsize=5.6, color=DARK, va='bottom', ha='left')
        ax.annotate('late: negative', xy=(xs_ig[-2], lo[-2] - 0.004),
                    ha='center', va='top', fontsize=5.6, color=DARK)
    else:   # COLLAB：全程为负，0 线标签放线下右侧，注释放左上空白区
        ax.text(0.97, 0.0055, '0 = no entropy reduction',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=5.6, color=DARK)
        ax.text(0.03, 0.70, 'IG < 0 at every step', transform=ax.transAxes,
                fontsize=5.6, color=DARK, ha='left', va='top')
    span = max(0.0, max(hi)) - min(0.0, min(lo))
    ax.set_ylim(min(0.0, min(lo)) - 0.30 * span,
                max(0.0, max(hi)) + 0.38 * span)
    ax.set_xticks(TICK_STEPS)
    ax.set_xticklabels(TICK_LABELS, fontsize=5.8)
    ax.set_xlabel('Reveal step (IG from step 2)')
    ax.set_ylabel('Info gain (nats)')
    ax.set_title(dataset, fontsize=7, pad=3, color='#1A1A1A')


def main():
    cif = aggregate(CIF_JSON, 'CIFAR10')
    col = aggregate(COL_JSON, 'COLLAB')

    matplotlib.rcParams.update(mpl_rc)
    fig, axs = plt.subplots(1, 4, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    fig.subplots_adjust(left=0.048, right=0.995, top=0.885, bottom=0.245,
                        wspace=0.42)
    ax_a, ax_b, ax_c, ax_d = axs

    draw_degree(ax_a, cif, 'CIFAR10', with_legend=True,
                explainer='Better connected early;\nless connected late')
    draw_ig(ax_b, cif, 'CIFAR10', 'cifar10')
    draw_degree(ax_c, col, 'COLLAB', with_legend=False,
                explainer='Chosen above average\nat every step')
    draw_ig(ax_d, col, 'COLLAB', 'collab')

    for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-11, 3), textcoords='offset points',
                    fontsize=8, fontweight='bold', va='bottom', ha='left')

    require_matplotlib_panel_alignment(
        fig,
        json_out=f'{OUT_PREFIX}.alignment.json',
        overlay_svg=f'{OUT_PREFIX}.alignment.svg',
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(f'{OUT_PREFIX}.svg', bbox_inches='tight')
    fig.savefig(f'{OUT_PREFIX}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_PREFIX}.tiff', dpi=600, bbox_inches='tight')
    fig.savefig(f'{OUT_PREFIX}.png', dpi=600, bbox_inches='tight')
    print(f"saved: {OUT_PREFIX}.{{svg,pdf,tiff,png}}")

    with open(f'{OUT_PREFIX}.sources.json', 'w', encoding='utf-8') as f:
        json.dump({
            'layout': '1x4 single row (a/b = CIFAR10, c/d = COLLAB); '
                      'IG panels plot real info gain (nats) of the chosen '
                      'node, rank-percentile version retired 2026-09-25',
            'cifar10': {
                'source': 'choice_behavior_b10_g100s4.json — resampled '
                          'locally (test first 100 graphs x 4 seeds, '
                          'epoch3 checkpoint, same protocol); original '
                          '10k-graph JSON lost (cleaned + not found on '
                          'servers 239.11:27016/27056)',
            },
            'collab': {
                'source': 'choice_behavior_b10.json — original full run '
                          '(test 501 graphs x 1 seed)',
            },
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

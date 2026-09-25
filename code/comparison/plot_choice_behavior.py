"""读者选择行为画像图（cifar10 主结果 evidence reader，B=10%）——直观版。

数据：analyze_choice_behavior.py 产出的 choice_behavior_b10_g100s4.json
（test 前 100 图 × 4 seeds 重采口径，2026-09-25；原 10k 全量 JSON 已删）。

图级主张：reader 只偏好高前沿度数节点（偏好转弱随进度衰减），不追
梯度归因，chosen 节点真实信息增益早正晚负（第 1 步无定义）；终判正误
两组行为无差。四面板全部直接编码：
  a. 三指标"被选节点平均排名分位"柱：1=总挑排名最靠前的候选，
     0.5=随机（虚线）——柱子过线即偏好（hero）
  b. 前沿度数：被选节点 vs 可选候选的平均度数（逐步两线对比：
     第 1–10 步逐步，≥11 步并入末桶）
  c. chosen 节点真实信息增益（nats）逐步：0 线=揭示不带来熵减
  d. 终判正误分组的同一柱状（与 a 同尺度）：两组几乎等高 = 行为与
     结果无关
不确定性统一为按图聚类 bootstrap（2,000 次重采样）的 95% CI。

产物（写入 JSON 同目录）：<JSON stem>.pdf/.svg/.tiff/.png
+ alignment / collision 审计 JSON。

用法:
  python code/comparison/plot_choice_behavior.py [--json <choice_behavior.json>]
"""

import argparse
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

BLUE = '#0F4D92'       # 梯度归因
ORANGE = '#B65A1F'     # 信息增益
TEAL = '#2E7D64'       # 前沿度数
GRAY = '#767676'       # 中性 / 可选参照
DARK = '#4D4D4D'

# (键, 主色, x 轴组标签)
METRICS_PLOT = (
    ('grad', BLUE, 'Gradient\nattribution'),
    ('ig', ORANGE, 'Info gain'),
    ('deg', TEAL, 'Frontier\ndegree'),
)
FIELD = {'grad': 'grad_rank_frac',
         'ig': 'ig_rank_frac',
         'deg': 'degree_rank_frac'}

FIG_W_MM, FIG_H_MM = 183, 118
BOOT_REPS = 2000


def boot_ci(num, den, rng, reps=BOOT_REPS):
    """按图聚类 bootstrap 的 95% CI（num/den 为逐图和向量，den>0 过滤）。"""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    keep = den > 0
    num, den = num[keep], den[keep]
    n = len(num)
    idx = rng.integers(0, n, size=(reps, n))
    m = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def main():
    parser = argparse.ArgumentParser(description='plot choice-behavior figure')
    parser.add_argument(
        '--json', default=str(Path(__file__).resolve().parents[1] / 'result' /
                               'cifar10' / 'experiments' / 'ev-R-read' /
                               'evidence' / 'rl' / 'behavior-analysis' /
                               'choice_behavior_b10_g100s4.json'))
    parser.add_argument('--output-prefix', default='')
    args = parser.parse_args()

    json_path = Path(args.json)
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    graphs = data['per_graph']
    out_prefix = Path(args.output_prefix) if args.output_prefix else \
        json_path.with_name(json_path.stem)

    keys = [k for k, _, _ in METRICS_PLOT]
    N_STEP_BUCKETS = 11        # 第 1–10 步逐步 + "≥11" 末桶
    per_graph = []
    for g in graphs:
        rank_sums = {k: 0.0 for k in keys}
        rank_ns = {k: 0 for k in keys}
        ch_deg = [0.0] * N_STEP_BUCKETS
        av_deg = [0.0] * N_STEP_BUCKETS
        n_q = [0] * N_STEP_BUCKETS
        ig_val = [0.0] * N_STEP_BUCKETS
        n_ig = [0] * N_STEP_BUCKETS
        for r in g['steps']:
            if r.get('teleport'):
                continue
            b = min(r['step'], N_STEP_BUCKETS - 1)   # 0 基步号；≥10 步并入末桶
            n_q[b] += 1
            ch_deg[b] += r['frontier_degree']
            av_deg[b] += r.get('cand_degree_mean', 0.0)
            if r.get('info_gain') is not None:
                ig_val[b] += r['info_gain']
                n_ig[b] += 1
            for k in keys:
                rf = r.get(FIELD[k])
                if rf is None:
                    continue
                rank_sums[k] += rf
                rank_ns[k] += 1
        per_graph.append((rank_sums, rank_ns, ch_deg, av_deg, n_q,
                          ig_val, n_ig, g['ev_correct']))

    n_g = len(per_graph)
    ev_ok = np.array([p[7] for p in per_graph])
    rng = np.random.default_rng(0)

    sums = {k: np.array([p[0][k] for p in per_graph]) for k in keys}
    ns = {k: np.array([p[1][k] for p in per_graph], dtype=float) for k in keys}
    means = {k: boot_ci(sums[k], ns[k], rng) for k in keys}

    ch_line, av_line, ig_line = [], [], []
    for b in range(N_STEP_BUCKETS):
        den_b = np.array([p[4][b] for p in per_graph], dtype=float)
        ch_line.append(boot_ci(np.array([p[2][b] for p in per_graph]), den_b, rng))
        av_line.append(boot_ci(np.array([p[3][b] for p in per_graph]), den_b, rng))
        ig_line.append(boot_ci(np.array([p[5][b] for p in per_graph]),
                               np.array([p[6][b] for p in per_graph],
                                        dtype=float), rng))
    # 各桶的横坐标（1 基步号；末桶 = "≥11"）；IG 无定义的桶（第 1 步）跳过
    xs_b = np.arange(1, N_STEP_BUCKETS + 1)
    tick_lbls = [str(i) for i in range(1, N_STEP_BUCKETS)] + [f'≥{N_STEP_BUCKETS}']

    groups = {}
    for val, name in ((1, 'correct'), (0, 'wrong')):
        m = ev_ok == val
        gm = {k: boot_ci(sums[k][m], ns[k][m], rng) for k in keys}
        groups[name] = (gm, int(m.sum()))

    n_steps = int(sum(ns[k].sum() for k in keys))
    print(f"graphs={n_g} steps={n_steps}")
    for k in keys:
        m = means[k]
        print(f"{k}: rank={m[0]:.4f} [{m[1]:.4f},{m[2]:.4f}]")
    print("chosen degree:", [round(x[0], 3) for x in ch_line])
    print("avail   degree:", [round(x[0], 3) for x in av_line])
    print("ig rank:", [round(x[0], 4) for x in ig_line])
    for name, (gm, ng) in groups.items():
        print(f"{name} (n={ng}):", {k: round(gm[k][0], 4) for k in keys})

    # ---- 图 ----
    matplotlib.rcParams.update(mpl_rc)
    fig, axs = plt.subplots(2, 2, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.10,
                        wspace=0.30, hspace=0.52)
    ax_a, ax_b, ax_c, ax_d = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # a) 三指标被选排名分位：0.5=随机，柱子过线即偏好
    xpos = np.arange(len(keys))
    a_means = [means[k][0] for k in keys]
    a_errs = np.array([[means[k][0] - means[k][1], means[k][2] - means[k][0]]
                       for k in keys]).T
    ax_a.bar(xpos, a_means, width=0.55,
             color=[c for _, c, _ in METRICS_PLOT], edgecolor='none')
    ax_a.errorbar(xpos, a_means, yerr=a_errs, fmt='none', ecolor=DARK,
                  elinewidth=0.8, capsize=2, capthick=0.8)
    a_top = max(means[k][2] for k in keys)
    ax_a.axhline(0.5, color=DARK, lw=0.9, ls=(0, (3, 2)))
    ax_a.text(1.3, 0.5 + 0.012, 'random choice', ha='center', va='bottom',
              fontsize=6.0, color=DARK)
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels([lbl for _, _, lbl in METRICS_PLOT])
    ax_a.set_ylabel('Rank percentile of chosen node')
    ax_a.set_ylim(0, a_top + 0.10)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8][:max(2, int((a_top + 0.10) / 0.2) + 1)])
    ax_a.set_xlim(-0.55, 2.55)
    ax_a.tick_params(axis='x', labelsize=6.2)
    ax_a.text(0.03, 0.965, '1.0 = always picks the best-ranked candidate; '
              'the dashed line = random choice',
              transform=ax_a.transAxes, fontsize=5.6, color=GRAY, va='top')

    # b) 前沿度数：被选 vs 可选（逐步；y 范围数据驱动）
    b_los = [x[1] for x in av_line + ch_line]
    b_his = [x[2] for x in av_line + ch_line]
    b_ylims = (0.9 * min(b_los), 1.06 * max(b_his))
    for line, ci, color, label, lw in (([x[0] for x in av_line], av_line, GRAY,
                                        'Available candidates', 1.0),
                                       ([x[0] for x in ch_line], ch_line, TEAL,
                                        'Chosen nodes', 1.4)):
        lo = [x[1] for x in ci]
        hi = [x[2] for x in ci]
        ax_b.fill_between(xs_b, lo, hi, color=color, alpha=0.15, lw=0)
        ax_b.plot(xs_b, line, color=color, lw=lw, marker='o', ms=2.4,
                  label=label)
    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels(tick_lbls, fontsize=5.8)
    ax_b.set_xlabel('Reveal step (≥11 pooled)')
    ax_b.set_ylabel('Frontier degree (mean)')
    ax_b.set_ylim(*b_ylims)
    ax_b.legend(loc='lower right', fontsize=6.0, handlelength=1.5)
    ax_b.text(0.03, 0.98, 'Early picks are better connected than the average '
              'candidate;\nlate picks are less connected',
              transform=ax_b.transAxes, fontsize=5.6, color=GRAY,
              ha='left', va='top')

    # c) chosen 节点真实信息增益（第 1 步 IG 无定义，从第 2 步起画）
    xs_ig = xs_b[1:]
    ms = [x[0] for x in ig_line[1:]]
    lo = [x[1] for x in ig_line[1:]]
    hi = [x[2] for x in ig_line[1:]]
    ax_c.fill_between(xs_ig, lo, hi, color=ORANGE, alpha=0.18, lw=0)
    ax_c.plot(xs_ig, ms, color=ORANGE, lw=1.4, marker='o', ms=2.4)
    ax_c.axhline(0, color=DARK, lw=0.9, ls=(0, (3, 2)))
    ax_c.text(xs_ig[-1], 0, '0 = no entropy reduction', ha='right',
              va='bottom', fontsize=6.0, color=DARK)
    ax_c.annotate('early: positive', xy=(1.7, hi[0] + 0.010),
                  fontsize=5.8, color=DARK, va='bottom', ha='left')
    ax_c.annotate('late: negative', xy=(xs_ig[-2], lo[-2] - 0.010),
                  ha='center', va='top', fontsize=5.8, color=DARK)
    ax_c.set_xticks(xs_b)
    ax_c.set_xticklabels(tick_lbls, fontsize=5.8)
    ax_c.set_xlabel('Reveal step (≥11 pooled; IG defined from step 2)')
    ax_c.set_ylabel('Info gain of chosen node (nats)')
    span = max(0.0, max(hi)) - min(0.0, min(lo))
    c_ylims = (min(0.0, min(lo)) - 0.30 * span,
               max(0.0, max(hi)) + 0.45 * span)
    ax_c.set_ylim(*c_ylims)

    # d) 读对 vs 读错：同一柱状（与 a 同尺度）
    w = 0.30
    for name, color, offset, filled, lbl in (
            ('correct', BLUE, -w / 2, True, 'Judged correct'),
            ('wrong', GRAY, w / 2, False, 'Judged wrong')):
        gm, ng = groups[name]
        xs_d = xpos + offset
        ms_d = [gm[k][0] for k in keys]
        errs_d = np.array([[gm[k][0] - gm[k][1], gm[k][2] - gm[k][0]]
                           for k in keys]).T
        ax_d.bar(xs_d, ms_d, width=w * 0.9,
                 color=color if filled else 'white', edgecolor=color,
                 linewidth=0.9, label=f'{lbl} (n={ng} graphs)')
        ax_d.errorbar(xs_d, ms_d, yerr=errs_d, fmt='none', ecolor=DARK,
                      elinewidth=0.6, capsize=1.2, capthick=0.6)
    ax_d.axhline(0.5, color=DARK, lw=0.9, ls=(0, (3, 2)))
    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels([lbl for _, _, lbl in METRICS_PLOT])
    ax_d.set_ylabel('Rank percentile of chosen node')
    d_top = max(gm[k][2] for gm, _ in groups.values() for k in keys)
    d_ylims = (0, d_top * 1.30)
    ax_d.set_ylim(*d_ylims)
    ax_d.set_yticks([round(t, 2) for t in np.arange(0, d_ylims[1] + 0.001, 0.2)])
    ax_d.set_xlim(-0.55, 2.55)
    ax_d.tick_params(axis='x', labelsize=6.2)
    ax_d.legend(loc='upper right', fontsize=5.8, handlelength=1.2)
    ax_d.text(0.03, 0.98, 'Behavior does not depend\non the outcome;\n'
              'dashed = random choice',
              transform=ax_d.transAxes, fontsize=5.6, color=GRAY, va='top')

    for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-14, 4), textcoords='offset points',
                    fontsize=8, fontweight='bold', va='bottom', ha='left')

    # ---- 面板对齐门禁 + 导出 ----
    require_matplotlib_panel_alignment(
        fig,
        json_out=f'{out_prefix}.alignment.json',
        overlay_svg=f'{out_prefix}.alignment.svg',
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(f'{out_prefix}.svg', bbox_inches='tight')
    fig.savefig(f'{out_prefix}.pdf', bbox_inches='tight')
    fig.savefig(f'{out_prefix}.tiff', dpi=600, bbox_inches='tight')
    fig.savefig(f'{out_prefix}.png', dpi=600, bbox_inches='tight')
    print(f"saved: {out_prefix}.{{svg,pdf,tiff,png}}")


if __name__ == '__main__':
    main()

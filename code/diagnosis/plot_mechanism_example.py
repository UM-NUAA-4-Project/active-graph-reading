"""机制例图（两联）：选点策略的功能性证据。

claim = expansion（cifar10）：选 fd 1~2 的节点换来更多新前沿边特征
        （ΔE = deg − fd；b 面板高亮揭示后新可见的前沿边）；
claim = densification（collab）：选 fd 最大的节点闭合社区中心
        （b 面板高亮 fd 辐条 + 已揭示邻居间的闭合三角形边）。

数据：dump_candidate_rollouts.py 的 candidate_rollouts_*.json 提供
表格数字；拓扑按同款 eval 协议重放 rollout（rollout_to_step）取揭示
前/后状态，重放被选节点/前沿集与 dump 不一致即报错。

面板：a = 揭示前（前沿按 reader 分数渐变成环、被选红虚线圈、对照
候选灰虚线圈）；b = 揭示后（按 claim 高亮）；c = 表格（被选 vs 对照
候选，纯文本、无数据几何，排除在等宽门禁外）。中心-辐射布局与
selection_process_b10_graph18_topo 同款。

用法（仓库根运行）:
  python code/diagnosis/plot_mechanism_example.py --claim expansion \
      --dataset cifar10 --root D:\\data\\superpixels \
      --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt \
      --dump code/result/cifar10/experiments/ev-R-read/evidence/rl/behavior-analysis/candidate_rollouts_b10_g100s4.json \
      --graph-idx 78 --seed-i 1 --step 5
  python code/diagnosis/plot_mechanism_example.py --claim densification \
      --dataset collab --root D:\\data\\COLLAB \
      --checkpoint code/result/collab/experiments/evidence1/evidence/rl/checkpoint_epoch6.pt \
      --dump code/result/collab/experiments/evidence1/evidence/rl/behavior-analysis/candidate_rollouts_b10_g100s1.json \
      --graph-idx 7 --seed-i 0 --step 5 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

SKILL_SCRIPTS = Path.home() / '.zcode' / 'skills' / 'nature-figure' / 'scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

from active_graph_reader.engine.reveal_action import reveal_node  # noqa: E402
from datasets import get_dataset, release_dataset  # noqa: E402
from train_reader.core import load_reader_from_checkpoint  # noqa: E402
from diagnosis.dump_candidate_rollouts import neighbors_map  # noqa: E402
from diagnosis.plot_selection_topology import (  # noqa: E402
    mpl_rc, FIG_W_MM, FIG_H_MM, AX_Y0, AX_H, AXA_X0, AXB_X0, AXC_X0,
    AX_W, AXC_W, BLUE, RED, GRAY, DARK, SEED, EDGE_RR, EDGE_RF,
    FRONTIER_EDGE, FRONTIER_EDGE_B, CMAP_STOPS, undirected_edges_pairs,
    radial_layout, set_limits_pts, node_xy, rollout_to_step)

NEW_EDGE = '#DE7E33'    # 新前沿边（expansion，暖橙）
SPOKE = '#C22432'       # fd 辐条（densification，深红）
CLOSURE = '#E3B23C'     # 闭合三角形边（densification，琥珀）
SEL_MAX_FD = '#4D4D4D'  # 对照候选灰虚线圈


def edge_key(u, v):
    return (min(u, v), max(u, v))


def draw_topology(ax, pos, pairs, revealed, seed_node, frontier,
                  facecolors, rim=FRONTIER_EDGE, node_s=60):
    """底图：边（内部实线/前沿虚线）+ 已揭示蓝圆 + seed 菱形 + 前沿环。"""
    R = set(revealed)
    seg_rr, seg_rf = [], []
    for u, v in pairs:
        if u in R and v in R:
            seg_rr.append((pos[u], pos[v]))
        elif u in R or v in R:
            seg_rf.append((pos[u], pos[v]))
    ax.add_collection(LineCollection(seg_rf, colors=EDGE_RF, linewidths=0.5,
                                     linestyles=(0, (2.5, 1.8)), zorder=1))
    ax.add_collection(LineCollection(seg_rr, colors=EDGE_RR, linewidths=0.6,
                                     zorder=1))
    for i, v in enumerate(frontier):
        ax.scatter(*pos[v], s=node_s, facecolors=facecolors[i],
                   edgecolors=rim, linewidths=0.5, zorder=3)
    ax.scatter(*node_xy(pos, [v for v in revealed if v != seed_node]),
               s=60, c=BLUE, edgecolors='white', linewidths=0.5, zorder=4)
    ax.scatter(*node_xy(pos, [seed_node]), marker='D', s=55, c=SEED,
               edgecolors='white', linewidths=0.5, zorder=6)


def dashed_circle(ax, pos, v, r, color):
    ax.add_patch(Circle(pos[v], r, fill=False, edgecolor=color,
                        linewidth=1.0, linestyle=(0, (3, 2)), zorder=5))


def draw_table(ax, cols, header, rows, foot_lines, hi_row=0):
    """纯文本表格：水平线 only，hi_row 淡红底。rows = [(label,color,bold,[cells])]"""
    ax.set_axis_off()
    y_hdr, y_rule = 0.90, 0.83
    n = len(rows)
    y_rows = [0.83 - 0.205 * (i + 1) for i in range(n)]
    y_bot = y_rows[-1] - 0.10
    for (x, ha), lab in zip(cols, header):
        ax.text(x, y_hdr, lab, transform=ax.transAxes, ha=ha, va='center',
                fontsize=5.6, color=GRAY, fontweight='bold')
    ax.plot([0.0, 1.0], [y_rule] * 2, transform=ax.transAxes,
            color='#B9B9B9', lw=0.5, clip_on=False)
    ax.add_patch(Rectangle((0.0, y_rows[hi_row] - 0.085), 1.0, 0.17,
                           transform=ax.transAxes, facecolor=RED,
                           alpha=0.055, edgecolor='none', zorder=0))
    for i, (label, color, bold, cells) in enumerate(rows):
        yr = y_rows[i]
        ax.text(cols[0][0], yr, label, transform=ax.transAxes, ha='left',
                va='center', fontsize=6.2, color=color,
                fontweight='bold' if bold else 'normal')
        for (x, ha), cell in zip(cols[1:], cells):
            ax.text(x, yr, cell, transform=ax.transAxes, ha=ha, va='center',
                    fontsize=6.2, color='#1A1A1A',
                    fontweight='bold' if bold else 'normal')
    ax.plot([0.0, 1.0], [y_bot] * 2, transform=ax.transAxes,
            color='#B9B9B9', lw=0.5, clip_on=False)
    for k, txt in enumerate(foot_lines):
        ax.text(0.0, y_bot - 0.055 - 0.075 * k, txt, transform=ax.transAxes,
                ha='left', va='top', fontsize=5.2, color=GRAY)


def main():
    parser = argparse.ArgumentParser(description='机制例图（expansion / densification）')
    parser.add_argument('--claim', required=True,
                        choices=['expansion', 'densification'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dump', required=True)
    parser.add_argument('--graph-idx', type=int, required=True)
    parser.add_argument('--seed-i', type=int, default=0)
    parser.add_argument('--step', type=int, required=True,
                        help='dump 里的 step 下标（0 基，含 teleport 槽位）')
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    with open(args.dump, encoding='utf-8') as f:
        dump = json.load(f)
    ep = next(e for e in dump['per_graph']
              if e['graph_idx'] == args.graph_idx and e['seed_i'] == args.seed_i)
    rec = next(r for r in ep['steps'] if r['step'] == args.step)
    if any(r.get('teleport') for r in ep['steps'][:args.step + 1]):
        raise SystemExit('目标步前有 teleport，重放会偏离 dump——另选例子')
    ch = next(c for c in rec['candidates'] if c['node'] == rec['chosen'])
    contrast = (max(rec['candidates'], key=lambda c: c['fd'])
                if args.claim == 'expansion'
                else min(rec['candidates'], key=lambda c: c['fd']))
    title_step = rec['revealed_count_pre']

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    reader = load_reader_from_checkpoint(args.checkpoint, device)
    reader.eval()
    for p_ in reader.parameters():
        p_.requires_grad_(False)
    ds = get_dataset(args.dataset, root=args.root, splits=('test',))
    g = ds.test_graphs[args.graph_idx]
    release_dataset(ds)
    adj = neighbors_map(g)

    snap = rollout_to_step(reader, g, None, args.graph_idx,
                           args.budget_ratio, device,
                           target_step=args.step + 1, seed_i=args.seed_i)
    if snap is None or snap['chosen'] != rec['chosen']:
        raise SystemExit('重放与 dump 不一致（chosen）——checkpoint/协议漂移')
    if not rec['frontier_truncated'] \
            and sorted(snap['frontier']) != sorted(c['node'] for c in rec['candidates']):
        raise SystemExit('重放与 dump 不一致（前沿集）')
    state = snap['state']
    frontier_b = list(snap['frontier'])
    probs = snap['probs']
    chosen = rec['chosen']
    revealed_b = sorted(state.V_revealed)
    seed_node = snap['seed_node']

    pmax = float(max(probs))
    cmap = LinearSegmentedColormap.from_list('sel_prob', CMAP_STOPS)
    norm = Normalize(vmin=0.0, vmax=pmax)
    fills = [cmap(norm(p)) for p in probs]

    if args.claim == 'expansion':
        # 揭示后全前沿入画（新前沿节点上环）
        reveal_node(state, chosen)
        frontier_a = frontier_b + [v for v in sorted(state.get_frontier_nodes())
                                   if v not in set(frontier_b)]
        revealed_a = sorted(state.V_revealed)
        active = sorted(set(revealed_a) | set(frontier_a) | set(frontier_b))
        pairs = undirected_edges_pairs(g, active)
        pos_a = radial_layout(pairs, active, revealed_center=revealed_b)
        pos_b = radial_layout(pairs, active, revealed_center=revealed_a)
        new_edges = [edge_key(chosen, u) for u in sorted(adj[chosen])
                     if u not in set(revealed_b)]
        if len(new_edges) != ch['new_fe']:
            raise SystemExit(f'ΔE 不一致: {len(new_edges)} vs {ch["new_fe"]}')
    else:
        # 致密化：布局固定（新前沿不入画），b 面板点亮揭示区内的轮辐+闭合边
        reveal_node(state, chosen)
        revealed_a = sorted(state.V_revealed)
        active = sorted(set(revealed_a) | set(frontier_b))
        pairs = undirected_edges_pairs(g, active)
        pos_a = radial_layout(pairs, active, revealed_center=revealed_b)
        pos_b = pos_a
        spokes = [edge_key(chosen, u) for u in sorted(adj[chosen])
                  if u in set(revealed_b)]
        rn = set(adj[chosen]) & set(revealed_b)
        closure = [(u, w) for u, w in pairs if u in rn and w in rn]
        if len(spokes) != ch['fd'] or len(closure) != ch['closed_tri']:
            raise SystemExit('辐条/闭合三角形与 dump 不一致')

    w_mm, h_mm = AX_W * FIG_W_MM, AX_H * FIG_H_MM
    matplotlib.rcParams.update(mpl_rc)
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax_a = fig.add_axes([AXA_X0, AX_Y0, AX_W, AX_H])
    ax_b = fig.add_axes([AXB_X0, AX_Y0, AX_W, AX_H])
    ax_c = fig.add_axes([AXC_X0, AX_Y0, AXC_W, AX_H])
    all_pts = np.array(list(pos_a.values()) + list(pos_b.values()))
    for ax in (ax_a, ax_b):
        set_limits_pts(ax, all_pts, w_mm, h_mm)
    x0, x1 = ax_a.get_xlim()
    r_node = 1.5 * (x1 - x0) / w_mm
    circ_r = 1.9 * r_node

    # ---- a：揭示前 ----
    draw_topology(ax_a, pos_a, pairs, revealed_b, seed_node, frontier_b,
                  fills)
    dashed_circle(ax_a, pos_a, chosen, circ_r, RED)
    if contrast['node'] != chosen:
        dashed_circle(ax_a, pos_a, contrast['node'], circ_r, SEL_MAX_FD)
    ax_a.set_title(f'Step {title_step}: before reveal (|F| = '
                   f'{rec["n_frontier"]})', fontsize=7, pad=4, color='#1A1A1A')
    from matplotlib import patheffects
    ax_a.annotate(f'p = {rec["chosen_prob"]:.3f}',
                  xy=(pos_a[chosen][0] + circ_r * 0.71,
                      pos_a[chosen][1] - circ_r * 0.71),
                  xytext=(9, 7), textcoords='offset points', fontsize=6,
                  color=RED, ha='left', va='bottom',
                  arrowprops=dict(arrowstyle='-', color=RED, lw=0.5,
                                  shrinkA=1, shrinkB=2),
                  path_effects=[patheffects.withStroke(linewidth=1.6,
                                                       foreground='white')])

    # ---- b：揭示后 ----
    R = set(revealed_a)
    if args.claim == 'expansion':
        draw_topology(ax_b, pos_b, pairs, revealed_a, seed_node, frontier_a,
                      ['white'] * len(frontier_a), rim=FRONTIER_EDGE_B)
        seg = [(pos_b[u], pos_b[v]) for u, v in new_edges]
        ax_b.add_collection(LineCollection(seg, colors=NEW_EDGE,
                                           linewidths=0.9, zorder=2))
        ax_b.set_title(f'Step {title_step + 1}: after reveal — '
                       f'+{ch["new_fe"]} new frontier edges',
                       fontsize=7, pad=4, color='#1A1A1A')
    else:
        draw_topology(ax_b, pos_b, pairs, revealed_a, seed_node, frontier_b,
                      ['white'] * len(frontier_b), rim=FRONTIER_EDGE_B)
        ax_b.add_collection(LineCollection(
            [(pos_b[u], pos_b[v]) for u, v in closure],
            colors=CLOSURE, linewidths=1.1, zorder=2))
        ax_b.add_collection(LineCollection(
            [(pos_b[u], pos_b[v]) for u, v in spokes],
            colors=SPOKE, linewidths=1.0, zorder=2))
        ax_b.set_title(f'Step {title_step + 1}: after reveal — '
                       f'{ch["closed_tri"]} triangles closed',
                       fontsize=7, pad=4, color='#1A1A1A')
    dashed_circle(ax_b, pos_b, chosen, circ_r, RED)

    for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c')):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-13, 1), textcoords='offset points',
                    fontsize=8, fontweight='bold', va='bottom', ha='left')

    ymid = AX_Y0 + AX_H / 2
    arrow = FancyArrowPatch((0.323, ymid), (0.355, ymid),
                            transform=fig.transFigure, arrowstyle='-|>',
                            mutation_scale=8, color=DARK, lw=1.0)
    fig.add_artist(arrow)
    fig.text((AXA_X0 + AX_W + AXB_X0) / 2, ymid + 0.055, 'reveal',
             ha='center', va='bottom', fontsize=6.2, color=DARK)

    # ---- c：表格 ----
    fmt_p = lambda v: f'{v:.3f}'
    if args.claim == 'expansion':
        cols = [(0.02, 'left'), (0.44, 'left'), (0.60, 'right'),
                (0.71, 'right'), (0.86, 'right'), (0.98, 'right')]
        header = ['candidate', 'node', 'fd', 'deg', 'ΔE', 'p']
        rows = [
            ('Reader-selected', RED, False,
             [f'n.{ch["node"]}', f'{ch["fd"]}', f'{ch["deg"]}',
              f'+{ch["new_fe"]}', fmt_p(ch['prob'])]),
            ('Max frontier degree', '#1A1A1A', False,
             [f'n.{contrast["node"]}', f'{contrast["fd"]}',
              f'{contrast["deg"]}', f'+{contrast["new_fe"]}',
              fmt_p(contrast['prob'])]),
        ]
        foot = [
            'fd = edges to revealed region; ΔE = new visible',
            'frontier edges gained by the reveal',
            'Aggregate: chosen fd ≤ 2 in 4516/4516 steps',
            '(100 graphs × 4 seeds, epoch-6 reader)',
        ]
        ax_c.set_title('New frontier edges per reveal', fontsize=7, pad=4,
                       color='#1A1A1A')
    else:
        cols = [(0.02, 'left'), (0.40, 'left'), (0.54, 'right'),
                (0.65, 'right'), (0.76, 'right'), (0.88, 'right'),
                (0.98, 'right')]
        header = ['candidate', 'node', 'fd', 'deg', 'ΔE', 'tri', 'p']
        rows = [
            ('Reader-selected', RED, False,
             [f'n.{ch["node"]}', f'{ch["fd"]}', f'{ch["deg"]}',
              f'+{ch["new_fe"]}', f'{ch["closed_tri"]}', fmt_p(ch['prob'])]),
            ('Min frontier degree', '#1A1A1A', False,
             [f'n.{contrast["node"]}', f'{contrast["fd"]}',
              f'{contrast["deg"]}', f'+{contrast["new_fe"]}',
              f'{contrast["closed_tri"]}', fmt_p(contrast['prob'])]),
        ]
        foot = [
            'fd = edges to revealed region; tri = triangles closed',
            'among revealed neighbors',
            'Panel b: |F| = '
            + f'{rec["n_frontier"]} -> {rec["n_frontier"] - 1 + ch["new_fn"]}'
            + '; new frontier nodes not shown',
            'Aggregate: chosen fd = step-max in 663/663 steps',
        ]
        ax_c.set_title('Community closure per reveal', fontsize=7, pad=4,
                       color='#1A1A1A')
    draw_table(ax_c, cols, header, rows, foot)

    # ---- 底条：色条 + 图例 ----
    cax = fig.add_axes([AXA_X0, 0.075, 51.0 / FIG_W_MM, 0.05])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([0.0, pmax / 2, pmax])
    cbar.set_ticklabels(['0.00', f'{pmax / 2:.2f}', f'{pmax:.2f}'])
    cbar.set_label('Reader selection probability (frontier fill)',
                   fontsize=6, color='#1A1A1A')
    cbar.ax.tick_params(labelsize=5.5, length=1.6, pad=1.2)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#B9B9B9')

    if args.claim == 'expansion':
        handles = [
            Line2D([], [], marker='o', ls='none', mfc=BLUE, mec=BLUE, ms=5,
                   label='Revealed'),
            Line2D([], [], marker='o', ls='none', mfc=cmap(0.55),
                   mec=FRONTIER_EDGE, ms=5, label='Scored frontier (panel a)'),
            Line2D([], [], marker='o', ls='none', mfc='white',
                   mec=FRONTIER_EDGE_B, ms=5, label='New frontier (panel b)'),
            Line2D([], [], color=NEW_EDGE, lw=1.2,
                   label='New frontier edge (panel b)'),
            Line2D([], [], marker='o', ls='none', mfc='none', mec=RED, ms=8,
                   mew=0.9, linestyle=(0, (2.5, 1.8)), label='Chosen (argmax)'),
            Line2D([], [], marker='o', ls='none', mfc='none',
                   mec=SEL_MAX_FD, ms=8, mew=0.9, linestyle=(0, (2.5, 1.8)),
                   label='Max-fd candidate'),
            Line2D([], [], marker='D', ls='none', mfc=SEED, mec=SEED,
                   ms=4.5, label='Seed node'),
        ]
    else:
        handles = [
            Line2D([], [], marker='o', ls='none', mfc=BLUE, mec=BLUE, ms=5,
                   label='Revealed'),
            Line2D([], [], marker='o', ls='none', mfc=cmap(0.55),
                   mec=FRONTIER_EDGE, ms=5, label='Scored frontier (panel a)'),
            Line2D([], [], color=SPOKE, lw=1.2,
                   label='Revealed spokes (fd, panel b)'),
            Line2D([], [], color=CLOSURE, lw=1.2,
                   label='Closed triangle edges (panel b)'),
            Line2D([], [], marker='o', ls='none', mfc='none', mec=RED, ms=8,
                   mew=0.9, linestyle=(0, (2.5, 1.8)), label='Chosen (argmax)'),
            Line2D([], [], marker='o', ls='none', mfc='none',
                   mec=SEL_MAX_FD, ms=8, mew=0.9, linestyle=(0, (2.5, 1.8)),
                   label='Min-fd candidate'),
            Line2D([], [], marker='D', ls='none', mfc=SEED, mec=SEED,
                   ms=4.5, label='Seed node'),
        ]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(AXB_X0 + AX_W, 0.035),
               ncol=2, fontsize=6, handlelength=1.4, columnspacing=0.9,
               handletextpad=0.4, labelspacing=0.62)

    # ---- 门禁 + 导出 ----
    out_prefix = (Path(args.output) if args.output else
                  Path(args.dump).parent
                  / f'mechanism_{args.claim}_{args.dataset}'
                  f'_g{args.graph_idx}s{args.seed_i}')
    require_matplotlib_panel_alignment(
        fig,
        json_out=f'{out_prefix}.alignment.json',
        overlay_svg=f'{out_prefix}.alignment.svg',
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        exclude_axes=[cax, ax_c],   # c = 表格面板（无数据几何，宽度有意不同）
        row_groups=[{'id': 'panels-row', 'panels': ['a', 'b']}],
    )
    fig.savefig(f'{out_prefix}.svg', bbox_inches='tight')
    fig.savefig(f'{out_prefix}.pdf', bbox_inches='tight')
    fig.savefig(f'{out_prefix}.tiff', dpi=600, bbox_inches='tight')
    fig.savefig(f'{out_prefix}.png', dpi=600, bbox_inches='tight')
    print(f'saved: {out_prefix}.{{svg,pdf,tiff,png}}')


if __name__ == '__main__':
    main()

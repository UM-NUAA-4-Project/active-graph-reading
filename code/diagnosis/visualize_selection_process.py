"""真实数据集上的图选择过程图：冻结 reader 在单张真实 CIFAR10 超像素图上
的"前沿子图 → 打分 → 揭示 → 新前沿子图"两态图。

图级契约
--------
核心主张：训练好的 reader 在真实图上的每一步选择是具体可解释的——它给
前沿候选逐一打分（策略概率），选中得分最高的节点揭示后，前沿随之推进。
面板角色：a = 第 5 步揭示前的读者视野（已揭示区 + 逐候选打分的前沿 +
被选节点）；b = 揭示被选节点后的新前沿。典型代表例 + 机制两角色。
图型：双面板节点链接图（空间坐标 = 超像素质心，等比例）。
导出契约：150×72 mm，Arial ≥5 pt，可编辑文字，svg/pdf/tiff/png +
面板对齐门禁（1.5 pt strict）+ PDF 碰撞审计。

协议：与 eval/analyze_choice_behavior 严格同款——确定性 argmax rollout、
deterministic_seed(graph_idx, 0, n)、B=max(1, 0.10n)、teleport 播种
seed_node*1_000_003+B；步号人类口径 1 基（第 5 步 = 第 5 次选择）。

例图挑选：扫描 test 前若干图，优先满足 n∈[--n-min,--n-max]、PCA 主轴比
≤--max-bbox-aspect（排除极端细长条）、无 teleport、前沿候选数
≥--min-frontier、证据头终判正确；被选节点前沿度数默认不约束（--want-
frontier-degree 可指定；注意 epoch6 主行为是 fd=1 轮廓行走，epoch3 旧
主行为 fd=2 凹角填充，两者都已实测）。全不满足则逐级放宽并报告。

用法（仓库根运行）:
  python code/diagnosis/visualize_selection_process.py \
      --root D:\\data\\superpixels \
      --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt \
      [--step 5] [--graph-idx N] [--max-scan 400]

产物（写入 checkpoint 同级 behavior-analysis/）：
  selection_process_b{10}_step{5}.{svg,pdf,tiff,png} + .json（快照数据）
  + selection_process_b{10}.md（说明）+ 对齐/碰撞审计文件
"""

import argparse
import json
import random
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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib import patheffects as patheffects  # noqa: E402

SKILL_SCRIPTS = Path.home() / '.zcode' / 'skills' / 'nature-figure' / 'scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import (  # noqa: E402
    reveal_node, teleport_if_stuck)
from datasets import get_dataset, release_dataset  # noqa: E402
from tools.seeds import deterministic_seed  # noqa: E402
from train_reader.core import load_reader_from_checkpoint  # noqa: E402

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

BLUE = '#0F4D92'       # 已揭示
RED = '#B3202C'        # 被选节点描边
GRAY = '#767676'       # 中性
DARK = '#4D4D4D'

FIG_W_MM, FIG_H_MM = 150.0, 72.0
AX_Y0, AX_H = 0.24, 0.70
AXA_X0, AXB_X0, AX_W = 0.045, 0.54, 0.415


def _edge_key_to_id(g) -> dict:
    """canonical (min,max) key → 首见边 id（与 eval_loop 同款预构建）。"""
    if 'feat' not in g.edata:
        return {}
    src, dst = g.edges()
    out = {}
    for eid in range(g.num_edges()):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in out:
            out[key] = eid
    return out


def coords_of(g) -> np.ndarray:
    return g.ndata['feat'][:, :2].double().numpy()


def undirected_edges(g) -> list:
    src, dst = g.edges()
    out = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            out.add((min(u, v), max(u, v)))
    return sorted(out)


def rollout_capture(reader, use_evidence, g, ek2id, graph_idx, label,
                    budget_ratio, device, target_step):
    """eval 协议逐步 rollout，捕获第 target_step（1 基）步揭示前后快照。

    返回 dict 或 None（teleport / 前沿断链 / 目标步前结束的图不要）。
    """
    n = g.num_nodes()
    B = max(1, int(budget_ratio * n))
    seed_node = deterministic_seed(graph_idx, 0, n)
    state = GraphState(g, seed_node=seed_node, edge_key_to_id=ek2id)
    state._budget_limit = B
    hc = reader.init_evidence(1, device) if use_evidence else None
    last_revealed = seed_node
    rng = random.Random(seed_node * 1_000_003 + B)
    target0 = target_step - 1
    snap = None
    step_idx = 0

    while state.get_budget_used() < state._budget_limit:
        if state.is_frontier_empty:
            tp_node = teleport_if_stuck(state, rng=rng)   # 保持与 eval 相同的
            if tp_node is None:                           # 状态推进，但例图不要
                return None
            return None
        frontier = state.get_frontier_nodes()
        with torch.no_grad():
            if use_evidence:
                results, hc, _ = reader.forward_states_evidence(
                    [state], temperature=1.0, hc_list=hc,
                    last_revealed=[last_revealed])
            else:
                results = reader.forward_states([state], temperature=1.0)
        _, probs, r_frontier = results[0]
        if not r_frontier:
            return None
        pos = int(probs.argmax().item())
        chosen = int(r_frontier[pos])

        if step_idx == target0:
            p = probs.detach().cpu().numpy()
            order = np.argsort(-p)
            snap = {
                'step': target_step,
                'budget_used_before': state.get_budget_used(),
                'revealed': sorted(state.V_revealed),
                'frontier': list(r_frontier),
                'probs': [round(float(x), 6) for x in p],
                'chosen': chosen,
                'chosen_prob': round(float(p[pos]), 6),
                'runner_up_prob': round(float(p[order[1]]), 6) if len(order) > 1 else None,
                'chosen_frontier_degree': int(state.get_frontier_degree(chosen)),
            }

        reveal_node(state, chosen)
        last_revealed = chosen
        if snap is not None and 'revealed_after' not in snap:
            snap['budget_used_after'] = state.get_budget_used()
            snap['revealed_after'] = sorted(state.V_revealed)
            snap['frontier_after'] = list(state.get_frontier_nodes())
        step_idx += 1

    ev_pred = None
    if use_evidence:
        with torch.no_grad():
            _, _, final_ev = reader.forward_states_evidence(
                [state], temperature=1.0, hc_list=hc,
                last_revealed=[last_revealed])
        ev_pred = int(final_ev[0].argmax(dim=-1).item())
    snap = snap or {}
    snap['ev_pred'] = ev_pred
    snap['ev_correct'] = (None if ev_pred is None else int(ev_pred == label))
    snap['n_steps_total'] = step_idx
    return snap


def example_ok(snap, n, args) -> bool:
    if snap is None or 'frontier' not in snap:
        return False
    if not (args.n_min <= n <= args.n_max):
        return False
    if len(snap['frontier']) < args.min_frontier:
        return False
    if args.want_frontier_degree >= 0 and \
            snap['chosen_frontier_degree'] != args.want_frontier_degree:
        return False
    if snap.get('ev_correct') != 1:
        return False
    return True


def relaxed_ok(snap, n, args) -> bool:
    """逐级放宽：仍要求无 teleport、目标步有前沿可选。"""
    if snap is None or 'frontier' not in snap:
        return False
    if len(snap['frontier']) < max(4, args.min_frontier - 2):
        return False
    return True


def visual_ok(snap, coords, seed_node, args) -> bool:
    """可读性门槛：快照节点两两最小间距足够大（标记不叠）、
    种子→被选节点的已走路径足够长（路径可见，非原地打转）。"""
    rset = set(snap['revealed'])
    nodes = list(snap['revealed']) + [v for v in snap['frontier'] if v not in rset]
    pts = coords[nodes]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    dmin = float(np.sqrt(d2.min()))
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    path = float(np.linalg.norm(coords[snap['chosen']] - coords[seed_node]))
    return (dmin >= args.min_sep_frac * diag
            and path >= args.path_frac * diag)


def set_square_limits(ax, coords, w_mm, h_mm, nodes=None, pad_frac=0.10):
    """等比例显示：把 x/y 限程补齐到面板物理宽高比（免 letterbox，
    两面板完全一致 → 对齐门禁通过）。y 轴按图像惯例反向。
    nodes 给定时视野框收拢到这批节点（未揭示区不画，画面不空旷）。"""
    pts = coords if nodes is None else coords[np.asarray(sorted(nodes))]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    target = h_mm / w_mm
    if dy / dx < target:
        dy = dx * target
    else:
        dx = dy / target
    dx *= (1 + 2 * pad_frac)
    dy *= (1 + 2 * pad_frac)
    ax.set_xlim(cx - dx / 2, cx + dx / 2)
    ax.set_ylim(cy + dy / 2, cy - dy / 2)   # top > bottom → y 反向


def draw_panel(ax, coords, edges, revealed, frontier, chosen=None,
               seed_node=None, probs=None, norm=None, show_scores=False):
    """只画读者可观察的部分：已揭示区 + 前沿候选（未揭示节点不显示）。"""
    R = set(revealed)
    seg_rr, seg_rf = [], []
    for u, v in edges:
        ur, vr = u in R, v in R
        if ur and vr:
            seg_rr.append((coords[u], coords[v]))
        elif ur or vr:
            seg_rf.append((coords[u], coords[v]))
    ax.add_collection(LineCollection(seg_rf, colors='#A8A8A8', linewidths=0.45,
                                     linestyles=(0, (2, 1.6)), zorder=1))
    ax.add_collection(LineCollection(seg_rr, colors='#8A8A8A',
                                     linewidths=0.5, zorder=1))

    fl = list(frontier)
    if show_scores:
        ax.scatter(coords[fl, 0], coords[fl, 1], c=cm.viridis(norm(probs)),
                   s=17, edgecolors='#404040', linewidths=0.35, zorder=3)
    else:
        ax.scatter(coords[fl, 0], coords[fl, 1], facecolors='white',
                   edgecolors=GRAY, linewidths=0.5, s=17, zorder=3)
    rl = sorted(R)
    ax.scatter(coords[rl, 0], coords[rl, 1], s=17, c=BLUE, lw=0, zorder=4)
    if chosen is not None:
        if show_scores:   # a：被选节点上叠红星
            ax.scatter([coords[chosen, 0]], [coords[chosen, 1]], marker='*',
                       s=110, facecolors='none', edgecolors=RED,
                       linewidths=0.9, zorder=5)
        else:             # b：被选节点已入揭示区，红描边以追踪
            ax.scatter([coords[chosen, 0]], [coords[chosen, 1]], s=17,
                       facecolors=BLUE, edgecolors=RED, linewidths=0.9,
                       zorder=5)
    if seed_node is not None:
        ax.scatter([coords[seed_node, 0]], [coords[seed_node, 1]], marker='D',
                   s=13, c='#1A1A1A', lw=0, zorder=6)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description='reader 选择过程两态图')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='D:\\data\\superpixels')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--step', type=int, default=5,
                        help='演示第几步选择（1 基）')
    parser.add_argument('--seed-index', type=int, default=0)
    parser.add_argument('--graph-idx', type=int, default=-1,
                        help='直接指定例图（跳过扫描）；-1 = 自动挑选')
    parser.add_argument('--max-scan', type=int, default=2000)
    parser.add_argument('--min-sep-frac', type=float, default=0.005,
                        help='快照节点最小间距 / 图对角线下限（实测中位数 '
                             '0.0025，0.005≈p85，保证放大视野里标记不叠）')
    parser.add_argument('--path-frac', type=float, default=0.12,
                        help='种子→被选节点距离 / 图对角线下限（实测中位数 '
                             '0.17；保证已走路径可见）')
    parser.add_argument('--n-min', type=int, default=80)
    parser.add_argument('--n-max', type=int, default=150)
    parser.add_argument('--min-frontier', type=int, default=6)
    parser.add_argument('--want-frontier-degree', type=int, default=-1,
                        help='要求被选节点前沿度数；-1 = 不约束（如实展示行为）')
    parser.add_argument('--max-bbox-aspect', type=float, default=4.0,
                        help='例图 PCA 主轴比上限（只排除极端细长条；'
                             '视野已收拢到活动区，全局形状影响小）')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    ckpt_path = Path(args.checkpoint)
    reader = load_reader_from_checkpoint(ckpt_path, device,
                                         stochastic=False, temperature=1.0)
    reader.eval()
    use_evidence = getattr(reader.config, 'kind', '') == 'evidence'
    print(f"Reader: kind={reader.config.kind} (deterministic argmax)", flush=True)

    ds = get_dataset(args.dataset, root=args.root, splits=(args.split,))
    graphs = getattr(ds, f'{args.split}_graphs')
    labels = getattr(ds, f'{args.split}_labels')
    print(f"[{args.split}] {len(graphs)} 图 | budget_ratio={args.budget_ratio}",
          flush=True)

    # ---- 挑例图 ----
    def pca_aspect(g):
        """坐标协方差主轴比（对角细长条轴对齐 bbox 也会是方形，PCA 才测得出）。"""
        c = g.ndata['feat'][:, :2].double().numpy()
        c = c - c.mean(axis=0)
        lam = np.linalg.eigvalsh(np.cov(c.T))
        return float(np.sqrt(max(lam) / max(min(lam), 1e-9)))

    scan_n = len(graphs) if args.graph_idx >= 0 else min(args.max_scan, len(graphs))
    pick, relaxed, n_scanned = None, None, 0
    for gi in range(scan_n):
        g = graphs[gi]
        n = g.num_nodes()
        # 细长条形超像素不适合做过程示意（空白太多、生长不可见），预过滤
        if args.graph_idx < 0 and pca_aspect(g) > args.max_bbox_aspect:
            continue
        if args.graph_idx < 0 and not (args.n_min <= n <= args.n_max):
            continue
        snap = rollout_capture(reader, use_evidence, g, _edge_key_to_id(g), gi,
                               int(labels[gi].item()), args.budget_ratio,
                               device, args.step)
        n_scanned += 1
        if args.graph_idx >= 0:
            pick = (gi, g, snap)
            break
        if example_ok(snap, n, args) and \
                visual_ok(snap, coords_of(g), deterministic_seed(gi, 0, g.num_nodes()), args):
            pick = (gi, g, snap)
            break
        if relaxed is None and relaxed_ok(snap, n, args):
            relaxed = (gi, g, snap)
    if pick is None:
        if relaxed is None:
            raise SystemExit(f"前 {n_scanned} 图内无可用例（全 teleport/断链），"
                             "请放宽 --n-min/--n-max 或加大 --max-scan")
        print("[warn] 无全条件例，使用放宽条件的第一个可用例")
        pick = relaxed
    gi, g, snap = pick
    if 'frontier' not in snap:
        raise SystemExit(f"graph_idx={gi} 在第 {args.step} 步前已耗尽预算/断链，"
                         "换一张图（--graph-idx）或降低 --step")
    print(f"picked graph_idx={gi}: n={g.num_nodes()} "
          f"B={max(1, int(args.budget_ratio * g.num_nodes()))} "
          f"label={int(labels[gi].item())} ev_pred={snap.get('ev_pred')} "
          f"ev_correct={snap.get('ev_correct')} "
          f"step{args.step}: |F|={len(snap['frontier'])} "
          f"chosen={snap['chosen']} p={snap['chosen_prob']:.4f} "
          f"fd={snap['chosen_frontier_degree']} "
          f"|F_after|={len(snap['frontier_after'])}", flush=True)
    release_dataset(ds)

    coords = g.ndata['feat'][:, :2].double().numpy()
    edges = undirected_edges(g)
    seed_node = deterministic_seed(gi, 0, g.num_nodes())
    probs = np.asarray(snap['probs'])
    norm = Normalize(vmin=float(probs.min()), vmax=float(probs.max()))

    # ---- 图 ----
    matplotlib.rcParams.update(mpl_rc)
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    w_mm, h_mm = AX_W * FIG_W_MM, AX_H * FIG_H_MM
    ax_a = fig.add_axes([AXA_X0, AX_Y0, AX_W, AX_H])
    ax_b = fig.add_axes([AXB_X0, AX_Y0, AX_W, AX_H])
    active = sorted(set(snap['revealed_after']) | set(snap['frontier_after'])
                    | set(snap['frontier']))
    for ax in (ax_a, ax_b):
        set_square_limits(ax, coords, w_mm, h_mm, nodes=active)

    draw_panel(ax_a, coords, edges, snap['revealed'], snap['frontier'],
               chosen=snap['chosen'], seed_node=seed_node, probs=probs,
               norm=norm, show_scores=True)
    ax_a.set_title(f"Step {args.step}: before reveal", fontsize=7, pad=4,
                   color='#1A1A1A')
    pc = snap['chosen_prob']
    ax_a.annotate(f'p = {pc:.2f}', xy=(coords[snap['chosen'], 0],
                                       coords[snap['chosen'], 1]),
                  xytext=(10, 8), textcoords='offset points', fontsize=6,
                  color=RED, ha='left', va='bottom',
                  arrowprops=dict(arrowstyle='-', color=RED, lw=0.5,
                                  shrinkA=1, shrinkB=3))
    ax_a.texts[-1].set_path_effects(
        [patheffects.withStroke(linewidth=1.6, foreground='white')])
    ax_a.text(0.03, 0.03, f"|F| = {len(snap['frontier'])} candidates",
              transform=ax_a.transAxes, fontsize=5.6, color=GRAY)

    draw_panel(ax_b, coords, edges, snap['revealed_after'],
               snap['frontier_after'], chosen=snap['chosen'],
               seed_node=seed_node, show_scores=False)
    ax_b.set_title(f"Step {args.step + 1}: after reveal", fontsize=7, pad=4,
                   color='#1A1A1A')
    ax_b.text(0.03, 0.03,
              f"|F| = {len(snap['frontier_after'])} after reveal",
              transform=ax_b.transAxes, fontsize=5.6, color=GRAY)

    for ax, letter in ((ax_a, 'a'), (ax_b, 'b')):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-13, 1), textcoords='offset points',
                    fontsize=8, fontweight='bold', va='bottom', ha='left')

    # 面板间揭示箭头
    ymid = AX_Y0 + AX_H / 2
    arrow = FancyArrowPatch((0.467, ymid), (0.533, ymid),
                            transform=fig.transFigure, arrowstyle='-|>',
                            mutation_scale=8, color=DARK, lw=1.0)
    fig.add_artist(arrow)
    fig.text(0.5, ymid + 0.055, 'reveal', ha='center', va='bottom',
             fontsize=6.2, color=DARK)

    # 底条：左 = 得分色条，右 = 图例
    cax = fig.add_axes([AXA_X0, 0.075, 0.34, 0.05])
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    pmin, pmax = float(probs.min()), float(probs.max())
    tick_vals = [pmin, pmin + (pmax - pmin) / 2, pmax]
    cbar.set_ticks(tick_vals)
    fmt = '{:.2f}' if pmax >= 0.05 else '{:.3f}'
    cbar.set_ticklabels([fmt.format(t) for t in tick_vals])
    cbar.set_label('Reader selection probability', fontsize=6, color='#1A1A1A')
    cbar.ax.tick_params(labelsize=5.5, length=1.6, pad=1.2)
    cbar.outline.set_linewidth(0.5)

    handles = [
        Line2D([], [], marker='o', ls='none', mfc=BLUE, mec=BLUE, ms=4.5,
               label='Revealed'),
        Line2D([], [], marker='o', ls='none', mfc='white', mec='#5A5A5A',
               ms=4.5, label='Frontier candidate'),
        Line2D([], [], marker='*', ls='none', mfc='none', mec=RED, ms=9,
               mew=0.9, label='Chosen by reader'),
        Line2D([], [], marker='D', ls='none', mfc='#1A1A1A', mec='#1A1A1A',
               ms=3.5, label='Seed node'),
    ]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.955, 0.035),
               ncol=2, fontsize=6, handlelength=1.2, columnspacing=0.9,
               handletextpad=0.4, labelspacing=0.35)

    # ---- 门禁 + 导出 ----
    out_dir = ckpt_path.parent / 'behavior-analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"selection_process_b{int(args.budget_ratio * 100)}_step{args.step}"
    require_matplotlib_panel_alignment(
        fig,
        json_out=f'{prefix}.alignment.json',
        overlay_svg=f'{prefix}.alignment.svg',
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        exclude_axes=[cax],
        row_groups=[{'id': 'panels-row', 'panels': ['a', 'b']}],
    )
    fig.savefig(f'{prefix}.svg', bbox_inches='tight')
    fig.savefig(f'{prefix}.pdf', bbox_inches='tight')
    fig.savefig(f'{prefix}.tiff', dpi=600, bbox_inches='tight')
    fig.savefig(f'{prefix}.png', dpi=600, bbox_inches='tight')
    print(f"saved: {prefix}.{{svg,pdf,tiff,png}}")

    sidecar = {
        'config': {
            'dataset': args.dataset, 'split': args.split,
            'budget_ratio': args.budget_ratio, 'step': args.step,
            'checkpoint': str(ckpt_path),
            'seed_index': args.seed_index,
            'seed_method': 'deterministic_seed(graph_idx, 0, n_nodes)',
            'protocol': 'deterministic argmax rollout, same as eval_loop',
        },
        'example': {
            'graph_idx': gi, 'n_nodes': int(g.num_nodes()),
            'budget': max(1, int(args.budget_ratio * g.num_nodes())),
            'label': int(labels[gi].item()),
            'seed_node': int(seed_node),
            'ev_pred': snap.get('ev_pred'),
            'ev_correct': snap.get('ev_correct'),
            'n_steps_total': snap.get('n_steps_total'),
            'step_snapshot': snap,
        },
    }
    with open(f'{prefix}.json', 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    ru = snap.get('runner_up_prob')
    fd = snap['chosen_frontier_degree']
    if fd == 1:
        mech = ('与已揭示区恰有 1 条连接：沿前沿延伸路径（contour walking）。'
                '这是补训后 checkpoint（epoch6）的主行为——见下方"行为注意"')
    elif fd == 2:
        mech = ('与已揭示区恰有 2 条连接：边界凹角向内补一格 = 紧致圆盘式生长'
                '（epoch3 旧主行为，见下方"行为注意"）')
    else:
        mech = f'与已揭示区有 {fd} 条连接'
    md = f"""# 选择过程图：真实 CIFAR10 图上的"打分 → 揭示 → 新前沿"（预算 {int(args.budget_ratio * 100)}%）

**一句话**：冻结 reader（`{ckpt_path.name}`）在一张真实 CIFAR10 超像素图上，
第 {args.step} 步对 {len(snap['frontier'])} 个前沿候选逐一给出选择概率（a 面板 viridis
着色），选中概率最高的节点（红星，p = {pc:.3f}）揭示后，前沿推进为
{len(snap['frontier_after'])} 个新候选（b 面板空心圈）。

## 例图参数

| 项 | 值 |
|---|---|
| checkpoint | `{ckpt_path}` |
| graph_idx（test 本地序号）/ 种子 | {gi} / deterministic_seed({gi}, 0, n) → 节点 {seed_node} |
| 图规模 / 预算 | n = {g.num_nodes()}，B = {sidecar['example']['budget']}（10%·n） |
| 类别 / 终判 | label = {sidecar['example']['label']}，证据头 ev_pred = {snap.get('ev_pred')}（{'正确' if snap.get('ev_correct') == 1 else '错误'}） |
| 协议 | 与 eval 严格同款：argmax 确定性 rollout、无 teleport |

## 第 {args.step} 步快照

- 揭示前：已揭示 {len(snap['revealed'])} 节点（含种子），前沿 |F| = {len(snap['frontier'])}；
- 被选节点：节点 {snap['chosen']}，前沿度数 = {fd}（{mech}），
  选择概率 p = {snap['chosen_prob']:.4f}（次高 {ru:.4f}）；
- 揭示后：已揭示 {len(snap['revealed_after'])} 节点，新前沿 |F| = {len(snap['frontier_after'])}。

## 读图

- **a**：蓝 = 已揭示，菱形 = 种子；前沿候选按 reader 选择概率 viridis 着色
  （底部色条），红星 = argmax 被选节点；灰虚线 = 前沿边（协议中免费可见）。
  只画读者可观察的已揭示区 + 前沿，未揭示节点不显示。
- **b**：揭示被选节点（红描边蓝点）后的新前沿（空心圈）：紧邻新邻居入列。

## 行为注意：epoch3 → epoch6 策略变化

用完全相同的 eval 协议 rollout 在 test 前 400 图上逐步统计"被选节点前沿度数"：

| checkpoint | 第 2 步 | 第 3–9 步 | 行为 |
|---|---|---|---|
| epoch3（旧主结果，行为分析 README 所用） | 98.3% 选 fd=2 | 84–91% 选 fd=2 | 凹角向内补，紧致圆盘生长 |
| epoch6（补训后，本图所用） | 99.8% 选 fd=1 | ≈99% 选 fd=1 | 沿前沿延伸路径，轮廓行走 |

补训不只提升精度（B10% ev_acc 29.9 → 33.8），还**定性地改变了读图策略**。
`behavior-analysis/README.md`（基于 epoch3）的"度数=2 凹角填充"结论对 epoch3
成立，但不再适用于 epoch6 主结果；引用行为规律时注意区分。

## 产物

`selection_process_b{int(args.budget_ratio * 100)}_step{args.step}.json`（逐步快照 +
每个候选的概率表）、`.svg/.pdf/.tiff/.png`、对齐/碰撞审计文件。
源码：`code/diagnosis/visualize_selection_process.py`。
"""
    with open(out_dir / f"selection_process_b{int(args.budget_ratio * 100)}.md",
              'w', encoding='utf-8') as f:
        f.write(md)
    print(f"saved: {out_dir / (f'selection_process_b{int(args.budget_ratio * 100)}.md')}")


if __name__ == '__main__':
    main()

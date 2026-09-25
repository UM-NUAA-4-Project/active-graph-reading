"""选择过程拓扑图（graph 18，第 2→3 步）：用户指定编码的 SVG 拓扑图。

数据：analyze 侧导出的 selection_process_b10_graph18_step2to3.json
（第 1/2 步转移：前沿节点+reader 分数+前沿度数、被选节点、揭示后前沿）。

图级契约
--------
核心主张：第 2 步 reader 面对 12 个前沿候选给出的分数高度分层，
argmax 选中节点揭示后前沿随之扩张（12→15）；c 面板表格进一步给出
三种选点口径（reader 选中 / 最大前沿度 / 前沿最大 IG）对 reader
证据分类头的一步前瞻信息增益——reader 的选择是否等于前沿最大 IG。
面板角色：a = 揭示前（已揭示区 + 白底分数环候选 + 红虚线选中圈）；
b = 揭示后（选中节点入揭示区，新前沿白圈）；
c = 表格（纯文本、无数据几何；宽度有意窄于 a/b，排除在等宽对齐
门禁外；上下边缘与 a/b 严格同带，同 y0/h）。
用户指定编码：已揭示 = 蓝色圆形（居中）；seed = 菱形（居中）；前沿 =
四周均匀成环，step 2 按分数 YlOrRd 渐变填充、step 3 白底；选中 =
红色虚线圆圈出。
布局 = 中心-辐射式：力导向角序决定环上顺序（减少交叉），前沿均匀分布。

c 面板 IG 定义（reader 分类头口径，非冻结分类器）：
  IG(v) = H(reader | R) - H(reader | R ∪ {v})
从 checkpoint 重放 eval rollout（argmax 确定性、deterministic_seed(gi,0,n)）
到第 2 步，取该步证据流的 (h, c) 与逐步预测分布为现状；对每个前沿候选
v 构造 R ∪ {v} 模拟态、以同款证据流前向一步，熵降即 IG(v)。重放的
被选节点/前沿集/逐点分数与 JSON 不一致（>5e-4）即报错退出。

导出契约：220×72 mm，Arial ≥5 pt，可编辑文字，svg/pdf/tiff/png +
面板对齐门禁（1.5 pt strict；c 为表格面板除外）+ PDF 碰撞审计；
另落盘 *_topo_ig.json（逐候选 IG + 表行）。

用法（仓库根运行）:
  python code/diagnosis/plot_selection_topology.py \
      --data code/result/cifar10/experiments/ev-R-read/evidence/rl/behavior-analysis/selection_process_b10_graph18_step2to3.json
checkpoint 默认从 JSON 目录的上级（rl/）按 JSON 内 checkpoint 字段解析，
--checkpoint 可覆盖。
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
import networkx as nx  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

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

BLUE = '#33608C'       # 已揭示（钢蓝）
RED = '#C22432'        # 被选虚线圈
GRAY = '#767676'
DARK = '#4D4D4D'
SEED = '#22304A'       # 种子菱形（深藏青）
EDGE_RR = '#8FA0B3'    # 已揭示区实线
EDGE_RF = '#B7C0CB'    # 前沿虚线
FRONTIER_EDGE = '#C9BBA4'   # 前沿节点暖沙描边
FRONTIER_EDGE_B = '#9AA5B1'  # panel b 白圈描边（冷灰）
# 分数渐变：奶油 → 琥珀 → 赭 → 深红（比 YlOrRd 更柔和、与揭示蓝互补）
CMAP_STOPS = ['#FBF3DF', '#F3C46B', '#DE7E33', '#B93A2A', '#8F1D2C']

FIG_W_MM, FIG_H_MM = 220.0, 72.0
AX_Y0, AX_H = 0.24, 0.70
_LEFT_MM, _PW_MM, _GUT_MM = 6.75, 62.25, 11.0   # a/b 面板物理尺寸不变
AXA_X0 = _LEFT_MM / FIG_W_MM
AXB_X0 = (_LEFT_MM + _PW_MM + _GUT_MM) / FIG_W_MM
AXC_X0 = (_LEFT_MM + 2 * _PW_MM + 2 * _GUT_MM) / FIG_W_MM
AX_W = _PW_MM / FIG_W_MM
AXC_W = 1.0 - _LEFT_MM / FIG_W_MM - AXC_X0      # 60 mm 表格面板
LAYOUT_SEED = 11


def undirected_edges_pairs(g, nodes=None) -> list:
    src, dst = g.edges()
    keep = set(int(v) for v in nodes) if nodes is not None else None
    out = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        if keep is not None and (u not in keep or v not in keep):
            continue
        out.add((min(u, v), max(u, v)))
    return sorted(out)


def spring_order(pairs, nodes, seed=LAYOUT_SEED):
    """力导向布局 → 返回各节点位置（仅用于确定环上的角序，减少交叉）。"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(pairs)
    return nx.spring_layout(G, iterations=500, seed=seed)


def radial_layout(pairs, active, revealed_center, R=1.0, r_center=0.20,
                  seed=LAYOUT_SEED):
    """中心 = 已揭示节点，四周 = 前沿节点均匀成环。

    环上顺序：前沿按"主归属中心节点"（spring 中最近的已揭示邻居）分组，
    组内按方位角排序，组依中心节点的 spring 角序首尾相接——每个中心节点
    的辐条落在同一扇区，连线不穿中心。中心节点坐向自己扇区的平均角。"""
    pairset = set(pairs)
    pos0 = spring_order(pairs, active, seed)
    rc = sorted(revealed_center)
    center = np.mean([pos0[v] for v in rc], axis=0)

    def _ang(v, ref):
        d = pos0[v] - ref
        return float(np.arctan2(d[1], d[0]))

    rc_ang = {v: _ang(v, center) for v in rc}
    frontier = [v for v in active if v not in set(revealed_center)]
    groups = {v: [] for v in rc}
    for v in frontier:
        neigh = [u for u in rc
                 if (min(u, v), max(u, v)) in pairset]
        if not neigh:
            neigh = sorted(rc, key=lambda u: float(np.linalg.norm(
                pos0[v] - pos0[u])))[:1]
        primary = min(neigh, key=lambda u: float(np.linalg.norm(
            pos0[v] - pos0[u])))
        groups[primary].append(v)
    order = []
    sector = {}
    for u in sorted(rc, key=lambda t: rc_ang[t]):
        members = sorted(groups[u], key=lambda t: _ang(t, pos0[u]))
        sector[u] = members
        order.extend(members)
    pos = {}
    n = max(len(order), 1)
    for i, v in enumerate(order):
        th = 2 * np.pi * i / n
        pos[v] = center + R * np.array([np.cos(th), np.sin(th)])
    for u in rc:
        if sector.get(u):
            idx = [order.index(v) for v in sector[u]]
            th_mean = 2 * np.pi * (np.mean(idx) + 0.5) / n
        else:   # 无前沿的中心节点：均布在剩余角度
            th_mean = 2 * np.pi * rc.index(u) / len(rc) + np.pi / 2
        pos[u] = center + r_center * np.array([np.cos(th_mean),
                                               np.sin(th_mean)])
    return pos


def set_limits_pts(ax, pts, w_mm, h_mm, pad=1.12):
    cx = (pts[:, 0].max() + pts[:, 0].min()) / 2
    cy = (pts[:, 1].max() + pts[:, 1].min()) / 2
    dx = (pts[:, 0].max() - pts[:, 0].min()) * pad
    dy = (pts[:, 1].max() - pts[:, 1].min()) * pad
    target = h_mm / w_mm
    if dy / dx < target:
        dy = dx * target
    else:
        dx = dy / target
    ax.set_xlim(cx - dx / 2, cx + dx / 2)
    ax.set_ylim(cy + dy / 2, cy - dy / 2)   # y 反向（图像惯例）
    ax.set_axis_off()


def draw_edges(ax, pos, pairs, revealed):
    R = set(revealed)
    seg_rr, seg_rf = [], []
    for u, v in pairs:
        ur, vr = u in R, v in R
        if ur and vr:
            seg_rr.append((pos[u], pos[v]))
        elif ur or vr:
            seg_rf.append((pos[u], pos[v]))
    ax.add_collection(LineCollection(seg_rf, colors=EDGE_RF, linewidths=0.5,
                                     linestyles=(0, (2.5, 1.8)), zorder=1))
    ax.add_collection(LineCollection(seg_rr, colors=EDGE_RR,
                                     linewidths=0.6, zorder=1))


def node_xy(pos, nodes):
    pts = np.array([pos[v] for v in nodes])
    return pts[:, 0], pts[:, 1]


def _entropy_nats(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-8)).sum())


def rollout_to_step(reader, g, ek2id, graph_idx, budget_ratio, device,
                    target_step, seed_i=0):
    """eval 协议 rollout 到第 target_step（1 基）选择点（揭示前返回）。

    循环语义与 visualize_selection_process.rollout_capture 逐字一致
    （argmax 确定性、teleport 同款——本例图要求无 teleport），额外
    返回该步证据流的 (h, c)、逐步预测 logits 与前沿顺序。"""
    n = g.num_nodes()
    B = max(1, int(budget_ratio * n))
    seed_node = deterministic_seed(graph_idx, seed_i, n)
    state = GraphState(g, seed_node=seed_node, edge_key_to_id=ek2id)
    state._budget_limit = B
    hc = reader.init_evidence(1, device)
    last_revealed = seed_node
    rng = random.Random(seed_node * 1_000_003 + B)
    step_idx = 0
    while state.get_budget_used() < state._budget_limit:
        if state.is_frontier_empty:
            teleport_if_stuck(state, rng=rng)
            return None
        frontier = state.get_frontier_nodes()
        with torch.no_grad():
            results, hc, ev_logits = reader.forward_states_evidence(
                [state], temperature=1.0, hc_list=hc,
                last_revealed=[last_revealed])
        _, probs, r_frontier = results[0]
        if not r_frontier:
            return None
        chosen = int(r_frontier[int(probs.argmax().item())])
        if step_idx == target_step - 1:
            return {'state': state, 'hc': hc, 'ev_logits': ev_logits[0],
                    'probs': probs.detach().cpu().numpy(),
                    'frontier': list(r_frontier), 'chosen': chosen,
                    'seed_node': seed_node, 'budget': B}
        reveal_node(state, chosen)
        last_revealed = chosen
        step_idx += 1
    return None


def compute_reader_head_ig(ckpt_path, g, graph_idx, budget_ratio, device,
                           step_rec, seed_expect):
    """第 2 步各前沿候选对 reader 证据分类头的一步前瞻信息增益。

    IG(v) = H(reader | R) - H(reader | R ∪ {v})：现状 = rollout 第 2 步
    的证据流分布；候选 = 以同款证据流（同 (h, c)）前进一步。先与 JSON
    逐项校验（seed/chosen/前沿集/逐点分数/前沿度数），漂移即报错。"""
    reader = load_reader_from_checkpoint(ckpt_path, device)
    reader.eval()
    for p_ in reader.parameters():
        p_.requires_grad_(False)
    snap = rollout_to_step(reader, g, None, graph_idx, budget_ratio, device,
                           target_step=2)
    if snap is None:
        raise SystemExit('rollout 未到达第 2 步（teleport/断链），与 JSON 例图不符')
    if snap['seed_node'] != seed_expect:
        raise SystemExit(f'seed 不一致: {snap["seed_node"]} vs {seed_expect}')
    json_by_node = {f['node']: f for f in step_rec['frontier']}
    if snap['chosen'] != step_rec['chosen']:
        raise SystemExit(f'被选节点不一致: {snap["chosen"]} vs '
                         f'{step_rec["chosen"]}——checkpoint/协议漂移')
    if sorted(snap['frontier']) != sorted(json_by_node):
        raise SystemExit('前沿节点集与 JSON 不一致——checkpoint/协议漂移')
    for i, v in enumerate(snap['frontier']):
        p_json = json_by_node[v]['prob']
        if abs(float(snap['probs'][i]) - p_json) > 5e-4:
            raise SystemExit(f'节点 {v} 分数漂移: {snap["probs"][i]:.6f} '
                             f'vs JSON {p_json}——checkpoint/协议漂移')
    fd = {v: int(snap['state'].get_frontier_degree(v)) for v in snap['frontier']}
    for v in fd:
        if fd[v] != json_by_node[v]['frontier_degree']:
            raise SystemExit(f'节点 {v} 前沿度数不一致: {fd[v]} vs '
                             f'{json_by_node[v]["frontier_degree"]}')

    H_cur = _entropy_nats(snap['ev_logits'])
    revealed = sorted(snap['state'].V_revealed)
    cand = list(snap['frontier'])
    cand_states = []
    for v in cand:
        s = GraphState(g, seed_node=snap['seed_node'])
        for u in revealed:
            if u != snap['seed_node']:
                reveal_node(s, u)
        reveal_node(s, v)
        cand_states.append(s)
    with torch.no_grad():
        _, _, cand_ev = reader.forward_states_evidence(
            cand_states, temperature=1.0,
            hc_list=[snap['hc'][0]] * len(cand_states),
            last_revealed=list(cand))
    ig_by_node = {v: H_cur - _entropy_nats(lg) for v, lg in zip(cand, cand_ev)}
    return {'H_cur': H_cur, 'ig_by_node': ig_by_node, 'fd': fd,
            'frontier': cand, 'probs': snap['probs']}


def build_ig_rows(ig, step_rec):
    """三行表格数据 + fd 并列脚注。最大前沿度组取组内 IG 最大者为代表
    （度数贪心在该并列下的最有利实现）。"""
    json_by_node = {f['node']: f for f in step_rec['frontier']}
    chosen = step_rec['chosen']
    max_fd = max(ig['fd'].values())
    fd_group = [v for v in ig['frontier'] if ig['fd'][v] == max_fd]
    v_fd = max(fd_group, key=lambda v: ig['ig_by_node'][v])
    v_ig = max(ig['frontier'], key=lambda v: ig['ig_by_node'][v])
    if v_ig == chosen or v_fd == chosen:      # 例图不出现；兜底防重复行
        raise SystemExit('行代表与 reader 选中节点重合，表格需去重后再出图')
    rows = []
    for role, v, color, bold in (
            ('Reader-selected', chosen, RED, False),
            ('Max frontier degree', v_fd, '#1A1A1A', False),
            ('Max IG in frontier', v_ig, '#1A1A1A', True)):
        rows.append((role, v, json_by_node[v]['prob'], ig['fd'][v],
                     ig['ig_by_node'][v], color, bold))
    note = None
    if len(fd_group) > 1:
        note = (f'{len(fd_group)} candidates tie at fd = {max_fd}; '
                'shown: their IG-max')
    return rows, note


def draw_ig_table(ax, rows, h_cur, fd_note):
    """c 面板：reader 分类头一步前瞻 IG 表（纯文本，水平线 only）。"""
    ax.set_axis_off()
    cols = [(0.02, 'left', 'candidate'), (0.45, 'left', 'node'),
            (0.69, 'right', 'p'), (0.79, 'right', 'fd'),
            (0.98, 'right', 'IG (nats)')]
    y_hdr, y_rule, y_rows, y_bot = 0.88, 0.815, (0.615, 0.415, 0.215), 0.115
    for x, ha, lab in cols:
        ax.text(x, y_hdr, lab, transform=ax.transAxes, ha=ha, va='center',
                fontsize=5.8, color=GRAY, fontweight='bold')
    ax.plot([0.0, 1.0], [y_rule] * 2, transform=ax.transAxes,
            color='#B9B9B9', lw=0.5, clip_on=False)
    # reader 选中行淡红底（呼应 a/b 的红虚线圈）
    ax.add_patch(Rectangle((0.0, y_rows[0] - 0.10), 1.0, 0.20,
                           transform=ax.transAxes, facecolor=RED,
                           alpha=0.055, edgecolor='none', zorder=0))
    for (role, v, p, fdv, igv, color, bold), yr in zip(rows, y_rows):
        ax.text(0.02, yr, role, transform=ax.transAxes, ha='left',
                va='center', fontsize=6.3, color=color)
        ax.text(0.45, yr, f'n.{v}', transform=ax.transAxes, ha='left',
                va='center', fontsize=6.3, color='#1A1A1A')
        ax.text(0.69, yr, f'{p:.3f}', transform=ax.transAxes, ha='right',
                va='center', fontsize=6.3, color='#1A1A1A')
        ax.text(0.79, yr, f'{fdv}', transform=ax.transAxes, ha='right',
                va='center', fontsize=6.3, color='#1A1A1A')
        ax.text(0.98, yr, f'{igv:+.4f}', transform=ax.transAxes, ha='right',
                va='center', fontsize=6.3, color='#1A1A1A',
                fontweight='bold' if bold else 'normal')
    ax.plot([0.0, 1.0], [y_bot] * 2, transform=ax.transAxes,
            color='#B9B9B9', lw=0.5, clip_on=False)
    foot = [
        (0.075, 'IG = entropy drop (nats) of the reader head when the node'),
        (-0.005, 'is revealed next; H now = {:.3f}; fd = frontier degree'
         .format(h_cur)),
    ]
    if fd_note:
        foot.append((-0.085, fd_note))
    for y, txt in foot:
        ax.text(0.0, y, txt, transform=ax.transAxes, ha='left', va='top',
                fontsize=5.2, color=GRAY)


def main():
    parser = argparse.ArgumentParser(description='选择过程拓扑图（graph18 第2→3步）')
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='D:\\data\\superpixels')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=None,
                        help='reader checkpoint（默认按 JSON 内字段从 rl/ 解析）')
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    gi = data['graph_idx']
    step_rec = next(r for r in data['steps'] if r['step'] == 2)
    revealed = step_rec['revealed_nodes']
    frontier = step_rec['frontier']
    frontier_after = step_rec['frontier_after']
    revealed_after = step_rec['revealed_after']
    chosen = step_rec['chosen']
    seed_node = data['seed_node']
    probs = np.array([f['prob'] for f in frontier])
    f_nodes = [f['node'] for f in frontier]

    # reader 证据头一步前瞻 IG（c 面板表格数据；先与 JSON 逐项校验）
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ck_name = data.get('checkpoint', 'checkpoint_epoch6.pt')
        ckpt_path = data_path.parent.parent / ck_name
        if not ckpt_path.exists():
            ckpt_path = data_path.parent / ck_name
    if not ckpt_path.exists():
        raise SystemExit(f'checkpoint 不存在: {ckpt_path}（--checkpoint 覆盖）')
    budget_B = max(1, int(args.budget_ratio * data['n_nodes']))
    if budget_B != data['budget']:
        raise SystemExit(f'预算不一致: ratio {args.budget_ratio} → {budget_B}，'
                         f'JSON budget = {data["budget"]}')

    # 图结构（布局 + 边）
    ds = get_dataset(args.dataset, root=args.root, splits=('test',))
    g = ds.test_graphs[gi]
    release_dataset(ds)
    ig = compute_reader_head_ig(ckpt_path, g, gi, args.budget_ratio,
                                args.device, step_rec, seed_expect=seed_node)
    rows, fd_note = build_ig_rows(ig, step_rec)
    print(f'IG@step2 (reader head, nats): H_now = {ig["H_cur"]:.4f}')
    for role, v, p, fdv, igv, *_ in rows:
        print(f'  {role:<22} n.{v:<4} p={p:.3f} fd={fdv} IG={igv:+.4f}')
    active = sorted(set(revealed_after) | set(frontier_after) | set(f_nodes))
    pairs = undirected_edges_pairs(g, active)
    pos_a = radial_layout(pairs, active, revealed_center=revealed)
    pos_b = radial_layout(pairs, active, revealed_center=revealed_after)
    w_mm, h_mm = AX_W * FIG_W_MM, AX_H * FIG_H_MM

    # 选中圈半径：节点半径（≈1.5mm）的 1.9 倍，紧套住节点（set_limits 后换算）
    norm = Normalize(vmin=0.0, vmax=float(probs.max()))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('sel_prob', CMAP_STOPS)

    matplotlib.rcParams.update(mpl_rc)
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax_a = fig.add_axes([AXA_X0, AX_Y0, AX_W, AX_H])
    ax_b = fig.add_axes([AXB_X0, AX_Y0, AX_W, AX_H])
    ax_c = fig.add_axes([AXC_X0, AX_Y0, AXC_W, AX_H])
    all_pts = np.array(list(pos_a.values()) + list(pos_b.values()))
    for ax in (ax_a, ax_b):
        set_limits_pts(ax, all_pts, w_mm, h_mm)
    x0, x1 = ax_a.get_xlim()
    mm_to_data = (x1 - x0) / w_mm
    r_node = 1.5 * mm_to_data            # 节点半径（直径≈3.1mm）
    circ_r = 1.9 * r_node                # 红色虚线选中圈半径

    # ---- a：第 2 步揭示前 ----
    draw_edges(ax_a, pos_a, pairs, revealed)
    ax_a.scatter(*node_xy(pos_a, [v for v in revealed if v != seed_node]),
                 s=60, c=BLUE, edgecolors='white', linewidths=0.5, zorder=4)
    ax_a.scatter(*node_xy(pos_a, [seed_node]), marker='D', s=55, c=SEED,
                 edgecolors='white', linewidths=0.5, zorder=6)
    for f in frontier:
        x, y = pos_a[f['node']]
        ax_a.scatter([x], [y], s=60, facecolors=cmap(norm(f['prob'])),
                     edgecolors=FRONTIER_EDGE, linewidths=0.5, zorder=3)
    ax_a.add_patch(Circle(pos_a[chosen], circ_r, fill=False, edgecolor=RED,
                          linewidth=1.0, linestyle=(0, (3, 2)), zorder=5))
    ax_a.set_title(f'Step 2: before reveal (|F| = {len(frontier)})',
                   fontsize=7, pad=4, color='#1A1A1A')
    ax_a.annotate(f"p = {step_rec['chosen_prob']:.3f}",
                  xy=(pos_a[chosen][0] + circ_r * 0.71,
                      pos_a[chosen][1] - circ_r * 0.71),
                  xytext=(9, 7), textcoords='offset points', fontsize=6,
                  color=RED, ha='left', va='bottom',
                  arrowprops=dict(arrowstyle='-', color=RED, lw=0.5,
                                  shrinkA=1, shrinkB=2))
    from matplotlib import patheffects
    ax_a.texts[-1].set_path_effects(
        [patheffects.withStroke(linewidth=1.6, foreground='white')])

    # ---- b：揭示后（第 3 步开始）----
    draw_edges(ax_b, pos_b, pairs, revealed_after)
    ax_b.scatter(*node_xy(pos_b,
                          [v for v in revealed_after if v != seed_node]),
                 s=60, c=BLUE, edgecolors='white', linewidths=0.5, zorder=4)
    ax_b.scatter(*node_xy(pos_b, [seed_node]), marker='D', s=55, c=SEED,
                 edgecolors='white', linewidths=0.5, zorder=6)
    ax_b.scatter(*node_xy(pos_b, frontier_after), s=60, facecolors='white',
                 edgecolors=FRONTIER_EDGE_B, linewidths=0.6, zorder=3)
    ax_b.add_patch(Circle(pos_b[chosen], circ_r, fill=False, edgecolor=RED,
                          linewidth=1.0, linestyle=(0, (3, 2)), zorder=5))
    ax_b.set_title(f'Step 3: after reveal (|F| = {len(frontier_after)})',
                   fontsize=7, pad=4, color='#1A1A1A')

    for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c')):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(-13, 1), textcoords='offset points',
                    fontsize=8, fontweight='bold', va='bottom', ha='left')

    # 面板间揭示箭头
    from matplotlib.patches import FancyArrowPatch
    ymid = AX_Y0 + AX_H / 2
    arrow = FancyArrowPatch((0.323, ymid), (0.355, ymid),
                            transform=fig.transFigure, arrowstyle='-|>',
                            mutation_scale=8, color=DARK, lw=1.0)
    fig.add_artist(arrow)
    fig.text((AXA_X0 + AX_W + AXB_X0) / 2, ymid + 0.055, 'reveal',
             ha='center', va='bottom', fontsize=6.2, color=DARK)

    # ---- c：第 2 步候选对 reader 分类头的一步前瞻 IG 表 ----
    ax_c.set_title('Information gain to reader head', fontsize=7, pad=4,
                   color='#1A1A1A')
    draw_ig_table(ax_c, rows, ig['H_cur'], fd_note)

    # 底条：左 = 分数色条，右 = 图例
    cax = fig.add_axes([AXA_X0, 0.075, 51.0 / FIG_W_MM, 0.05])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    pmax = float(probs.max())
    cbar.set_ticks([0.0, pmax / 2, pmax])
    cbar.set_ticklabels(['0.00', f'{pmax / 2:.2f}', f'{pmax:.2f}'])
    cbar.set_label('Reader selection probability (frontier fill)',
                   fontsize=6, color='#1A1A1A')
    cbar.ax.tick_params(labelsize=5.5, length=1.6, pad=1.2)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#B9B9B9')

    handles = [
        Line2D([], [], marker='o', ls='none', mfc=BLUE, mec=BLUE, ms=5,
               label='Revealed'),
        Line2D([], [], marker='o', ls='none', mfc=cmap(0.55),
               mec=FRONTIER_EDGE, ms=5, label='Scored frontier (panel a)'),
        Line2D([], [], marker='o', ls='none', mfc='white',
               mec=FRONTIER_EDGE_B, ms=5, label='New frontier (panel b)'),
        Line2D([], [], marker='o', ls='none', mfc='none', mec=RED, ms=9,
               mew=0.9, linestyle=(0, (2.5, 1.8)), label='Chosen (argmax)'),
        Line2D([], [], marker='D', ls='none', mfc=SEED, mec=SEED,
               ms=4.5, label='Seed node'),
    ]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(AXB_X0 + AX_W, 0.035),
               ncol=2, fontsize=6, handlelength=1.4, columnspacing=0.9,
               handletextpad=0.4, labelspacing=0.35)

    # ---- 门禁 + 导出 ----
    out_prefix = data_path.with_name(
        data_path.name.replace('_step2to3.json', '_topo'))
    ig_dump = {
        'graph_idx': gi, 'step': 2, 'checkpoint': str(ckpt_path),
        'budget': budget_B, 'seed_node': seed_node,
        'definition': 'IG(v) = H(reader head | R) - H(reader head | R+{v}), '
                      'one-step evidence-stream lookahead on the eval '
                      'rollout (nats); reader head = checkpoint_epoch6 '
                      'evidence_head, NOT the frozen classifier',
        'H_current_nats': round(ig['H_cur'], 6),
        'rows': [
            {'role': role, 'node': int(v), 'prob': round(float(p), 6),
             'frontier_degree': int(fdv), 'ig_nats': round(float(gv), 6)}
            for (role, v, p, fdv, gv, *_rest) in rows],
        'per_frontier_node': [
            {'node': int(v), 'prob': round(float(ig['probs'][i]), 6),
             'frontier_degree': int(ig['fd'][v]),
             'ig_nats': round(float(ig['ig_by_node'][v]), 6)}
            for i, v in enumerate(ig['frontier'])],
    }
    with open(f'{out_prefix}_ig.json', 'w', encoding='utf-8') as f:
        json.dump(ig_dump, f, ensure_ascii=False, indent=2)
    print(f'IG table data: {out_prefix}_ig.json')
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
    print(f"saved: {out_prefix}.{{svg,pdf,tiff,png}}")


if __name__ == '__main__':
    main()

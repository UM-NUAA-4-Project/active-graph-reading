"""Phase 4: 分类质量与信息积累分析。

追踪每一步揭示后的分类器输出，分析：
  1. 信息积累曲线（CE_t, R_t, top1_prob_t vs t/B）
  2. ΔCE_t 信息增益递减分析
  3. 分类锁定点分布（首次正确后不再翻错的步数）
  4. 预测翻转分析
  5. per-class 分类质量 + 混淆矩阵（37×37）

输出：
  - raw_data/phase4_accumulation.json  原始逐步数据
  - figures/phase4_accumulation_curve.png  信息积累曲线
  - figures/phase4_delta_ce.png  ΔCE_t 信息增益曲线
  - figures/phase4_confusion_matrix.png  混淆矩阵
  - figures/phase4_lock_point.png  锁定点分布
  - figures/phase4_per_class.png  per-class 准确率
"""

import argparse
import gc
import random
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import pandas as pd
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from training_utils import patch_old_checkpoint

K = 37
LN_K = math.log(K)


# ---------------------------------------------------------------------------
# per-step data collection for one graph
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_per_step(
    reader: LearnableReader,
    g_raw: dgl.DGLGraph,
    seed_node: int,
    label: int,
    budget_abs: int,
    device: torch.device,
    classifier: GraphClassifier,
    builder: SubgraphBuilder,
) -> List[dict]:
    """运行 reveal 循环，每一步后对当前子图分类并记录指标。"""
    state = GraphState(g_raw, seed_node=seed_node)
    state._budget_limit = budget_abs
    reader.reset()
    initial_frontier = list(state._initial_frontier)
    if initial_frontier:
        reader.on_new_frontier(initial_frontier)

    steps_data = []
    prev_ce = None

    for t in range(budget_abs):
        if state.is_frontier_empty:
            break

        # reader 选择节点
        chosen_node = reader.select(state)
        if chosen_node is None:
            break

        new_frontier = reveal_node(state, chosen_node)
        if new_frontier:
            reader.on_new_frontier(new_frontier)

        # 对当前 V_revealed 子图分类
        sub_g, _ = builder.build(state)
        sub_g = sub_g.to(device)
        nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(sub_g, nids)
        probs = F.softmax(logits, dim=0)

        ce_t = F.cross_entropy(
            logits, torch.tensor([label], device=device),
        ).item()
        r_t = math.tanh(1.0 - ce_t / LN_K)

        # ΔCE_t = CE_{t-1} - CE_t（正=信息增益，负=信息损失）
        delta_ce = prev_ce - ce_t if prev_ce is not None else 0.0
        prev_ce = ce_t

        top1_prob, top1_label = probs.max(dim=1)
        top1_prob = top1_prob.item()
        top1_label = top1_label.item()
        top2_vals = probs.topk(2, dim=1).values
        top2_val = top2_vals[0, 1].item() if top2_vals.shape[1] > 1 else 0.0

        step = {
            'step': t + 1,
            'CE_t': ce_t,
            'delta_CE_t': delta_ce,
            'R_t': r_t,
            'pred_correct_t': 1 if top1_label == label else 0,
            'top1_prob_t': top1_prob,
            'top2_prob_t': top2_val,
            'top1_top2_gap_t': top1_prob - top2_val,
            'pred_label_t': top1_label,
            'true_label': label,
        }
        steps_data.append(step)

    return steps_data


# ---------------------------------------------------------------------------
# lock point analysis
# ---------------------------------------------------------------------------

def compute_lock_point(steps_data: List[dict]) -> int:
    """返回锁定点 t*（1-indexed）：首次 pred_correct=1 且之后不再翻错的步。
    若从未正确返回 -1，若从不锁定（最后一步才正确且之前有错）返回 budget_abs。
    """
    if not steps_data:
        return -1

    budget = len(steps_data)
    # 从最后一步往前找：最后一个"正确且之后不再错"的起点
    # 简单定义：从末尾起第一个 pred_correct=1，然后往前找到首次连续的 1 的起点
    correct_from_end = 0
    for i in range(budget - 1, -1, -1):
        if steps_data[i]['pred_correct_t'] == 1:
            correct_from_end += 1
        else:
            break

    if correct_from_end == 0:
        return -1  # 从未正确

    # 锁定点 = 正确段开始的步（1-indexed）
    lock_step = budget - correct_from_end + 1
    return lock_step


def count_flips(steps_data: List[dict]) -> Dict:
    """统计预测翻转。"""
    if len(steps_data) < 2:
        return {'n_flips': 0, 'flip_to_correct': 0, 'flip_to_wrong': 0}

    n_flips = 0
    to_correct = 0
    to_wrong = 0

    for t in range(1, len(steps_data)):
        prev_label = steps_data[t - 1]['pred_label_t']
        curr_label = steps_data[t]['pred_label_t']
        if prev_label != curr_label:
            n_flips += 1
            true_label = steps_data[t]['true_label']
            if curr_label == true_label:
                to_correct += 1
            else:
                to_wrong += 1

    return {'n_flips': n_flips, 'flip_to_correct': to_correct, 'flip_to_wrong': to_wrong}


# ---------------------------------------------------------------------------
# interpolation to uniform t/B grid (for averaging across graphs with different B)
# ---------------------------------------------------------------------------

def interpolate_to_grid(steps_data: List[dict], n_points: int = 20) -> Dict[str, List[float]]:
    """将 steps_data 插值到均匀的 t/B 网格。"""
    if not steps_data:
        return {k: [0.0] * n_points for k in
                ['CE', 'delta_CE', 'R', 'top1_prob', 'top1_top2_gap', 'pred_correct']}

    B = len(steps_data)
    t_frac = [(i + 1) / B for i in range(B)]

    result = {}
    for key, extract in [
        ('CE', lambda d: d['CE_t']),
        ('delta_CE', lambda d: d['delta_CE_t']),
        ('R', lambda d: d['R_t']),
        ('top1_prob', lambda d: d['top1_prob_t']),
        ('top1_top2_gap', lambda d: d['top1_top2_gap_t']),
        ('pred_correct', lambda d: d['pred_correct_t']),
    ]:
        values = [extract(d) for d in steps_data]
        grid_x = np.linspace(1.0 / n_points, 1.0, n_points)
        interpolated = np.interp(grid_x, t_frac, values)
        result[key] = interpolated.tolist()

    return result


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_accumulation_curve(
    grid_data: Dict[str, np.ndarray],
    grid_std: Dict[str, np.ndarray],
    output_path: str,
):
    """信息积累曲线：CE_t, R_t, top1_prob_t vs t/B。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    n_points = len(grid_data['CE'])
    t_frac = np.linspace(1.0 / n_points, 1.0, n_points)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CE curve
    ax = axes[0]
    ax.plot(t_frac, grid_data['CE'], color='#C44E52', linewidth=2)
    ax.fill_between(t_frac,
                    grid_data['CE'] - grid_std['CE'],
                    grid_data['CE'] + grid_std['CE'],
                    alpha=0.2, color='#C44E52')
    ax.set_xlabel('t / B', fontsize=11)
    ax.set_ylabel('Cross-Entropy', fontsize=11)
    ax.set_title('CE_t vs Budget Fraction', fontsize=12)
    ax.grid(alpha=0.3)

    # R curve
    ax = axes[1]
    ax.plot(t_frac, grid_data['R'], color='#55A868', linewidth=2)
    ax.fill_between(t_frac,
                    grid_data['R'] - grid_std['R'],
                    grid_data['R'] + grid_std['R'],
                    alpha=0.2, color='#55A868')
    ax.set_xlabel('t / B', fontsize=11)
    ax.set_ylabel('R_t = tanh(1 - CE/ln K)', fontsize=11)
    ax.set_title('Reward vs Budget Fraction', fontsize=12)
    ax.grid(alpha=0.3)

    # top1_prob curve
    ax = axes[2]
    ax.plot(t_frac, grid_data['top1_prob'], color='#4C72B0', linewidth=2)
    ax.fill_between(t_frac,
                    grid_data['top1_prob'] - grid_std['top1_prob'],
                    grid_data['top1_prob'] + grid_std['top1_prob'],
                    alpha=0.2, color='#4C72B0')
    ax.set_xlabel('t / B', fontsize=11)
    ax.set_ylabel('Top-1 Probability', fontsize=11)
    ax.set_title('Classifier Confidence vs Budget Fraction', fontsize=12)
    ax.axhline(y=1.0 / K, color='gray', linestyle=':', alpha=0.5, label=f'Random (1/{K})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Phase 4: Information Accumulation Curves', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Accumulation curve saved to {output_path}")


def plot_lock_point_histogram(
    lock_points: List[int],
    budget_abs_list: List[int],
    output_path: str,
):
    """锁定点 t*/B 分布直方图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    # 计算 t*/B 比例
    ratios = []
    for lp, B in zip(lock_points, budget_abs_list):
        if lp > 0 and B > 0:
            ratios.append(lp / B)

    if not ratios:
        print("  No lock points to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    ax = axes[0]
    ax.hist(ratios, bins=20, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(ratios), color='#C44E52', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(ratios):.2f}')
    ax.axvline(x=np.median(ratios), color='#55A868', linestyle='--',
               linewidth=2, label=f'Median = {np.median(ratios):.2f}')
    ax.set_xlabel('Lock Point t* / B', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Classification Lock Points', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 累积分布
    ax = axes[1]
    sorted_ratios = np.sort(ratios)
    cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    ax.plot(sorted_ratios, cumulative, color='#4C72B0', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90%')
    ax.set_xlabel('Lock Point t* / B', fontsize=11)
    ax.set_ylabel('Cumulative Fraction', fontsize=11)
    ax.set_title('CDF of Lock Points', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Phase 4: Classification Lock Point Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Lock point histogram saved to {output_path}")


def plot_per_class(
    per_class_stats: Dict[int, Dict[str, float]],
    output_path: str,
):
    """per-class 准确率排序图 — 识别容易/中等/困难类。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    if not per_class_stats:
        return

    classes = sorted(per_class_stats.keys())
    accs = [per_class_stats[c]['accuracy_mean'] for c in classes]
    n_list = [per_class_stats[c]['n'] for c in classes]

    # 按准确率排序
    sorted_idx = np.argsort(accs)
    sorted_classes = [classes[i] for i in sorted_idx]
    sorted_accs = [accs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(sorted_classes))

    # 颜色：困难类（<40%）红色，中等（40-80%）橙色，容易（>80%）绿色
    colors = []
    for acc in sorted_accs:
        if acc < 0.4:
            colors.append('#C44E52')
        elif acc < 0.8:
            colors.append('#DD8452')
        else:
            colors.append('#55A868')

    bars = ax.bar(x, sorted_accs, color=colors, edgecolor='white', width=0.7)

    # 标注
    for bar, acc, cls_id in zip(bars, sorted_accs, sorted_classes):
        if acc > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{cls_id}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.axhline(y=0.4, color='#C44E52', linestyle='--', alpha=0.5, label='Hard (<40%)')
    ax.axhline(y=0.8, color='#55A868', linestyle='--', alpha=0.5, label='Easy (>80%)')

    ax.set_xlabel('Class (sorted by accuracy)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Phase 4: Per-Class Classification Accuracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 底部标注样本数
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # 只标注少量类别避免拥挤
    step = max(1, len(sorted_classes) // 10)
    tick_positions = x[::step]
    tick_labels = [f'n={n_list[sorted_idx[i]]}' for i in range(0, len(sorted_idx), step)]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=6, rotation=45)
    ax2.set_xlabel('Sample count per class (top axis)', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Per-class plot saved to {output_path}")


def plot_delta_ce_curve(
    grid_data: Dict[str, np.ndarray],
    grid_std: Dict[str, np.ndarray],
    output_path: str,
):
    """ΔCE_t 信息增益递减曲线。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    n_points = len(grid_data['delta_CE'])
    t_frac = np.linspace(1.0 / n_points, 1.0, n_points)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ΔCE_t 均值曲线
    ax = axes[0]
    ax.plot(t_frac, grid_data['delta_CE'], color='#4C72B0', linewidth=2)
    ax.fill_between(t_frac,
                    grid_data['delta_CE'] - grid_std['delta_CE'],
                    grid_data['delta_CE'] + grid_std['delta_CE'],
                    alpha=0.2, color='#4C72B0')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('t / B', fontsize=11)
    ax.set_ylabel('ΔCE_t = CE_{t-1} - CE_t', fontsize=11)
    ax.set_title('Per-Step Information Gain vs Budget Fraction', fontsize=12)
    ax.grid(alpha=0.3)

    # 累积信息增益
    ax = axes[1]
    cumsum = np.cumsum(grid_data['delta_CE'])
    cumsum_std = np.sqrt(np.cumsum(grid_std['delta_CE'] ** 2))
    ax.plot(t_frac, cumsum, color='#55A868', linewidth=2)
    ax.fill_between(t_frac,
                    cumsum - cumsum_std,
                    cumsum + cumsum_std,
                    alpha=0.2, color='#55A868')
    ax.set_xlabel('t / B', fontsize=11)
    ax.set_ylabel('Cumulative ΣΔCE_t', fontsize=11)
    ax.set_title('Cumulative Information Gain', fontsize=12)
    ax.grid(alpha=0.3)

    fig.suptitle('Phase 4: Information Gain Decay Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Delta-CE curve saved to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_list: List[int],
    output_path: str,
):
    """混淆矩阵热力图（37×37）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    n_classes = len(class_list)
    if n_classes == 0:
        return

    # 归一化（每行除以其总和）
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # 原始计数
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_title('Confusion Matrix (count)', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 归一化
    ax = axes[1]
    im2 = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_title('Confusion Matrix (row-normalized)', fontsize=12)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # 标注容易混淆的类对（off-diagonal > 0.3）
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm_norm[i, j] > 0.3:
                axes[1].text(j, i, f'{cm_norm[i, j]:.2f}',
                             ha='center', va='center', fontsize=5,
                             color='white' if cm_norm[i, j] > 0.6 else 'black')

    fig.suptitle('Phase 4: Confusion Matrix (37 Classes)', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Information accumulation analysis')
    parser.add_argument('--checkpoint', default='m5/base/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--budget', type=float, default=0.10,
                        help='Budget ratio (default 0.10)')
    parser.add_argument('--max-graphs', type=int, default=500,
                        help='Max graphs from val split (0=all)')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir
    raw_dir = output_dir / 'raw_data'
    fig_dir = output_dir / 'figures'
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 classifier ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 2. 加载 reader ----
    print("\n=== Loading reader ===")
    reader_ckpt = torch.load(str(repo_root / args.checkpoint),
                             map_location='cpu', weights_only=False)
    reader_sd = reader_ckpt['reader_state_dict']
    hidden_dim = reader_sd['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in reader_sd:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}")

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=0.1,
    ).to(device)
    reader_sd, _ = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
    reader.load_state_dict(reader_sd)
    reader.eval()
    print(f"  Params: {sum(p.numel() for p in reader.parameters()):,}")

    # ---- 3. 加载 val 图 ----
    processed_path = str(Path(args.root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    split_path = Path(args.root) / 'ogbg_ppa' / 'split' / 'species' / 'valid.csv.gz'
    val_indices = pd.read_csv(split_path, compression='gzip', header=None).values.flatten().tolist()

    print(f"\n=== Loading {len(val_indices)} val graphs ===")
    graphs, label_dict = dgl.load_graphs(processed_path, idx_list=val_indices)
    print(f"  Loaded {len(graphs)} graphs")

    if args.max_graphs > 0 and args.max_graphs < len(graphs):
        # 分层抽样：每类均匀覆盖
        n_per_class = max(1, args.max_graphs // K)
        class_to_local_idx = defaultdict(list)
        for local_i, ogb_idx in enumerate(val_indices):
            lbl = int(label_dict['labels'][ogb_idx].item())
            class_to_local_idx[lbl].append(local_i)

        sampled_local = []
        for cls in range(K):
            pool = class_to_local_idx.get(cls, [])
            n = min(n_per_class, len(pool))
            if n > 0:
                sampled_local.extend(random.sample(pool, n))

        if len(sampled_local) < args.max_graphs:
            remaining_pool = [i for i in range(len(graphs)) if i not in set(sampled_local)]
            extra_needed = args.max_graphs - len(sampled_local)
            sampled_local.extend(random.sample(remaining_pool, min(extra_needed, len(remaining_pool))))
        else:
            sampled_local = random.sample(sampled_local, args.max_graphs)

        graphs = [graphs[i] for i in sampled_local]
        val_indices = [val_indices[i] for i in sampled_local]
        print(f"  Stratified sampling: {len(graphs)} graphs "
              f"({len(set(int(label_dict['labels'][idx].item()) for idx in val_indices))} classes covered)")

    builder = SubgraphBuilder()

    # ---- 4. 逐图收集逐步数据 ----
    all_steps = []          # per-graph step data lists
    all_grid_data = defaultdict(list)  # interpolated grid data
    all_lock_points = []
    all_budget_abs = []
    all_flip_stats = []
    per_class_results = defaultdict(lambda: {'correct': [], 'total': 0})
    confusion_pairs = []    # (true_label, pred_label) at final step
    all_graph_props = []    # graph-level properties

    for i, ogb_idx in enumerate(tqdm(val_indices, desc='Phase 4 accumulation')):
        g_raw = dgl.add_self_loop(graphs[i])
        label = int(label_dict['labels'][ogb_idx].item())
        n_nodes = g_raw.num_nodes()
        n_edges = g_raw.num_edges()
        density = 2.0 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        indeg = g_raw.in_degrees().float()
        outdeg = g_raw.out_degrees().float()
        total_deg = indeg + outdeg
        avg_degree = float(total_deg.mean().item())
        best_seed = int(total_deg.argmax().item())
        budget_abs = max(1, int(args.budget * n_nodes))

        steps_data = collect_per_step(
            reader, g_raw, best_seed, label, budget_abs,
            device, classifier, builder,
        )

        # graph properties
        all_graph_props.append({
            'graph_idx': ogb_idx,
            'label': label,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
        })

        all_steps.append({
            'graph_idx': ogb_idx,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'label': label,
            'budget_abs': budget_abs,
            'budget_ratio': args.budget,
            'steps': steps_data,
        })

        # 插值到均匀网格
        grid = interpolate_to_grid(steps_data, n_points=20)
        for k, v in grid.items():
            all_grid_data[k].append(v)

        # 锁定点
        lp = compute_lock_point(steps_data)
        all_lock_points.append(lp)
        all_budget_abs.append(budget_abs)

        # 翻转统计
        flips = count_flips(steps_data)
        all_flip_stats.append(flips)

        # per-class & confusion matrix
        if steps_data:
            final = steps_data[-1]
            final_correct = final['pred_correct_t']
            final_pred = final['pred_label_t']
            per_class_results[label]['correct'].append(final_correct)
            per_class_results[label]['total'] += 1
            confusion_pairs.append((label, final_pred))

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 5. 汇总统计 ----
    # 网格均值/std
    grid_mean = {}
    grid_std = {}
    for k, v_list in all_grid_data.items():
        arr = np.array(v_list)
        grid_mean[k] = arr.mean(axis=0)
        grid_std[k] = arr.std(axis=0)

    # 锁定点统计
    lock_ratios = [lp / B for lp, B in zip(all_lock_points, all_budget_abs)
                   if lp > 0 and B > 0]
    never_locked = sum(1 for lp in all_lock_points if lp < 0)
    n_graphs = len(all_steps)

    # 翻转统计
    total_flips = sum(f['n_flips'] for f in all_flip_stats)
    total_to_correct = sum(f['flip_to_correct'] for f in all_flip_stats)
    total_to_wrong = sum(f['flip_to_wrong'] for f in all_flip_stats)

    # per-class 统计
    per_class_stats = {}
    for cls in sorted(per_class_results.keys()):
        correct_list = per_class_results[cls]['correct']
        n = per_class_results[cls]['total']
        per_class_stats[cls] = {
            'accuracy_mean': float(np.mean(correct_list)) if correct_list else 0.0,
            'accuracy_std': float(np.std(correct_list)) if correct_list else 0.0,
            'n': n,
        }

    # 最终准确率
    final_corrects = []
    for gdata in all_steps:
        steps = gdata['steps']
        if steps:
            final_corrects.append(steps[-1]['pred_correct_t'])
    final_acc = float(np.mean(final_corrects)) if final_corrects else 0.0

    # 混淆矩阵
    cm = np.zeros((K, K), dtype=int)
    for true_lbl, pred_lbl in confusion_pairs:
        if 0 <= true_lbl < K and 0 <= pred_lbl < K:
            cm[true_lbl, pred_lbl] += 1

    # ΔCE_t 分阶段统计
    all_delta_ce_per_step = [[] for _ in range(3)]  # early / mid / late
    for gdata in all_steps:
        steps = gdata['steps']
        B = len(steps)
        if B == 0:
            continue
        for i, s in enumerate(steps):
            phase = min(2, int(i / max(1, B / 3)))
            all_delta_ce_per_step[phase].append(s['delta_CE_t'])
    delta_ce_phase_means = [float(np.mean(v)) if v else 0.0 for v in all_delta_ce_per_step]

    # 图属性 vs 分类结果（用于困难类分析）
    graph_props_by_result = {'correct': [], 'wrong': []}
    for gdata in all_steps:
        steps = gdata['steps']
        outcome = 'correct' if (steps and steps[-1]['pred_correct_t'] == 1) else 'wrong'
        graph_props_by_result[outcome].append({
            'n_nodes': gdata['n_nodes'],
            'n_edges': gdata['n_edges'],
            'density': gdata['density'],
            'avg_degree': gdata['avg_degree'],
            'label': gdata['label'],
        })

    # ---- 6. 输出 JSON ----
    output = {
        'summary': {
            'n_graphs': n_graphs,
            'budget_ratio': args.budget,
            'final_accuracy': final_acc,
            'lock_point': {
                'mean_t_over_B': float(np.mean(lock_ratios)) if lock_ratios else 0.0,
                'median_t_over_B': float(np.median(lock_ratios)) if lock_ratios else 0.0,
                'never_locked': never_locked,
                'never_locked_pct': never_locked / n_graphs * 100 if n_graphs > 0 else 0,
            },
            'flip_analysis': {
                'total_flips': total_flips,
                'flip_to_correct': total_to_correct,
                'flip_to_wrong': total_to_wrong,
                'avg_flips_per_graph': total_flips / n_graphs if n_graphs > 0 else 0,
            },
            'delta_ce': {
                'phase_means': {
                    'early': delta_ce_phase_means[0],
                    'mid': delta_ce_phase_means[1],
                    'late': delta_ce_phase_means[2],
                },
            },
            'confusion_matrix': cm.tolist(),
            'grid_curves': {k: v.tolist() for k, v in grid_mean.items()},
            'grid_stds': {k: v.tolist() for k, v in grid_std.items()},
            'per_class': per_class_stats,
            'graph_props_by_result': {
                'correct_count': len(graph_props_by_result['correct']),
                'wrong_count': len(graph_props_by_result['wrong']),
            },
        },
        'per_graph': all_steps,
        'confusion_pairs': [[int(t), int(p)] for t, p in confusion_pairs],
        'config': {
            'checkpoint': str(repo_root / args.checkpoint),
            'classifier': str(repo_root / args.classifier),
            'budget': args.budget,
            'n_graphs': n_graphs,
        },
    }

    json_path = raw_dir / 'phase4_accumulation.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {json_path}")

    # 打印汇总
    print(f"\n=== Phase 4 Summary ===")
    print(f"  Graphs analyzed: {n_graphs}")
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Lock point t*/B: mean={np.mean(lock_ratios):.3f}, median={np.median(lock_ratios):.3f}")
    print(f"  Never locked: {never_locked} ({never_locked / n_graphs * 100:.1f}%)")
    print(f"  Avg flips/graph: {total_flips / n_graphs:.2f}")
    print(f"  Flip→correct: {total_to_correct}, Flip→wrong: {total_to_wrong}")

    # 困难/中等/容易类
    easy = sum(1 for s in per_class_stats.values() if s['accuracy_mean'] >= 0.8)
    mid = sum(1 for s in per_class_stats.values() if 0.4 <= s['accuracy_mean'] < 0.8)
    hard = sum(1 for s in per_class_stats.values() if s['accuracy_mean'] < 0.4)
    print(f"  Per-class: easy={easy}, medium={mid}, hard={hard} (of {len(per_class_stats)} classes)")
    print(f"  ΔCE phase means: early={delta_ce_phase_means[0]:.4f}, "
          f"mid={delta_ce_phase_means[1]:.4f}, late={delta_ce_phase_means[2]:.4f}")

    # 混淆矩阵 top-5 误分类对
    off_diag = []
    for i in range(K):
        for j in range(K):
            if i != j and cm[i, j] > 0:
                off_diag.append((cm[i, j], i, j))
    off_diag.sort(reverse=True)
    print(f"  Top-5 confused class pairs (true→pred):")
    for count, ti, pj in off_diag[:5]:
        print(f"    class {ti} → class {pj}: {count} times")

    # ---- 7. 可视化 ----
    plot_accumulation_curve(grid_mean, grid_std, str(fig_dir / 'phase4_accumulation_curve.png'))
    plot_delta_ce_curve(grid_mean, grid_std, str(fig_dir / 'phase4_delta_ce.png'))
    plot_lock_point_histogram(all_lock_points, all_budget_abs, str(fig_dir / 'phase4_lock_point.png'))
    if confusion_pairs:
        plot_confusion_matrix(cm, sorted(per_class_results.keys()), str(fig_dir / 'phase4_confusion_matrix.png'))
    if per_class_stats:
        plot_per_class(per_class_stats, str(fig_dir / 'phase4_per_class.png'))

    print("\nPhase 4 done.")


if __name__ == '__main__':
    main()

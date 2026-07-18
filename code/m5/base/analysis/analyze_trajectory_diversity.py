"""分层采样 n 张图，统计模型在指定 budget 下采样 x 次的子图相似度和决策熵。

相似度：同一张图 x 条轨迹两两之间的 |intersection| / B
决策熵：每步 -Σ p_i log(p_i)，按 step 对齐，mean ± std 曲线 + 半透明填充

用法:
  python code/m5/base/analysis/analyze_trajectory_diversity.py \
    --checkpoint m5/base/rl/binary/checkpoint_epoch10.pt \
    --budget 0.10 --n-graphs 100 --samples-per-graph 10 \
    --temperature 0.2
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent.parent
sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import torch
import dgl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m5.training_utils import infer_reader_config, patch_old_checkpoint

K = 37


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


# ---------------------------------------------------------------------------
# 分层采样
# ---------------------------------------------------------------------------

def stratified_sample_graphs(all_indices, labels, n_graphs, rng):
    """从 validation set 按类别比例分层采样 n_graphs 张图。

    Returns: list of (local_index, ogb_index, label)
    """
    class_to_local = defaultdict(list)
    for i, ogb_idx in enumerate(all_indices):
        lbl = int(labels[ogb_idx].item())
        class_to_local[lbl].append(i)

    total = len(all_indices)
    sampled = []
    for cls in range(K):
        pool = class_to_local.get(cls, [])
        n = max(1, int(n_graphs * len(pool) / total))
        n = min(n, len(pool))
        if n > 0:
            for local_i in rng.choice(pool, size=n, replace=False):
                sampled.append((int(local_i), int(all_indices[local_i]), cls))

    if len(sampled) > n_graphs:
        sampled = rng.choice(sampled, size=n_graphs, replace=False).tolist()
    elif len(sampled) < n_graphs:
        remaining = [i for i in range(len(all_indices)) if i not in {s[0] for s in sampled}]
        n_extra = min(n_graphs - len(sampled), len(remaining))
        for local_i in rng.choice(remaining, size=n_extra, replace=False):
            sampled.append((int(local_i), int(all_indices[local_i]),
                            int(labels[all_indices[local_i]].item())))

    return [(int(local_i), int(ogb_idx), int(lbl)) for local_i, ogb_idx, lbl in sampled]


# ---------------------------------------------------------------------------
# 单条采样轨迹
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_trajectory(reader, g, seed_node, budget, temperature, device):
    """运行一条采样轨迹，返回 (revealed_nodes_set, entropies_list)。

    entropies[i] = -Σ_j p_j log(p_j)  at step i
    如果 frontier 提前为空，后续 step 熵记为 NaN。
    """
    state = GraphState(g, seed_node=seed_node)
    state._budget_limit = budget
    reader.reset()
    initial_frontier = list(state._initial_frontier)
    if initial_frontier:
        reader.on_new_frontier(initial_frontier)

    revealed = {seed_node}
    entropies = []

    for _step in range(budget):
        if state.is_frontier_empty:
            entropies.append(float('nan'))
            continue

        scores, probs, frontier_nodes, action_idx, log_prob = reader.forward(
            state, sample=True
        )

        n = len(frontier_nodes)
        if n == 0:
            entropies.append(float('nan'))
            continue

        # 计算熵
        p = probs.detach().cpu().numpy()
        p = np.clip(p, 1e-12, 1.0)
        entropy = float(-np.sum(p * np.log(p)))
        entropies.append(entropy)

        chosen = frontier_nodes[action_idx.item()]
        revealed.add(chosen)
        reveal_node(state, chosen)

    return revealed, entropies


# ---------------------------------------------------------------------------
# 相似度计算
# ---------------------------------------------------------------------------

def compute_pairwise_similarity(node_sets):
    """计算 x 个 revealed node sets 的 pairwise 相似度。

    每对 (i, j): |V_i ∩ V_j| / budget（budget = |V_i| 同图固定）。
    Returns: list of similarity scores (length = C(x, 2))
    """
    if len(node_sets) < 2:
        return []

    budgets = [len(s) for s in node_sets]
    B = budgets[0]

    similarities = []
    for i in range(len(node_sets)):
        for j in range(i + 1, len(node_sets)):
            intersection = len(node_sets[i] & node_sets[j])
            similarities.append(intersection / B)

    return similarities


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_similarity(all_similarities, all_graph_info, output_path):
    """相似度分布：左半边散点图（每图一个点=mean） + 右半边直方图。

    all_similarities: list of list of float (n_graphs × C(x,2))
    all_graph_info: list of (ogb_idx, label) per graph
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：每张图 mean similarity 散点
    ax = axes[0]
    per_graph_means = [np.mean(sims) if sims else 0.0 for sims in all_similarities]
    x_pos = np.arange(len(per_graph_means))
    ax.scatter(x_pos, per_graph_means, s=8, c='#4ECDC4', alpha=0.6, edgecolors='none')
    ax.set_ylabel('Mean Similarity = |intersection| / B')
    ax.set_title('Per-Graph Trajectory Similarity (mean per graph)')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis='y', alpha=0.3)
    # 隐藏 x 轴刻度（图太多时无意义）
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

    # 右：总体直方图
    ax = axes[1]
    all_flat = [s for sims in all_similarities for s in sims]
    ax.hist(all_flat, bins=30, color='#4ECDC4', edgecolor='white', alpha=0.8)
    mean_sim = np.mean(all_flat) if all_flat else 0
    ax.axvline(mean_sim, color='#FF6B6B', linestyle='--', linewidth=2,
               label=f'Mean = {mean_sim:.3f}')
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Count')
    ax.set_title(f'Overall Pairwise Similarity Distribution\n(n={len(all_flat)} pairs)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Similarity plot saved: {output_path}")


def plot_per_class_similarity(all_similarities, all_graph_info, output_path):
    """按类别统计平均相似度和方差，柱状图 + error bar。

    all_similarities: list of list of float (n_graphs × C(x,2))
    all_graph_info: list of (ogb_idx, label) per graph
    """
    from collections import defaultdict

    # 按 label 汇总所有 pairwise similarity
    class_sims = defaultdict(list)
    for sims, (ogb_idx, lbl) in zip(all_similarities, all_graph_info):
        class_sims[lbl].extend(sims)

    # 计算每类 mean ± std
    class_labels = sorted(class_sims.keys(), key=int)
    means = [np.mean(class_sims[lbl]) for lbl in class_labels]
    stds = [np.std(class_sims[lbl]) for lbl in class_labels]
    counts = [len(class_sims[lbl]) for lbl in class_labels]

    # 排序：按 mean 降序
    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(class_labels))

    bars = ax.bar(x_pos, [means[i] for i in order], yerr=[stds[i] for i in order],
                  capsize=3, color='#4ECDC4', edgecolor='white', alpha=0.85,
                  error_kw={'linewidth': 1, 'alpha': 0.6})

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"c{class_labels[i]}\n(n={counts[i]})" for i in order],
                       fontsize=7, rotation=90)
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Per-Class Trajectory Similarity (mean ± std)')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # 总体均值参考线
    all_flat = [s for sims in all_similarities for s in sims]
    overall_mean = np.mean(all_flat)
    ax.axhline(overall_mean, color='#FF6B6B', linestyle='--', linewidth=1.5,
               label=f'Overall mean = {overall_mean:.3f}')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Per-class similarity plot saved: {output_path}")


def plot_entropy(all_entropies, output_path):
    """决策熵 mean ± std 曲线。

    all_entropies: list of list of float (n_trajectories × steps), 可能含 NaN

    按有效 step 计算 mean/std，NaN 不参与统计。
    """
    if not all_entropies:
        print("  WARNING: no entropy data to plot")
        return

    max_steps = max(len(e) for e in all_entropies)
    n_trajs = len(all_entropies)

    # 构建 (n_trajs, max_steps) 矩阵，NaN 留空
    matrix = np.full((n_trajs, max_steps), np.nan)
    for i, ent in enumerate(all_entropies):
        matrix[i, :len(ent)] = ent

    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    counts = np.sum(~np.isnan(matrix), axis=0)  # 每步有效样本数
    steps = np.arange(max_steps)

    # 标准化 x 轴为 budget 比例
    x_pct = steps / max(max_steps - 1, 1) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # ±std 半透明填充
    upper = means + stds
    lower = np.maximum(means - stds, 0)  # entropy >= 0
    ax.fill_between(x_pct, lower, upper, alpha=0.2, color='#4ECDC4',
                    label=f'±1 std (n={n_trajs} trajectories)')

    # mean 曲线
    ax.plot(x_pct, means, color='#4ECDC4', linewidth=2, label='Mean entropy')

    # mean±std 虚线
    ax.plot(x_pct, upper, color='#4ECDC4', linestyle='--', linewidth=1, alpha=0.6)
    ax.plot(x_pct, lower, color='#4ECDC4', linestyle='--', linewidth=1, alpha=0.6)

    # 每步有效样本数标注（底部小字）
    for i, (s, c) in enumerate(zip(steps, counts)):
        if i % max(1, max_steps // 10) == 0 and c < n_trajs:
            ax.annotate(f'n={int(c)}', (x_pct[i], 0.02), fontsize=6,
                        ha='center', color='gray', alpha=0.7)

    ax.set_xlabel('Step (% of budget)')
    ax.set_ylabel('Decision Entropy (nats)')
    ax.set_title(f'Decision Entropy over Trajectory Steps\n'
                 f'(mean ± std, {n_trajs} trajectories)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # y 轴从 0 开始
    ax.set_ylim(bottom=-0.05)

    # 理论最大熵标注
    ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Entropy plot saved: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze trajectory diversity & decision entropy'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Reader model checkpoint path (relative to repo root)')
    parser.add_argument('--budget', type=float, default=0.10,
                        help='Budget ratio (default: 0.10)')
    parser.add_argument('--n-graphs', type=int, default=100,
                        help='Number of graphs to sample (default: 100)')
    parser.add_argument('--samples-per-graph', type=int, default=10,
                        help='Trajectories per graph (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Sampling temperature (default: 0.2)')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset',
                        help='OGB dataset root')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='train',
                        help='Data split to analyze (default: train)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='m4/analysis/output')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读 split + 标签（不加载图数据） ----
    split_map = {
        'train': 'train.csv.gz',
        'valid': 'valid.csv.gz',
        'test': 'test.csv.gz',
    }
    split_file = split_map[args.split]
    root_path = Path(args.root)

    split_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / split_file
    all_indices = pd.read_csv(split_csv, compression='gzip', header=None
                              ).values.flatten().tolist()
    print(f"\n=== Split '{args.split}': {len(all_indices)} graph indices ===")

    # 读全量标签用于分层采样（只读标签列，不加载图数据）
    label_csv = root_path / 'ogbg_ppa' / 'raw' / 'graph-label.csv.gz'
    label_arr = pd.read_csv(label_csv, compression='gzip', header=None
                            ).values.flatten()  # (158100,) int64
    print(f"  Labels loaded: {len(label_arr)}")

    # ---- 2. 分层采样 ----
    sampled = stratified_sample_graphs(all_indices, label_arr,
                                       args.n_graphs, rng)
    print(f"\n=== Stratified sampled {len(sampled)} graphs ===")
    class_counts = defaultdict(int)
    for _, _, lbl in sampled:
        class_counts[lbl] += 1
    print(f"  Classes covered: {len(class_counts)}/{K}")

    # ---- 3. 只加载采样到的图 ----
    sampled_indices = [ogb_idx for _, ogb_idx, _ in sampled]
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    all_graphs, all_labels = dgl.load_graphs(processed_path, idx_list=sampled_indices)
    print(f"  Loaded {len(all_graphs)} graphs (sampled subset only)")

    # ---- 4. 加载 Reader ----
    print("\n=== Loading LearnableReader ===")
    ckpt_path = repo_root / args.checkpoint
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    reader_sd = ckpt.get('reader_state_dict', ckpt)
    reader_sd, was_patched = patch_old_checkpoint(reader_sd)
    hidden_dim, n_layers = infer_reader_config(reader_sd)

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=0.1, stochastic=True, temperature=args.temperature,
    ).to(device)
    reader.load_state_dict(reader_sd)
    reader.eval()
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  Temperature: {args.temperature}")

    # ---- 5. 逐图采样轨迹 ----
    print(f"\n=== Sampling trajectories (budget={args.budget}) ===")
    all_similarities = []   # list of list of float, per-graph pairwise similarities
    all_graph_info = []     # list of (ogb_idx, label)
    all_entropies = []      # list of list of float, per-trajectory entropies

    for i, (_local_i, ogb_idx, label) in enumerate(tqdm(sampled, desc="  Graphs")):
        g_raw = all_graphs[i]
        g = dgl.add_self_loop(g_raw).to(device)
        n_nodes = g.num_nodes()
        B = max(1, int(args.budget * n_nodes))

        revealed_sets = []
        graph_entropies = []

        for seed_i in range(args.samples_per_graph):
            seed_node = deterministic_seed(ogb_idx, seed_i, n_nodes)

            # 重新设置 temperature 确保 stochastic 采样
            reader.temperature = args.temperature
            reader.stochastic = True

            revealed, entropies = run_trajectory(
                reader, g, seed_node, B, args.temperature, device
            )
            revealed_sets.append(revealed)
            graph_entropies.append(entropies)
            all_entropies.append(entropies)

        sims = compute_pairwise_similarity(revealed_sets)
        all_similarities.append(sims)
        all_graph_info.append((ogb_idx, label))

    # ---- 6. 汇总统计 ----
    all_flat_sims = [s for sims in all_similarities for s in sims]
    n_pairs = len(all_flat_sims)
    mean_sim = np.mean(all_flat_sims) if all_flat_sims else 0.0
    std_sim = np.std(all_flat_sims) if all_flat_sims else 0.0

    # 熵统计
    ent_lens = [len(e) for e in all_entropies]
    max_ent_len = max(ent_lens) if ent_lens else 0

    print(f"\n=== Results ===")
    print(f"  Graphs analyzed: {len(sampled)}")
    print(f"  Trajectories total: {len(all_entropies)}")
    print(f"  Pairwise similarity pairs: {n_pairs}")
    print(f"  Mean similarity: {mean_sim:.4f} ± {std_sim:.4f}")
    print(f"  Entropy: max steps = {max_ent_len}")

    # 熵的汇总统计（平均第一步熵、最后一步熵）
    first_entropies = [e[0] for e in all_entropies if len(e) > 0 and not math.isnan(e[0])]
    last_entropies = []
    for e in all_entropies:
        for val in reversed(e):
            if not math.isnan(val):
                last_entropies.append(val)
                break

    if first_entropies:
        print(f"  First-step entropy: {np.mean(first_entropies):.4f} ± {np.std(first_entropies):.4f}")
    if last_entropies:
        print(f"  Last-step entropy:  {np.mean(last_entropies):.4f} ± {np.std(last_entropies):.4f}")

    # ---- 7. 可视化 ----
    budget_tag = f"b{int(args.budget * 1000):03d}"  # e.g. 0.10 → b100, 0.05 → b050

    print(f"\n=== Saving plots (budget={args.budget}, tag={budget_tag}) ===")
    plot_similarity(all_similarities, all_graph_info,
                    str(output_dir / f'similarity_distribution_{budget_tag}.png'))
    plot_per_class_similarity(all_similarities, all_graph_info,
                              str(output_dir / f'similarity_per_class_{budget_tag}.png'))
    plot_entropy(all_entropies, str(output_dir / f'entropy_curve_{budget_tag}.png'))

    # ---- 8. 保存原始数据 ----
    results = {
        'config': {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'budget': args.budget,
            'n_graphs': args.n_graphs,
            'samples_per_graph': args.samples_per_graph,
            'temperature': args.temperature,
        },
        'summary': {
            'mean_similarity': float(mean_sim),
            'std_similarity': float(std_sim),
            'n_pairs': n_pairs,
            'n_trajectories': len(all_entropies),
            'max_entropy_steps': max_ent_len,
        },
        'per_graph': [
            {
                'ogb_idx': ogb_idx,
                'label': lbl,
                'similarities': [float(s) for s in sims],
                'entropies': [
                    [float(v) if not math.isnan(v) else None for v in ent]
                    for ent in graph_entropies
                ],
            }
            for (ogb_idx, lbl), sims, graph_entropies in zip(
                all_graph_info, all_similarities,
                # Group entropies back per graph
                [all_entropies[i*args.samples_per_graph:(i+1)*args.samples_per_graph]
                 for i in range(len(sampled))]
            )
        ],
    }

    results_path = output_dir / f'analysis_results_{budget_tag}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {results_path}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()

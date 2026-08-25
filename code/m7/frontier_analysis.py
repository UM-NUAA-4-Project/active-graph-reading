"""M7 前沿规模分析：随机游走测量每步 frontier 规模，估计 BC 效果基线。

原理（恢复自被删的 belief/frontier_analysis.py + figures/compute_bc_baseline.py,
f62e00d，用途是估计 BC 学习效果的可达上限）：
  BC 训练每步让 reader 在 frontier 中选 teacher_node，损失 ≈ -log P(teacher_node)。
  若策略近似在 frontier 上均匀分布，则每步 NLL ≈ ln(mean|frontier|)——随机游走
  装出 frontier 规模后即得"随机 BC 基线"：平均损失低于它才有意义，否则劣于随机猜测。

与原版差异（通用化，PPA 特有逻辑已去除）：
  - 数据走 m7 registry（--dataset/--root 任意切换，origin 断言 PPA 37 类/CSV 等）
  - 图默认已由加载器加自环，不再手动 add_self_loop
  - graph_idx 语义 = ds.train_graphs 本地位置（同 train_bc.py / train_rl.py）
  - 只加载 train split（同 m7 训练脚本）
  - BC 基线计算并入本脚本（原版依赖独立 compute_bc_baseline.py + JSON）

用法:
  python code/m7/frontier_analysis.py --dataset cifar10 --root D:\\superpixels \\
      --graphs-per-class 10 --seeds 10 --budget-ratio 0.10
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
sys.path.insert(0, str(_repo_root))
# 移除 code/m7 自身：dgl→torchdata 的 `import datasets` 不得解析到 code/m7/datasets
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import numpy as np
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m6.training_utils import deterministic_seed
from m7.datasets.registry import get_dataset


def run_random_walk(state: GraphState, budget: int) -> list:
    """随机游走一条轨迹，返回每步 reveal 前的 frontier 规模。

    sizes[0] = 初始 frontier（尚未 reveal）；sizes[t] = t 次 reveal 后。
    """
    frontier_sizes = []
    rng = random.Random()

    for _ in range(budget):
        frontier = state.get_frontier_nodes()
        if not frontier:
            break
        frontier_sizes.append(len(frontier))
        reveal_node(state, rng.choice(frontier))

    return frontier_sizes


def compute_bc_baseline(all_sizes: list) -> list:
    """随机 BC 基线表：每步 (n_traj, mean|frontier|, ln(mean), 均匀匹配概率)。

    ln(mean) 即该步均匀策略下匹配 teacher_node 的 NLL；
    基线 l = 各步 ln(mean) 的均值（平均每步损失必须低于它）。
    """
    max_len = max(len(s) for s in all_sizes)
    rows = []
    for t in range(max_len):
        vals = [s[t] for s in all_sizes if t < len(s)]
        m = float(np.mean(vals))
        rows.append({
            'step': t + 1,
            'n_traj': len(vals),
            'mean_frontier': m,
            'ln_mean': float(np.log(m)),
            'uniform_match_prob': 1.0 / m if m > 0 else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='M7 前沿规模分析（随机游走 → BC 基线估计）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--graphs-per-class', type=int, default=10)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--output-dir', default='m7/data/frontier_analysis',
                        help='输出目录（相对 code/ 或绝对路径）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = get_dataset(args.dataset, root=args.root or None, splits=['train'])
    spec = ds.spec
    train_graphs, train_labels = ds.train_graphs, ds.train_labels
    print(f"Dataset: {spec.name} | n_classes={spec.n_classes} | "
          f"train={len(train_graphs)} | node_feat_dim={spec.node_feat_dim}")

    # ---- 1. 按类分组并采样 ----
    by_label = defaultdict(list)
    for i in range(len(train_graphs)):
        by_label[int(train_labels[i])].append(i)

    sampled = []
    for label in range(spec.n_classes):
        pool = by_label.get(label, [])
        n_sample = min(args.graphs_per_class, len(pool))
        for pos in random.sample(pool, n_sample):
            sampled.append((label, pos, train_graphs[pos]))
        print(f"  class {label:2d}: {len(pool)} graphs → sampled {n_sample}")
    print(f"\nTotal sampled: {len(sampled)} graph-images")

    # ---- 2. 随机游走 rollout ----
    print(f"\n=== Random walks "
          f"(budget={args.budget_ratio:.0%}, seeds={args.seeds}) ===")
    results = {}
    for label, pos, g in tqdm(sampled, desc="  Random walks"):
        n_nodes = g.num_nodes()
        B = max(1, int(args.budget_ratio * n_nodes))
        for seed_i in range(args.seeds):
            seed_node = deterministic_seed(pos, seed_i, n_nodes)
            state = GraphState(g, seed_node=seed_node)
            state._budget_limit = B
            frontier_sizes = run_random_walk(state, B)
            key = f"{pos}_s{seed_i}"
            results[key] = {
                'graph_idx': pos,
                'label': label,
                'n_nodes': n_nodes,
                'budget': B,
                'seed_i': seed_i,
                'seed_node': seed_node,
                'frontier_sizes': frontier_sizes,
                'n_steps': len(frontier_sizes),
            }
    print(f"Total trajectories: {len(results)}")

    # ---- 3. 保存 JSON + 基线 ----
    output_dir = _repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'frontier_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    all_sizes = [r['frontier_sizes'] for r in results.values()]
    rows = compute_bc_baseline(all_sizes)
    baseline_l = float(np.mean([r['ln_mean'] for r in rows]))

    print(f"\n{'Step':>5s} {'N_traj':>7s} {'Mean':>8s} {'ln(mean)':>9s} "
          f"{'P_uniform':>9s}")
    print('-' * 46)
    for r in rows:
        print(f"{r['step']:5d} {r['n_traj']:7d} {r['mean_frontier']:8.1f} "
              f"{r['ln_mean']:9.4f} {r['uniform_match_prob']:9.4f}")
    print('=' * 46)
    print(f"Random BC baseline ≈ {baseline_l:.4f}  "
          f"(平均每步约 {np.exp(baseline_l):.1f} 个 frontier 节点的均匀匹配 NLL)")
    print("任何 BC 平均损失高于此基线 = 劣于随机猜测")

    txt_path = output_dir / 'bc_baseline.txt'
    lines = [f"dataset={spec.name} budget_ratio={args.budget_ratio:.2f} "
             f"graphs_per_class={args.graphs_per_class} seeds={args.seeds}",
             f"l = {baseline_l:.4f}",
             "Random BC baseline ≈ %.2f 等价于每步约 %.1f 个前沿节点"
             % (baseline_l, np.exp(baseline_l))]
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Saved: {txt_path}")

    # ---- 4. 可视化（两联图：散点 + 均值±std）----
    print("=== Generating visualization ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    means = np.array([r['mean_frontier'] for r in rows])
    stds = np.array([
        np.std([s[t] for s in all_sizes if t < len(s)])
        for t in range(len(rows))
    ])
    steps = np.arange(1, len(rows) + 1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))
    n_plot = min(200, len(all_sizes))
    for sizes in all_sizes[:n_plot]:
        ax_left.scatter(range(1, len(sizes) + 1), sizes,
                        s=3, alpha=0.3, color='steelblue', edgecolors='none')
    ax_left.set_xlabel('Step')
    ax_left.set_ylabel('Frontier node count')
    ax_left.set_title(f'Random Walk Frontier Sizes\n'
                      f'({n_plot} shown, {len(all_sizes)} total)')
    ax_left.grid(alpha=0.3)

    ax_right.plot(steps, means, color='steelblue', linewidth=1.5, label='Mean')
    ax_right.fill_between(steps, means - stds, means + stds,
                          color='steelblue', alpha=0.15, label='±1 std')
    counts = [r['n_traj'] for r in rows]
    for t_idx in {0, len(rows) // 2, len(rows) - 1}:
        ax_right.annotate(f"n={counts[t_idx]}", (steps[t_idx], means[t_idx]),
                          textcoords='offset points', xytext=(0, -15),
                          fontsize=8, ha='center', color='gray')
    ax_right.set_xlabel('Step')
    ax_right.set_ylabel('Frontier node count')
    ax_right.set_title(f'Mean ± Std Frontier Size\n'
                       f'({spec.name}, budget={args.budget_ratio:.0%}, '
                       f'n_trajs={len(all_sizes)})')
    ax_right.legend(fontsize=9)
    ax_right.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = output_dir / 'frontier_analysis.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == '__main__':
    main()

"""Analyze frontier node counts during random walk on PPA graphs.

Samples 10 graphs per class from the training set, runs random-walk rollouts
at 10% budget with 10 seeds each, and records frontier sizes per step.

用法:
  python belief/frontier_analysis.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import K, _project_root, setup_paths
setup_paths()

import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from training_utils import deterministic_seed


def run_random_walk(state: GraphState, budget: int) -> list:
    """Run random walk on one state, return frontier sizes at each step.

    Step 0 = initial frontier (before any reveal). Step t = after t reveals.
    """
    frontier_sizes = []
    rng = random.Random()

    for step in range(budget):
        frontier = state.get_frontier_nodes()
        if not frontier:
            break
        frontier_sizes.append(len(frontier))

        # Random choice from frontier
        v = rng.choice(frontier)
        reveal_node(state, v)

    return frontier_sizes


def main():
    parser = argparse.ArgumentParser(
        description='Analyze frontier node counts during random walk')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--graphs-per-class', type=int, default=10)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'

    # ---- 1. load graphs grouped by label ----
    print("=== Loading training graphs ===")
    train_indices = pd.read_csv(train_csv, compression='gzip', header=None
                                ).values.flatten().tolist()
    train_graphs_list, train_labels = dgl.load_graphs(
        processed_path, idx_list=train_indices)
    labels_tensor = train_labels['labels']
    # Handle both 1D (N,) and 2D (N,1) label tensors
    if labels_tensor.dim() == 2:
        labels_tensor = labels_tensor.squeeze(-1)

    by_label = defaultdict(list)
    for i, g in enumerate(train_graphs_list):
        label = int(labels_tensor[i].item())
        by_label[label].append((train_indices[i], g))

    # ---- 2. sample graphs per class ----
    sampled = []
    for label in range(K):
        pool = by_label[label]
        n_sample = min(args.graphs_per_class, len(pool))
        chosen = random.sample(pool, n_sample)
        sampled.extend([(label, ogb_idx, g) for ogb_idx, g in chosen])
        print(f"  class {label:2d}: {len(pool)} graphs → sampled {n_sample}")

    print(f"\nTotal sampled: {len(sampled)} graphs")

    # ---- 3. run random walks ----
    print(f"\n=== Running random walks "
          f"(budget={args.budget_ratio:.0%}, seeds={args.seeds}) ===")
    results = {}  # {index: [t1_count, t2_count, ...]}

    for label, ogb_idx, g_raw in tqdm(sampled, desc="  Random walks"):
        g = dgl.add_self_loop(g_raw)
        n_nodes = g.num_nodes()
        B = max(1, int(args.budget_ratio * n_nodes))

        for seed_i in range(args.seeds):
            seed_node = deterministic_seed(ogb_idx, seed_i, n_nodes)
            state = GraphState(g, seed_node=seed_node)
            state._budget_limit = B
            frontier_sizes = run_random_walk(state, B)
            # Trajectory key: ogb_idx + seed offset
            traj_key = f"{ogb_idx}_s{seed_i}"
            results[traj_key] = {
                "ogb_index": ogb_idx,
                "label": label,
                "n_nodes": n_nodes,
                "budget": B,
                "seed_i": seed_i,
                "seed_node": seed_node,
                "frontier_sizes": frontier_sizes,
                "n_steps": len(frontier_sizes),
            }

    # ---- 4. save JSON ----
    output_dir = _project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frontier_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {json_path}")
    print(f"  Total trajectories: {len(results)}")

    # ---- 5. stats ----
    all_sizes = [r['frontier_sizes'] for r in results.values()]
    max_len = max(len(s) for s in all_sizes)
    print(f"  Max steps: {max_len}")

    # Per-step stats with correct denominators
    step_means, step_stds = [], []
    for t in range(max_len):
        vals = [s[t] for s in all_sizes if t < len(s)]
        step_means.append(np.mean(vals))
        step_stds.append(np.std(vals))

    # ---- 6. visualize ----
    print("=== Generating visualization ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={'width_ratios': [1, 1]},
    )

    # --- Left: scatter ---
    # Plot first N trajectories for readability
    n_plot = min(200, len(all_sizes))
    for i, sizes in enumerate(all_sizes[:n_plot]):
        steps = list(range(1, len(sizes) + 1))
        ax_left.scatter(steps, sizes, s=3, alpha=0.3, color='steelblue',
                        edgecolors='none')

    ax_left.set_xlabel('Step', fontsize=12)
    ax_left.set_ylabel('Frontier node count', fontsize=12)
    ax_left.set_title(
        f'Random Walk Frontier Sizes\n'
        f'({n_plot} trajectories shown, {len(all_sizes)} total)',
        fontsize=12)
    ax_left.grid(alpha=0.3)

    # --- Right: mean ± std curve ---
    steps = np.arange(1, max_len + 1)
    means = np.array(step_means)
    stds = np.array(step_stds)

    ax_right.plot(steps, means, color='steelblue', linewidth=1.5, label='Mean')
    ax_right.fill_between(steps, means - stds, means + stds,
                          color='steelblue', alpha=0.15, label='±1 std')

    # Add trajectory count annotation at a few points
    counts = [sum(1 for s in all_sizes if t < len(s)) for t in range(max_len)]
    for t_idx in [0, max_len // 2, max_len - 1]:
        ax_right.annotate(
            f'n={counts[t_idx]}',
            (steps[t_idx], means[t_idx]),
            textcoords="offset points", xytext=(0, -15),
            fontsize=8, ha='center', color='gray',
        )

    ax_right.set_xlabel('Step', fontsize=12)
    ax_right.set_ylabel('Frontier node count', fontsize=12)
    ax_right.set_title(
        f'Mean ± Std Frontier Size (n_trajs={len(all_sizes)})',
        fontsize=12)
    ax_right.legend(fontsize=9)
    ax_right.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = output_dir / "frontier_analysis.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Print summary stats
    print(f"\n=== Summary ===")
    print(f"  Step   N_traj   Mean     Std      Min    Max")
    print(f"  {'─' * 50}")
    for t in range(max_len - 1):
        if t >= max_len:
            continue
        vals = [s[t] for s in all_sizes if t < len(s)]
        print(f"  {t + 1:4d}   {len(vals):6d}   {np.mean(vals):6.1f}   "
              f"{np.std(vals):6.1f}   {np.min(vals):5d}   {np.max(vals):5d}")


if __name__ == '__main__':
    main()

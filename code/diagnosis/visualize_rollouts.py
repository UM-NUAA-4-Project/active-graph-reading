"""可视化 Reader 采样结果：每类 1 图 × 4 次采样，节点网格 + 相似矩阵。

用法:
  python visualize_rollouts.py --checkpoint code/m5/base/rl/binary/ex1/checkpoint_epoch4.pt
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

# 标准引导：把 <repo>/code 插到 sys.path，移除脚本自身目录（防包名遮蔽）
_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [s for s in sys.path
            if Path(s).resolve() != Path(__file__).resolve().parent]

from tools.paths import CODE_DIR as _repo_root  # noqa: E402
from tools.seeds import deterministic_seed  # noqa: E402  （PPA 专用工具；图加载沿用 OGB 直读）

import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

from active_graph_reader.reader.learnable_reader import LearnableReader  # noqa: E402
from classifier import GraphClassifier  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import reveal_node  # noqa: E402

K = 37
BUDGET_RATIO = 0.15
N_SEEDS = 4
GRID_COLS = 20
GRID_ROWS = 15  # 15×20 = 300 = max PPA nodes

ROLLOUT_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']  # red, blue, teal, orange
UNREVEALED_COLOR = '#E0E0E0'
SEED_COLOR = '#FFD700'  # gold for seed node


@torch.no_grad()
def rollout(reader, g, graph_idx, seed_i, budget_ratio, device):
    """Run a single deterministic rollout, return set of revealed nodes (excluding seed)."""
    n_nodes = g.num_nodes()
    B = max(1, int(budget_ratio * n_nodes))
    seed_node = deterministic_seed(graph_idx, seed_i, n_nodes)
    state = GraphState(g, seed_node=seed_node)
    state._budget_limit = B

    for _step in range(B):
        scores, probs, frontier_nodes = reader.forward(state)
        if not frontier_nodes:
            break
        chosen = frontier_nodes[probs.argmax().item()]
        reveal_node(state, chosen)

    return set(state.V_revealed), seed_node


def jaccard_similarity(sets):
    """Compute 4×4 Jaccard similarity matrix."""
    n = len(sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(sets[i]) == 0 and len(sets[j]) == 0:
                mat[i, j] = 1.0
            else:
                mat[i, j] = len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
    return mat


def draw_node_grid(ax, n_nodes, revealed_sets, seed_nodes):
    """Draw the 15×20 node grid with 4-color rollouts."""
    # Build per-node color info: which rollouts revealed it
    for idx in range(n_nodes):
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        # Determine which rollouts cover this node
        coverage = [idx in rs for rs in revealed_sets]
        is_seed = idx in seed_nodes
        revealed_by = [i for i, c in enumerate(coverage) if c]

        x = col
        y = GRID_ROWS - 1 - row  # flip y so idx 0 is top-left

        if not revealed_by:
            # Unrevealed — gray
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=UNREVEALED_COLOR,
                                   edgecolor='#CCCCCC', linewidth=0.3))
            if is_seed:
                # Seed but unrevealed (should not happen normally)
                ax.add_patch(Rectangle((x + 0.15, y + 0.15), 0.7, 0.7,
                                       facecolor=SEED_COLOR, edgecolor='none'))
        elif len(revealed_by) == 1:
            # Single rollout — solid color
            color = ROLLOUT_COLORS[revealed_by[0]]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='#888888', linewidth=0.3))
            if is_seed:
                ax.add_patch(Rectangle((x + 0.2, y + 0.2), 0.6, 0.6,
                                       facecolor=SEED_COLOR, edgecolor='none'))
        elif len(revealed_by) == 2:
            # Two rollouts — split vertically
            for half, ri in enumerate(revealed_by):
                ax.add_patch(Rectangle((x + half * 0.5, y), 0.5, 1,
                                       facecolor=ROLLOUT_COLORS[ri],
                                       edgecolor='#888888', linewidth=0.2))
            if is_seed:
                ax.add_patch(Rectangle((x + 0.3, y + 0.3), 0.4, 0.4,
                                       facecolor=SEED_COLOR, edgecolor='none', zorder=5))
        elif len(revealed_by) == 3:
            # Three rollouts — horizontal thirds
            for tier, ri in enumerate(revealed_by):
                ax.add_patch(Rectangle((x, y + tier * 0.333), 1, 0.333,
                                       facecolor=ROLLOUT_COLORS[ri],
                                       edgecolor='#888888', linewidth=0.2))
            if is_seed:
                ax.add_patch(Rectangle((x + 0.3, y + 0.3), 0.4, 0.4,
                                       facecolor=SEED_COLOR, edgecolor='none', zorder=5))
        else:
            # Four rollouts — 2×2 split
            for qi, ri in enumerate(revealed_by):
                qr, qc = qi // 2, qi % 2
                ax.add_patch(Rectangle((x + qc * 0.5, y + (1 - qr) * 0.5), 0.5, 0.5,
                                       facecolor=ROLLOUT_COLORS[ri],
                                       edgecolor='#888888', linewidth=0.2))
            if is_seed:
                ax.add_patch(Rectangle((x + 0.3, y + 0.3), 0.4, 0.4,
                                       facecolor=SEED_COLOR, edgecolor='none', zorder=5))

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def draw_similarity_matrix(ax, sim_mat, accuracies):
    """Draw 4×4 similarity matrix with accuracy on diagonal."""
    im = ax.imshow(sim_mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')

    for i in range(4):
        for j in range(4):
            text = f'{sim_mat[i, j]:.2f}'
            color = 'white' if sim_mat[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    # Label diagonal with accuracy
    for i in range(4):
        ax.text(i, i - 0.35, f'acc={accuracies[i]:.2f}' if accuracies[i] >= 0 else '',
                ha='center', va='top', fontsize=7, color='#333333')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=8)
    ax.set_yticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=8)
    ax.set_title('Jaccard Similarity', fontsize=10)


def build_figure(graph_idx, label, n_nodes, revealed_sets, seed_nodes,
                 accuracies, sim_mat, class_name=""):
    """Build one figure: node grid (left) + similarity matrix (right)."""
    fig = plt.figure(figsize=(16, 7))

    # Left: node grid
    ax_grid = fig.add_axes([0.05, 0.08, 0.58, 0.87])
    draw_node_grid(ax_grid, n_nodes, revealed_sets, seed_nodes)

    # Budget line indicator
    budget_line_y = GRID_ROWS - 1 - ((max(len(rs) for rs in revealed_sets) - 1) // GRID_COLS)
    ax_grid.axhline(y=budget_line_y + 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.3)

    title = f'Graph {graph_idx}  |V|={n_nodes}  Class {label}'
    if class_name:
        title += f' ({class_name})'
    ax_grid.set_title(title, fontsize=12, fontweight='bold')

    # Legend
    legend_y = -0.8
    for i, (color, label_text) in enumerate(zip(ROLLOUT_COLORS, ['Seed1', 'Seed2', 'Seed3', 'Seed4'])):
        ax_grid.add_patch(Rectangle((i * 5.5, legend_y), 0.8, 0.5, facecolor=color,
                                    edgecolor='#888888', linewidth=0.3, transform=ax_grid.transData))
        ax_grid.text(i * 5.5 + 1.0, legend_y + 0.25, label_text, fontsize=8, va='center')

    ax_grid.add_patch(Rectangle((22, legend_y), 0.8, 0.5, facecolor=UNREVEALED_COLOR,
                                edgecolor='#CCCCCC', linewidth=0.3, transform=ax_grid.transData))
    ax_grid.text(23, legend_y + 0.25, 'Unrevealed', fontsize=8, va='center')

    # Right: similarity matrix
    ax_sim = fig.add_axes([0.68, 0.25, 0.28, 0.55])
    draw_similarity_matrix(ax_sim, sim_mat, accuracies)

    # Stats text
    stats_text = (
        f"Mean Jaccard: {sim_mat[~np.eye(4, dtype=bool)].mean():.3f}\n"
        f"Nodes covered by ≥1 rollout: {len(set.union(*revealed_sets))} / {n_nodes}\n"
        f"Nodes covered by all 4: {len(set.intersection(*revealed_sets))}"
    )
    fig.text(0.68, 0.12, stats_text, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize reader rollouts')
    parser.add_argument('--checkpoint', default='m5/base/rl/binary/ex1/checkpoint_epoch4.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='visualization_output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = _script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load training graphs ----
    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'

    train_indices = pd.read_csv(train_csv, compression='gzip', header=None).values.flatten().tolist()
    print(f"Training set: {len(train_indices)} graphs")

    # Read labels for all training graphs
    labels_csv = root_path / 'ogbg_ppa' / 'raw' / 'graph-label.csv.gz'
    all_labels = pd.read_csv(labels_csv, compression='gzip', header=None).values.flatten()

    # Group by class
    class_to_indices = defaultdict(list)
    for idx in train_indices:
        label = int(all_labels[idx])
        class_to_indices[label].append(idx)

    print(f"Classes in training set: {len(class_to_indices)}")

    # Sample 1 graph per class
    sampled = {}
    for label, indices in sorted(class_to_indices.items()):
        sampled[label] = random.choice(indices)

    all_sampled = sorted(sampled.values())
    print(f"Sampled {len(all_sampled)} graphs (1 per class)")

    # Load all sampled graphs
    print("\n=== Loading graphs ===")
    graphs, label_dict = dgl.load_graphs(processed_path, idx_list=all_sampled)
    graphs = [dgl.add_self_loop(g) for g in graphs]
    print(f"  Loaded {len(graphs)} graphs")

    # ---- 2. Load classifier ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 3. Load reader ----
    print("\n=== Loading reader ===")
    checkpoint_path = repo_root / args.checkpoint
    reader_ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    reader_sd = reader_ckpt['reader_state_dict']

    hidden_dim = reader_sd['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in reader_sd:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=0.1,
    ).to(device)
    reader.load_state_dict(reader_sd)
    reader.eval()
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, epoch={reader_ckpt.get('epoch', 'N/A')}")

    builder = SubgraphBuilder()

    # ---- 4. Run rollouts and generate visualizations ----
    print(f"\n=== Running rollouts & generating visualizations ===")

    for i, ogb_idx in enumerate(tqdm(all_sampled, desc="  Processing")):
        g = graphs[i]
        label = int(label_dict['labels'][ogb_idx].item())
        n_nodes = g.num_nodes()

        # 4 deterministic rollouts
        revealed_sets = []
        seed_nodes_list = []
        for seed_i in range(N_SEEDS):
            rs, sn = rollout(reader, g, ogb_idx, seed_i, BUDGET_RATIO, device)
            revealed_sets.append(rs)
            seed_nodes_list.append(sn)

        # Compute similarity
        sim_mat = jaccard_similarity(revealed_sets)

        # Classify each rollout's subgraph for accuracy
        accuracies = []
        for seed_i in range(N_SEEDS):
            seed_node = seed_nodes_list[seed_i]
            n_nodes_g = g.num_nodes()
            B = max(1, int(BUDGET_RATIO * n_nodes_g))
            state = GraphState(g, seed_node=seed_node)
            state._budget_limit = B

            for _step in range(B):
                scores, probs, frontier_nodes = reader.forward(state)
                if not frontier_nodes:
                    break
                chosen = frontier_nodes[probs.argmax().item()]
                reveal_node(state, chosen)

            sub_g, _ = builder.build(state)
            batch_g = dgl.batch([sub_g]).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(batch_g, nids)
            pred = logits[0].argmax(dim=-1).item()
            accuracies.append(1.0 if pred == label else 0.0)

        # Build and save figure
        fig = build_figure(ogb_idx, label, n_nodes, revealed_sets, seed_nodes_list,
                          accuracies, sim_mat)
        fig.savefig(output_dir / f'class_{label:02d}_graph_{ogb_idx}.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    print(f"\nDone. {len(all_sampled)} figures saved to {output_dir}/")


if __name__ == '__main__':
    main()

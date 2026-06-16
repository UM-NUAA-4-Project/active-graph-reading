"""Reader 效果验证脚本：在验证集和测试集上评测 LearnableReader vs Random/BFS。

- 每次只加载一个 split，使用完释放
- 每张图给所有节点按 degree 打分，选最高分节点作 best seed、最低分作 worst seed
- 分别统计 best-seed 和 worst-seed 的准确率
- 对比 LearnableReader、Random、BFS 三种策略

输出 JSON 日志和可视化图像。
"""

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader import FixedStrategyReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from ogb.graphproppred import DglGraphPropPredDataset

BUDGETS_REL = [0.05, 0.10, 0.15]
FIXED_STRATEGIES = ['random', 'bfs']
ALL_STRATEGIES = FIXED_STRATEGIES + ['learnable']
STRATEGY_COLORS = {'random': '#999999', 'bfs': '#55A868', 'learnable': '#4C72B0'}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def infer_reader_config(state_dict: dict) -> Tuple[int, int]:
    hidden_dim = state_dict['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in state_dict:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)
    return hidden_dim, n_layers


def compute_budget(n_nodes: int, ratio: float) -> int:
    return max(1, int(ratio * n_nodes))


def pick_seeds_by_degree(g: dgl.DGLGraph) -> Tuple[int, int]:
    indeg = g.in_degrees().float()
    outdeg = g.out_degrees().float()
    total_deg = indeg + outdeg

    best_seed = int(total_deg.argmax().item())
    worst_seed = int(total_deg.argmin().item())
    return best_seed, worst_seed


# ---------------------------------------------------------------------------
# run one seed through reveal + classify
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_one_seed(
    reader,
    g_raw: dgl.DGLGraph,
    seed_node: int,
    label: int,
    device: torch.device,
    classifier: GraphClassifier,
    builder: SubgraphBuilder,
    budgets_rel: List[float],
    budgets_abs: List[int],
) -> List[float]:
    state = GraphState(g_raw, seed_node=seed_node)

    reader.reset()
    initial_frontier = list(state._initial_frontier)
    if initial_frontier:
        reader.on_new_frontier(initial_frontier)

    per_budget_correct = []

    for abs_b in budgets_abs:
        while state.get_budget_used() < abs_b:
            nxt = reader.select(state)
            if nxt is None:
                break
            new_frontier = reveal_node(state, nxt)
            if new_frontier:
                reader.on_new_frontier(new_frontier)

        sub_g, _ = builder.build(state)
        sub_g = sub_g.to(device)
        nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(sub_g, nids)
        per_budget_correct.append(1.0 if logits.argmax().item() == label else 0.0)

    return per_budget_correct


# ---------------------------------------------------------------------------
# evaluate one split for one strategy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_strategy_split(
    reader,
    strategy_name: str,
    dataset: DglGraphPropPredDataset,
    indices: List[int],
    classifier: GraphClassifier,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, Dict[float, List[float]]]:
    builder = SubgraphBuilder()
    budgets_rel_sorted = sorted(BUDGETS_REL)

    results: Dict[str, Dict[float, List[float]]] = {
        'best': defaultdict(list),
        'worst': defaultdict(list),
    }

    for idx in tqdm(indices, desc=desc):
        g_raw, label = dataset[int(idx)]
        if hasattr(label, 'item'):
            label = label.item()
        g_raw = dgl.add_self_loop(g_raw)
        n_nodes = g_raw.num_nodes()
        budgets_abs = [compute_budget(n_nodes, r) for r in budgets_rel_sorted]

        best_seed, worst_seed = pick_seeds_by_degree(g_raw)

        # best seed
        best_rec = run_one_seed(
            reader, g_raw, best_seed, label, device, classifier, builder,
            budgets_rel_sorted, budgets_abs,
        )
        for i, rel_b in enumerate(budgets_rel_sorted):
            results['best'][rel_b].append(best_rec[i])

        # worst seed
        worst_rec = run_one_seed(
            reader, g_raw, worst_seed, label, device, classifier, builder,
            budgets_rel_sorted, budgets_abs,
        )
        for i, rel_b in enumerate(budgets_rel_sorted):
            results['worst'][rel_b].append(worst_rec[i])

    return {
        'best': dict(results['best']),
        'worst': dict(results['worst']),
    }


# ---------------------------------------------------------------------------
# stats & plot
# ---------------------------------------------------------------------------

def compute_stats(
    split_results: Dict[str, Dict[str, Dict[float, List[float]]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """strategy -> seed_type -> budget_key -> {mean, std, n}"""
    stats = {}
    for strategy in ALL_STRATEGIES:
        if strategy not in split_results:
            continue
        stats[strategy] = {}
        for kind in ('best', 'worst'):
            stats[strategy][kind] = {}
            for rel_b in sorted(BUDGETS_REL):
                accs = split_results[strategy].get(kind, {}).get(rel_b, [])
                stats[strategy][kind][f'{rel_b:.2f}'] = {
                    'mean': float(np.mean(accs)) if accs else 0.0,
                    'std': float(np.std(accs)) if accs else 0.0,
                    'n': len(accs),
                }
    return stats


def plot_results(
    valid_stats: Dict[str, Dict[str, Dict[str, float]]],
    test_stats: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
    ref_valid: float = None,
    ref_test: float = None,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budgets = sorted(BUDGETS_REL)
    budget_labels = [f'{b:.0%}' for b in budgets]
    x = np.arange(len(budgets))
    n_strategies = len(ALL_STRATEGIES)
    width = 0.8 / n_strategies
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_strategies)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey='row')
    seed_types = ['best', 'worst']
    split_names = ['valid', 'test']

    for row, seed_type in enumerate(seed_types):
        for col, split_name in enumerate(split_names):
            ax = axes[row][col]
            stats = valid_stats if split_name == 'valid' else test_stats

            for si, strategy in enumerate(ALL_STRATEGIES):
                if strategy not in stats:
                    continue
                seed_stats = stats[strategy].get(seed_type, {})
                means = [seed_stats[f'{b:.2f}']['mean'] for b in budgets]
                stds = [seed_stats[f'{b:.2f}']['std'] for b in budgets]
                color = STRATEGY_COLORS[strategy]
                label = strategy if split_name == 'valid' else None

                bars = ax.bar(
                    x + offsets[si], means, width, yerr=stds,
                    label=label, capsize=4, color=color,
                )
                for bar, mean in zip(bars, means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{mean:.4f}', ha='center', va='bottom', fontsize=6,
                    )

            ax.set_title(f'{split_name.capitalize()} — {seed_type.capitalize()} Seed')
            ax.set_xlabel('Budget Ratio')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(x)
            ax.set_xticklabels(budget_labels)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            ref = ref_valid if split_name == 'valid' else ref_test
            if ref is not None:
                ref_frac = ref / 100 if ref > 1 else ref
                ref_pct = ref if ref > 1 else ref * 100
                ax.axhline(y=ref_frac, color='red', linestyle='--', linewidth=1.2,
                           label=f'Full-graph ({ref_pct:.2f}%)')
                ax.legend(fontsize=8)

    plt.suptitle('LearnableReader vs Random / BFS (best/worst seed by degree)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LearnableReader vs Random/BFS on valid/test sets')
    parser.add_argument('--checkpoint', default='m5/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='Limit graphs per split (0 = all)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--strategies', nargs='+',
                        default=['random', 'bfs', 'learnable'],
                        help='Strategies to evaluate (default: random bfs learnable)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random strategy reproducibility')
    parser.add_argument('--ref-valid', type=float, default=None,
                        help='Full-graph classifier valid accuracy '
                             '(auto-read from checkpoint if omitted)')
    parser.add_argument('--ref-test', type=float, default=None,
                        help='Full-graph classifier test accuracy')
    args = parser.parse_args()

    repo_root = _script_dir.parent
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies_to_run = [s for s in args.strategies if s in ALL_STRATEGIES]

    # ---- 1. 加载 classifier（全局复用） ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    ref_valid = args.ref_valid if args.ref_valid is not None else ckpt.get('val_acc', None)
    ref_test = args.ref_test  # not in checkpoint, user must provide
    if ref_valid is not None or ref_test is not None:
        def _fmt(v):
            if v is None:
                return 'N/A'
            return f'{v * 100:.2f}%' if v <= 1 else f'{v:.2f}%'
        print(f"  Full-graph ref: valid={_fmt(ref_valid)}, test={_fmt(ref_test)}")

    # ---- 2. 加载 learnable reader（如果需要） ----
    learnable_reader = None
    hidden_dim, n_layers = None, None
    if 'learnable' in strategies_to_run:
        print("\n=== Loading learnable reader ===")
        reader_ckpt = torch.load(str(repo_root / args.checkpoint),
                                 map_location='cpu', weights_only=False)
        hidden_dim, n_layers = infer_reader_config(reader_ckpt['reader_state_dict'])
        print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, "
              f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")

        learnable_reader = LearnableReader(
            edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers,
            dropout=args.dropout,
        ).to(device)
        learnable_reader.load_state_dict(reader_ckpt['reader_state_dict'])
        learnable_reader.eval()
        print(f"  Params: {sum(p.numel() for p in learnable_reader.parameters()):,}")

    # ---- 3. 按 split + strategy 分别评测 ----
    all_stats: Dict[str, Dict] = {}

    for split_name in ('valid', 'test'):
        all_stats[split_name] = {}

        print(f"\n{'='*60}")
        print(f"=== Loading OGBG-PPA dataset ({split_name} split) ===")
        dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
        split_indices = dataset.get_idx_split()[split_name].tolist()

        if args.max_graphs > 0:
            split_indices = split_indices[:args.max_graphs]

        print(f"  {split_name}: {len(split_indices)} graphs")

        for strategy_name in strategies_to_run:
            print(f"\n  --- {strategy_name} ---")

            if strategy_name == 'learnable':
                reader = learnable_reader
                reader.eval()
            else:
                random.seed(args.seed)
                reader = FixedStrategyReader(strategy_name)

            split_results = evaluate_strategy_split(
                reader, strategy_name, dataset, split_indices,
                classifier, device,
                desc=f"  {split_name.capitalize()} / {strategy_name}",
            )
            all_stats[split_name][strategy_name] = split_results

            # 即时输出 stats
            stats = compute_stats({strategy_name: split_results})
            for kind in ('best', 'worst'):
                for b_key, s in stats[strategy_name][kind].items():
                    print(f"    {kind:5s} budget={b_key}: "
                          f"mean={s['mean']:.4f}, std={s['std']:.4f} (n={s['n']})")

        del dataset
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 4. 组装全局 stats ----
    final_stats = {}
    for split_name in ('valid', 'test'):
        final_stats[split_name] = compute_stats(all_stats[split_name])

    # ---- 5. 输出 JSON ----
    output = {
        'valid': final_stats.get('valid', {}),
        'test': final_stats.get('test', {}),
        'config': {
            'checkpoint': str(repo_root / args.checkpoint),
            'classifier': str(repo_root / args.classifier),
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'strategies': strategies_to_run,
            'seed_method': 'max-degree (best) / min-degree (worst)',
            'budgets': BUDGETS_REL,
            'full_graph_ref': {
                'valid': ref_valid,
                'test': ref_test,
            },
        },
    }

    json_path = output_dir / 'eval_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {json_path}")

    # ---- 6. 可视化 ----
    plot_path = output_dir / 'eval_plot.png'
    plot_results(
        final_stats.get('valid', {}),
        final_stats.get('test', {}),
        str(plot_path),
        ref_valid=ref_valid,
        ref_test=ref_test,
    )


if __name__ == '__main__':
    main()

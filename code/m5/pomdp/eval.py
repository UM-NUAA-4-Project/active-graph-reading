"""POMDP Reader 效果验证脚本：在验证集和测试集上评测 POMDP vs Random/BFS。

POMDP reader 使用信念条件化策略选择节点：
- 主指标：外部 GraphClassifier 分类准确率（与 random/BFS 公平对比）
- 副指标：Reader 自主裁判准确率（argmax b_B^total）

按 split 逐次加载，只用 dgl.load_graphs 加载需要的图（不加载训练集）。
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
import pandas as pd
import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

from active_graph_reader.reader.pomdp_reader import POMDPReader
from active_graph_reader.reader import FixedStrategyReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder

BUDGETS_REL = [0.05, 0.10, 0.15]
FIXED_STRATEGIES = ['random', 'bfs']
ALL_STRATEGIES = FIXED_STRATEGIES + ['pomdp']
STRATEGY_COLORS = {'random': '#999999', 'bfs': '#55A868', 'pomdp': '#C44E52'}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
def run_one_seed_pomdp(
    reader: POMDPReader,
    g_raw: dgl.DGLGraph,
    seed_node: int,
    label: int,
    device: torch.device,
    classifier: GraphClassifier,
    builder: SubgraphBuilder,
    budgets_abs: List[int],
) -> Tuple[List[float], List[float]]:
    """POMDP reader: 返回 (external_acc, belief_acc) 两个指标。"""
    state = GraphState(g_raw, seed_node=seed_node)
    reader.reset()

    ext_correct = []
    belief_correct = []

    for abs_b in budgets_abs:
        while state.get_budget_used() < abs_b:
            nxt = reader.select(state)
            if nxt is None:
                break
            reveal_node(state, nxt)

        # 外部 classifier 分类
        sub_g, _ = builder.build(state)
        sub_g = sub_g.to(device)
        nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(sub_g, nids)
        ext_correct.append(1.0 if logits.argmax().item() == label else 0.0)

        # Reader 自主裁判
        final_out = reader.get_terminal_belief(state)
        belief_correct.append(1.0 if final_out.y_hat.item() == label else 0.0)

    return ext_correct, belief_correct


@torch.no_grad()
def run_one_seed_fixed(
    reader,
    g_raw: dgl.DGLGraph,
    seed_node: int,
    label: int,
    device: torch.device,
    classifier: GraphClassifier,
    builder: SubgraphBuilder,
    budgets_abs: List[int],
) -> List[float]:
    """固定策略: 返回外部 classifier 准确率。"""
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
    graphs: List[dgl.DGLGraph],
    labels_tensor: torch.Tensor,
    indices: List[int],
    classifier: GraphClassifier,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict:
    builder = SubgraphBuilder()
    budgets_rel_sorted = sorted(BUDGETS_REL)

    results: Dict[str, Dict[float, List[float]]] = {
        'best': defaultdict(list),
        'worst': defaultdict(list),
    }
    belief_results: Dict[str, Dict[float, List[float]]] = {
        'best': defaultdict(list),
        'worst': defaultdict(list),
    }
    is_pomdp = (strategy_name == 'pomdp')

    for i, ogb_idx in enumerate(tqdm(indices, desc=desc)):
        g_raw = dgl.add_self_loop(graphs[i])
        label = int(labels_tensor[ogb_idx].item())
        n_nodes = g_raw.num_nodes()
        budgets_abs = [compute_budget(n_nodes, r) for r in budgets_rel_sorted]

        best_seed, worst_seed = pick_seeds_by_degree(g_raw)

        for seed_type, seed in [('best', best_seed), ('worst', worst_seed)]:
            if is_pomdp:
                reader.reset()
                ext_rec, belief_rec = run_one_seed_pomdp(
                    reader, g_raw, seed, label, device, classifier, builder,
                    budgets_abs,
                )
                for j, rel_b in enumerate(budgets_rel_sorted):
                    results[seed_type][rel_b].append(ext_rec[j])
                    belief_results[seed_type][rel_b].append(belief_rec[j])
            else:
                reader.reset()
                rec = run_one_seed_fixed(
                    reader, g_raw, seed, label, device, classifier, builder,
                    budgets_abs,
                )
                for j, rel_b in enumerate(budgets_rel_sorted):
                    results[seed_type][rel_b].append(rec[j])

    out = {
        'best': dict(results['best']),
        'worst': dict(results['worst']),
    }
    if is_pomdp:
        out['belief_best'] = dict(belief_results['best'])
        out['belief_worst'] = dict(belief_results['worst'])
    return out


# ---------------------------------------------------------------------------
# stats & plot
# ---------------------------------------------------------------------------

def compute_stats(
    split_results: Dict[str, Dict],
    key: str = 'best',
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
    valid_stats: Dict,
    test_stats: Dict,
    output_path: str,
    ref_valid: float = None,
    ref_test: float = None,
    belief_valid: Dict = None,
    belief_test: Dict = None,
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
                label = strategy if col == 0 else None

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

            # POMDP belief accuracy (dashed overlay)
            bstats = belief_valid if split_name == 'valid' else belief_test
            if bstats and 'pomdp' in bstats:
                b_seed_stats = bstats['pomdp'].get(seed_type, {})
                b_means = [b_seed_stats[f'{b:.2f}']['mean'] for b in budgets]
                ax.plot(x, b_means, 'o--', color=STRATEGY_COLORS['pomdp'],
                        markersize=6, linewidth=1.2, alpha=0.7,
                        label='pomdp (belief)' if col == 0 else None)
                for xi, bm in zip(x, b_means):
                    ax.annotate(f'{bm:.4f}', (xi, bm),
                                textcoords="offset points", xytext=(0, 12),
                                ha='center', fontsize=6, color=STRATEGY_COLORS['pomdp'])

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

    plt.suptitle('POMDP Reader vs Random / BFS (best/worst seed by degree)',
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
        description='Evaluate POMDP Reader vs Random/BFS on valid/test sets')
    parser.add_argument('--checkpoint', default='m5/phase2/checkpoint_epoch20.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='m5/pomdp/eval_results')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='Limit graphs per split (0 = all)')
    parser.add_argument('--strategies', nargs='+',
                        default=['random', 'bfs', 'pomdp'],
                        help='Strategies to evaluate (default: random bfs pomdp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random strategy reproducibility')
    parser.add_argument('--ref-valid', type=float, default=None,
                        help='Full-graph classifier valid accuracy')
    parser.add_argument('--ref-test', type=float, default=None,
                        help='Full-graph classifier test accuracy')
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _m5_dir / 'pomdp' / 'eval_results'
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
    ref_test = args.ref_test
    if ref_valid is not None or ref_test is not None:
        def _fmt(v):
            if v is None:
                return 'N/A'
            return f'{v * 100:.2f}%' if v <= 1 else f'{v:.2f}%'
        print(f"  Full-graph ref: valid={_fmt(ref_valid)}, test={_fmt(ref_test)}")

    # ---- 2. 加载 POMDP reader（如果需要） ----
    pomdp_reader = None
    if 'pomdp' in strategies_to_run:
        print("\n=== Loading POMDP Reader ===")
        pomdp_reader = POMDPReader(
            edge_feat_dim=7, hidden_dim=300, k_state=128, k_evidence=37,
            n_layers=3, dropout=0.5,
            use_struct=not args.no_struct if hasattr(args, 'no_struct') else True,
        ).to(device)
        ckpt_path = repo_root / args.checkpoint
        if ckpt_path.exists():
            reader_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            pomdp_reader.load_state_dict(reader_ckpt['reader_state_dict'])
            print(f"  Checkpoint epoch: {reader_ckpt.get('epoch', 'N/A')}")
        else:
            print(f"  WARNING: {ckpt_path} not found, using random init")
        pomdp_reader.eval()
        print(f"  Params: {sum(p.numel() for p in pomdp_reader.parameters()):,}")

    # ---- 3. 按 split + strategy 分别评测（逐 split 加载图，释放训练集） ----
    all_results: Dict[str, Dict] = {}
    all_belief: Dict[str, Dict] = {}
    processed_path = str(Path(args.root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')

    for split_name in ('valid', 'test'):
        all_results[split_name] = {}

        # 从 CSV 读 split 索引（不加载图数据）
        split_path = Path(args.root) / 'ogbg_ppa' / 'split' / 'species' / f'{split_name}.csv.gz'
        split_indices = pd.read_csv(split_path, compression='gzip', header=None).values.flatten().tolist()

        if args.max_graphs > 0:
            split_indices = split_indices[:args.max_graphs]

        # 仅加载当前 split 的图
        print(f"\n{'='*60}")
        print(f"=== Loading {split_name} split ({len(split_indices)} graphs) ===")
        graphs, label_dict = dgl.load_graphs(processed_path, idx_list=split_indices)
        print(f"  Loaded {len(graphs)} graphs")

        for strategy_name in strategies_to_run:
            print(f"\n  --- {strategy_name} ---")

            if strategy_name == 'pomdp':
                reader = pomdp_reader
                reader.eval()
            else:
                random.seed(args.seed)
                reader = FixedStrategyReader(strategy_name)

            split_results = evaluate_strategy_split(
                reader, strategy_name, graphs, label_dict['labels'], split_indices,
                classifier, device,
                desc=f"  {split_name.capitalize()} / {strategy_name}",
            )
            all_results[split_name][strategy_name] = split_results

            # 即时输出 stats
            for kind in ('best', 'worst'):
                if kind not in split_results:
                    continue
                for b_key in sorted(BUDGETS_REL):
                    accs = split_results[kind].get(b_key, [])
                    mean_v = float(np.mean(accs)) if accs else 0.0
                    std_v = float(np.std(accs)) if accs else 0.0
                    print(f"    {kind:5s} budget={b_key:.2f}: "
                          f"mean={mean_v:.4f}, std={std_v:.4f} (n={len(accs)})")

            # POMDP belief 指标
            if 'belief_best' in split_results:
                print(f"    --- belief (reader自主裁判) ---")
                for kind in ('belief_best', 'belief_worst'):
                    label_kind = kind.replace('belief_', '')
                    for b_key in sorted(BUDGETS_REL):
                        accs = split_results[kind].get(b_key, [])
                        mean_v = float(np.mean(accs)) if accs else 0.0
                        std_v = float(np.std(accs)) if accs else 0.0
                        print(f"    {label_kind:5s} budget={b_key:.2f}: "
                              f"belief_acc={mean_v:.4f}, std={std_v:.4f} (n={len(accs)})")

            # Store belief results separately for plotting
            if 'belief_best' in split_results:
                if split_name not in all_belief:
                    all_belief[split_name] = {}
                all_belief[split_name]['pomdp'] = {
                    'best': split_results['belief_best'],
                    'worst': split_results['belief_worst'],
                }

        # 释放当前 split 的图
        del graphs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 4. 组装全局 stats ----
    final_stats = {}
    for split_name in ('valid', 'test'):
        final_stats[split_name] = compute_stats(all_results[split_name])

    # Build belief stats for plotting
    belief_stats = {}
    for split_name in ('valid', 'test'):
        if split_name in all_belief:
            belief_stats[split_name] = all_belief[split_name]

    # ---- 5. 输出 JSON ----
    output = {
        'valid': final_stats.get('valid', {}),
        'test': final_stats.get('test', {}),
        'config': {
            'checkpoint': str(repo_root / args.checkpoint),
            'classifier': str(repo_root / args.classifier),
            'strategies': strategies_to_run,
            'seed_method': 'max-degree (best) / min-degree (worst)',
            'budgets': BUDGETS_REL,
            'full_graph_ref': {
                'valid': ref_valid,
                'test': ref_test,
            },
            'metrics': 'Primary: external classifier acc (fair comparison). '
                       'Secondary (pomdp only): belief acc (reader自主裁判).',
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
        belief_valid=belief_stats.get('valid'),
        belief_test=belief_stats.get('test'),
    )


if __name__ == '__main__':
    main()

"""Phase 1: 泛化验证 — val/test 全量评测。

评测 LearnableReader vs Random/BFS，输出 JSON 日志和可视化图像。

输出文件（按策略类型分开）：
  - eval_fixed.json          random + bfs（checkpoint 无关，可复用）
  - eval_learnable.json      learnable（checkpoint 相关）

用法：
  python phase1_eval.py                                          # 默认三种策略
  python phase1_eval.py --skip-fixed                             # 仅 learnable
  python phase1_eval.py --checkpoint m5/base/rl/checkpoint_epoch7.pt  # 指定 checkpoint
"""

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader import FixedStrategyReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from training_utils import patch_old_checkpoint

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
    state._budget_limit = max(budgets_abs)

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
):
    builder = SubgraphBuilder()
    budgets_rel_sorted = sorted(BUDGETS_REL)

    results: Dict[str, Dict[float, List[float]]] = {
        'best': defaultdict(list),
        'worst': defaultdict(list),
    }
    per_class: Dict[str, Dict[int, Dict[float, List[float]]]] = {
        'best': defaultdict(lambda: defaultdict(list)),
        'worst': defaultdict(lambda: defaultdict(list)),
    }

    for i, ogb_idx in enumerate(tqdm(indices, desc=desc)):
        g_raw = dgl.add_self_loop(graphs[i])
        label = int(labels_tensor[ogb_idx].item())
        n_nodes = g_raw.num_nodes()
        budgets_abs = [compute_budget(n_nodes, r) for r in budgets_rel_sorted]

        best_seed, worst_seed = pick_seeds_by_degree(g_raw)

        # best seed
        best_rec = run_one_seed(
            reader, g_raw, best_seed, label, device, classifier, builder,
            budgets_rel_sorted, budgets_abs,
        )
        for j, rel_b in enumerate(budgets_rel_sorted):
            results['best'][rel_b].append(best_rec[j])
            per_class['best'][label][rel_b].append(best_rec[j])

        # worst seed
        worst_rec = run_one_seed(
            reader, g_raw, worst_seed, label, device, classifier, builder,
            budgets_rel_sorted, budgets_abs,
        )
        for j, rel_b in enumerate(budgets_rel_sorted):
            results['worst'][rel_b].append(worst_rec[j])
            per_class['worst'][label][rel_b].append(worst_rec[j])

    aggregated = {
        'best': dict(results['best']),
        'worst': dict(results['worst']),
    }
    per_class_clean = {
        'best': {cls: dict(budgets) for cls, budgets in per_class['best'].items()},
        'worst': {cls: dict(budgets) for cls, budgets in per_class['worst'].items()},
    }
    return aggregated, per_class_clean


# ---------------------------------------------------------------------------
# stats & plot
# ---------------------------------------------------------------------------

def compute_stats(
    split_results: Dict[str, Dict[str, Dict[float, List[float]]]],
    strategies: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats = {}
    for strategy in strategies:
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


def compute_per_class_stats(
    per_class_results: Dict[str, Dict[str, Dict[int, Dict[float, List[float]]]]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    stats = {}
    for strategy, seed_dict in per_class_results.items():
        stats[strategy] = {}
        for seed_type, class_dict in seed_dict.items():
            stats[strategy][seed_type] = {}
            for cls, budget_dict in class_dict.items():
                cls_str = str(cls)
                stats[strategy][seed_type][cls_str] = {}
                for rel_b in sorted(BUDGETS_REL):
                    accs = budget_dict.get(rel_b, [])
                    stats[strategy][seed_type][cls_str][f'{rel_b:.2f}'] = {
                        'mean': float(np.mean(accs)) if accs else 0.0,
                        'std': float(np.std(accs)) if accs else 0.0,
                        'n': len(accs),
                    }
    return stats


def plot_per_class(
    per_class_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    split_name: str,
    output_dir: Path,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budgets = sorted(BUDGETS_REL)
    budget_labels = [f'{b:.0%}' for b in budgets]

    for strategy, seed_dict in per_class_stats.items():
        for seed_type, class_dict in seed_dict.items():
            if not class_dict:
                continue
            classes = sorted(class_dict.keys(), key=lambda x: int(x))
            n_classes = len(classes)
            fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.35), 6))

            x = np.arange(n_classes)
            n_budgets = len(budgets)
            width = 0.8 / n_budgets
            offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_budgets)
            colors = ['#4C72B0', '#55A868', '#C44E52']

            for bi, rel_b in enumerate(budgets):
                b_key = f'{rel_b:.2f}'
                means = [class_dict[cls][b_key]['mean'] for cls in classes]
                stds = [class_dict[cls][b_key]['std'] for cls in classes]
                ax.bar(
                    x + offsets[bi], means, width, yerr=stds,
                    label=f'budget={rel_b:.0%}', capsize=2,
                    color=colors[bi % len(colors)],
                )

            ax.set_title(f'{split_name} — {strategy} / {seed_type} seed (per-class)')
            ax.set_xlabel('Class')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=90, fontsize=5)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)

            fig.tight_layout()
            fname = f'per_class_{split_name}_{strategy}_{seed_type}.png'
            fig.savefig(output_dir / fname, dpi=150)
            plt.close(fig)
            print(f"Per-class plot saved to {output_dir / fname}")


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

    # 聚合所有出现的策略
    all_strats = set()
    for s in (valid_stats, test_stats):
        all_strats.update(s.keys())
    strategies_to_plot = [s for s in ALL_STRATEGIES if s in all_strats]

    budgets = sorted(BUDGETS_REL)
    budget_labels = [f'{b:.0%}' for b in budgets]
    x = np.arange(len(budgets))
    n_strategies = len(strategies_to_plot)
    width = 0.8 / n_strategies if n_strategies > 0 else 0.8
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_strategies) if n_strategies > 0 else [0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey='row')
    seed_types = ['best', 'worst']
    split_names = ['valid', 'test']

    for row, seed_type in enumerate(seed_types):
        for col, split_name in enumerate(split_names):
            ax = axes[row][col]
            stats = valid_stats if split_name == 'valid' else test_stats

            for si, strategy in enumerate(strategies_to_plot):
                if strategy not in stats:
                    continue
                seed_stats = stats[strategy].get(seed_type, {})
                means = [seed_stats.get(f'{b:.2f}', {}).get('mean', 0.0) for b in budgets]
                stds = [seed_stats.get(f'{b:.2f}', {}).get('std', 0.0) for b in budgets]
                color = STRATEGY_COLORS.get(strategy, '#999999')
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
# load cached fixed-strategy results
# ---------------------------------------------------------------------------

def load_fixed_results(output_dir: Path) -> Optional[Dict]:
    """加载已有的 fixed 策略结果（如果存在）。"""
    fixed_path = output_dir / 'eval_fixed.json'
    if fixed_path.exists():
        with open(fixed_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Evaluate LearnableReader vs Random/BFS on valid/test sets')
    parser.add_argument('--checkpoint', default='m5/base/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='m5/base/analysis/eval_results')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='Limit graphs per split (0 = all)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--skip-fixed', action='store_true',
                        help='Skip random/bfs evaluation, only run learnable')
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

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _m5_dir / 'base' / 'analysis' / 'eval_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要跑的策略
    if args.skip_fixed:
        strategies_to_run = ['learnable']
    else:
        strategies_to_run = [s for s in args.strategies if s in ALL_STRATEGIES]

    fixed_strategies_to_run = [s for s in strategies_to_run if s in FIXED_STRATEGIES]
    run_learnable = 'learnable' in strategies_to_run

    print(f"Strategies to run: {strategies_to_run}")
    if fixed_strategies_to_run:
        print(f"  Fixed: {fixed_strategies_to_run}")
    if run_learnable:
        print(f"  Learnable: yes (checkpoint={args.checkpoint})")

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

    # ---- 2. 加载 learnable reader（如果需要） ----
    learnable_reader = None
    hidden_dim, n_layers = None, None
    if run_learnable:
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
        reader_sd, _ = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
        learnable_reader.load_state_dict(reader_sd)
        learnable_reader.eval()
        print(f"  Params: {sum(p.numel() for p in learnable_reader.parameters()):,}")

    # ---- 3. 加载已有的 fixed 结果（如果 skip-fixed 且有缓存） ----
    cached_fixed = None
    if args.skip_fixed or not fixed_strategies_to_run:
        cached_fixed = load_fixed_results(output_dir)
        if cached_fixed:
            print(f"\n=== Loaded cached fixed results from eval_fixed.json ===")

    # ---- 4. 按 split + strategy 分别评测 ----
    all_stats: Dict[str, Dict] = {}
    all_per_class: Dict[str, Dict] = {}
    processed_path = str(Path(args.root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')

    for split_name in ('valid', 'test'):
        all_stats[split_name] = {}
        all_per_class[split_name] = {}

        # 从 CSV 读 split 索引
        split_path = Path(args.root) / 'ogbg_ppa' / 'split' / 'species' / f'{split_name}.csv.gz'
        split_indices = pd.read_csv(split_path, compression='gzip', header=None).values.flatten().tolist()

        if args.max_graphs > 0:
            split_indices = split_indices[:args.max_graphs]

        print(f"\n{'='*60}")
        print(f"=== Loading {split_name} split ({len(split_indices)} graphs) ===")
        graphs, label_dict = dgl.load_graphs(processed_path, idx_list=split_indices)
        print(f"  Loaded {len(graphs)} graphs")

        for strategy_name in strategies_to_run:
            print(f"\n  --- {strategy_name} ---")

            if strategy_name == 'learnable':
                reader = learnable_reader
                reader.eval()
            else:
                random.seed(args.seed)
                reader = FixedStrategyReader(strategy_name)

            aggregated, per_class = evaluate_strategy_split(
                reader, strategy_name, graphs, label_dict['labels'], split_indices,
                classifier, device,
                desc=f"  {split_name.capitalize()} / {strategy_name}",
            )
            all_stats[split_name][strategy_name] = aggregated
            all_per_class[split_name][strategy_name] = per_class

            # 即时输出 overall stats
            stats = compute_stats({strategy_name: aggregated}, [strategy_name])
            for kind in ('best', 'worst'):
                for b_key, s in stats[strategy_name][kind].items():
                    print(f"    {kind:5s} budget={b_key}: "
                          f"mean={s['mean']:.4f}, std={s['std']:.4f} (n={s['n']})")

            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        del graphs

    # ---- 5. 组装 stats ----
    final_stats = {}
    for split_name in ('valid', 'test'):
        final_stats[split_name] = compute_stats(all_stats[split_name], strategies_to_run)

    final_per_class_stats = {}
    for split_name in ('valid', 'test'):
        final_per_class_stats[split_name] = compute_per_class_stats(all_per_class[split_name])

    # 合并 cached_fixed 到 stats 中供绘图用
    if cached_fixed:
        for split_name in ('valid', 'test'):
            if split_name in cached_fixed:
                for strategy in FIXED_STRATEGIES:
                    if strategy in cached_fixed[split_name]:
                        final_stats[split_name][strategy] = cached_fixed[split_name][strategy]
    merged_final_stats = final_stats

    # ---- 6. 输出 JSON（按策略类型分开） ----
    base_config = {
        'classifier': str(repo_root / args.classifier),
        'budgets': BUDGETS_REL,
        'seed_method': 'max-degree (best) / min-degree (worst)',
        'full_graph_ref': {
            'valid': ref_valid,
            'test': ref_test,
        },
    }

    # 6a. Fixed 策略 JSON（random + bfs）
    if fixed_strategies_to_run:
        fixed_output = {}
        fixed_per_class = {}
        for split_name in ('valid', 'test'):
            fixed_output[split_name] = {
                s: final_stats[split_name].get(s, {})
                for s in FIXED_STRATEGIES if s in final_stats[split_name]
            }
            if split_name in final_per_class_stats:
                fixed_per_class[split_name] = {
                    s: final_per_class_stats[split_name].get(s, {})
                    for s in FIXED_STRATEGIES if s in final_per_class_stats[split_name]
                }

        fixed_json = {
            **fixed_output,
            'per_class_valid': fixed_per_class.get('valid', {}),
            'per_class_test': fixed_per_class.get('test', {}),
            'config': {
                **base_config,
                'strategies': FIXED_STRATEGIES,
                'seed': args.seed,
            },
        }
        fixed_path = output_dir / 'eval_fixed.json'
        with open(fixed_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_json, f, indent=2, ensure_ascii=False)
        print(f"\nFixed-strategy JSON saved to {fixed_path}")

    # 6b. Learnable JSON
    if run_learnable:
        learnable_output = {}
        learnable_per_class = {}
        for split_name in ('valid', 'test'):
            learnable_output[split_name] = {
                'learnable': final_stats[split_name].get('learnable', {})
            }
            if split_name in final_per_class_stats:
                learnable_per_class[split_name] = {
                    'learnable': final_per_class_stats[split_name].get('learnable', {})
                }

        learnable_json = {
            **learnable_output,
            'per_class_valid': learnable_per_class.get('valid', {}),
            'per_class_test': learnable_per_class.get('test', {}),
            'config': {
                **base_config,
                'checkpoint': str(repo_root / args.checkpoint),
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'strategies': ['learnable'],
            },
        }
        learnable_path = output_dir / 'eval_learnable.json'
        with open(learnable_path, 'w', encoding='utf-8') as f:
            json.dump(learnable_json, f, indent=2, ensure_ascii=False)
        print(f"Learnable JSON saved to {learnable_path}")

    # ---- 7. 可视化（合并 fixed + learnable） ----
    plot_path = output_dir / 'eval_plot.png'
    plot_results(
        merged_final_stats.get('valid', {}),
        merged_final_stats.get('test', {}),
        str(plot_path),
        ref_valid=ref_valid,
        ref_test=ref_test,
    )

    # per-class plots（合并所有策略）
    merged_per_class = {}
    for split_name in ('valid', 'test'):
        merged_per_class[split_name] = {}
        for s in ALL_STRATEGIES:
            split_data = final_per_class_stats.get(split_name, {})
            if s in split_data:
                merged_per_class[split_name][s] = split_data[s]

    for split_name in ('valid', 'test'):
        if merged_per_class.get(split_name):
            plot_per_class(merged_per_class[split_name], split_name, output_dir)


if __name__ == '__main__':
    main()

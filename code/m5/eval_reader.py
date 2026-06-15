"""Reader 效果验证脚本：在验证集和测试集上评测 LearnableReader。

每张图用 5 个 seed 在 [0.05, 0.10, 0.15] 三种预算下 greedy reveal，
分类器推理后统计各预算下的平均准确率和方差。
输出 JSON 日志和可视化图像。
"""

import argparse
import json
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
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from ogb.graphproppred import DglGraphPropPredDataset

BUDGETS_REL = [0.05, 0.10, 0.15]
N_SEEDS = 5


def infer_reader_config(state_dict: dict) -> Tuple[int, int]:
    """从 state_dict 推断 hidden_dim 和 n_layers。"""
    hidden_dim = state_dict['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in state_dict:
        if k.startswith('frontier_encoder.convs.'):
            layer_idx = int(k.split('.')[2])
            n_layers = max(n_layers, layer_idx + 1)
    return hidden_dim, n_layers


def compute_budget(n_nodes: int, ratio: float) -> int:
    return max(1, int(ratio * n_nodes))


@torch.no_grad()
def evaluate_split(
    reader: LearnableReader,
    dataset: DglGraphPropPredDataset,
    indices: List[int],
    classifier: GraphClassifier,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "Evaluating",
) -> Dict[float, List[float]]:
    """在指定图索引集上评测 reader + classifier。

    每张图跑 N_SEEDS 个 seed，每个 seed 从 0 开始逐步 reveal 到各预算档位。
    子图批量送入 classifier 推理以利用 GPU。

    Returns
    -------
    Dict[float, List[float]]  budget_ratio → [per_graph_accuracy, ...]
    """
    reader.eval()
    builder = SubgraphBuilder()
    budgets_rel_sorted = sorted(BUDGETS_REL)
    results: Dict[float, List[float]] = defaultdict(list)

    pbar = tqdm(range(0, len(indices), batch_size), desc=desc)
    for batch_start in pbar:
        batch_indices = indices[batch_start:batch_start + batch_size]

        subgraphs: List[dgl.DGLGraph] = []
        meta: List[Tuple[int, float, int]] = []  # (batch_i, rel_b, label)

        for batch_i, idx in enumerate(batch_indices):
            g_raw, label = dataset[int(idx)]
            if hasattr(label, 'item'):
                label = label.item()
            g_raw = dgl.add_self_loop(g_raw)
            n_nodes = g_raw.num_nodes()
            budgets_abs = [compute_budget(n_nodes, r) for r in budgets_rel_sorted]

            for seed_i in range(N_SEEDS):
                seed_node = (idx * 31 + seed_i * 17 + 7) % max(n_nodes, 1)
                state = GraphState(g_raw, seed_node=seed_node)

                for abs_b, rel_b in zip(budgets_abs, budgets_rel_sorted):
                    while state.get_budget_used() < abs_b:
                        next_node = reader.select(state)
                        if next_node is None:
                            break
                        reveal_node(state, next_node)

                    sub_g, _ = builder.build(state)
                    subgraphs.append(sub_g)
                    meta.append((batch_i, rel_b, label))

        if not subgraphs:
            continue

        batch_g = dgl.batch(subgraphs).to(device)
        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, node_ids)
        preds = logits.argmax(dim=1).cpu().tolist()

        # 按 (graph, budget) 聚合 seed 结果
        graph_budget_correct: Dict[Tuple[int, float], List[float]] = defaultdict(list)
        for flat_i, (batch_i, rel_b, label) in enumerate(meta):
            graph_budget_correct[(batch_i, rel_b)].append(
                1.0 if preds[flat_i] == label else 0.0
            )

        for (_batch_i, rel_b), corrects in graph_budget_correct.items():
            results[rel_b].append(np.mean(corrects))

        # 更新进度条
        if results:
            latest_budget = budgets_rel_sorted[-1]
            if results[latest_budget]:
                pbar.set_postfix({f'acc@{latest_budget:.2f}': f'{np.mean(results[latest_budget]):.4f}'})

    reader.train()
    return dict(results)


def compute_stats(results: Dict[float, List[float]]) -> Dict[str, Dict[str, float]]:
    """计算各预算下的均值和标准差。"""
    stats = {}
    for rel_b in sorted(results):
        accs = results[rel_b]
        stats[f'{rel_b:.2f}'] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'n_graphs': len(accs),
        }
    return stats


def plot_results(
    valid_stats: Dict[str, Dict[str, float]],
    test_stats: Dict[str, Dict[str, float]],
    output_path: str,
):
    """绘制分组柱状图：valid vs test 在不同预算下的准确率对比。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budgets = sorted(BUDGETS_REL)
    budget_labels = [f'{b:.0%}' for b in budgets]

    valid_means = [valid_stats[f'{b:.2f}']['mean'] for b in budgets]
    valid_stds = [valid_stats[f'{b:.2f}']['std'] for b in budgets]
    test_means = [test_stats[f'{b:.2f}']['mean'] for b in budgets]
    test_stds = [test_stats[f'{b:.2f}']['std'] for b in budgets]

    x = np.arange(len(budgets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, valid_means, width, yerr=valid_stds,
                   label='Valid', capsize=5, color='#4C72B0')
    bars2 = ax.bar(x + width / 2, test_means, width, yerr=test_stds,
                   label='Test', capsize=5, color='#DD8452')

    ax.set_xlabel('Budget Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_title('LearnableReader Performance on OGBG-PPA')
    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar, mean in zip(bars1, valid_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=8)
    for bar, mean in zip(bars2, test_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LearnableReader on valid/test sets')
    parser.add_argument('--checkpoint', default='m5/checkpoint/best_reader.pt',
                        help='Reader checkpoint path (relative to repo root)')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt',
                        help='Classifier checkpoint path')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset',
                        help='OGB dataset root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Graphs per classifier inference batch')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same dir as this script)')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='Limit graphs per split for quick test (0 = all)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Reader dropout (only affects construction, disabled in eval)')
    args = parser.parse_args()

    repo_root = _script_dir.parent
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 Classifier ----
    print("\n=== Loading classifier ===")
    classifier_ckpt_path = repo_root / args.classifier
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(classifier_ckpt_path), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 2. 加载 Reader ----
    print("\n=== Loading reader ===")
    reader_ckpt_path = repo_root / args.checkpoint
    reader_ckpt = torch.load(str(reader_ckpt_path), map_location='cpu', weights_only=False)
    hidden_dim, n_layers = infer_reader_config(reader_ckpt['reader_state_dict'])
    print(f"  Inferred: hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  Checkpoint epoch={reader_ckpt.get('epoch', 'N/A')}, "
          f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=args.dropout,
    ).to(device)
    reader.load_state_dict(reader_ckpt['reader_state_dict'])
    reader.eval()
    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Params: {total_params:,}")

    # ---- 3. 加载数据集 ----
    print("\n=== Loading OGBG-PPA dataset ===")
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    valid_indices = dataset.get_idx_split()['valid'].tolist()
    test_indices = dataset.get_idx_split()['test'].tolist()

    if args.max_graphs > 0:
        valid_indices = valid_indices[:args.max_graphs]
        test_indices = test_indices[:args.max_graphs]

    print(f"  Valid: {len(valid_indices)} graphs, Test: {len(test_indices)} graphs")
    print(f"  Per graph: {N_SEEDS} seeds x {len(BUDGETS_REL)} budgets = {N_SEEDS * len(BUDGETS_REL)} subgraphs")

    # ---- 4. 评测 ----
    print("\n=== Evaluating on validation set ===")
    valid_results = evaluate_split(
        reader, dataset, valid_indices, classifier, device,
        batch_size=args.batch_size, desc="Valid",
    )
    valid_stats = compute_stats(valid_results)
    for b, s in valid_stats.items():
        print(f"  Valid budget={b}: mean={s['mean']:.4f}, std={s['std']:.4f} (n={s['n_graphs']})")

    print("\n=== Evaluating on test set ===")
    test_results = evaluate_split(
        reader, dataset, test_indices, classifier, device,
        batch_size=args.batch_size, desc="Test",
    )
    test_stats = compute_stats(test_results)
    for b, s in test_stats.items():
        print(f"  Test  budget={b}: mean={s['mean']:.4f}, std={s['std']:.4f} (n={s['n_graphs']})")

    # ---- 5. 输出 ----
    output = {
        'valid': valid_stats,
        'test': test_stats,
        'config': {
            'checkpoint': str(reader_ckpt_path),
            'classifier': str(classifier_ckpt_path),
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_seeds': N_SEEDS,
            'budgets': BUDGETS_REL,
            'valid_n_graphs': len(valid_indices),
            'test_n_graphs': len(test_indices),
        },
    }

    json_path = output_dir / 'eval_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {json_path}")

    plot_path = output_dir / 'eval_plot.png'
    plot_results(valid_stats, test_stats, str(plot_path))


if __name__ == '__main__':
    main()

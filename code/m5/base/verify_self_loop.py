"""验证：classifier 子图无自环条件下的准确率（批处理版）。

输出 JSON（与 eval 一致的格式）：
  - verify_no_self_loop.json     per-budget / per-seed-type 准确率统计

用法：
  python code/m5/base/verify_self_loop.py
  python code/m5/base/verify_self_loop.py --checkpoint m5/base/rl/asymmetric/ex1/checkpoint_epoch10.pt
  python code/m5/base/verify_self_loop.py --output-dir code/m5/base/analysis/eval_results_no_self_loop
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
sys.path.insert(0, str(_repo_root))

import torch
import dgl
import numpy as np
import pandas as pd
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m5.training_utils import patch_old_checkpoint

K = 37
BUDGETS_REL = [0.05, 0.10, 0.15]
CHUNK_SIZE = 32


def pick_seeds_by_degree(g):
    indeg = g.in_degrees().float()
    outdeg = g.out_degrees().float()
    total_deg = indeg + outdeg
    return int(total_deg.argmax().item()), int(total_deg.argmin().item())


def remove_self_loops(g):
    src, dst = g.edges()
    mask = src != dst
    g2 = dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes())
    if 'feat' in g.edata:
        g2.edata['feat'] = g.edata['feat'][mask]
    return g2


def compute_budget(n_nodes: int, ratio: float) -> int:
    return max(1, int(ratio * n_nodes))


@torch.no_grad()
def evaluate(reader, classifier, graphs, ogb_indices, labels, device):
    """Lockstep batched evaluation: reader GNN + classifier both batched.

    返回 per-seed-type / per-budget 的原始正确性列表，
    以及 per-class 的原始正确性列表，供 compute_stats 汇总。
    """
    reader.eval()
    classifier.eval()
    builder = SubgraphBuilder()

    results: Dict[str, Dict[float, List[float]]] = {
        'best': defaultdict(list),
        'worst': defaultdict(list),
    }
    per_class: Dict[str, Dict[int, Dict[float, List[float]]]] = {
        'best': defaultdict(lambda: defaultdict(list)),
        'worst': defaultdict(lambda: defaultdict(list)),
    }
    classify_queue: List[Tuple[dgl.DGLGraph, int, str, float]] = []

    budgets_rel_sorted = sorted(BUDGETS_REL)

    for chunk_start in tqdm(range(0, len(graphs), CHUNK_SIZE), desc="  eval"):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(graphs))

        # ---- 创建该 chunk 的所有 state ----
        state_infos: List[dict] = []
        for i in range(chunk_start, chunk_end):
            g = dgl.add_self_loop(graphs[i])
            ogb_idx = ogb_indices[i]
            label = int(labels[ogb_idx].item())
            n_nodes = g.num_nodes()
            budgets_abs = [compute_budget(n_nodes, r) for r in budgets_rel_sorted]

            best_seed, worst_seed = pick_seeds_by_degree(g)

            for seed_type, seed_node in [('best', best_seed), ('worst', worst_seed)]:
                state = GraphState(g, seed_node=seed_node)
                state._budget_limit = max(budgets_abs)
                state_infos.append({
                    'state': state, 'label': label, 'seed_type': seed_type,
                    'budgets_abs': budgets_abs,
                })

        active = [True] * len(state_infos)
        budget_recorded = [[False] * len(budgets_rel_sorted) for _ in state_infos]
        max_steps = max(max(s['budgets_abs']) for s in state_infos)

        # ---- lockstep reveal 循环 ----
        for _step in range(max_steps):
            active_idx = [j for j in range(len(state_infos))
                          if active[j] and not state_infos[j]['state'].is_frontier_empty]
            if not active_idx:
                break

            batch_results = reader.forward_states(
                [state_infos[j]['state'] for j in active_idx],
            )

            for k, j in enumerate(active_idx):
                scores, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue

                si = state_infos[j]
                state = si['state']
                chosen = frontier_nodes[probs.argmax().item()]
                reveal_node(state, chosen)

                budget_used = state.get_budget_used()
                for bi, abs_b in enumerate(si['budgets_abs']):
                    if not budget_recorded[j][bi] and budget_used >= abs_b:
                        budget_recorded[j][bi] = True
                        sub_g, _ = builder.build(state)
                        classify_queue.append(
                            (sub_g, si['label'], si['seed_type'], budgets_rel_sorted[bi])
                        )

        # 补录未达到 budget 阈值的（frontier 提前耗尽）
        for j, si in enumerate(state_infos):
            for bi in range(len(budgets_rel_sorted)):
                if not budget_recorded[j][bi]:
                    sub_g, _ = builder.build(si['state'])
                    classify_queue.append(
                        (sub_g, si['label'], si['seed_type'], budgets_rel_sorted[bi])
                    )

    # ---- 批量分类所有收集的子图 ----
    for c_start in tqdm(range(0, len(classify_queue), 512), desc="  Classifying"):
        c_end = min(c_start + 512, len(classify_queue))
        chunk = classify_queue[c_start:c_end]

        batch_g = dgl.batch([remove_self_loops(item[0]) for item in chunk]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)

        for j, (_, lbl, seed_type, rel_b) in enumerate(chunk):
            pred = logits[j].argmax(dim=-1).item()
            correct = 1.0 if pred == lbl else 0.0
            results[seed_type][rel_b].append(correct)
            per_class[seed_type][lbl][rel_b].append(correct)

    return results, per_class


def compute_stats(
    split_results: Dict[str, Dict[float, List[float]]],
) -> Dict[str, Dict[str, float]]:
    """将原始正确性列表汇总为 mean/std/n。"""
    stats: Dict[str, Dict[str, float]] = {}
    for kind in ('best', 'worst'):
        stats[kind] = {}
        for rel_b in sorted(BUDGETS_REL):
            accs = split_results.get(kind, {}).get(rel_b, [])
            stats[kind][f'{rel_b:.2f}'] = {
                'mean': float(np.mean(accs)) if accs else 0.0,
                'std': float(np.std(accs)) if accs else 0.0,
                'n': len(accs),
            }
    return stats


def compute_per_class_stats(
    per_class_raw: Dict[str, Dict[int, Dict[float, List[float]]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """汇总 per-class 原始正确性为 mean/std/n。"""
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed_type, class_dict in per_class_raw.items():
        stats[seed_type] = {}
        for cls, budget_dict in class_dict.items():
            cls_str = str(cls)
            stats[seed_type][cls_str] = {}
            for rel_b in sorted(BUDGETS_REL):
                accs = budget_dict.get(rel_b, [])
                stats[seed_type][cls_str][f'{rel_b:.2f}'] = {
                    'mean': float(np.mean(accs)) if accs else 0.0,
                    'std': float(np.std(accs)) if accs else 0.0,
                    'n': len(accs),
                }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='m5/base/rl/asymmetric/ex1/checkpoint_epoch10.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-graphs', type=int, default=0, help='0=全量')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='m5/base/analysis/eval_results_no_self_loop')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode: classifier subgraph WITHOUT self-loop")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    proc = str(root / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')

    # ---- 加载 valid split ----
    csv = root / 'ogbg_ppa' / 'split' / 'species' / 'valid.csv.gz'
    indices = pd.read_csv(csv, compression='gzip', header=None).values.flatten().tolist()
    if args.n_graphs > 0:
        indices = random.sample(indices, args.n_graphs)

    print(f"Loading {len(indices)} graphs ...")
    graphs, labels = dgl.load_graphs(proc, idx_list=indices)
    graphs = [dgl.add_self_loop(g) for g in graphs]

    # ---- 加载 classifier ----
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # ---- 加载 reader ----
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=128, n_layers=2, dropout=0.1, stochastic=False,
    ).to(device)
    r_ckpt = torch.load(str(repo_root / args.checkpoint), map_location=device, weights_only=False)
    sd, _ = patch_old_checkpoint(r_ckpt['reader_state_dict'])
    reader.load_state_dict(sd)
    reader.eval()

    # ---- 评测 ----
    results, per_class_raw = evaluate(reader, classifier, graphs, indices, labels['labels'], device)

    # ---- 汇总统计 ----
    stats = compute_stats(results)
    per_class_stats = compute_per_class_stats(per_class_raw)

    print("\n=== Results (no self-loop) ===")
    total_correct = 0
    total_n = 0
    for kind in ('best', 'worst'):
        for b_key, s in stats[kind].items():
            print(f"  {kind:5s} budget={b_key}: "
                  f"mean={s['mean']:.4f}, std={s['std']:.4f} (n={s['n']})")
            total_correct += s['mean'] * s['n']
            total_n += s['n']
    print(f"  OVERALL: {total_correct / total_n:.4f}  ({total_n} samples)")

    # ---- 输出 JSON ----
    output_json = {
        'valid': stats,
        'per_class_valid': per_class_stats,
        'config': {
            'classifier': str(repo_root / args.classifier),
            'checkpoint': str(repo_root / args.checkpoint),
            'budgets': BUDGETS_REL,
            'seed_method': 'max-degree (best) / min-degree (worst)',
            'mode': 'no_self_loop',
            'seed': args.seed,
        },
    }

    output_path = output_dir / 'verify_no_self_loop.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {output_path}")


if __name__ == '__main__':
    main()

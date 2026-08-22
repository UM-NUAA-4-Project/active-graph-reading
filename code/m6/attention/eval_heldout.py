"""Held-out evaluation for AttentionReader (no classification head).

Evaluates a reader checkpoint on held-out graphs (train and test splits
sequentially, releasing each split's memory before the next),
reporting accuracy at each budget level (5%, 10%, 15%).  Classification is
done entirely by the external GraphClassifier — the selector reader has no
classification head of its own.
Saves results to eval_heldout_{split}.json, structure aligned with
code/m5/base/analysis/eval.py's eval_learnable.json.

用法:
  python m6/attention/eval_heldout.py --checkpoint m6/attention/bc/reader_sel_final.pt   # 依次验证 train + test
  python m6/attention/eval_heldout.py --checkpoint PATH --split train                  # 仅训练集
  python m6/attention/eval_heldout.py --checkpoint PATH --split test --max-graphs 5000 # 仅测试集
  python m6/attention/eval_heldout.py --checkpoint PATH --argmax                       # argmax 选择（默认按分布采样）
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

from common import K, BUDGET_OPTIONS, _project_root, _repo_root, setup_paths
setup_paths()

import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.reader.attention.attention_reader import AttentionReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from m5.training_utils import infer_reader_config, patch_old_checkpoint
from training_utils import (
    deterministic_seed, reveal_and_update_beliefs_batch, strip_classifier_keys,
)


@torch.no_grad()
def evaluate_heldout(reader, classifier, builder, graphs, ogb_indices,
                     labels_tensor, device, chunk_size=64,
                     argmax=False, seed=42, n_seeds=4):
    """Cross-graph lockstep held-out eval, per-budget accuracy.

    Returns:
        (results, per_class_outcomes, per_seed_outcomes, per_seed_per_class)：
        results              — 合并所有 seed 的 acc_*/outcomes；
        per_class_outcomes   — 合并所有 seed 的逐类别 outcomes；
        per_seed_outcomes    — {seed_i: {budget: [0/1...]}}；
        per_seed_per_class   — {seed_i: {budget: {label: [0/1...]}}}。
    """
    reader.eval()

    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    all_subgraphs = []
    all_labels = []
    all_budgets = []
    all_seed_ids = []

    for chunk_start in tqdm(range(0, len(graphs), chunk_size), desc="  Revealing"):
        chunk_end = min(chunk_start + chunk_size, len(graphs))
        chunk_graphs = graphs[chunk_start:chunk_end]
        chunk_indices = ogb_indices[chunk_start:chunk_end]

        # ---- Pass 1: create states (n_graphs × n_budgets × n_seeds) ----
        all_states = []
        state_info = []  # (label_tensor, label_int, budget_ratio, seed_i)

        for local_i in range(len(chunk_graphs)):
            g_raw = chunk_graphs[local_i]
            graph_idx = chunk_indices[local_i]
            label = int(labels_tensor[graph_idx].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()

            for budget_ratio in BUDGET_OPTIONS:
                B = max(1, int(budget_ratio * n_nodes))
                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(graph_idx, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node)
                    state._budget_limit = B
                    all_states.append(state)
                    state_info.append((y, label, budget_ratio, seed_i))

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)
        all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
        cache_subgraphs = [None] * n_total

        # ---- Pass 2: cross-graph lockstep (deterministic argmax) ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break

            active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
            active_subgraphs = None
            if cache_subgraphs[active_idx[0]] is not None:
                active_subgraphs = [cache_subgraphs[j] for j in active_idx]

            batch_results = reader.forward_states(
                [all_states[j] for j in active_idx], temperature=1.0,
                beliefs=active_beliefs,
                subgraphs=active_subgraphs,
            )

            # ---- Phase A: collect chosen nodes ----
            batch_states = []
            batch_nodes = []
            batch_local_idx = []

            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                if argmax:
                    chosen = frontier_nodes[probs.argmax().item()]
                else:
                    chosen = frontier_nodes[
                        torch.distributions.Categorical(probs).sample().item()]
                batch_states.append(all_states[j])
                batch_nodes.append(chosen)
                batch_local_idx.append(j)

            # ---- Phase B: batch reveal + encode + update ----
            if batch_states:
                batch_beliefs = torch.cat(
                    [all_beliefs[j] for j in batch_local_idx], dim=0)
                new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                    reader, batch_states, batch_nodes, batch_beliefs, device,
                )
                for k, j in enumerate(batch_local_idx):
                    all_beliefs[j] = new_beliefs[k:k + 1]
                    cache_subgraphs[j] = new_subgraphs[k]

            # ---- Phase C: check budget exhaustion ----
            for j in active_idx:
                if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        # ---- Pass 3: collect subgraphs ----
        for j in range(n_total):
            sub_g, _ = builder.build(all_states[j])
            all_subgraphs.append(sub_g)
            all_labels.append(state_info[j][:2])  # (y, label)
            all_budgets.append(state_info[j][2])   # budget_ratio
            all_seed_ids.append(state_info[j][3])  # seed_i

    # ---- Batch classify ----
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_class_outcomes = defaultdict(lambda: defaultdict(list))
    per_seed_outcomes = {i: defaultdict(list) for i in range(n_seeds)}
    per_seed_per_class = {i: defaultdict(lambda: defaultdict(list))
                          for i in range(n_seeds)}

    for i in tqdm(range(0, len(all_subgraphs), 512), desc="  Classifying"):
        chunk_sub = all_subgraphs[i:i + 512]
        chunk_labels = all_labels[i:i + 512]
        chunk_budgets = all_budgets[i:i + 512]
        chunk_seed_ids = all_seed_ids[i:i + 512]

        batch_g = dgl.batch(chunk_sub).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)

        for j, (y, label) in enumerate(chunk_labels):
            budget = chunk_budgets[j]
            seed_i = chunk_seed_ids[j]
            correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
            per_budget_correct[budget] += correct
            per_budget_total[budget] += 1
            per_budget_outcomes[budget].append(correct)
            per_class_outcomes[budget][label].append(correct)
            per_seed_outcomes[seed_i][budget].append(correct)
            per_seed_per_class[seed_i][budget][label].append(correct)

    del all_subgraphs, all_labels, all_budgets, all_seed_ids

    results = {}
    for budget in BUDGET_OPTIONS:
        acc = per_budget_correct[budget] / max(per_budget_total[budget], 1)
        results[f'acc_{int(budget * 100)}%'] = acc

    total_correct = sum(per_budget_correct.values())
    total_all = sum(per_budget_total.values())
    results['acc_overall'] = total_correct / max(total_all, 1)
    results['n_graphs'] = len(graphs)
    results['n_trajectories'] = total_all
    results['outcomes'] = {b: list(per_budget_outcomes[b]) for b in BUDGET_OPTIONS}

    return results, per_class_outcomes, per_seed_outcomes, per_seed_per_class


# ---------------------------------------------------------------------------
# 可视化（风格对齐 code/m5/base/analysis/eval.py，不绘制误差棒）
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Held-out eval for attention reader')
    parser.add_argument('--checkpoint', required=True, help='Path to reader checkpoint')
    parser.add_argument('--split', default='both', choices=['train', 'valid', 'test', 'both'],
                        help="Data split to evaluate: 'both' = train + test sequentially (default)")
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='Limit eval graphs (0 = all)')
    parser.add_argument('--m', type=int, default=4,
                        help='number of belief tokens (selection only)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='number of attention heads in FrontierPointer and BeliefState')
    parser.add_argument('--chunk-size', type=int, default=64,
                        help='Graphs per lockstep chunk')
    parser.add_argument('--argmax', action='store_true',
                        help='Select argmax node each step (default: sample from distribution)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for node sampling reproducibility')
    parser.add_argument('--n-seeds', type=int, default=4,
                        help='Number of deterministic seeds per graph (default: 4)')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'eval_results'),
                        help='Directory for eval_heldout_{split}.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')

    # ---- 1. load classifier (frozen, 所有 split 复用) ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(_repo_root / args.classifier), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 2. load reader (复用) ----
    print("\n=== Loading reader ===")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _project_root / args.checkpoint

    reader_ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    reader_sd = reader_ckpt.get('reader_state_dict', reader_ckpt)

    hidden_dim, n_layers = infer_reader_config(reader_sd)
    dropout = reader_ckpt.get('config', {}).get('dropout_reader', 0.1)

    reader = AttentionReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
        m=args.m, n_heads=args.n_heads,
    ).to(device)

    reader_sd_patched, _ = patch_old_checkpoint(reader_sd)
    reader_sd_patched = strip_classifier_keys(reader_sd_patched)
    missing, unexpected = reader.load_state_dict(reader_sd_patched, strict=False)
    if missing:
        print(f"  Warning: missing keys ({len(missing)}), using random init")
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"checkpoint_epoch={reader_ckpt.get('epoch', 'N/A')}")

    builder = SubgraphBuilder()

    def _budget_stats(outcomes):
        arr = np.array(outcomes, dtype=float)
        return {
            'mean': float(arr.mean()) if len(arr) else 0.0,
            'std': float(arr.std()) if len(arr) else 0.0,
            'n': len(arr),
        }

    # ---- 3. evaluate splits sequentially, release memory after each ----
    for split in splits:
        # ---- 3a. load eval graphs ----
        csv_path = root_path / 'ogbg_ppa' / 'split' / 'species' / f'{split}.csv.gz'
        all_indices = pd.read_csv(csv_path, compression='gzip', header=None
                                  ).values.flatten().tolist()
        if args.max_graphs > 0 and args.max_graphs < len(all_indices):
            all_indices = all_indices[:args.max_graphs]

        print(f"\n=== Loading {len(all_indices)} {split} graphs ===")
        eval_graphs, eval_labels = dgl.load_graphs(processed_path, idx_list=all_indices)
        eval_graphs = [dgl.add_self_loop(g) for g in eval_graphs]
        print(f"  Loaded {len(eval_graphs)} graphs (CPU)")

        # ---- 3b. run eval ----
        print(f"\n=== Held-out evaluation ({split}) ===")
        print(f"  Selection: {'argmax' if args.argmax else 'sample'} "
              f"(seed={args.seed}, n_seeds={args.n_seeds})")
        results, per_class_outcomes, per_seed_outcomes, per_seed_per_class = \
            evaluate_heldout(
                reader, classifier, builder,
                eval_graphs, all_indices, eval_labels['labels'],
                device, chunk_size=args.chunk_size,
                argmax=args.argmax, seed=args.seed, n_seeds=args.n_seeds,
            )

        print(f"\n  Graphs evaluated: {results['n_graphs']}")
        print(f"  Trajectories:     {results['n_trajectories']}")
        print(f"  ─────────────────────────")
        for budget in BUDGET_OPTIONS:
            key = f'acc_{int(budget * 100)}%'
            print(f"  Budget {int(budget * 100):2d}%:       {results[key]:.4f}  "
                  f"({results[key] * 100:.2f}%)")
        print(f"  ─────────────────────────")
        print(f"  Overall:         {results['acc_overall']:.4f}  "
              f"({results['acc_overall'] * 100:.2f}%)")

        # ---- 3c. save JSON (aligned with m5/base/analysis/eval.py) ----
        # 每 seed 一个键 + 'mean'（合并所有 seed 的 outcomes 池）
        all_labels = sorted({lbl for b in BUDGET_OPTIONS for lbl in per_class_outcomes[b]})
        per_seed_stats = {}
        per_seed_class_stats = {}
        pooled_outcomes = {b: [] for b in BUDGET_OPTIONS}
        pooled_per_class = {b: defaultdict(list) for b in BUDGET_OPTIONS}
        for seed_i in range(args.n_seeds):
            seed_outcomes = per_seed_outcomes[seed_i]
            per_seed_stats[f's{seed_i}'] = {
                f'{b:.2f}': _budget_stats(seed_outcomes[b]) for b in BUDGET_OPTIONS
            }
            per_seed_class_stats[f's{seed_i}'] = {
                str(lbl): {
                    f'{b:.2f}': _budget_stats(
                        per_seed_per_class[seed_i][b].get(lbl, []))
                    for b in BUDGET_OPTIONS
                }
                for lbl in all_labels
            }
            for b in BUDGET_OPTIONS:
                pooled_outcomes[b] += seed_outcomes[b]
                for lbl in all_labels:
                    pooled_per_class[b][lbl] += per_seed_per_class[seed_i][b].get(lbl, [])
        per_seed_stats['mean'] = {
            f'{b:.2f}': _budget_stats(pooled_outcomes[b]) for b in BUDGET_OPTIONS
        }
        per_seed_class_stats['mean'] = {
            str(lbl): {
                f'{b:.2f}': _budget_stats(pooled_per_class[b][lbl])
                for b in BUDGET_OPTIONS
            }
            for lbl in all_labels
        }

        json_out = {
            split: {'learnable': per_seed_stats},
            f'per_class_{split}': {'learnable': per_seed_class_stats},
            'config': {
                'classifier': str(_repo_root / args.classifier),
                'budgets': BUDGET_OPTIONS,
                'n_seeds': args.n_seeds,
                'seed_method': "deterministic formula (PPA): (idx*31 + seed_i*17 + 7) % n_nodes",
                'full_graph_ref': {'valid': ckpt.get('val_acc', None), 'test': None},
                'checkpoint': str(checkpoint_path),
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'm': args.m,
                'n_heads': args.n_heads,
                'temperature': 1.0,
                'selection': 'argmax' if args.argmax else 'sample',
                'seed': args.seed,
                'split': split,
                'max_graphs': args.max_graphs,
                'chunk_size': args.chunk_size,
                'strategies': ['learnable'],
            },
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'eval_heldout_{split}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_out, f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved to {output_path}")

        # ---- 3d. release memory before next split ----
        del eval_graphs, eval_labels, all_indices, results, per_class_outcomes
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  Memory released after {split} split.")


if __name__ == '__main__':
    main()

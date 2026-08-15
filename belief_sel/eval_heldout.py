"""Held-out evaluation for selector LearnableReader (no classification head).

Evaluates a reader checkpoint on held-out graphs (test or valid split),
reporting accuracy at each budget level (5%, 10%, 15%).  Classification is
done entirely by the external GraphClassifier.
Saves results to eval_heldout_{split}.json, structure aligned with
code/m5/base/analysis/eval.py's eval_learnable.json.

用法:
  python belief_sel/eval_heldout.py --checkpoint belief_sel/bc/reader_sel_final.pt --split test
  python belief_sel/eval_heldout.py --checkpoint belief_sel/rl/checkpoint_epoch5.pt --split valid
  python belief_sel/eval_heldout.py --checkpoint PATH --split test --max-graphs 5000
"""

import argparse
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

from learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from m5.training_utils import infer_reader_config, patch_old_checkpoint
from training_utils import (
    deterministic_seed, reveal_and_update_beliefs_batch, strip_classifier_keys,
)


@torch.no_grad()
def evaluate_heldout(reader, classifier, builder, graphs, ogb_indices,
                     labels_tensor, device, chunk_size=64):
    """Cross-graph lockstep held-out eval, per-budget accuracy."""
    reader.eval()

    all_subgraphs = []
    all_labels = []
    all_budgets = []

    for chunk_start in tqdm(range(0, len(graphs), chunk_size), desc="  Revealing"):
        chunk_end = min(chunk_start + chunk_size, len(graphs))
        chunk_graphs = graphs[chunk_start:chunk_end]
        chunk_indices = ogb_indices[chunk_start:chunk_end]

        # ---- Pass 1: create states (n_graphs × 3 budgets) ----
        all_states = []
        state_info = []  # (label_tensor, label_int, budget_ratio)

        for local_i in range(len(chunk_graphs)):
            g_raw = chunk_graphs[local_i]
            graph_idx = chunk_indices[local_i]
            label = int(labels_tensor[graph_idx].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()

            for budget_ratio in BUDGET_OPTIONS:
                B = max(1, int(budget_ratio * n_nodes))
                seed_node = deterministic_seed(graph_idx, 0, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                all_states.append(state)
                state_info.append((y, label, budget_ratio))

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
                chosen = frontier_nodes[probs.argmax().item()]
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

    # ---- Batch classify ----
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_class_outcomes = defaultdict(lambda: defaultdict(list))

    for i in tqdm(range(0, len(all_subgraphs), 512), desc="  Classifying"):
        chunk_sub = all_subgraphs[i:i + 512]
        chunk_labels = all_labels[i:i + 512]
        chunk_budgets = all_budgets[i:i + 512]

        batch_g = dgl.batch(chunk_sub).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)

        for j, (y, label) in enumerate(chunk_labels):
            budget = chunk_budgets[j]
            correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
            per_budget_correct[budget] += correct
            per_budget_total[budget] += 1
            per_budget_outcomes[budget].append(correct)
            per_class_outcomes[budget][label].append(correct)

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

    return results, per_class_outcomes


def main():
    parser = argparse.ArgumentParser(description='Held-out eval for attention reader')
    parser.add_argument('--checkpoint', required=True, help='Path to reader checkpoint')
    parser.add_argument('--split', default='test', choices=['test', 'valid'],
                        help='Data split to evaluate on')
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
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'eval_results'),
                        help='Directory for eval_heldout_{split}.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 1. load eval graphs ----
    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    split_file = f'{args.split}.csv.gz'
    csv_path = root_path / 'ogbg_ppa' / 'split' / 'species' / split_file

    all_indices = pd.read_csv(csv_path, compression='gzip', header=None
                              ).values.flatten().tolist()
    if args.max_graphs > 0 and args.max_graphs < len(all_indices):
        all_indices = all_indices[:args.max_graphs]

    print(f"\n=== Loading {len(all_indices)} {args.split} graphs ===")
    eval_graphs, eval_labels = dgl.load_graphs(processed_path, idx_list=all_indices)
    eval_graphs = [dgl.add_self_loop(g) for g in eval_graphs]
    print(f"  Loaded {len(eval_graphs)} graphs (CPU)")

    # ---- 2. load classifier (frozen) ----
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

    # ---- 3. load reader ----
    print("\n=== Loading reader ===")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _project_root / args.checkpoint

    reader_ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    reader_sd = reader_ckpt.get('reader_state_dict', reader_ckpt)

    hidden_dim, n_layers = infer_reader_config(reader_sd)
    dropout = reader_ckpt.get('config', {}).get('dropout_reader', 0.1)

    reader = LearnableReader(
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

    # ---- 4. run eval ----
    print(f"\n=== Held-out evaluation ({args.split}) ===")
    results, per_class_outcomes = evaluate_heldout(
        reader, classifier, builder,
        eval_graphs, all_indices, eval_labels['labels'],
        device, chunk_size=args.chunk_size,
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

    # ---- 5. save JSON (aligned with m5/base/analysis/eval.py) ----
    def _budget_stats(outcomes):
        arr = np.array(outcomes, dtype=float)
        return {
            'mean': float(arr.mean()) if len(arr) else 0.0,
            'std': float(arr.std()) if len(arr) else 0.0,
            'n': len(arr),
        }

    outcomes = results['outcomes']
    best_stats = {f'{b:.2f}': _budget_stats(outcomes[b]) for b in BUDGET_OPTIONS}

    all_labels = sorted({lbl for b in BUDGET_OPTIONS for lbl in per_class_outcomes[b]})
    per_class_best = {
        str(lbl): {
            f'{b:.2f}': _budget_stats(per_class_outcomes[b].get(lbl, []))
            for b in BUDGET_OPTIONS
        }
        for lbl in all_labels
    }

    json_out = {
        args.split: {'learnable': {'best': best_stats}},
        f'per_class_{args.split}': {'learnable': {'best': per_class_best}},
        'config': {
            'classifier': str(_repo_root / args.classifier),
            'budgets': BUDGET_OPTIONS,
            'seed_method': "single deterministic seed (seed_i=0, "
                           "same formula as eval.py's s0/'best')",
            'full_graph_ref': {'valid': ckpt.get('val_acc', None), 'test': None},
            'checkpoint': str(checkpoint_path),
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'm': args.m,
            'n_heads': args.n_heads,
            'temperature': 1.0,
            'split': args.split,
            'max_graphs': args.max_graphs,
            'chunk_size': args.chunk_size,
            'strategies': ['learnable'],
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'eval_heldout_{args.split}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {output_path}")


if __name__ == '__main__':
    main()

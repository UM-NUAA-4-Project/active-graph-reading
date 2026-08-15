"""Selector Reader — Self-Sampled REINFORCE（外部分类器 reward）。

Loss = mean_i(-R_i · Σ_{t=1}^{B_i} log π_t(a_t))
R_i 由外部 GraphClassifier 对最终子图分类给出（binary/tanh/asymmetric），
policy 与 reward 来源解耦：R 断图，只更新选择路径。

用法:
  python belief_sel/train_rl.py --bootstrap belief_sel/bc/reader_sel_final.pt --epochs 10
  python belief_sel/train_rl.py --reward binary --epochs 10
  python belief_sel/train_rl.py --bootstrap belief/bc/reader_belief_epoch5.pt --epochs 10
  --reward asymmetric --graphs-per-batch 32 --seeds-per-graph 2 
  --output-dir belief_sel/rl/ex_external
"""

import argparse
import json
import math
import random
from pathlib import Path

from common import K, BUDGET_OPTIONS, _project_root, _repo_root, setup_paths
setup_paths()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.gnn.classifier import GraphClassifier

from m5.training_utils import patch_old_checkpoint
from training_utils import (
    deterministic_seed, reveal_and_update_beliefs_batch, strip_classifier_keys,
)

LN_K = math.log(K)


# ---------------------------------------------------------------------------
# REINFORCE with external classifier reward
# ---------------------------------------------------------------------------

def compute_reward(logits: torch.Tensor, label: torch.Tensor,
                   reward_type: str = 'binary') -> tuple:
    """Compute reward from external classifier logits.

    Returns (R, correct) where:
      binary:     R=1.0 if correct, else 0.0
      tanh:       R=tanh(1 - CE/ln37)
      asymmetric: R=1.0 if correct, else -0.1
    """
    pred = logits.argmax(dim=-1).item()
    correct = 1.0 if pred == label.item() else 0.0

    if reward_type == 'binary':
        R = 1.0 if correct else 0.0
    elif reward_type == 'asymmetric':
        R = 1.0 if correct else -0.1
    else:  # tanh
        ce = F.cross_entropy(logits, label).item()
        R = math.tanh(1.0 - ce / LN_K)

    return R, correct


def train_one_epoch(
    reader, optimizer, classifier,
    graphs, ogb_indices, labels_tensor,
    n_seeds, batch_size, device,
    reward_type='binary',
):
    """REINFORCE with external classifier reward: self-sampled + lockstep.

    Loss per trajectory i (budget B_i):
      L_i = -R_i · Σ_{t=1}^{B_i} log π_t(a_t)

    R_i = reward from external GraphClassifier on the final revealed subgraph
          (binary/tanh/asymmetric).  R is detached — only the selection path
          (frontier_encoder, frontier_pointer, belief_update) receives gradients.

    Loss = mean_i(L_i).
    """
    reader.train()

    total_R = 0.0
    total_trajs = 0
    total_correct = 0
    total_loss = 0.0

    graph_order = list(range(len(graphs)))
    random.shuffle(graph_order)

    micro_batch_size = 16  # process in sub-batches to limit BPTT memory
    pbar = tqdm(range(0, len(graph_order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(graph_order))
        batch_indices = graph_order[batch_start:batch_end]

        optimizer.zero_grad()
        n_micro = (len(batch_indices) + micro_batch_size - 1) // micro_batch_size

        for mb_i in range(0, len(batch_indices), micro_batch_size):
            mb_indices = batch_indices[mb_i:mb_i + micro_batch_size]

            # ---- Pass 1: create states + beliefs (mb only) ----
            all_states = []
            state_labels = []
            per_state_log_probs = []

            for local_i in mb_indices:
                g_raw = graphs[local_i]
                graph_idx = ogb_indices[local_i]
                label = int(labels_tensor[graph_idx].item())
                y = torch.tensor([label], device=device)
                n_nodes = g_raw.num_nodes()

                budget_ratio = random.choice(BUDGET_OPTIONS)
                B = max(1, int(budget_ratio * n_nodes))

                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(graph_idx, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node)
                    state._budget_limit = B
                    all_states.append(state)
                    state_labels.append(y)
                    per_state_log_probs.append([])

            n_total = len(all_states)
            active = [True] * n_total
            max_budget = max(s._budget_limit for s in all_states)
            all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
            all_subgraphs = [None] * n_total

            # ---- Pass 2: cross-graph lockstep ----
            for _step in range(max_budget):
                active_idx = [j for j in range(n_total)
                              if active[j] and not all_states[j].is_frontier_empty]
                if not active_idx:
                    break

                active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
                active_subgraphs = None
                if all_subgraphs[active_idx[0]] is not None:
                    active_subgraphs = [all_subgraphs[j] for j in active_idx]

                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx],
                    temperature=reader.temperature,
                    beliefs=active_beliefs,
                    subgraphs=active_subgraphs,
                )

                # ---- Phase A: sample actions + collect log_probs ----
                batch_states = []
                batch_nodes = []
                batch_local_idx = []

                for k, j in enumerate(active_idx):
                    _, probs, frontier_nodes = batch_results[k]
                    if not frontier_nodes:
                        active[j] = False
                        continue

                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample()
                    per_state_log_probs[j].append(dist.log_prob(action_idx))

                    v_chosen = frontier_nodes[action_idx.item()]
                    batch_states.append(all_states[j])
                    batch_nodes.append(v_chosen)
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
                        all_subgraphs[j] = new_subgraphs[k]

                # ---- Phase C: check budget exhaustion ----
                for j in active_idx:
                    if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                        active[j] = False

            # ---- Pass 3: external classifier rewards ----
            final_states = []
            final_log_probs = []
            final_labels = []
            for j in range(n_total):
                if per_state_log_probs[j]:
                    final_states.append(all_states[j])
                    final_log_probs.append(per_state_log_probs[j])
                    final_labels.append(state_labels[j])

            if not final_states:
                continue

            # Batch external classifier on final subgraphs (zeros-nids protocol)
            subgraph_builder = SubgraphBuilder()
            all_logits = []
            with torch.no_grad():
                for c_start in range(0, len(final_states), 512):
                    chunk = final_states[c_start:c_start + 512]
                    sub_graphs = [subgraph_builder.build(s)[0] for s in chunk]
                    batch_g = dgl.batch(sub_graphs).to(device)
                    nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                                       device=device)
                    all_logits.append(classifier(batch_g, nids))
            logits_all = torch.cat(all_logits, dim=0)  # (n_final, K)

            batch_losses = []
            for k in range(len(final_states)):
                R, correct = compute_reward(
                    logits_all[k:k + 1], final_labels[k], reward_type)

                total_R += R
                total_trajs += 1
                if correct:
                    total_correct += 1

                sum_log_prob = torch.stack(final_log_probs[k]).sum()
                batch_losses.append(-R * sum_log_prob)

            rl_loss = torch.stack(batch_losses).mean()
            loss = rl_loss / n_micro
            loss.backward()

            total_loss += loss.item() * len(batch_losses) * n_micro

        optimizer.step()
        n = max(total_trajs, 1)
        pbar.set_postfix({
            'loss': f'{total_loss / n:.4f}',
            'avg_R': f'{total_R / n:.3f}',
            'acc': f'{total_correct / n:.3f}',
        })

    n = max(total_trajs, 1)
    return {
        'loss': total_loss / n,
        'avg_R': total_R / n,
        'acc': total_correct / n,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Selector Reader — Self-Sampled REINFORCE (External Classifier Reward)')
    parser.add_argument('--bootstrap-checkpoint',
                        default='belief_sel/bc/reader_sel_final.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--graphs-per-batch', type=int, default=16)
    parser.add_argument('--seeds-per-graph', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--m', type=int, default=4,
                        help='number of belief tokens (shared for selection and classification)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='number of attention heads in FrontierPointer and BeliefState')
    parser.add_argument('--max-graphs', type=int, default=0)
    parser.add_argument('--output-dir', default='belief_sel/rl')
    parser.add_argument('--resume-checkpoint', default='')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt',
                        help='External classifier checkpoint (relative to code/)')
    parser.add_argument('--reward', default='binary',
                        choices=['binary', 'tanh', 'asymmetric'],
                        help='reward type: binary=0/1, tanh=tanh(1-CE/ln37), '
                             'asymmetric=+1/-0.1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load training graphs ----
    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'

    train_indices = pd.read_csv(train_csv, compression='gzip', header=None
                                ).values.flatten().tolist()
    if args.max_graphs > 0 and args.max_graphs < len(train_indices):
        train_indices = train_indices[:args.max_graphs]

    print(f"\n=== Loading {len(train_indices)} training graphs ===")
    train_graphs, train_labels = dgl.load_graphs(processed_path, idx_list=train_indices)
    train_graphs = [dgl.add_self_loop(g) for g in train_graphs]
    print(f"  Loaded {len(train_graphs)} graphs (CPU)")

    # ---- 2. load reader ----
    print("\n=== Loading LearnableReader (Phase 2) ===")
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, dropout=args.dropout,
        stochastic=True, temperature=args.temperature,
        m=args.m,
        n_heads=args.n_heads,
    ).to(device)

    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    training_log = []
    start_epoch = 0

    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint
        if not Path(resume_path).is_absolute():
            resume_path = _project_root / args.resume_checkpoint
        else:
            resume_path = Path(resume_path)
        print(f"  Resuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        reader_sd, was_patched = patch_old_checkpoint(ckpt['reader_state_dict'])
        reader_sd = strip_classifier_keys(reader_sd)
        missing, unexpected = reader.load_state_dict(reader_sd, strict=False)
        if missing:
            print(f"  New params (random init): {len(missing)} keys")
        if was_patched:
            print("  NOTE: checkpoint architecture patched, using fresh optimizer")
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, RuntimeError):
                print("  WARNING: optimizer state incompatible, using fresh optimizer")
        training_log = ckpt.get('training_log', [])
        start_epoch = ckpt['epoch'] + 1
    else:
        bootstrap_path = _project_root / args.bootstrap_checkpoint
        if bootstrap_path.exists():
            print(f"  Loading BC bootstrap from {bootstrap_path}")
            reader_ckpt = torch.load(str(bootstrap_path), map_location=device,
                                     weights_only=False)
            reader_sd, _ = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
            reader_sd = strip_classifier_keys(reader_sd)
            missing, unexpected = reader.load_state_dict(reader_sd, strict=False)
            if missing:
                print(f"  New RL params (random init): {len(missing)} keys")
            print(f"  BC checkpoint epoch: {reader_ckpt.get('epoch', 'N/A')}")
        else:
            print(f"  WARNING: {bootstrap_path} not found, training from scratch")

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Reader params: {total_params:,}")

    # ---- 2.5 load external classifier (frozen, reward source) ----
    print("\n=== Loading external classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    cls_ckpt = torch.load(str(_repo_root / args.classifier), map_location=device,
                          weights_only=False)
    classifier.load_state_dict(cls_ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={cls_ckpt.get('val_acc', 'N/A')}")
    print(f"  Belief tokens: {args.m}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Reward: {args.reward}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Graphs/batch: {args.graphs_per_batch}, Seeds/graph: {args.seeds_per_graph}")

    # Save config
    config = {
        'hidden_dim': reader.frontier_encoder.hidden_dim,
        'n_layers': reader.frontier_encoder.n_layers,
        'dropout': reader.frontier_encoder.dropout,
        'm': args.m,
        'n_heads': args.n_heads,
        'temperature': args.temperature,
        'reward_type': args.reward,
        'classifier': str(_repo_root / args.classifier),
        'lr': args.lr,
        'graphs_per_batch': args.graphs_per_batch,
        'seeds_per_graph': args.seeds_per_graph,
        'budget_options': BUDGET_OPTIONS,
        'epochs': args.epochs,
        'total_params': total_params,
    }
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config saved: {output_dir / 'config.json'}")

    # ---- 5. training loop ----
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        metrics = train_one_epoch(
            reader, optimizer, classifier,
            train_graphs, train_indices, train_labels['labels'],
            args.seeds_per_graph, args.graphs_per_batch, device,
            reward_type=args.reward,
        )
        print(f"  Train: loss={metrics['loss']:.4f}  "
              f"avg_R={metrics['avg_R']:.3f}  acc={metrics['acc']:.3f}")

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': metrics['loss'],
            'train_avg_R': metrics['avg_R'],
            'train_acc': metrics['acc'],
        }
        training_log.append(log_entry)

        ckpt = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': {
                'hidden_dim': reader.frontier_encoder.hidden_dim,
                'lr': args.lr,
                'temperature': args.temperature,
                'reward_type': args.reward,
                'graphs_per_batch': args.graphs_per_batch,
                'seeds_per_graph': args.seeds_per_graph,
                'm': args.m,
                'n_heads': args.n_heads,
            },
        }
        ckpt_path = output_dir / f'checkpoint_epoch{epoch + 1}.pt'
        torch.save(ckpt, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nREINFORCE complete. Output: {output_dir}")


if __name__ == '__main__':
    main()

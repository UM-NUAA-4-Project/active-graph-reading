"""Attention-Enhanced LearnableReader — Behavior Cloning 训练（Phase 2: Belief State）。

基于 M4 teacher 轨迹训练 FrontierPointer + BeliefState。用法:
  python belief/train_bc.py --epochs 10 --lr 1e-3
  python belief/train_bc.py --finetune PATH --lr 1e-4 --epochs 5
"""

import argparse
import json
import random
from pathlib import Path

from common import K, _project_root, _repo_root, setup_paths
setup_paths()

import torch
import torch.nn.functional as F
from tqdm import tqdm

from learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState

from m5.training_utils import (
    merge_trajectories, load_graphs_for_trajectories,
    infer_reader_config, patch_old_checkpoint,
)
from training_utils import reveal_and_update_belief, reveal_and_update_beliefs_batch


def get_alpha_for_epoch(epoch, total_epochs, schedule):
    """Map epoch to alpha from evenly-distributed schedule.

    e.g. schedule=[0.9, 0.8, 0.7] with 9 epochs → epochs 0-2:0.9, 3-5:0.8, 6-8:0.7
    """
    n = len(schedule)
    idx = min(epoch * n // total_epochs, n - 1)
    return schedule[idx]


# ---------------------------------------------------------------------------
# Phase 2: belief-aware BC training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    reader, optimizer, graph_cache, train_trajs,
    batch_size, device, epoch, total_epochs,
    temperature=1.0, alpha=0.7,
):
    """Single epoch BC training with belief state + label supervision.

    Lockstep pattern: Pass 1 create states+beliefs, Pass 2 cross-graph lockstep
    with belief scoring, Pass 3 α·L_BC + (1-α)·L_cls backward.

    L_cls = CE(reader_logits, label) — direct ground-truth supervision.
    """
    reader.train()

    total_loss = 0.0
    total_bc = 0.0
    total_cls = 0.0
    total_correct = 0
    total_samples = 0
    total_steps = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        # ---- Pass 1: create states + beliefs ----
        all_states = []
        reveal_orders = []
        step_counters = []
        state_labels = []

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            all_states.append(state)
            reveal_orders.append(traj.reveal_order)
            step_counters.append(0)
            state_labels.append(label)

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(len(ro) for ro in reveal_orders)
        all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
        all_subgraphs = [None] * n_total

        step_losses = []

        # ---- Pass 2: cross-graph lockstep ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty
                          and step_counters[j] < len(reveal_orders[j])]
            if not active_idx:
                break

            active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
            active_subgraphs = None
            if all_subgraphs[active_idx[0]] is not None:
                active_subgraphs = [all_subgraphs[j] for j in active_idx]

            batch_results = reader.forward_states(
                [all_states[j] for j in active_idx],
                temperature=temperature,
                beliefs=active_beliefs,
                subgraphs=active_subgraphs,
            )

            # ---- Phase A: collect actions + BC losses ----
            batch_states = []
            batch_nodes = []
            batch_local_idx = []  # maps batch_states index → active_idx index

            for k, j in enumerate(active_idx):
                scores, probs, frontier_nodes, logits = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue

                teacher_node = reveal_orders[j][step_counters[j]]
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    active[j] = False
                    continue

                step_losses.append(-torch.log(probs[target_idx] + 1e-8))
                step_counters[j] += 1

                batch_states.append(all_states[j])
                batch_nodes.append(teacher_node)
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
                if active[j] and step_counters[j] >= len(reveal_orders[j]):
                    active[j] = False

        if not step_losses:
            continue

        # ---- Pass 3: L_BC + (1-α)·L_cls ----
        bc_loss = torch.stack(step_losses).mean()

        # Classification loss: CE against ground-truth label
        final_beliefs = []
        final_labels = []
        for j in range(n_total):
            if reveal_orders[j]:
                final_beliefs.append(all_beliefs[j])
                final_labels.append(state_labels[j])

        if len(final_beliefs) > 0:
            h_batch = torch.cat(final_beliefs, dim=0)          # (n_valid, m, d)
            reader_logits = reader.classify(h_batch)            # (n_valid, K)
            label_tensor = torch.tensor(final_labels, device=device)

            cls_loss = F.cross_entropy(reader_logits, label_tensor)

            reader_pred = reader_logits.argmax(dim=-1)          # (n_valid,)
            total_correct += (reader_pred == label_tensor).sum().item()
            total_samples += len(reader_pred)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        loss = alpha * bc_loss + (1 - alpha) * cls_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = len(step_losses)
        bc_val = bc_loss.item()
        cls_val = cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
        loss_val = loss.item()
        total_loss += loss_val * n
        total_bc += bc_val * n
        total_cls += cls_val * n
        total_steps += n

        avg_loss = total_loss / max(total_steps, 1)
        n_steps = max(total_steps, 1)
        acc = total_correct / max(total_samples, 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'bc': f'{total_bc / n_steps:.4f}',
            'cls': f'{total_cls / n_steps:.4f}',
            'acc': f'{acc:.3f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
        })

    n = max(total_steps, 1)
    return {
        'loss': total_loss / n, 'bc': total_bc / n, 'cls': total_cls / n,
        'acc': total_correct / max(total_samples, 1),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BC Training for Attention LearnableReader (Phase 2)')
    parser.add_argument('--data-dir', default='m4/data/stage7_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='belief/bc')
    parser.add_argument('--n-layers-reader', type=int, default=2, dest='n_layers_reader')
    parser.add_argument('--dropout-reader', type=float, default=0.25, dest='dropout_reader')
    parser.add_argument('--m', type=int, default=4,
                        help='number of belief tokens (shared for selection and classification)')
    parser.add_argument('--resume', default='')
    parser.add_argument('--finetune', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teacher', default='',
                        help='filter by teacher type, comma-separated; '
                             'empty=all. e.g. gradient,hybrid-a0.70')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='softmax temperature for node selection')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='BC loss weight: loss = alpha*BC + (1-alpha)*CLS '
                             '(ignored when --alpha-schedule is set)')
    parser.add_argument('--alpha-schedule', default='',
                        help='comma-separated alpha values evenly distributed '
                             'across epochs, e.g. "0.9,0.8,0.7"')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='number of attention heads in FrontierPointer and BeliefState')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    is_finetune = bool(args.finetune)
    resume_path = args.finetune if is_finetune else args.resume

    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load trajectories & graphs ----
    data_dir = _repo_root / args.data_dir
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir)

    if args.teacher:
        teachers = [t.strip() for t in args.teacher.split(',')]
        all_trajs = [t for t in all_trajs if t.teacher_type in teachers]
        from collections import Counter
        cnt = Counter(t.teacher_type for t in all_trajs)
        for k, v in sorted(cnt.items()):
            print(f"  Filtered: {k} → {v} trajectories")
        print(f"Total after filter: {len(all_trajs)} trajectories")

    print("\n=== Loading graphs ===")
    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    train_trajs = all_trajs
    print(f"\nUsing {len(train_trajs)} trajectories for training")

    # ---- 2. create / resume Reader ----
    print("\n=== Creating LearnableReader (attention + belief) ===")
    start_epoch = 0
    training_log = []

    if resume_path:
        print(f"  Loading from {resume_path}")
        reader_ckpt = torch.load(str(resume_path), map_location='cpu', weights_only=False)

        if is_finetune:
            hidden_dim, n_layers = infer_reader_config(reader_ckpt['reader_state_dict'])
            print(f"  Inferred: hidden_dim={hidden_dim}, n_layers={n_layers}")
            dropout = args.dropout_reader
        else:
            hidden_dim = args.hidden_dim
            n_layers = args.n_layers_reader
            dropout = args.dropout_reader

        print(f"  Checkpoint epoch={reader_ckpt.get('epoch', 'N/A')}")

        reader = LearnableReader(
            edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers,
            dropout=dropout, m=args.m,
            n_heads=args.n_heads,
        ).to(device)
        reader_sd, was_patched = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
        missing, unexpected = reader.load_state_dict(reader_sd, strict=False)
        if missing:
            belief_keys = [k for k in missing if 'belief_state' in k]
            ptr_keys = [k for k in missing if 'frontier_pointer.cross_attn' in k or
                       'frontier_pointer.norm_cross' in k or 'frontier_pointer.score_proj' in k]
            other_keys = [k for k in missing if k not in belief_keys + ptr_keys]
            if belief_keys:
                print(f"  New belief params (random init): {len(belief_keys)} keys")
            if ptr_keys:
                print(f"  New pointer params (random init): {len(ptr_keys)} keys")
            if other_keys:
                print(f"  Other missing: {other_keys}")
        if unexpected:
            print(f"  Unused old params: {[k for k in unexpected if 'reader_mlp' in k or 'query' in k]}")

        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
        if was_patched:
            print("  NOTE: checkpoint architecture patched, using fresh optimizer")
        elif not missing:
            try:
                optimizer.load_state_dict(reader_ckpt['optimizer_state_dict'])
            except (KeyError, ValueError, RuntimeError) as e:
                print(f"  WARNING: optimizer state unavailable ({e}), using fresh optimizer")

        start_epoch = reader_ckpt.get('epoch', -1) + 1
        training_log = reader_ckpt.get('training_log', [])
    else:
        reader = LearnableReader(
            edge_feat_dim=7, hidden_dim=args.hidden_dim,
            n_layers=args.n_layers_reader, dropout=args.dropout_reader,
            m=args.m,
            n_heads=args.n_heads,
        ).to(device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Params: {total_params:,}")

    # Parse alpha schedule
    alpha_schedule = None
    if args.alpha_schedule:
        alpha_schedule = [float(x.strip()) for x in args.alpha_schedule.split(',')]
        print(f"  Alpha schedule: {alpha_schedule}")

    # ---- 3. training loop ----
    mode = "Fine-tuning" if is_finetune else ("Resuming" if resume_path else "Training")
    print(f"\n=== {mode}: epoch {start_epoch + 1} → {args.epochs}, lr={args.lr} ===")
    print(f"  Train trajectories: {len(train_trajs)}")
    print(f"  Belief tokens: {args.m}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Loss: {args.alpha}*BC + {1 - args.alpha:.1f}*CE(label)")
    if alpha_schedule:
        print(f"  Alpha schedule: {alpha_schedule}")
    if args.teacher:
        print(f"  Teacher filter: {args.teacher}")

    # Save config
    config = {
        'hidden_dim': reader.frontier_encoder.hidden_dim,
        'n_layers_reader': reader.frontier_encoder.n_layers,
        'dropout_reader': reader.frontier_encoder.dropout,
        'm': args.m,
        'n_heads': args.n_heads,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'alpha_schedule': alpha_schedule,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'teacher_filter': args.teacher or 'all',
        'total_params': total_params,
    }
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config saved: {output_dir / 'config.json'}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        alpha = get_alpha_for_epoch(epoch, args.epochs, alpha_schedule) \
                if alpha_schedule else args.alpha
        if alpha_schedule:
            print(f"  alpha={alpha:.2f}")

        metrics = train_one_epoch(
            reader, optimizer, graph_cache, train_trajs,
            args.batch_size, device, epoch, args.epochs,
            temperature=args.temperature,
            alpha=alpha,
        )
        print(f"  Train: loss={metrics['loss']:.4f}  "
              f"bc={metrics['bc']:.4f}  cls={metrics['cls']:.4f}  "
              f"acc={metrics['acc']:.3f}")

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': metrics['loss'],
            'train_bc': metrics['bc'],
            'train_cls': metrics['cls'],
            'train_acc': metrics['acc'],
        }
        training_log.append(log_entry)

        checkpoint = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': {
                'hidden_dim': reader.frontier_encoder.hidden_dim,
                'lr': args.lr, 'batch_size': args.batch_size,
                'n_layers_reader': reader.frontier_encoder.n_layers,
                'dropout_reader': reader.frontier_encoder.dropout,
                'm': args.m,
                'n_heads': args.n_heads,
                'temperature': args.temperature,
            },
        }

        prefix = 'reader_belief'
        ckpt_path = output_dir / f'{prefix}_epoch{epoch + 1}.pt'
        torch.save(checkpoint, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    final_path = output_dir / 'reader_belief_final.pt'
    checkpoint['epoch'] = args.epochs - 1
    torch.save(checkpoint, str(final_path))
    print(f"\n{mode} complete. Final model: {final_path}")


if __name__ == '__main__':
    main()

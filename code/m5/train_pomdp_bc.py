"""Phase 2: POMDP Reader BC 联合训练。

BC Loss = (1/B) Σ CE(π_t, a*_t) + β · CE(b_B^total, y)

从 Phase 0+1 checkpoint 加载 GSAE + EvidenceNet。
"""

import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_graph_reader.reader.pomdp_reader import POMDPReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node

from training_utils import (
    TEACHER_NAMES, BUDGET_RATIOS,
    merge_trajectories, load_graphs_for_trajectories,
)


def train_one_epoch(reader, optimizer, graph_cache, train_trajs,
                    batch_size, device, beta_belief, freeze_gsae):
    reader.train()
    if freeze_gsae:
        reader.gsae.eval()
        for p in reader.gsae.parameters():
            p.requires_grad = False

    total_loss = 0.0

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        batch_losses = []

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            y = torch.tensor(label, device=device)

            reader.reset()
            state = GraphState(g, seed_node=traj.seed_node)
            step_losses = []

            for teacher_node in traj.reveal_order:
                scores, probs, frontier_nodes = reader.forward(state)
                nf = len(frontier_nodes)

                if nf == 0:
                    break

                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    break

                loss_t = F.cross_entropy(
                    scores.unsqueeze(0),
                    torch.tensor([target_idx], device=device),
                )
                step_losses.append(loss_t)
                reveal_node(state, teacher_node)

            # 终端信念 loss
            final_out = reader.get_terminal_belief(state)
            belief_loss = F.cross_entropy(
                final_out.b_total,
                y.unsqueeze(0),
            )

            if step_losses:
                bc_loss = torch.stack(step_losses).mean()
                traj_loss = bc_loss + beta_belief * belief_loss
                batch_losses.append(traj_loss)

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item() * len(batch_losses)
        pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

    return total_loss / max(len(train_trajs), 1)


def main():
    parser = argparse.ArgumentParser(description='Phase 2: POMDP Reader BC training')
    parser.add_argument('--gsae-checkpoint', default='m5/phase01/checkpoint_epoch20.pt')
    parser.add_argument('--data-dir', default='m4/data/stage5_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--beta-belief', type=float, default=0.5)
    parser.add_argument('--freeze-gsae', action='store_true')
    parser.add_argument('--limit-trajs', type=int, default=0)
    parser.add_argument('--output-dir', default='m5/phase2/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repo_root = _script_dir.parent
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载轨迹和图
    data_dir = repo_root / args.data_dir
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir)
    if args.limit_trajs > 0:
        all_trajs = all_trajs[:args.limit_trajs]
        print(f"  Limited to {len(all_trajs)} trajectories")

    print("\n=== Loading graphs ===")
    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    # 2. 创建 POMDPReader + 加载 Phase 0+1 checkpoint
    print("\n=== Creating POMDPReader ===")
    reader = POMDPReader(edge_feat_dim=7, hidden_dim=300, k_state=128,
                         k_evidence=37, n_layers=3, dropout=0.5).to(device)

    ckpt_path = repo_root / args.gsae_checkpoint
    if ckpt_path.exists():
        print(f"  Loading GSAE+Evidence from {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        reader.gsae.load_state_dict(ckpt['gsae_state_dict'])
        reader.evidence_net.load_state_dict(ckpt['evidence_state_dict'])
        print(f"  Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    else:
        print(f"  WARNING: {ckpt_path} not found, training from scratch")

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    training_log = []

    # 3. 训练
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg_loss = train_one_epoch(
            reader, optimizer, graph_cache, all_trajs,
            args.batch_size, device, args.beta_belief, args.freeze_gsae,
        )
        print(f"  Avg loss: {avg_loss:.4f}")

        training_log.append({'epoch': epoch + 1, 'avg_loss': avg_loss})

        ckpt = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
        }
        ckpt_path = output_dir / f'checkpoint_epoch{epoch + 1}.pt'
        torch.save(ckpt, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 2 complete. Output: {output_dir}")


if __name__ == '__main__':
    main()

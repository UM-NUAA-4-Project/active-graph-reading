"""Phase 2: POMDP Reader BC 联合训练。

BC Loss = (1/B) Σ CE(π_t, a*_t) + β · CE(b_B^total, y)

从 Phase 0+1 checkpoint 加载 GSAE + EvidenceNet。
"""

import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_graph_reader.reader.pomdp_reader import POMDPReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node

from training_utils import (
    TEACHER_NAMES, BUDGET_RATIOS,
    merge_trajectories, load_graphs_for_trajectories,
    split_trajectories,
)


def train_one_epoch(reader, optimizer, graph_cache, train_trajs,
                    batch_size, device, beta_belief, freeze_gsae):
    reader.train()
    if freeze_gsae:
        reader.gsae.eval()
        for p in reader.gsae.parameters():
            p.requires_grad = False

    sum_bc = 0.0
    sum_belief = 0.0
    sum_total = 0.0
    n_trajs = 0

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
                scores, probs, frontier_nodes = reader.forward(state, action_override=teacher_node)
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
                sum_bc += bc_loss.item()
                sum_belief += belief_loss.item()
                sum_total += traj_loss.item()
                n_trajs += 1

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = max(n_trajs, 1)
        pbar.set_postfix({
            'bc': f'{sum_bc / n:.4f}',
            'belief': f'{sum_belief / n:.4f}',
            'total': f'{sum_total / n:.4f}',
        })

    return {
        'bc': sum_bc / max(n_trajs, 1),
        'belief': sum_belief / max(n_trajs, 1),
        'total': sum_total / max(n_trajs, 1),
    }


@torch.no_grad()
def val_one_epoch(reader, graph_cache, val_trajs, batch_size, device, beta_belief):
    """单轮 BC 验证（POMDP），返回各 loss 均值。"""
    reader.eval()
    sum_bc = 0.0
    sum_belief = 0.0
    sum_total = 0.0
    n_trajs = 0

    pbar = tqdm(range(0, len(val_trajs), batch_size), desc="  Validating")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(val_trajs))
        batch_trajs = val_trajs[batch_start:batch_end]

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            y = torch.tensor(label, device=device)

            reader.reset()
            state = GraphState(g, seed_node=traj.seed_node)
            step_losses = []

            for teacher_node in traj.reveal_order:
                scores, probs, frontier_nodes = reader.forward(
                    state, action_override=teacher_node)
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

            final_out = reader.get_terminal_belief(state)
            belief_loss = F.cross_entropy(final_out.b_total, y.unsqueeze(0))

            if step_losses:
                bc_val = torch.stack(step_losses).mean().item()
                bl_val = belief_loss.item()
                sum_bc += bc_val
                sum_belief += bl_val
                sum_total += bc_val + beta_belief * bl_val
                n_trajs += 1

    n = max(n_trajs, 1)
    return {'bc': sum_bc / n, 'belief': sum_belief / n, 'total': sum_total / n}


def main():
    parser = argparse.ArgumentParser(description='Phase 2: POMDP Reader BC training')
    parser.add_argument('--gsae-checkpoint', default='m5/phase01/checkpoint_epoch20.pt')
    parser.add_argument('--data-dir', default='m4/data/stage7_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--beta-belief', type=float, default=0.5)
    parser.add_argument('--freeze-gsae', action='store_true')
    parser.add_argument('--limit-trajs', type=int, default=0)
    parser.add_argument('--output-dir', default='m5/pomdp/phase2/')
    parser.add_argument('--no-struct', action='store_true', help='禁用结构特征+结构注意力池化')
    parser.add_argument('--stochastic', action='store_true', help='使用随机策略')
    parser.add_argument('--resume-checkpoint', default='', help='从指定 checkpoint 恢复训练')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (0 = all train, no val)')
    args = parser.parse_args()

    repo_root = _repo_root
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

    # 划分 train/val
    if args.val_split > 0:
        print("\n=== Splitting trajectories ===")
        train_trajs, val_trajs = split_trajectories(
            all_trajs, args.root, val_ratio=args.val_split, random_seed=args.seed,
        )
    else:
        train_trajs, val_trajs = all_trajs, []
        print(f"\nUsing all {len(train_trajs)} trajectories for training (no val split)")

    # 2. 创建 POMDPReader
    print("\n=== Creating POMDPReader ===")
    reader = POMDPReader(edge_feat_dim=7, hidden_dim=args.hidden_dim, k_state=128,
                         k_evidence=37, n_layers=args.n_layers, dropout=args.dropout,
                         use_struct=not args.no_struct, stochastic=args.stochastic).to(device)

    # 初始化
    start_epoch = 0
    training_log = []

    if args.resume_checkpoint:
        # 从 Phase 2 checkpoint 恢复
        resume_path = repo_root / args.resume_checkpoint
        print(f"\n=== Resuming from {resume_path} ===")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        reader.load_state_dict(ckpt['reader_state_dict'])
        training_log = ckpt.get('training_log', [])
        start_epoch = ckpt['epoch'] + 1
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError:
            print("  WARNING: optimizer state incompatible, starting with fresh optimizer")
        print(f"  Resumed at epoch {start_epoch + 1}/{args.epochs}")
    else:
        # 全新训练：加载 Phase 0+1 checkpoint 初始化 GSAE + EvidenceNet
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
        ckpt_path = repo_root / args.gsae_checkpoint
        if ckpt_path.exists():
            print(f"  Loading GSAE+Evidence from {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            reader.gsae.load_state_dict(ckpt['gsae_state_dict'])
            reader.evidence_net.load_state_dict(ckpt['evidence_state_dict'])
            print(f"  Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
        else:
            print(f"  WARNING: {ckpt_path} not found, training from new initialization")


    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Total params: {total_params:,}")

    # 3. 训练
    print(f"  Train trajectories: {len(train_trajs)}")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train = train_one_epoch(
            reader, optimizer, graph_cache, train_trajs,
            args.batch_size, device, args.beta_belief, args.freeze_gsae,
        )
        print(f"  Train: bc={train['bc']:.4f}  belief={train['belief']:.4f}  total={train['total']:.4f}")

        log_entry = {'epoch': epoch + 1, **{f'train_{k}': v for k, v in train.items()}}

        if val_trajs:
            val = val_one_epoch(
                reader, graph_cache, val_trajs,
                args.batch_size, device, args.beta_belief,
            )
            print(f"  Val:   bc={val['bc']:.4f}  belief={val['belief']:.4f}  total={val['total']:.4f}")
            log_entry.update({f'val_{k}': v for k, v in val.items()})

        training_log.append(log_entry)

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

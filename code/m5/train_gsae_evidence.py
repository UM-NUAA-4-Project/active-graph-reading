"""Phase 0+1: GSAE + EvidenceNetwork 联合训练。

Loss = ||v̂ - v||² + λ · EDL_loss(e, y)

可选 --fclass-init 从 f_class 初始化 GSAE 前 3 层。
"""

import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))

import torch
from tqdm import tqdm

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder

from training_utils import (
    TEACHER_NAMES, BUDGET_RATIOS,
    merge_trajectories, load_graphs_for_trajectories, edl_loss,
)


def train_one_epoch(gsae, evidence_net, optimizer, graph_cache, train_trajs,
                    batch_size, device, lambda_edl, lambda_kl):
    gsae.train()
    evidence_net.train()
    builder = FrontierGraphBuilder()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        batch_losses = []
        batch_steps = 0

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            y = torch.zeros(evidence_net.k_evidence, device=device)
            y[label] = 1.0

            state = GraphState(g, seed_node=traj.seed_node)
            traj_losses = []

            for node in traj.reveal_order:
                subgraph = builder.build(state)
                gsae_out = gsae(subgraph)

                rec_loss = ((gsae_out.v_hat - gsae_out.v) ** 2).mean()

                ev_out = evidence_net(gsae_out.z)
                edl = edl_loss(ev_out.e, ev_out.alpha, ev_out.S,
                               y.unsqueeze(0), evidence_net.k_evidence, lambda_kl)

                step_loss = rec_loss + lambda_edl * edl
                traj_losses.append(step_loss)

                reveal_node(state, node)

            if traj_losses:
                traj_loss = torch.stack(traj_losses).mean()
                batch_losses.append(traj_loss)
                batch_steps += len(traj_losses)

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item() * len(batch_losses)
        total_steps += batch_steps

        pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

    return total_loss / max(total_steps, 1)


def main():
    parser = argparse.ArgumentParser(description='Phase 0+1: GSAE + EvidenceNet joint training')
    parser.add_argument('--data-dir', default='m4/data/stage5_filtered')
    parser.add_argument('--fclass-init', default='', help='f_class checkpoint (optional)')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--lambda-edl', type=float, default=1.0)
    parser.add_argument('--lambda-kl', type=float, default=0.01)
    parser.add_argument('--limit-trajs', type=int, default=0, help='0=all')
    parser.add_argument('--output-dir', default='m5/phase01/')
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

    # 2. 创建 GSAE + EvidenceNet
    print("\n=== Creating models ===")
    gsae = GSAE(edge_feat_dim=7, hidden_dim=args.hidden_dim, k_state=128,
                n_layers=3, dropout=0.5).to(device)
    if args.fclass_init:
        print(f"  Initializing GSAE from {args.fclass_init}")
        gsae.init_from_fclass(str(repo_root / args.fclass_init), device)

    evidence_net = EvidenceNetwork(k_state=128, k_evidence=37).to(device)

    total_params = sum(p.numel() for p in gsae.parameters()) + \
                   sum(p.numel() for p in evidence_net.parameters())
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(gsae.parameters()) + list(evidence_net.parameters()), lr=args.lr)
    training_log = []

    # 3. 训练
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg_loss = train_one_epoch(
            gsae, evidence_net, optimizer, graph_cache, all_trajs,
            args.batch_size, device, args.lambda_edl, args.lambda_kl,
        )
        print(f"  Avg loss: {avg_loss:.4f}")

        training_log.append({'epoch': epoch + 1, 'avg_loss': avg_loss})

        ckpt = {
            'epoch': epoch,
            'gsae_state_dict': gsae.state_dict(),
            'evidence_state_dict': evidence_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': {
                'hidden_dim': args.hidden_dim,
                'lr': args.lr,
                'lambda_edl': args.lambda_edl,
                'lambda_kl': args.lambda_kl,
            },
        }
        ckpt_path = output_dir / f'checkpoint_epoch{epoch + 1}.pt'
        torch.save(ckpt, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 0+1 complete. Output: {output_dir}")


if __name__ == '__main__':
    main()

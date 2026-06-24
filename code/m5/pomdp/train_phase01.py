"""Phase 0+1: GSAE + EvidenceNetwork 联合训练。

Loss = CrossEntropy(b, y)

EvidenceNetwork 吸收原 GSAE encoder，形成 300→128→37 两层投影。
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

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder

from training_utils import (
    TEACHER_NAMES, BUDGET_RATIOS,
    merge_trajectories, load_graphs_for_trajectories,
)


def train_one_epoch(gsae, evidence_net, optimizer, graph_cache, train_trajs,
                    batch_size, device):
    gsae.train()
    evidence_net.train()
    builder = FrontierGraphBuilder()
    sum_ce = 0.0
    sum_correct = 0
    total_trajs = 0

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        batch_losses = []

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            label_t = torch.tensor(label, device=device)

            state = GraphState(g, seed_node=traj.seed_node)
            for node in traj.reveal_order:
                reveal_node(state, node)

            # 只用最终子图（B 步揭示后）计算损失
            subgraph = builder.build(state)
            gsae_out = gsae(subgraph)
            ev_out = evidence_net(gsae_out.v)
            b = ev_out.b.squeeze(0)  # (K,)
            ce_loss = F.cross_entropy(b.unsqueeze(0), label_t.unsqueeze(0))

            batch_losses.append(ce_loss)
            sum_ce += ce_loss.item()
            sum_correct += int(b.argmax().item() == label)
            total_trajs += 1

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = max(total_trajs, 1)
        pbar.set_postfix({
            'ce': f'{sum_ce / n:.4f}',
            'acc': f'{sum_correct / n:.4f}',
        })

    return {
        'ce': sum_ce / max(total_trajs, 1),
        'acc': sum_correct / max(total_trajs, 1),
    }


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
    parser.add_argument('--limit-trajs', type=int, default=0, help='0=all')
    parser.add_argument('--output-dir', default='m5/pomdp/phase01/')
    parser.add_argument('--seed', type=int, default=42)
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

    # 2. 创建 GSAE + EvidenceNet
    print("\n=== Creating models ===")
    gsae = GSAE(edge_feat_dim=7, hidden_dim=args.hidden_dim, k_state=128,
                n_layers=2, dropout=0.5).to(device)
    if args.fclass_init:
        print(f"  Initializing GSAE from {args.fclass_init}")
        gsae.init_from_fclass(str(repo_root / args.fclass_init), device)

    evidence_net = EvidenceNetwork(hidden_dim=args.hidden_dim, k_evidence=37).to(device)

    total_params = sum(p.numel() for p in gsae.parameters()) + \
                   sum(p.numel() for p in evidence_net.parameters())
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(gsae.parameters()) + list(evidence_net.parameters()), lr=args.lr)

    # 断点恢复：自动检测 output_dir 中最新的 checkpoint
    start_epoch = 0
    training_log = []
    existing = sorted(output_dir.glob('checkpoint_epoch*.pt'))
    if existing:
        latest = existing[-1]
        print(f"\n=== Resuming from {latest} ===")
        ckpt = torch.load(str(latest), map_location=device, weights_only=False)
        gsae.load_state_dict(ckpt['gsae_state_dict'])
        evidence_net.load_state_dict(ckpt['evidence_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError:
            print("  WARNING: optimizer state incompatible (old checkpoint format?), "
                  "starting with fresh optimizer")
        training_log = ckpt.get('training_log', [])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch + 1}/{args.epochs}")

    # 训练
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg = train_one_epoch(
            gsae, evidence_net, optimizer, graph_cache, all_trajs,
            args.batch_size, device,
        )
        print(f"  ce={avg['ce']:.4f}  acc={avg['acc']:.4f}")

        training_log.append({'epoch': epoch + 1, **avg})

        ckpt = {
            'epoch': epoch,
            'gsae_state_dict': gsae.state_dict(),
            'evidence_state_dict': evidence_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': {
                'hidden_dim': args.hidden_dim,
                'lr': args.lr,
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

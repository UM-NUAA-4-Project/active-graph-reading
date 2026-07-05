"""Phase 0+1: GSAE + EvidenceNetwork + LSTM 逐步加权信念训练。

每步：
  G_t → FrontierSubgraph → GSAE → v, frontier_embeds
      → EvidenceNetwork(v) → e, b_raw, u
      → LSTMBeliefUpdater(b_{t-1}, e, u, emb(a_{t-1})) → alpha_accum_t
      → L_t = EDL(alpha_accum_t, y_true)

加权：w_t = 2t / (B(B+1))，L_final = Σ w_t · L_t

不训练 policy_mlp 和 final_classifier。
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
import torch.nn as nn
from tqdm import tqdm

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.reader.belief_updater import LSTMBeliefUpdater
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder

from training_utils import (
    merge_trajectories, load_graphs_for_trajectories, edl_loss,
)


def train_one_epoch(gsae, evidence_net, belief_updater, frontier_proj,
                    optimizer, graph_cache, train_trajs,
                    batch_size, device, lambda_kl):
    gsae.train()
    evidence_net.train()
    belief_updater.train()
    frontier_proj.train()
    builder = FrontierGraphBuilder()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_var = 0.0
    sum_kl = 0.0
    sum_correct = 0
    total_trajs = 0

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        batch_traj_count = 0

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            B = len(traj.reveal_order)
            if B == 0:
                continue

            y = torch.zeros(evidence_net.k_evidence, device=device)
            y[label] = 1.0

            state = GraphState(g, seed_node=traj.seed_node)
            belief_updater.reset()
            prev_action_emb = torch.zeros(128, device=device)

            step_losses = []

            for t, teacher_node in enumerate(traj.reveal_order, start=1):
                subgraph = builder.build(state)
                gsae_out = gsae(subgraph)

                if len(gsae_out.frontier_original_ids) == 0:
                    break

                ev_out = evidence_net(gsae_out.v)

                # b_prev from current alpha_accum
                alpha_prev = belief_updater.alpha_accum.to(device)
                b_prev = alpha_prev / alpha_prev.sum().clamp(min=1)

                belief_updater(
                    b_prev.unsqueeze(0),
                    ev_out.e,
                    ev_out.u,
                    prev_action_emb.unsqueeze(0),
                )

                alpha_t = belief_updater.alpha_accum  # (K,) 累积 Dirichlet 参数
                S_t = alpha_t.sum()

                L_t, mse_t, var_t, kl_t = edl_loss(
                    alpha_t - 1,            # evidence (unused by fn, for API compat)
                    alpha_t.unsqueeze(0),    # (1, K)
                    S_t.unsqueeze(0),        # (1,)
                    y.unsqueeze(0),          # (1, K)
                    evidence_net.k_evidence, lambda_kl,
                    return_components=True,
                )

                w_t = 2.0 * t / (B * (B + 1))
                step_losses.append((w_t * L_t, w_t * mse_t, w_t * var_t, w_t * kl_t))

                # 编码 teacher 节点为下一步动作嵌入（detach 避免跨步梯度）
                try:
                    idx = gsae_out.frontier_original_ids.index(teacher_node)
                    emb_300 = gsae_out.frontier_embeds[idx:idx + 1]
                    prev_action_emb = frontier_proj(emb_300).detach().squeeze(0)
                except ValueError:
                    break

                reveal_node(state, teacher_node)

            if not step_losses:
                continue

            losses_only = torch.stack([s[0] for s in step_losses])
            mse_only = torch.stack([s[1] for s in step_losses])
            var_only = torch.stack([s[2] for s in step_losses])
            kl_only = torch.stack([s[3] for s in step_losses])
            traj_loss = losses_only.sum()

            # 逐轨迹 backward：B 步 GSAE forward 已在图中，
            # 合并多条轨迹的图再 backward 内存过大（B×batch_size 次 GSAE forward）
            (traj_loss / len(batch_trajs)).backward()

            # 最终信念准确率
            final_alpha = belief_updater.alpha_accum.to(device)
            final_b = final_alpha / final_alpha.sum().clamp(min=1)
            sum_correct += int(final_b.argmax().item() == label)
            sum_loss += traj_loss.item()
            sum_mse += mse_only.sum().item()
            sum_var += var_only.sum().item()
            sum_kl += kl_only.sum().item()
            total_trajs += 1
            batch_traj_count += 1

        if batch_traj_count == 0:
            continue

        optimizer.step()
        optimizer.zero_grad()

        n = max(total_trajs, 1)
        pbar.set_postfix({
            'loss': f'{sum_loss / n:.4f}',
            'mse': f'{sum_mse / n:.4f}',
            'var': f'{sum_var / n:.4f}',
            'kl': f'{sum_kl / n:.4f}',
            'acc': f'{sum_correct / n:.4f}',
        })

    return {
        'loss': sum_loss / max(total_trajs, 1),
        'mse': sum_mse / max(total_trajs, 1),
        'var': sum_var / max(total_trajs, 1),
        'kl': sum_kl / max(total_trajs, 1),
        'acc': sum_correct / max(total_trajs, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Phase 0+1: GSAE + EvidenceNet + LSTM step-weighted belief training')
    parser.add_argument('--data-dir', default='m4/data/stage7_filtered')
    parser.add_argument('--fclass-init', default='', help='f_class checkpoint (optional)')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size（逐步训练每条轨迹 B 步 forward，内存消耗大，建议 ≤8）')
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lambda-kl', type=float, default=0.01, help='KL 正则权重')
    parser.add_argument('--limit-trajs', type=int, default=0, help='0=all')
    parser.add_argument('--output-dir', default='m5/pomdp/phase01/')
    parser.add_argument('--no-struct', action='store_true', help='禁用结构特征+结构注意力池化')
    parser.add_argument('--resume-checkpoint', default='', help='从指定 checkpoint 恢复训练')
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

    # 2. 创建模型
    print("\n=== Creating models ===")
    use_struct = not args.no_struct
    gsae = GSAE(edge_feat_dim=7, hidden_dim=args.hidden_dim, k_state=128,
                n_layers=args.n_layers, dropout=args.dropout,
                use_struct=use_struct).to(device)
    if args.fclass_init:
        print(f"  Initializing GSAE from {args.fclass_init}")
        gsae.init_from_fclass(str(repo_root / args.fclass_init), device)

    evidence_net = EvidenceNetwork(hidden_dim=args.hidden_dim, k_evidence=37, dropout=args.dropout).to(device)
    belief_updater = LSTMBeliefUpdater(k_evidence=37, lstm_hidden=128, action_emb_dim=128).to(device)
    frontier_proj = nn.Linear(args.hidden_dim, 128).to(device)

    total_params = sum(p.numel() for m in [gsae, evidence_net, belief_updater, frontier_proj]
                       for p in m.parameters())
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(gsae.parameters()) +
        list(evidence_net.parameters()) +
        list(belief_updater.parameters()) +
        list(frontier_proj.parameters()),
        lr=args.lr,
    )

    # 断点恢复
    start_epoch = 0
    training_log = []
    if args.resume_checkpoint:
        resume_path = repo_root / args.resume_checkpoint
        print(f"\n=== Resuming from {resume_path} ===")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        gsae.load_state_dict(ckpt['gsae_state_dict'])
        evidence_net.load_state_dict(ckpt['evidence_state_dict'])
        belief_updater.load_state_dict(ckpt['belief_updater_state_dict'])
        frontier_proj.load_state_dict(ckpt['frontier_proj_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError:
            print("  WARNING: optimizer state incompatible, starting with fresh optimizer")
        training_log = ckpt.get('training_log', [])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch + 1}/{args.epochs}")

    # 训练
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg = train_one_epoch(
            gsae, evidence_net, belief_updater, frontier_proj,
            optimizer, graph_cache, all_trajs,
            args.batch_size, device, args.lambda_kl,
        )
        avg_loss = avg['loss']
        avg_mse = avg['mse']
        avg_var = avg['var']
        avg_kl = avg['kl']
        avg_acc = avg['acc']
        mse_baseline = 0.973  # uniform MSE
        signal = 'YES' if avg_mse < 0.95 else 'NO'
        print(f"  loss={avg_loss:.4f}  mse={avg_mse:.4f}  var={avg_var:.4f}  kl={avg_kl:.4f}  acc={avg_acc:.4f}  "
              f"graph_signal={signal}  (MSE_uniform={mse_baseline:.3f})")

        training_log.append({'epoch': epoch + 1, **avg})

        ckpt = {
            'epoch': epoch,
            'gsae_state_dict': gsae.state_dict(),
            'evidence_state_dict': evidence_net.state_dict(),
            'belief_updater_state_dict': belief_updater.state_dict(),
            'frontier_proj_state_dict': frontier_proj.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': {
                'hidden_dim': args.hidden_dim,
                'n_layers': args.n_layers,
                'lr': args.lr,
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

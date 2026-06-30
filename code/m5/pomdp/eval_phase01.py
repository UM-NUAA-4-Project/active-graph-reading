"""Phase 0+1 评估：逐步信念质量。

Usage:
  python m5/pomdp/eval_phase01.py \
    --checkpoint m5/pomdp/phase01/checkpoint_epoch20.pt \
    --root D:\\ogb-master\\dataset \
    --device cuda:0

输出：每步平均 EDL loss | 最终信念准确率 | 证据统计。
"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

import torch
from tqdm import tqdm

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.reader.belief_updater import SimpleDirichletAccumulator
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder

from training_utils import merge_trajectories, load_graphs_for_trajectories, edl_loss


@torch.no_grad()
def evaluate(gsae, evidence_net, belief_updater,
             graph_cache, trajs, device, lambda_kl):
    gsae.eval()
    evidence_net.eval()
    belief_updater.eval()
    builder = FrontierGraphBuilder()

    total_trajs = 0
    final_correct = 0

    sum_weighted_loss = 0.0
    sum_unweighted_loss = 0.0
    e_max_sum = 0.0
    u_mean_sum = 0.0
    step_count = 0

    pbar = tqdm(trajs, desc="  Evaluating")
    for traj in pbar:
        g, label = graph_cache[traj.graph_idx]
        B = len(traj.reveal_order)
        if B == 0:
            continue

        y = torch.zeros(evidence_net.k_evidence, device=device)
        y[label] = 1.0

        state = GraphState(g, seed_node=traj.seed_node)
        belief_updater.reset()

        traj_losses = []

        for t, teacher_node in enumerate(traj.reveal_order, start=1):
            subgraph = builder.build(state)
            gsae_out = gsae(subgraph)

            if len(gsae_out.frontier_original_ids) == 0:
                break

            ev_out = evidence_net(gsae_out.v)

            # 简单 Dirichlet 累积
            belief_updater.update(ev_out.e.squeeze(0))

            alpha_t = belief_updater.alpha
            S_t = alpha_t.sum()

            L_t = edl_loss(
                alpha_t - 1,
                alpha_t.unsqueeze(0),
                S_t.unsqueeze(0),
                y.unsqueeze(0),
                evidence_net.k_evidence, lambda_kl,
                return_components=False,
            )

            w_t = 2.0 * t / (B * (B + 1))
            traj_losses.append((w_t * L_t).item())
            sum_unweighted_loss += L_t.item()
            step_count += 1

            e_max_sum += ev_out.e.max().item()
            u_mean_sum += ev_out.u.mean().item()

            reveal_node(state, teacher_node)

        if traj_losses:
            sum_weighted_loss += sum(traj_losses)
            total_trajs += 1

            final_alpha = belief_updater.alpha.to(device)
            final_b = final_alpha / final_alpha.sum().clamp(min=1)
            final_correct += int(final_b.argmax().item() == label)

    n_traj = max(total_trajs, 1)
    n_step = max(step_count, 1)
    return {
        'weighted_loss': sum_weighted_loss / n_traj,
        'avg_step_loss': sum_unweighted_loss / n_step,
        'final_acc': final_correct / n_traj,
        'e_max': e_max_sum / n_step,
        'u_mean': u_mean_sum / n_step,
        'n_trajs': total_trajs,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 0+1 checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='m4/data/stage7_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-struct', action='store_true', help='禁用结构特征+结构注意力池化')
    parser.add_argument('--limit-trajs', type=int, default=0)
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载 checkpoint
    ckpt = torch.load(str(repo_root / args.checkpoint), map_location='cpu',
                      weights_only=False)
    cfg = ckpt['config']
    print(f"Checkpoint: epoch {ckpt['epoch']}")
    print(f"Config: hidden_dim={cfg['hidden_dim']}, n_layers={cfg.get('n_layers', 2)}, "
          f"lr={cfg['lr']}, lambda_kl={cfg.get('lambda_kl', 0.01)}")

    # 加载数据
    data_dir = repo_root / args.data_dir
    print(f"\nLoading trajectories from {data_dir}...")
    all_trajs = merge_trajectories(data_dir)
    if args.limit_trajs > 0:
        all_trajs = all_trajs[:args.limit_trajs]

    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    # 创建模型并加载权重
    use_struct = not args.no_struct
    gsae = GSAE(edge_feat_dim=7, hidden_dim=cfg['hidden_dim'], k_state=128,
                n_layers=cfg.get('n_layers', 2), dropout=0.5,
                use_struct=use_struct).to(device)
    gsae.load_state_dict(ckpt['gsae_state_dict'])

    evidence_net = EvidenceNetwork(hidden_dim=cfg['hidden_dim'], k_evidence=37).to(device)
    evidence_net.load_state_dict(ckpt['evidence_state_dict'])

    belief_updater = SimpleDirichletAccumulator(k_evidence=37).to(device)
    belief_updater.load_state_dict(ckpt.get('belief_updater_state_dict', {}))

    n_params = sum(p.numel() for m in [gsae, evidence_net, belief_updater]
                   for p in m.parameters())
    print(f"Model params: {n_params:,}")

    # 评估
    lambda_kl = cfg.get('lambda_kl', 0.01)
    print(f"\n=== Evaluating on {len(all_trajs)} trajectories ===")
    results = evaluate(
        gsae, evidence_net, belief_updater,
        graph_cache, all_trajs, device, lambda_kl,
    )

    print(f"\n--- Results ---")
    print(f"  Weighted loss  = {results['weighted_loss']:.4f}  (Σ w_t·L_t)")
    print(f"  Avg step loss  = {results['avg_step_loss']:.4f}  (mean L_t)")
    print(f"  Final accuracy = {results['final_acc']:.4f}")
    print(f"  e_max          = {results['e_max']:.4f}  (max evidence per step)")
    print(f"  u_mean         = {results['u_mean']:.4f}  (mean uncertainty)")
    print(f"  n_trajs        = {results['n_trajs']}")
    print(f"\n  CE baseline     = {3.6109:.4f}  (ln(37), uniform)")

    if results['final_acc'] > 1.0 / 37:
        print(f"  → above random ({1.0/37:.4f})")


if __name__ == '__main__':
    main()

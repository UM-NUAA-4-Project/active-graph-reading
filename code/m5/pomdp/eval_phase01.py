"""Phase 0+1 评估脚本：加载 checkpoint，计算 CE loss 和准确率。

Usage:
  python m5/pomdp/eval_phase01.py \
    --checkpoint m5/phase01/checkpoint_epoch1.pt \
    --root D:\\ogb-master\\dataset
    --data-dir m4/data/stage7_filtered \
    --device cuda:0 \

输出：CE loss | accuracy | 证据统计（e_max, u_mean）。
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
import torch.nn.functional as F
from tqdm import tqdm

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder

from training_utils import merge_trajectories, load_graphs_for_trajectories


@torch.no_grad()
def evaluate(gsae, evidence_net, graph_cache, trajs, device):
    """返回 CE loss 及证据统计的平均值。"""
    gsae.eval()
    evidence_net.eval()
    builder = FrontierGraphBuilder()

    total_ce = 0.0
    total_correct = 0
    total_steps = 0
    e_max_sum = 0.0
    u_mean_sum = 0.0

    pbar = tqdm(trajs, desc="  Evaluating")
    for traj in pbar:
        g, label = graph_cache[traj.graph_idx]
        label_t = torch.tensor(label, device=device)

        state = GraphState(g, seed_node=traj.seed_node)

        for node in traj.reveal_order:
            subgraph = builder.build(state)
            gsae_out = gsae(subgraph)

            ev_out = evidence_net(gsae_out.v)
            b = ev_out.b.squeeze(0)  # (K,)
            ce = F.cross_entropy(b.unsqueeze(0), label_t.unsqueeze(0))

            total_ce += ce.item()
            total_correct += int(b.argmax().item() == label)
            total_steps += 1
            e_max_sum += ev_out.e.max().item()
            u_mean_sum += ev_out.u.mean().item()

            reveal_node(state, node)

    n = max(total_steps, 1)
    return {
        'ce': total_ce / n,
        'acc': total_correct / n,
        'e_max': e_max_sum / n,
        'u_mean': u_mean_sum / n,
        'n_steps': total_steps,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 0+1 checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='m4/data/stage5_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
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
    print(f"Config: hidden_dim={cfg['hidden_dim']}, lr={cfg['lr']}")

    # 加载数据
    data_dir = repo_root / args.data_dir
    print(f"\nLoading trajectories from {data_dir}...")
    all_trajs = merge_trajectories(data_dir)
    if args.limit_trajs > 0:
        all_trajs = all_trajs[:args.limit_trajs]

    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    # 创建模型并加载权重
    gsae = GSAE(edge_feat_dim=7, hidden_dim=cfg['hidden_dim'], k_state=128,
                n_layers=2, dropout=0.5).to(device)
    gsae.load_state_dict(ckpt['gsae_state_dict'])

    evidence_net = EvidenceNetwork(hidden_dim=cfg['hidden_dim'], k_evidence=37).to(device)
    evidence_net.load_state_dict(ckpt['evidence_state_dict'])

    n_params = sum(p.numel() for p in gsae.parameters()) + \
               sum(p.numel() for p in evidence_net.parameters())
    print(f"Model params: {n_params:,}")

    # 评估
    print(f"\n=== Evaluating on {len(all_trajs)} trajectories ===")
    results = evaluate(gsae, evidence_net, graph_cache, all_trajs, device)

    print(f"\n--- Results ---")
    print(f"  CE         = {results['ce']:.4f}")
    print(f"  Accuracy   = {results['acc']:.4f}")
    print(f"  e_max      = {results['e_max']:.4f}  (max evidence per step)")
    print(f"  u_mean     = {results['u_mean']:.4f}  (mean uncertainty)")
    print(f"  n_steps    = {results['n_steps']}")


if __name__ == '__main__':
    main()

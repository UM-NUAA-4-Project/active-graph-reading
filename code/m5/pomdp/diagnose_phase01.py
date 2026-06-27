"""Phase 0+1 诊断脚本：检查数据正确性 + 编码器输出质量。

用法：
  python code/m5/pomdp/diagnose_phase01.py \
    --data-dir m4/data/stage5_filtered \
    --root /mnt/data \
    --device cuda:0 \
    [--checkpoint m5/pomdp/phase01/checkpoint_epoch1.pt]
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
import dgl
import numpy as np
from tqdm import tqdm

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder
from training_utils import merge_trajectories, load_graphs_for_trajectories


def check_data(graph_cache, trajs, n_samples=200):
    """检查数据正确性。"""
    print("\n" + "=" * 60)
    print("CHECK 1: 数据正确性")
    print("=" * 60)

    import random
    random.seed(42)
    samples = random.sample(trajs, min(n_samples, len(trajs)))

    # 1.1 图索引和标签
    label_counts = {}
    idx_mismatch = 0
    n_node_list = []
    label_list = []

    for traj in samples[:50]:
        g, label = graph_cache[traj.graph_idx]
        n_node_list.append(g.num_nodes())
        label_list.append(label)

    n_nodes = np.array(n_node_list)
    print(f"\n  Node count: min={n_nodes.min()} max={n_nodes.max()} "
          f"mean={n_nodes.mean():.0f} median={np.median(n_nodes):.0f}")

    unique_labels = set(label_list)
    all_labels = set()
    label_counts = {}
    for traj in samples:
        _, l = graph_cache[traj.graph_idx]
        all_labels.add(l)
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"  Labels: {len(unique_labels)} unique in first 50, "
          f"{len(all_labels)} unique in {len(samples)} samples (expected 37)")

    # Check full dataset label distribution
    all_full_labels = {}
    for traj in trajs:
        _, l = graph_cache[traj.graph_idx]
        all_full_labels[l] = all_full_labels.get(l, 0) + 1
    n_classes = len(all_full_labels)
    counts = sorted(all_full_labels.values())
    print(f"  Full dataset: {n_classes} classes, "
          f"samples per class: min={counts[0]} max={counts[-1]} "
          f"median={counts[n_classes//2]} mean={np.mean(counts):.0f}")

    # 1.2 trajectory reveal_order
    empty_reveal = 0
    bad_seed = 0
    bad_reveal = 0
    for traj in samples[:20]:
        if len(traj.reveal_order) == 0:
            empty_reveal += 1
        g, _ = graph_cache[traj.graph_idx]
        if traj.seed_node >= g.num_nodes():
            bad_seed += 1
        reveal_within = [n for n in traj.reveal_order if n < g.num_nodes()]
        if len(reveal_within) != len(traj.reveal_order):
            bad_reveal += 1

    if empty_reveal or bad_seed or bad_reveal:
        print(f"  FAIL: empty_reveal={empty_reveal}, bad_seed={bad_seed}, bad_reveal={bad_reveal}")
    else:
        print(f"  PASS: trajectory reveal_order")

    # 1.3 subgraph build
    print(f"\n  Subgraph build samples:")
    builder = FrontierGraphBuilder()
    for traj in samples[:5]:
        g, label = graph_cache[traj.graph_idx]
        state = GraphState(g, seed_node=traj.seed_node)
        for j, node in enumerate(traj.reveal_order):
            reveal_node(state, node)
            if j == len(traj.reveal_order) - 1:
                sub = builder.build(state)
                print(f"    graph_idx={traj.graph_idx}: |V|={g.num_nodes()}, "
                      f"final sub: revealed={sub.n_revealed}, "
                      f"frontier={len(sub.frontier_original_ids)}, "
                      f"total_nodes={sub.g.num_nodes()}")


def check_encoder(gsae, evidence_net, graph_cache, trajs, device, n_samples=500):
    """检查编码器输出质量，诊断退化。"""
    print("\n" + "=" * 60)
    print("CHECK 2: 编码器输出质量")
    print("=" * 60)

    gsae.eval()
    evidence_net.eval()
    builder = FrontierGraphBuilder()

    import random
    random.seed(42)
    samples = random.sample(trajs, min(n_samples, len(trajs)))

    v_norms = []
    e_means = []
    e_maxs = []
    S_vals = []
    u_vals = []
    predictions = []
    labels = []
    v_diffs = []  # 不同图的 v 差异

    @torch.no_grad()
    def get_embedding(traj):
        g, label = graph_cache[traj.graph_idx]
        state = GraphState(g, seed_node=traj.seed_node)
        for node in traj.reveal_order:
            reveal_node(state, node)
        sub = builder.build(state)
        out = gsae(sub)
        ev = evidence_net(out.v)
        return out.v.squeeze(0), ev

    prev_v = None
    prev_label = None
    for i, traj in enumerate(tqdm(samples, desc="  Encoding")):
        v, ev = get_embedding(traj)
        g, label = graph_cache[traj.graph_idx]

        v_norms.append(v.norm().item())
        e_means.append(ev.e.mean().item())
        e_maxs.append(ev.e.max().item())
        S_vals.append(ev.S.item())
        u_vals.append(ev.u.item() if ev.u.dim() == 0 else ev.u.mean().item())
        predictions.append(ev.b.squeeze(0).argmax().item())
        labels.append(label)

        if prev_v is not None and label != prev_label:
            v_diffs.append((v - prev_v).norm().item())
        prev_v = v
        prev_label = label

    print(f"\n  Graph embedding v (300-dim):")
    print(f"    norm: min={np.min(v_norms):.4f} max={np.max(v_norms):.4f} "
          f"mean={np.mean(v_norms):.4f} std={np.std(v_norms):.4f}")

    print(f"\n  Evidence e:")
    print(f"    mean: {np.mean(e_means):.6f}  (expected >0)")
    print(f"    max:  {np.mean(e_maxs):.6f}  (expected >0)")
    print(f"    zero_ratio: {np.mean([1.0 if m == 0 else 0.0 for m in e_means]):.2%} "
          f"(expected 0%)")

    print(f"\n  Dirichlet stats:")
    print(f"    S (total evidence): mean={np.mean(S_vals):.2f} (K=37, degraded when ~37)")
    print(f"    u (uncertainty): mean={np.mean(u_vals):.4f} (degraded when ~1.0)")

    acc = np.mean([1.0 if p == l else 0.0 for p, l in zip(predictions, labels)])
    print(f"\n  Accuracy: {acc:.4f} (random baseline 1/37 = {1/37:.4f})")

    print(f"\n  Cross-class v diffs:")
    if v_diffs:
        print(f"    ||v_a - v_b|| cross-class: mean={np.mean(v_diffs):.4f} "
              f"min={np.min(v_diffs):.4f} max={np.max(v_diffs):.4f}")
        print(f"    (expected >0, 0 means collapse to same embedding)")
    else:
        print(f"    (not enough cross-class pairs)")

    # diagnostic
    print(f"\n  --- Diagnosis ---")
    issues = []
    if np.mean(v_norms) < 1e-3:
        issues.append("FAIL: v_norm ~ 0 -> GSAE output no signal")
    if np.mean(e_maxs) < 1e-3:
        issues.append("FAIL: e_max ~ 0 -> evidence head outputs all zeros")
    if np.mean(u_vals) > 0.95:
        issues.append("FAIL: u ~ 1.0 -> model completely uncertain")
    if np.mean(S_vals) < 40:
        issues.append("WARN: S ~ K -> total evidence near zero, EDL degraded")
    if acc < 0.05:
        issues.append("WARN: acc ~ random -> cannot learn discriminative info")
    if len(v_diffs) > 0 and np.mean(v_diffs) < 1e-3:
        issues.append("FAIL: v cross-class diffs ~ 0 -> GSAE collapsed, same embedding for different graphs")

    if not issues:
        print("  PASS: no obvious problems detected")
    else:
        for issue in issues:
            print(f"  {issue}")


def check_gnn_gradients(gsae, evidence_net, graph_cache, trajs, device):
    """检查 GNN 梯度是否正常。"""
    print("\n" + "=" * 60)
    print("CHECK 3: 梯度流检查")
    print("=" * 60)

    gsae.train()
    evidence_net.train()
    builder = FrontierGraphBuilder()

    import random
    random.seed(42)
    sample = random.sample(trajs, 8)

    from training_utils import edl_loss

    for traj in sample:
        g, label = graph_cache[traj.graph_idx]
        y = torch.zeros(evidence_net.k_evidence, device=device)
        y[label] = 1.0

        state = GraphState(g, seed_node=traj.seed_node)
        for node in traj.reveal_order:
            reveal_node(state, node)

        sub = builder.build(state)
        out = gsae(sub)
        ev = evidence_net(out.v)
        loss, _, _, _ = edl_loss(ev.e, ev.alpha, ev.S, y.unsqueeze(0),
                                  evidence_net.k_evidence, 0.01,
                                  return_components=True)

    loss.backward()

    grad_info = {}
    for name, module in [('gsae', gsae), ('evidence', evidence_net)]:
        for p_name, p in module.named_parameters():
            if p.grad is not None:
                g_norm = p.grad.norm().item()
                if g_norm > 0:
                    key = f'{name}.{p_name}'
                    grad_info[key] = g_norm

    if not grad_info:
        print("  FAIL: no parameters have gradients! Gradient flow is broken")
        return

    norms = list(grad_info.values())
    zero_grads = sum(1 for v in norms if v < 1e-8)
    print(f"  Params with grad: {len(grad_info)}, zero_grad: {zero_grads}")
    print(f"  Grad norm: min={np.min(norms):.2e} max={np.max(norms):.2e} "
          f"mean={np.mean(norms):.2e}")

    top_params = [k for k in grad_info if 'evidence' in k]
    bottom_params = [k for k in grad_info if 'convs.0' in k]
    if top_params:
        top_norms = [grad_info[k] for k in top_params[:3]]
        print(f"  Evidence head grad: {' '.join(f'{k}:{grad_info[k]:.2e}' for k in top_params[:3])}")
    if bottom_params:
        bot_norms = [grad_info[k] for k in bottom_params[:3]]
        print(f"  GNN layer0 grad: {' '.join(f'{k}:{grad_info[k]:.2e}' for k in bottom_params[:3])}")
        if np.mean(bot_norms) < 1e-6:
            print("  FAIL: GNN bottom layer grad ~ 0 -> vanishing gradient, encoder not learning")
        else:
            print("  PASS: GNN gradients ok")

    gsae.zero_grad()
    evidence_net.zero_grad()


def main():
    parser = argparse.ArgumentParser(description='Diagnose Phase 0+1 training')
    parser.add_argument('--data-dir', default='m4/data/stage5_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', default='',
                        help='加载 checkpoint 后的编码器（不指定则用随机初始化）')
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--no-struct', action='store_true')
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    data_dir = repo_root / args.data_dir
    print("\n=== Loading data ===")
    all_trajs = merge_trajectories(data_dir)
    needed = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed)

    # 创建模型
    print("\n=== Creating models ===")
    gsae = GSAE(edge_feat_dim=7, hidden_dim=args.hidden_dim, k_state=128,
                n_layers=2, dropout=0.5,
                use_struct=not args.no_struct).to(device)
    evidence_net = EvidenceNetwork(hidden_dim=args.hidden_dim, k_evidence=37).to(device)

    if args.checkpoint:
        ckpt = torch.load(str(repo_root / args.checkpoint), map_location=device,
                          weights_only=False)
        gsae.load_state_dict(ckpt['gsae_state_dict'])
        evidence_net.load_state_dict(ckpt['evidence_state_dict'])
        print(f"  Loaded checkpoint: epoch {ckpt.get('epoch', 'N/A')}")

    # 检查
    check_data(graph_cache, all_trajs)
    check_encoder(gsae, evidence_net, graph_cache, all_trajs, device)
    check_gnn_gradients(gsae, evidence_net, graph_cache, all_trajs, device)


if __name__ == '__main__':
    main()

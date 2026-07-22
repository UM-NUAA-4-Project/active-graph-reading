"""冒烟测试：跨图锁步训练 — 单 batch 验证预算、退出、梯度。

用法:
  python code/m5/base/smoke_test_cross_graph.py
  python code/m5/base/smoke_test_cross_graph.py --graphs 8 --seeds 3
"""

import argparse
import math
import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

import numpy as np
import pandas as pd
import torch
import dgl

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from training_utils import patch_old_checkpoint

K = 37
BUDGET_OPTIONS = [0.05, 0.10, 0.15]


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


def train_one_batch_smoke(
    reader, classifier, builder, optimizer,
    graphs, ogb_indices, labels_tensor,
    n_seeds, device, reward_type='binary',
):
    """单 batch 训练 + 诊断打印。"""
    reader.train()
    classifier.eval()

    g_raws = []
    graph_indices = []
    n_nodes_list = []
    budget_list = []
    labels_list = []

    for local_i in range(len(graphs)):
        g_raw = graphs[local_i]
        graph_idx = ogb_indices[local_i]
        label = int(labels_tensor[graph_idx].item())
        n_nodes = g_raw.num_nodes()
        budget_ratio = random.choice(BUDGET_OPTIONS)
        B = max(1, int(budget_ratio * n_nodes))

        g_raws.append(g_raw)
        graph_indices.append(graph_idx)
        n_nodes_list.append(n_nodes)
        budget_list.append(B)
        labels_list.append(label)

    n_graphs = len(graphs)
    print(f"\n  Graphs: {n_graphs}")
    for gi in range(n_graphs):
        print(f"    graph[{gi}]: ogb_idx={graph_indices[gi]}, "
              f"|V|={n_nodes_list[gi]}, B={budget_list[gi]}, label={labels_list[gi]}")

    # ---- Pass 1: create ALL states ----
    all_states = []
    per_state_log_probs = []
    state_labels = []
    state_graph_idx = []  # which graph this state belongs to

    for gi in range(n_graphs):
        g_raw = g_raws[gi]
        n_nodes = n_nodes_list[gi]
        B = budget_list[gi]
        y = torch.tensor([labels_list[gi]], device=device)

        for seed_i in range(n_seeds):
            seed_node = deterministic_seed(graph_indices[gi], seed_i, n_nodes)
            state = GraphState(g_raw, seed_node=seed_node)
            state._budget_limit = B
            all_states.append(state)
            per_state_log_probs.append([])
            state_labels.append(y)
            state_graph_idx.append(gi)

    n_total = len(all_states)
    print(f"\n  Total states: {n_total} ({n_graphs} graphs × {n_seeds} seeds)")

    # Verify budgets
    for j in range(n_total):
        gi = state_graph_idx[j]
        expected_B = budget_list[gi]
        actual_B = all_states[j]._budget_limit
        assert actual_B == expected_B, \
            f"State {j} (graph {gi}): budget {actual_B} != expected {expected_B}"
    print(f"  [PASS] All state budgets match graph budgets")

    # ---- Pass 2: cross-graph lockstep ----
    active = [True] * n_total
    max_budget = max(budget_list)
    steps_run = 0
    early_exits = set()  # track which graphs had states exit early

    for step in range(max_budget):
        active_idx = [j for j in range(n_total)
                      if active[j] and not all_states[j].is_frontier_empty]
        if not active_idx:
            break

        steps_run += 1
        active_graphs = len(set(state_graph_idx[j] for j in active_idx))
        if step == 0:
            print(f"\n  Lockstep step 0: {len(active_idx)}/{n_total} states active "
                  f"({active_graphs}/{n_graphs} graphs)")

        batch_results = reader.forward_states(
            [all_states[j] for j in active_idx],
            temperature=reader.temperature,
        )

        for k, j in enumerate(active_idx):
            scores, probs, frontier_nodes = batch_results[k]
            if not frontier_nodes:
                active[j] = False
                early_exits.add(state_graph_idx[j])
                continue

            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            per_state_log_probs[j].append(dist.log_prob(action_idx))
            reveal_node(all_states[j], frontier_nodes[action_idx.item()])

            if all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                active[j] = False

    # How many still active?
    still_active = sum(active)
    print(f"  Lockstep done: {steps_run}/{max_budget} steps, "
          f"{still_active} states still active "
          f"(frontier-emptied: {len(early_exits)} graphs)")

    # Verify: states with B < max_budget should have been deactivated
    for j in range(n_total):
        B = all_states[j]._budget_limit
        if B < max_budget and all_states[j].get_budget_used() >= B:
            assert not active[j] or all_states[j].is_frontier_empty, \
                f"State {j} with B={B} < max_budget={max_budget} should be inactive"
    print(f"  [PASS] Budget drop-out triggered correctly")

    # ---- Pass 3: collect trajectories ----
    traj_subgraphs = []
    traj_labels = []
    traj_log_probs = []
    trajectory_graph_ids = []

    for j in range(n_total):
        if per_state_log_probs[j]:
            sub_g, _ = builder.build(all_states[j])
            traj_subgraphs.append(sub_g)
            traj_labels.append(state_labels[j])
            traj_log_probs.append(per_state_log_probs[j])
            trajectory_graph_ids.append(state_graph_idx[j])

    n_trajs = len(traj_subgraphs)
    assert n_trajs <= n_total, \
        f"Trajectories {n_trajs} > states {n_total}"
    assert n_trajs > 0, "No trajectories collected!"
    print(f"  [PASS] Collected {n_trajs}/{n_total} trajectories "
          f"(≤ n_graphs × n_seeds = {n_graphs * n_seeds})")

    # ---- classify + loss + backward ----
    batch_g = dgl.batch(traj_subgraphs).to(device)
    nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
    with torch.no_grad():
        all_logits = classifier(batch_g, nids)

    batch_losses = []
    for i, log_probs in enumerate(traj_log_probs):
        logits_i = all_logits[i:i + 1]
        correct = 1.0 if logits_i.argmax(dim=-1).item() == traj_labels[i].item() else 0.0
        if reward_type == 'asymmetric':
            R = 1.0 if correct else -0.1
        else:
            R = 1.0 if correct else 0.0
        batch_losses.append(-R * torch.stack(log_probs).sum())

    batch_loss = torch.stack(batch_losses).mean()

    # Verify loss is finite
    assert math.isfinite(batch_loss.item()), f"Loss is not finite: {batch_loss.item()}"
    print(f"  [PASS] Loss is finite: {batch_loss.item():.6f}")

    # Check gradients before step
    param_before = next(reader.parameters()).clone().detach()

    batch_loss.backward()

    # Verify gradients exist and are non-zero
    has_grad = False
    for name, p in reader.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No non-zero gradients found!"
    print(f"  [PASS] Gradients are non-zero")

    optimizer.step()
    optimizer.zero_grad()

    # Verify parameters changed
    param_after = next(reader.parameters()).clone().detach()
    param_diff = (param_after - param_before).abs().max().item()
    assert param_diff > 0, f"Parameters did not change after optimizer.step()"
    print(f"  [PASS] Parameters updated (max Δ = {param_diff:.6f})")

    print(f"\n  === All smoke tests PASSED ===")


def main():
    parser = argparse.ArgumentParser(description='Smoke test: cross-graph lockstep training')
    parser.add_argument('--graphs', type=int, default=4,
                        help='Number of graphs in the batch')
    parser.add_argument('--seeds', type=int, default=2,
                        help='Seeds per graph')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--bootstrap', default='m5/base/checkpoint/best_reader.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- load a few training graphs ----
    root_path = Path(args.root)
    train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'
    all_train_indices = pd.read_csv(train_csv, compression='gzip', header=None
                                    ).values.flatten().tolist()
    print(f"Train set: {len(all_train_indices)} graphs, using first {args.graphs}")

    sampled = all_train_indices[:args.graphs]
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    train_graphs, train_labels = dgl.load_graphs(processed_path, idx_list=sampled)
    train_graphs = [dgl.add_self_loop(g) for g in train_graphs]
    print(f"  Loaded {len(train_graphs)} graphs (CPU)")

    # ---- load classifier ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device,
                      weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- load reader ----
    print("\n=== Loading LearnableReader ===")
    reader_ckpt = torch.load(str(repo_root / args.bootstrap), map_location='cpu',
                             weights_only=False)
    hidden_dim = reader_ckpt['reader_state_dict']['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in reader_ckpt['reader_state_dict']:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=0.1, stochastic=True, temperature=0.2,
    ).to(device)
    reader_sd, _ = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
    reader.load_state_dict(reader_sd)
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")

    optimizer = torch.optim.Adam(reader.parameters(), lr=1e-4)
    builder = SubgraphBuilder()

    # ---- run smoke test ----
    train_one_batch_smoke(
        reader, classifier, builder, optimizer,
        train_graphs, sampled, train_labels['labels'],
        n_seeds=args.seeds, device=device, reward_type='asymmetric',
    )


if __name__ == '__main__':
    main()

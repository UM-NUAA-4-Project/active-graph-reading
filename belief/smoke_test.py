"""Phase 2 smoke test: belief state + classifier distillation + Phase 1 compat.

用法:
  python belief/smoke_test.py
  python belief/smoke_test.py --graphs 8 --seeds 3
"""

import argparse
import math
import random
from pathlib import Path

from common import K, BUDGET_OPTIONS, _repo_root, setup_paths
setup_paths()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl

from learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from belief_state import BeliefState
from training_utils import deterministic_seed, reveal_and_update_beliefs_batch


def test_belief_state_module(device):
    """Test BeliefState sub-modules in isolation."""
    print("\n--- Test 1: BeliefState modules ---")

    bs = BeliefState(d_model=128, m=4, n_heads=4, dropout=0.1).to(device)

    h0 = bs.init_beliefs(3)
    assert h0.shape == (3, 4, 128), f"init shape: {h0.shape}"
    print(f"  [PASS] init_beliefs shape: {h0.shape}")

    new_info = torch.randn(3, 1, 128, device=device)
    h1 = bs.update_belief(h0.to(device), new_info)
    assert h1.shape == (3, 4, 128)
    print(f"  [PASS] update_belief shape: {h1.shape}")

    logits = bs.classify(h1)
    assert logits.shape == (3, K), f"classify shape: {logits.shape}"
    print(f"  [PASS] classify shape: {logits.shape}")

    # single state
    h_s = bs.init_belief().to(device)
    ni_s = torch.randn(1, 1, 128, device=device)
    h2 = bs.update_belief(h_s, ni_s)
    assert h2.shape == (1, 4, 128)
    print(f"  [PASS] single-state update shape: {h2.shape}")

    # gradients
    bs.train()
    h = bs.init_belief().to(device).requires_grad_()
    ni = torch.randn(1, 1, 128, device=device)
    h_up = bs.update_belief(h, ni)
    logits = bs.classify(h_up)
    logits.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in bs.parameters())
    assert has_grad, "No gradients in BeliefState!"
    print(f"  [PASS] gradients flow through BeliefState")


def test_frontier_pointer(device):
    """Test frontier_pointer: belief cross-attention."""
    print("\n--- Test 2: FrontierPointer ---")

    from frontier_pointer import FrontierPointer
    fp = FrontierPointer(d_in=132, d_model=128, n_heads=4, dropout=0.1).to(device)

    x = torch.randn(14, 132, device=device)
    belief = torch.randn(4, 128, device=device)

    scores = fp(x, belief=belief)
    assert scores.shape == (14,), f"shape: {scores.shape}"
    print(f"  [PASS] scores shape: {scores.shape}")

    fp.zero_grad()
    s = fp(x, belief=belief)
    s.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in fp.parameters()), "No gradients!"
    print(f"  [PASS] gradients flow")


def test_forward_states_belief(reader, graphs, ogb_indices, labels_tensor, device):
    """Test forward_states() with beliefs."""
    print("\n--- Test 3: forward_states with belief ---")

    reader.eval()
    states = []
    for local_i in range(min(4, len(graphs))):
        g = graphs[local_i]
        n_nodes = g.num_nodes()
        B = max(1, int(0.10 * n_nodes))
        seed_node = deterministic_seed(ogb_indices[local_i], 0, n_nodes)
        state = GraphState(g, seed_node=seed_node)
        state._budget_limit = B
        states.append(state)

    beliefs = reader.init_beliefs(len(states)).to(device)

    results = reader.forward_states(states, temperature=1.0, beliefs=beliefs)
    for i, r in enumerate(results):
        scores, probs, frontier_nodes, logits = r
        assert logits.shape == (K,), f"logits shape: {logits.shape}"
    print(f"  [PASS] forward_states with belief: {len(states)} states, all have logits")


def test_lockstep_belief(reader, classifier, graphs, ogb_indices, labels_tensor, device):
    """Full lockstep loop: belief update + distillation loss."""
    print("\n--- Test 4: Lockstep + belief update + distillation ---")

    reader.train()
    classifier.eval()
    builder = SubgraphBuilder()

    n_graphs = min(4, len(graphs))
    n_seeds = 2
    n_total = n_graphs * n_seeds

    all_states, all_log_probs, all_labels = [], [], []
    all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
    all_subgraphs = [None] * n_total

    for gi in range(n_graphs):
        g = graphs[gi]
        graph_idx = ogb_indices[gi]
        label = int(labels_tensor[graph_idx].item())
        n_nodes = g.num_nodes()
        B = max(1, int(0.10 * n_nodes))

        for seed_i in range(n_seeds):
            seed_node = deterministic_seed(graph_idx, seed_i, n_nodes)
            state = GraphState(g, seed_node=seed_node)
            state._budget_limit = B
            all_states.append(state)
            all_log_probs.append([])
            all_labels.append(torch.tensor([label], device=device))

    active = [True] * n_total
    max_budget = max(s._budget_limit for s in all_states)
    steps_run = 0

    for _step in range(max_budget):
        active_idx = [j for j in range(n_total)
                      if active[j] and not all_states[j].is_frontier_empty]
        if not active_idx:
            break
        steps_run += 1

        active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
        active_subgraphs = None
        if all_subgraphs[active_idx[0]] is not None:
            active_subgraphs = [all_subgraphs[j] for j in active_idx]

        results = reader.forward_states(
            [all_states[j] for j in active_idx],
            temperature=reader.temperature,
            beliefs=active_beliefs,
            subgraphs=active_subgraphs,
        )

        # Phase A: sample actions + collect log_probs
        batch_states = []
        batch_nodes = []
        batch_local_idx = []

        for k, j in enumerate(active_idx):
            scores, probs, frontier_nodes, logits = results[k]
            if not frontier_nodes:
                active[j] = False
                continue

            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            all_log_probs[j].append(dist.log_prob(action_idx))

            v_chosen = frontier_nodes[action_idx.item()]
            batch_states.append(all_states[j])
            batch_nodes.append(v_chosen)
            batch_local_idx.append(j)

        # Phase B: batch reveal + encode + update
        if batch_states:
            batch_beliefs = torch.cat(
                [all_beliefs[j] for j in batch_local_idx], dim=0)
            new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                reader, batch_states, batch_nodes, batch_beliefs, device,
            )
            for k, j in enumerate(batch_local_idx):
                all_beliefs[j] = new_beliefs[k:k + 1]
                all_subgraphs[j] = new_subgraphs[k]

        # Phase C: check budget
        for j in active_idx:
            if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                active[j] = False

    assert steps_run > 0, "No lockstep steps!"
    print(f"  [PASS] Lockstep: {steps_run}/{max_budget} steps, "
          f"{n_total - sum(active)}/{n_total} states completed")

    # Beliefs evolved
    all_beliefs_t = torch.cat(all_beliefs, dim=0)
    init_ref = reader.init_beliefs(n_total).to(device)
    delta = (all_beliefs_t - init_ref).abs().max().item()
    assert delta > 0, "Beliefs did not change!"
    print(f"  [PASS] Beliefs evolved (max delta: {delta:.4f})")

    # Collect trajectories
    traj_subgraphs, traj_labels, traj_log_probs = [], [], []
    for j in range(n_total):
        if all_log_probs[j]:
            sub_g, _ = builder.build(all_states[j])
            traj_subgraphs.append(sub_g)
            traj_labels.append(all_labels[j])
            traj_log_probs.append(all_log_probs[j])
    assert len(traj_subgraphs) > 0, "No trajectories!"
    print(f"  [PASS] Collected {len(traj_subgraphs)}/{n_total} trajectories")

    # REINFORCE loss
    batch_g = dgl.batch(traj_subgraphs).to(device)
    nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
    with torch.no_grad():
        ext_logits = classifier(batch_g, nids)

    rl_losses = []
    for i, log_probs in enumerate(traj_log_probs):
        correct = 1.0 if ext_logits[i].argmax() == traj_labels[i] else 0.0
        R = 1.0 if correct else -0.1
        rl_losses.append(-R * torch.stack(log_probs).sum())
    rl_loss = torch.stack(rl_losses).mean()

    # Classification loss: CE against ground-truth labels
    final_beliefs = [all_beliefs[j] for j in range(n_total) if all_log_probs[j]]
    final_labels = [all_labels[j].item() for j in range(n_total) if all_log_probs[j]]
    h_batch = torch.cat(final_beliefs, dim=0)
    reader_logits = reader.classify(h_batch)
    cls_loss = F.cross_entropy(
        reader_logits, torch.tensor(final_labels, device=h_batch.device))

    loss = rl_loss + 0.5 * cls_loss
    assert math.isfinite(loss.item()), f"Loss not finite: {loss.item()}"
    print(f"  [PASS] Loss: {loss.item():.4f} (RL={rl_loss.item():.4f}, CLS={cls_loss.item():.4f})")

    # Gradients
    param_before = next(reader.parameters()).clone().detach()
    reader.zero_grad()
    loss.backward()
    has_belief_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in reader.named_parameters() if 'belief_state' in n)
    assert has_belief_grad, "No gradients in belief_state!"
    print(f"  [PASS] Gradients flow through belief_state")

    opt = torch.optim.Adam(reader.parameters(), lr=1e-4)
    opt.step()
    param_after = next(reader.parameters()).clone().detach()
    assert (param_after - param_before).abs().max() > 0
    print(f"  [PASS] Parameters updated (Δmax={(param_after - param_before).abs().max().item():.6f})")


def main():
    parser = argparse.ArgumentParser(description='Phase 2 smoke test')
    parser.add_argument('--graphs', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--bootstrap', default='m5/base/checkpoint/best_reader.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- load graphs ----
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
    ckpt = torch.load(str(_repo_root / args.classifier), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # ---- Test 1: BeliefState module ----
    test_belief_state_module(device)

    # ---- Test 2: FrontierPointer ----
    test_frontier_pointer(device)

    # ---- load reader ----
    print("\n=== Loading LearnableReader (Phase 2) ===")
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=128, n_layers=2, dropout=0.1,
        stochastic=True, temperature=0.2, m=4,
    ).to(device)

    bootstrap_path = _repo_root / args.bootstrap
    if bootstrap_path.exists():
        print(f"  Loading bootstrap from {bootstrap_path}")
        reader_ckpt = torch.load(str(bootstrap_path), map_location=device, weights_only=False)
        from m5.training_utils import patch_old_checkpoint as _pc
        sd, _ = _pc(reader_ckpt['reader_state_dict'])
        missing, unexpected = reader.load_state_dict(sd, strict=False)
        if missing:
            print(f"  New params (random init): {len(missing)} keys")
        if unexpected:
            print(f"  Unused old params: {len(unexpected)} keys")
    else:
        print(f"  WARNING: bootstrap not found, all params random")

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Reader params: {total_params:,}")

    # ---- Test 3: forward_states ----
    test_forward_states_belief(reader, train_graphs, sampled, train_labels['labels'], device)

    # ---- Test 4: full lockstep ----
    test_lockstep_belief(reader, classifier, train_graphs, sampled, train_labels['labels'], device)

    print(f"\n  === All Phase 2 smoke tests PASSED ===")


if __name__ == '__main__':
    main()

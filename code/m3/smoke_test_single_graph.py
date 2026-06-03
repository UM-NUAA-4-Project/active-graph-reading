"""M3 训练管线单图冒烟测试 — 验证 加载→forward→loss→backward 流程。

Usage:
  python code/m3/smoke_test_single_graph.py
  python code/m3/smoke_test_single_graph.py --device cuda:0 --label-smoothing 0.1
"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import torch
import dgl
import numpy as np
from ogb.graphproppred import DglGraphPropPredDataset

from active_graph_reader.encoder.classifier import GraphClassifier

SEPARATOR = "=" * 60


def parse_args():
    p = argparse.ArgumentParser(description='M3 single-graph smoke test')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--class-weight', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # -------------------------------------------------------------------
    # 1. Load 1 graph from PPA training set
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 1: Load single PPA graph")
    print(SEPARATOR)
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    split = dataset.get_idx_split()
    train_idx = split['train']
    g, lbl = dataset[int(train_idx[0])]
    g = dgl.add_self_loop(g)
    label = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
    edge_feat_dim = g.edata['feat'].shape[1]
    n_classes = int(dataset.num_classes)
    print(f"  graph |V|={g.num_nodes()}, |E|={g.num_edges()}, "
          f"edge_dim={edge_feat_dim}, class={label}, n_classes={n_classes}")

    # -------------------------------------------------------------------
    # 2. Create model
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 2: Create GraphClassifier")
    print(SEPARATOR)
    model = GraphClassifier(
        edge_feat_dim=edge_feat_dim,
        n_classes=n_classes,
        hidden_dim=300,
        n_layers=5,
        dropout=0.5,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    model.train()

    # -------------------------------------------------------------------
    # 3. Forward pass
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 3: Forward pass")
    print(SEPARATOR)
    bg = dgl.batch([g]).to(device)
    node_ids = torch.zeros(bg.num_nodes(), dtype=torch.long, device=device)
    logits = model(bg, node_ids)
    print(f"  logits shape: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
    assert logits.shape == (1, n_classes), f"Expected (1, {n_classes}), got {logits.shape}"

    # -------------------------------------------------------------------
    # 4. Compute loss (label_smoothing + class_weight)
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 4: Loss computation")
    print(SEPARATOR)

    class_weights = None
    if args.class_weight:
        # Simulate: compute class weights from 10 training labels
        sample_labels = []
        for i in range(min(100, len(train_idx))):
            _, lbl_i = dataset[int(train_idx[i])]
            sample_labels.append(int(lbl_i.item()) if hasattr(lbl_i, 'item') else int(lbl_i))
        label_counts = np.bincount(sample_labels, minlength=n_classes)
        n_samples = len(sample_labels)
        # pad zeros to avoid division by zero for unseen classes in small sample
        label_counts = np.maximum(label_counts, 1)
        class_weights = n_samples / (n_classes * label_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
        print(f"  class_weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

    target = torch.tensor([label], device=device)
    loss = torch.nn.functional.cross_entropy(
        logits, target,
        label_smoothing=args.label_smoothing,
        weight=class_weights,
    )
    print(f"  label_smoothing={args.label_smoothing}, "
          f"class_weight={'on' if class_weights is not None else 'off'}")
    print(f"  loss={loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be > 0!"

    # -------------------------------------------------------------------
    # 5. Backward pass
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 5: Backward pass")
    print(SEPARATOR)
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    n_params_with_grad = len(grad_norms)
    n_params_total = sum(1 for _ in model.parameters())
    print(f"  params with grad: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad > 0, "No parameters received gradients!"

    # Check a few key layers have non-zero gradients
    zero_grad_params = [name for name, gn in grad_norms.items() if gn == 0.0]
    if zero_grad_params:
        print(f"  [WARN] {len(zero_grad_params)} params have zero grad: {zero_grad_params[:5]}...")
    else:
        print(f"  All params have non-zero gradients")

    # Check gradient in classifier head
    head_grad = grad_norms.get('head.weight', 0.0)
    print(f"  |grad(head.weight)|={head_grad:.6f}")
    assert head_grad > 0, "Classifier head received zero gradient!"

    # -------------------------------------------------------------------
    # 6. Optimizer step (sanity: loss should change)
    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("Test 6: Optimizer step")
    print(SEPARATOR)
    loss_before = loss.item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.step()
    optimizer.zero_grad()

    # Forward again to verify loss changed
    with torch.no_grad():
        logits2 = model(bg, node_ids)
    loss_after = torch.nn.functional.cross_entropy(
        logits2, target,
        label_smoothing=args.label_smoothing,
        weight=class_weights,
    ).item()
    print(f"  loss before step: {loss_before:.4f}")
    print(f"  loss after  step: {loss_after:.4f}")
    assert loss_after != loss_before, "Loss did not change after optimizer step!"

    # -------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("M3 Single-Graph Smoke Test — ALL TESTS PASSED")
    print(SEPARATOR)


if __name__ == '__main__':
    main()

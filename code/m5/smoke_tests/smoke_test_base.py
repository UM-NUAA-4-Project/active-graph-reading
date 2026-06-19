"""Smoke test: 单条轨迹验证 forward→loss→backward→参数更新管线。"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _repo_root.parent
sys.path.insert(0, str(_repo_root))

import torch
import dgl

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.classifier import GraphClassifier
from ogb.graphproppred import DglGraphPropPredDataset


def main():
    parser = argparse.ArgumentParser(description="Smoke test: 1 trajectory forward→backward→update")
    parser.add_argument('--data-dir', default='m4/data/stage4_balanced')
    parser.add_argument('--pretrained', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hidden-dim', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 加载 1 条 trajectory
    data_path = _repo_root / args.data_dir / 'trajectories_balanced_r0_050.pt'
    data = torch.load(str(data_path), map_location='cpu', weights_only=False)
    teacher_names = data['config']['teacher_names']
    print(f"Teachers: {teacher_names}")

    traj = None
    for teacher_name in teacher_names:
        for per_graph_list in data['trajectories'][teacher_name]:
            if per_graph_list:
                traj = per_graph_list[0]
                break
        if traj:
            break

    assert traj is not None, "未找到可用轨迹"
    print(f"Trajectory: graph_idx={traj.graph_idx}, seed={traj.seed_node}, "
          f"teacher={traj.teacher_type}, reveal_order_len={len(traj.reveal_order)}, "
          f"ratio={traj.reveal_ratio:.4f}")

    # 2. 加载全图
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    full_graph, label = dataset[traj.graph_idx]
    full_graph = dgl.add_self_loop(full_graph)
    print(f"Graph: |V|={full_graph.num_nodes()}, |E|={full_graph.num_edges()}, "
          f"label={label.item()}")

    # 3. 加载冻结分类器
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(_repo_root / args.pretrained), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    # require grad for gradient teacher, but we don't need it for BC training
    # Keep grads enabled for parameters so the graph is usable
    print(f"Classifier loaded, val_acc={ckpt.get('val_acc', 'N/A')}")

    # 4. 创建 LearnableReader
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=args.hidden_dim, n_layers=2, dropout=0.1,
    ).to(device)
    reader.train()
    total_params = sum(p.numel() for p in reader.parameters())
    print(f"Reader params: {total_params:,}")

    # 5. Optimizer
    optimizer = torch.optim.Adam(reader.parameters(), lr=1e-3)

    # --- 记录参数前快照 ---
    before_params = {}
    for name, p in reader.named_parameters():
        before_params[name] = p.detach().clone()

    # 6. 逐步 replay 轨迹（full_graph 留在 CPU，reader 内部处理 device）
    state = GraphState(full_graph, seed_node=traj.seed_node)
    step_losses = []

    print(f"\n--- Step-by-step reveal ---")
    for step, teacher_node in enumerate(traj.reveal_order):
        scores, probs, frontier_nodes = reader.forward(state)
        n_frontier = len(frontier_nodes)

        if n_frontier == 0:
            print(f"  Step {step+1}: frontier empty, break")
            break

        # teacher_node 必须在前沿中（相同 state 重放）
        try:
            target_idx = frontier_nodes.index(teacher_node)
        except ValueError:
            print(f"  Step {step+1}: teacher_node={teacher_node} NOT in frontier "
                  f"(frontier={frontier_nodes[:5]}...), skip")
            break

        # CE = -log(p_target)，其中 p = softmax(scores)
        loss_t = -torch.log(probs[target_idx] + 1e-8)
        step_losses.append(loss_t)

        max_prob = probs.max().item()
        teacher_prob = probs[target_idx].item()
        print(f"  Step {step+1}: frontier={n_frontier}, teacher_node={teacher_node}, "
              f"target_idx={target_idx}, teacher_prob={teacher_prob:.4f}, "
              f"max_prob={max_prob:.4f}, loss={loss_t.item():.4f}")

        reveal_node(state, teacher_node)

    # 7. Backward
    total_loss = torch.stack(step_losses).mean()
    print(f"\nMean step loss: {total_loss.item():.4f}")
    print("Running backward...")
    total_loss.backward()

    # 8. 验证梯度
    print("\n--- Gradient check ---")
    grad_ok = True
    for name, p in reader.named_parameters():
        if p.grad is None:
            print(f"  {name}: grad is None!")
            grad_ok = False
        elif p.grad.abs().sum() == 0:
            print(f"  {name}: grad is all zero!")
            grad_ok = False
        else:
            print(f"  {name}: grad norm={p.grad.norm().item():.6f}, "
                  f"shape={list(p.grad.shape)}")

    assert grad_ok, "Some parameters have no gradient!"

    # 9. Optimizer step
    print("\n--- Optimizer step ---")
    optimizer.step()

    # 10. 验证参数变化
    print("\n--- Parameter change ---")
    changed = False
    for name, p in reader.named_parameters():
        diff = (p.detach() - before_params[name]).abs().sum().item()
        if diff > 0:
            changed = True
            print(f"  {name}: changed, total_abs_diff={diff:.8f}")
        else:
            print(f"  {name}: NOT changed!")
    assert changed, "No parameters changed after optimizer.step()!"

    print("\nSmoke test PASSED: forward -> loss -> backward -> update all work.")


if __name__ == '__main__':
    main()

"""Base Learnable Reader — Behavior Cloning 训练。

基于 M4 teacher 轨迹，使用 BC 训练 LearnableReader：
每步 teacher 选择的 frontier node 转为 one-hot target，
与 reader softmax 输出计算 cross-entropy loss。
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))

import numpy as np
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.encoder.classifier import GraphClassifier
from active_graph_reader.encoder.subgraph_builder import SubgraphBuilder
from ogb.graphproppred import DglGraphPropPredDataset

TEACHER_NAMES = [
    'gradient', 'structural-degree', 'structural-pagerank',
    'hybrid-a0.30', 'hybrid-a0.70',
]
BUDGET_RATIOS = ['0_050', '0_100', '0_150', '0_200']
BUDGETS_REL = [0.05, 0.10, 0.15, 0.20]


def merge_trajectories(data_dir: Path) -> List:
    """加载 stage4_balanced 中所有 .pt 文件，展平为 List[Trajectory]。"""
    from m4.teachers.base import Trajectory

    all_trajs: List[Trajectory] = []
    for ratio_key in BUDGET_RATIOS:
        fname = f'trajectories_balanced_r{ratio_key}.pt'
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        trajs_dict = data['trajectories']
        n_before = len(all_trajs)
        for teacher_name in TEACHER_NAMES:
            if teacher_name not in trajs_dict:
                continue
            for per_graph_list in trajs_dict[teacher_name]:
                all_trajs.extend(per_graph_list)
        print(f"  {fname}: +{len(all_trajs) - n_before} trajectories")
    print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs


def split_by_graph(trajectories: List, val_ratio: float = 0.2, seed: int = 42):
    """按 graph_idx 分层分割 train/val，确保同一图不跨集合。"""
    rng = random.Random(seed)
    graph_to_trajs: Dict[int, List] = defaultdict(list)
    for t in trajectories:
        graph_to_trajs[t.graph_idx].append(t)

    unique_graphs = sorted(graph_to_trajs.keys())
    rng.shuffle(unique_graphs)
    n_val = max(1, int(len(unique_graphs) * val_ratio))
    val_graphs = set(unique_graphs[:n_val])
    train_graphs = set(unique_graphs[n_val:])

    train_trajs = []
    val_trajs = []
    for gid, t_list in graph_to_trajs.items():
        if gid in val_graphs:
            val_trajs.extend(t_list)
        else:
            train_trajs.extend(t_list)

    print(f"Train: {len(train_trajs)} trajectories ({len(train_graphs)} graphs)")
    print(f"Val:   {len(val_trajs)} trajectories ({len(val_graphs)} graphs)")
    return train_trajs, val_trajs


def load_graphs_for_trajectories(root: str, needed_graphs: set):
    """只加载轨迹引用的训练集图，加自环后缓存，释放不相关图。

    轨迹全部来自训练集，无需加载 val/test 图。
    """
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
    train_indices = set(dataset.get_idx_split()['train'].tolist())

    # 只加载训练集中被轨迹引用的图
    to_load = needed_graphs & train_indices
    skipped = needed_graphs - train_indices
    if skipped:
        print(f"  WARNING: {len(skipped)} graph_idx not in training set, ignored")

    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]] = {}
    print(f"Loading {len(to_load)} graphs (training set subset)...")
    for i in tqdm(sorted(to_load), desc="  Loading graphs"):
        g, lbl = dataset[i]
        g = dgl.add_self_loop(g)
        graph_cache[i] = (g, lbl.item())
    return graph_cache


def compute_budget(n_nodes: int, ratio: float) -> int:
    return max(1, int(ratio * n_nodes))


def validate(
    reader: LearnableReader,
    classifier: GraphClassifier,
    builder: SubgraphBuilder,
    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]],
    val_trajs: List,
    device: torch.device,
) -> Dict[float, float]:
    """在 val 集上评估 reader 性能：每个图一个 seed，greedy reveal。

    为减少推理开销，每张图只用第一条轨迹的 seed。
    """
    # 每张图只取第一个 seed
    seen_graphs = set()
    tasks: List[Tuple[int, int, float]] = []  # (graph_idx, seed, budget_ratio)
    for t in val_trajs:
        if t.graph_idx in seen_graphs:
            continue
        seen_graphs.add(t.graph_idx)
        for br in BUDGETS_REL:
            tasks.append((t.graph_idx, t.seed_node, br))

    print(f"  Validating on {len(seen_graphs)} graphs x {len(BUDGETS_REL)} budgets = {len(tasks)} tasks")

    reader.eval()
    results: Dict[float, List[float]] = {br: [] for br in BUDGETS_REL}

    # 批量处理：先跑完所有 reveal，再批量分类
    batch_size = 64
    for batch_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[batch_start:batch_start + batch_size]

        states = []
        budgets_used = []
        for graph_idx, seed, br in batch_tasks:
            g, _label = graph_cache[graph_idx]
            b = compute_budget(g.num_nodes(), br)
            state = GraphState(g, seed_node=seed)

            for _step in range(b):
                next_node = reader.select(state)
                if next_node is None:
                    break
                reveal_node(state, next_node)

            states.append(state)
            budgets_used.append(br)

        # 批量分类
        if states:
            batched_g, _all_ids = builder.build_batch(states)
            batched_g = batched_g.to(device)
            node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = classifier(batched_g, node_ids)
                preds = logits.argmax(dim=1).cpu()

            for i, (graph_idx, _seed, br) in enumerate(batch_tasks):
                _, label = graph_cache[graph_idx]
                results[br].append(float(preds[i] == label))

    reader.train()
    return {br: float(np.mean(accs)) if accs else 0.0 for br, accs in results.items()}


def train_one_epoch(
    reader: LearnableReader,
    optimizer: torch.optim.Optimizer,
    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]],
    train_trajs: List,
    batch_size: int,
    device: torch.device,
) -> float:
    """单轮 BC 训练，返回平均 step loss。

    batch_size 条轨迹为一个 batch：每条轨迹内部 step losses 求均值，
    再跨 batch 求均值，单次 backward + optimizer.step()。
    """
    reader.train()
    total_loss = 0.0
    total_steps = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]
        batch_losses = []
        batch_steps = 0

        for traj in batch_trajs:
            g, _label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)

            step_losses = []
            for teacher_node in traj.reveal_order:
                scores, probs, frontier_nodes = reader.forward(state)
                n_frontier = len(frontier_nodes)

                if n_frontier == 0:
                    break

                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    break

                # CE = -log(p_target)，其中 p = softmax(scores)
                loss_t = -torch.log(probs[target_idx] + 1e-8)
                step_losses.append(loss_t)
                reveal_node(state, teacher_node)

            if step_losses:
                traj_loss = torch.stack(step_losses).mean()
                batch_losses.append(traj_loss)
                batch_steps += len(step_losses)

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item() * len(batch_losses)
        total_steps += batch_steps

        avg_step_loss = total_loss / max(total_steps, 1)
        pbar.set_postfix({
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_step_loss': f'{avg_step_loss:.4f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
        })

    return total_loss / max(total_steps, 1)


def main():
    parser = argparse.ArgumentParser(description='BC Training for LearnableReader')
    parser.add_argument('--data-dir', default='m4/data/stage4_balanced')
    parser.add_argument('--pretrained', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-dir', default='m5/')
    parser.add_argument('--resume', default='')
    parser.add_argument('--n_layers_reader', type=int, default=2)
    parser.add_argument('--dropout_reader', type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = _script_dir.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 合并轨迹
    data_dir = _script_dir.parent / args.data_dir
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir)

    # 2. Train/val split
    print("\n=== Splitting train/val ===")
    train_trajs, val_trajs = split_by_graph(all_trajs, val_ratio=0.2)

    # 3. 只加载轨迹引用的训练集图
    print("\n=== Loading graphs ===")
    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(_script_dir.parent / args.pretrained), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # 4. 创建 Reader 和 optimizer
    print("\n=== Creating LearnableReader ===")
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=args.hidden_dim, n_layers=args.n_layers_reader, dropout=args.dropout_reader,
    ).to(device)
    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Params: {total_params:,}")

    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    start_epoch = 0
    best_val_acc = 0.0
    training_log = []

    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device)
        reader.load_state_dict(ckpt['reader_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        training_log = ckpt.get('training_log', [])

    builder = SubgraphBuilder()

    # 5. Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        avg_loss = train_one_epoch(
            reader, optimizer, graph_cache, train_trajs,
            args.batch_size, device,
        )
        print(f"  Avg step loss: {avg_loss:.4f}")

        # Validation
        print("  Running validation...")
        val_accs = validate(reader, classifier, builder, graph_cache, val_trajs, device)
        val_summary = {f'r{br:.2f}': acc for br, acc in val_accs.items()}
        avg_val_acc = float(np.mean(list(val_accs.values())))
        print(f"  Val acc: {val_summary}, avg={avg_val_acc:.4f}")

        log_entry = {
            'epoch': epoch + 1,
            'avg_step_loss': avg_loss,
            'val_acc': val_summary,
            'val_acc_avg': avg_val_acc,
        }
        training_log.append(log_entry)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': max(best_val_acc, avg_val_acc),
            'training_log': training_log,
            'config': {
                'hidden_dim': args.hidden_dim,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'n_layers_reader': args.n_layers_reader,
                'dropout_reader': args.dropout_reader,
            },
        }

        # 每轮保存
        ckpt_path = output_dir / f'reader_checkpoint_epoch{epoch + 1}.pt'
        torch.save(checkpoint, str(ckpt_path))

        # 最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_path = output_dir / 'best_reader.pt'
            torch.save(checkpoint, str(best_path))
            print(f"  => Best model saved (val_acc={best_val_acc:.4f})")

        # 写日志
        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()

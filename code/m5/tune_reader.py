"""BC 微调 LearnableReader：从 best_reader.pt 恢复，用更低学习率继续模仿 teacher 轨迹。

与 train_bc.py 共享核心训练逻辑（BC + CrossEntropy），差异在于：
  - 必须从 checkpoint 恢复（不从零训练）
  - 自动从 state_dict 推断 hidden_dim / n_layers
  - 更低默认学习率（1e-4 vs 1e-3）
  - 输出到独立目录（m5/finetune/）
  - 可选学习率调度（ReduceLROnPlateau）
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

# 复用 train_bc 的数据加载和训练逻辑
from m4.teachers.base import Trajectory

TEACHER_NAMES = [
    'gradient', 'structural-degree', 'structural-pagerank',
    'hybrid-a0.30', 'hybrid-a0.70',
]
BUDGET_RATIOS = ['0_050', '0_100', '0_150', '0_200']
BUDGETS_REL = [0.05, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# 从 train_bc.py 复用的数据 / 训练函数（避免跨文件 import 的路径问题）
# ---------------------------------------------------------------------------

def merge_trajectories(data_dir: Path) -> List[Trajectory]:
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


def split_by_graph(trajectories: List[Trajectory], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    graph_to_trajs: Dict[int, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        graph_to_trajs[t.graph_idx].append(t)

    unique_graphs = sorted(graph_to_trajs.keys())
    rng.shuffle(unique_graphs)
    n_val = max(1, int(len(unique_graphs) * val_ratio))
    val_graphs = set(unique_graphs[:n_val])
    train_graphs = set(unique_graphs[n_val:])

    train_trajs, val_trajs = [], []
    for gid, t_list in graph_to_trajs.items():
        (val_trajs if gid in val_graphs else train_trajs).extend(t_list)

    print(f"Train: {len(train_trajs)} trajectories ({len(train_graphs)} graphs)")
    print(f"Val:   {len(val_trajs)} trajectories ({len(val_graphs)} graphs)")
    return train_trajs, val_trajs


def load_graphs_for_trajectories(root: str, needed_graphs: set) -> Dict[int, Tuple[dgl.DGLGraph, int]]:
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
    train_indices = set(dataset.get_idx_split()['train'].tolist())
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
    val_trajs: List[Trajectory],
    device: torch.device,
) -> Dict[float, float]:
    seen_graphs = set()
    tasks: List[Tuple[int, int, float]] = []
    for t in val_trajs:
        if t.graph_idx in seen_graphs:
            continue
        seen_graphs.add(t.graph_idx)
        for br in BUDGETS_REL:
            tasks.append((t.graph_idx, t.seed_node, br))

    print(f"  Validating on {len(seen_graphs)} graphs x {len(BUDGETS_REL)} budgets = {len(tasks)} tasks")

    reader.eval()
    results: Dict[float, List[float]] = {br: [] for br in BUDGETS_REL}

    batch_size = 64
    for batch_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[batch_start:batch_start + batch_size]
        states = []
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
    train_trajs: List[Trajectory],
    batch_size: int,
    device: torch.device,
) -> float:
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


# ---------------------------------------------------------------------------
# 从 state_dict 推断 reader 配置
# ---------------------------------------------------------------------------

def infer_reader_config(state_dict: dict) -> Tuple[int, int]:
    hidden_dim = state_dict['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in state_dict:
        if k.startswith('frontier_encoder.convs.'):
            layer_idx = int(k.split('.')[2])
            n_layers = max(n_layers, layer_idx + 1)
    return hidden_dim, n_layers


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BC fine-tune LearnableReader from checkpoint')
    parser.add_argument('--checkpoint', default='m5/checkpoint/best_reader.pt',
                        help='Reader checkpoint to fine-tune from')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--data-dir', default='m4/data/stage4_balanced')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr-patience', type=int, default=2,
                        help='ReduceLROnPlateau patience (0=disabled)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repo_root = _script_dir.parent
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载轨迹 & 图 ----
    data_dir = repo_root / args.data_dir
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir)

    print("\n=== Splitting train/val ===")
    train_trajs, val_trajs = split_by_graph(all_trajs, val_ratio=0.2, seed=args.seed)

    print("\n=== Loading graphs ===")
    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    # ---- 2. 加载 classifier（冻结） ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 3. 从 checkpoint 恢复 reader ----
    print(f"\n=== Loading reader from {args.checkpoint} ===")
    reader_ckpt = torch.load(str(repo_root / args.checkpoint), map_location='cpu', weights_only=False)
    hidden_dim, n_layers = infer_reader_config(reader_ckpt['reader_state_dict'])
    print(f"  Inferred: hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  Checkpoint epoch={reader_ckpt.get('epoch', 'N/A')}, "
          f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=args.dropout,
    ).to(device)
    reader.load_state_dict(reader_ckpt['reader_state_dict'])
    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Params: {total_params:,}")

    # 恢复 optimizer state（继续用 Adam 动量）
    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    try:
        optimizer.load_state_dict(reader_ckpt['optimizer_state_dict'])
    except (KeyError, ValueError) as e:
        print(f"  WARNING: Could not load optimizer state ({e}), using fresh optimizer")

    start_epoch = reader_ckpt.get('epoch', -1) + 1
    best_val_acc = reader_ckpt.get('best_val_acc', 0.0)
    training_log = reader_ckpt.get('training_log', [])

    # 学习率调度
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.lr_factor,
            patience=args.lr_patience, verbose=True,
        )
        print(f"  LR scheduler: ReduceLROnPlateau(patience={args.lr_patience}, "
              f"factor={args.lr_factor})")

    builder = SubgraphBuilder()

    # ---- 4. 训练循环 ----
    print(f"\n=== BC fine-tuning: epoch {start_epoch + 1} → {args.epochs}, lr={args.lr} ===")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        avg_loss = train_one_epoch(
            reader, optimizer, graph_cache, train_trajs, args.batch_size, device,
        )
        print(f"  Avg step loss: {avg_loss:.4f}")

        # 验证
        print("  Running validation...")
        val_accs = validate(reader, classifier, builder, graph_cache, val_trajs, device)
        val_summary = {f'r{br:.2f}': acc for br, acc in val_accs.items()}
        avg_val_acc = float(np.mean(list(val_accs.values())))
        print(f"  Val acc: {val_summary}, avg={avg_val_acc:.4f}")

        # LR 调度
        if scheduler is not None:
            scheduler.step(avg_val_acc)

        log_entry = {
            'epoch': epoch + 1,
            'avg_step_loss': avg_loss,
            'val_acc': val_summary,
            'val_acc_avg': avg_val_acc,
        }
        training_log.append(log_entry)

        # 保存
        checkpoint = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': max(best_val_acc, avg_val_acc),
            'training_log': training_log,
            'config': {
                'hidden_dim': hidden_dim,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'n_layers_reader': n_layers,
                'dropout_reader': args.dropout,
                'finetune_from': str(repo_root / args.checkpoint),
            },
        }

        ckpt_path = output_dir / f'reader_ft_epoch{epoch + 1}.pt'
        torch.save(checkpoint, str(ckpt_path))

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_path = output_dir / 'best_reader_ft.pt'
            torch.save(checkpoint, str(best_path))
            print(f"  => Best model saved (val_acc={best_val_acc:.4f})")

        log_path = output_dir / 'finetune_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nFine-tuning complete. Best val acc: {best_val_acc:.4f}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()

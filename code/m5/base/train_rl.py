"""v0 LearnableReader — Self-Sampled REINFORCE 微调。

不再使用 teacher 轨迹。每 batch 从训练集加载 x 张图，每张图用 n 个确定性 seed
分别采样一条轨迹（temperature 控制探索程度），classifier 的 CE 作为 reward 信号。

REINFORCE: loss = -R · Σ_t log π_t(a_t)  (batch 平均)
  R (binary): 1 if correct else 0
  R (tanh):   tanh(1 - CE / ln(37))
  CE = cross_entropy(classifier(subgraph), y)  仅用于计算 R，loss 本身用 log_prob
  budget ∈ {0.05, 0.10, 0.15}  随机抽取

batch size = x · n 条轨迹

用法:
python code/m5/base/train_rl.py --root /mnt/data --reward-type binary --epochs 10 
--temperature 0.2 --resume-checkpoint m5/base/rl/binary/checkpoint_epoch1.pt 
--graphs-per-batch 16 --seeds-per-graph 2 --output-dir m5/base/rl/binary
  python code/m5/base/train_rl.py --root /mnt/data --epochs 10 --temperature 1.5
  python code/m5/base/train_rl.py --root /mnt/data --temperature 0.2 --val-graphs 10000
  python code/m5/base/train_rl.py --root /mnt/data --reward-type tanh --epochs 10 --temperature 0.2
"""
import argparse
import json
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
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from training_utils import patch_old_checkpoint

K = 37
LN_K = math.log(K)
BUDGET_OPTIONS = [0.05, 0.10, 0.15]


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    """确定性种子公式（与 CLAUDE.md 一致）。"""
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


# ---------------------------------------------------------------------------
# self-sampled REINFORCE training
# ---------------------------------------------------------------------------

def train_one_epoch(
    reader, classifier, builder, optimizer,
    graphs, ogb_indices, labels_tensor,
    n_seeds, batch_size, device,
    reward_type='binary',
):
    """Self-sampled REINFORCE。每张图用 n_seeds 个 seed 各采样一条轨迹。"""
    reader.train()
    classifier.eval()

    total_R = 0.0
    total_trajs = 0
    total_correct = 0
    total_loss = 0.0

    # shuffle graph indices for batching
    graph_order = list(range(len(graphs)))
    random.shuffle(graph_order)

    pbar = tqdm(range(0, len(graph_order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(graph_order))
        batch_indices = graph_order[batch_start:batch_end]

        traj_subgraphs = []
        traj_labels = []
        traj_log_probs = []

        for local_i in batch_indices:
            g_raw = graphs[local_i]
            graph_idx = ogb_indices[local_i]
            label = int(labels_tensor[graph_idx].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()

            # 随机 budget
            budget_ratio = random.choice(BUDGET_OPTIONS)
            B = max(1, int(budget_ratio * n_nodes))

            for seed_i in range(n_seeds):
                seed_node = deterministic_seed(graph_idx, seed_i, n_nodes)

                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                reader.reset()
                initial_frontier = list(state._initial_frontier)
                if initial_frontier:
                    reader.on_new_frontier(initial_frontier)

                log_probs = []
                for _step in range(B):
                    if state.is_frontier_empty:
                        break

                    scores, probs, frontier_nodes, action_idx, log_prob = reader.forward(state, sample=True)

                    if len(frontier_nodes) == 0:
                        break
                    log_probs.append(log_prob)
                    chosen = frontier_nodes[action_idx.item()]
                    reveal_node(state, chosen)

                if not log_probs:
                    continue

                sub_g, _ = builder.build(state)
                traj_subgraphs.append(sub_g)
                traj_labels.append(y)
                traj_log_probs.append(log_probs)

        if not traj_subgraphs:
            continue

        # 批量 classifier 前向
        batch_g = dgl.batch(traj_subgraphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            all_logits = classifier(batch_g, nids)  # (n_trajs, K)

        # 逐 trajectory 计算 reward 和 loss
        batch_losses = []
        for i, log_probs in enumerate(traj_log_probs):
            logits_i = all_logits[i:i + 1]
            ce = F.cross_entropy(logits_i, traj_labels[i]).item()
            correct = 1.0 if logits_i.argmax(dim=-1).item() == traj_labels[i].item() else 0.0

            if correct:
                total_correct += 1

            if reward_type == 'tanh':
                R = math.tanh(1.0 - ce / LN_K)
            elif reward_type == 'asymmetric':
                R = 1.0 if correct else -0.1
            else:  # binary
                R = 1.0 if correct else 0.0

            total_R += R
            total_trajs += 1

            # REINFORCE: loss = -R * Σ log π_t(a_t)
            sum_log_prob = torch.stack(log_probs).sum()
            rl_loss = -R * sum_log_prob
            batch_losses.append(rl_loss)

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item() * len(batch_losses)
        n = max(total_trajs, 1)
        pbar.set_postfix({
            'loss': f'{total_loss / n:.4f}',
            'avg_R': f'{total_R / n:.3f}',
            'acc': f'{total_correct / n:.3f}',
        })

    n = max(total_trajs, 1)
    return {
        'loss': total_loss / n,
        'avg_R': total_R / n,
        'acc': total_correct / n,
    }


# ---------------------------------------------------------------------------
# validation (deterministic argmax)
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_one_epoch(
    reader, classifier, builder,
    graphs, ogb_indices, labels_tensor,
    device,
):
    reader.eval()
    total_ce = 0.0
    total_trajs = 0
    total_correct = 0

    # 第一阶段：收集所有 (sub_g, y, label) — reveal 仍逐轨迹串行
    val_subgraphs = []
    val_labels = []

    for local_i in tqdm(range(len(graphs)), desc="  Validating"):
        g_raw = graphs[local_i]  # 已预加载到 GPU
        graph_idx = ogb_indices[local_i]
        label = int(labels_tensor[graph_idx].item())
        y = torch.tensor([label], device=device)
        n_nodes = g_raw.num_nodes()

        for budget_ratio in BUDGET_OPTIONS:
            B = max(1, int(budget_ratio * n_nodes))
            seed_node = deterministic_seed(graph_idx, 0, n_nodes)

            state = GraphState(g_raw, seed_node=seed_node)
            state._budget_limit = B
            reader.reset()
            initial_frontier = list(state._initial_frontier)
            if initial_frontier:
                reader.on_new_frontier(initial_frontier)

            for _step in range(B):
                if state.is_frontier_empty:
                    break
                scores, probs, frontier_nodes = reader.forward(state)
                if len(frontier_nodes) == 0:
                    break
                chosen = frontier_nodes[probs.argmax().item()]
                reveal_node(state, chosen)

            sub_g, _ = builder.build(state)
            val_subgraphs.append(sub_g)
            val_labels.append((y, label))

    # 第二阶段：批量 classifier 前向
    for i in tqdm(range(0, len(val_subgraphs), 512), desc="  Classifying"):
        chunk_sub = val_subgraphs[i:i + 512]
        chunk_labels = val_labels[i:i + 512]

        batch_g = dgl.batch(chunk_sub).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)  # (chunk_size, K)

        for j, (y, label) in enumerate(chunk_labels):
            ce = F.cross_entropy(logits[j:j + 1], y).item()
            if logits[j].argmax(dim=-1).item() == label:
                total_correct += 1
            total_ce += ce
            total_trajs += 1

    n = max(total_trajs, 1)
    return {
        'avg_CE': total_ce / n,
        'acc': total_correct / n,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='v0 LearnableReader Self-Sampled REINFORCE')
    parser.add_argument('--bootstrap-checkpoint', default='m5/base/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--graphs-per-batch', type=int, default=32,
                        help='每 batch 的图数量 (x)')
    parser.add_argument('--seeds-per-graph', type=int, default=4,
                        help='每张图用几个 seed (n), batch size = x * n')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='采样温度 (T>1 更多探索, T<1 更多利用)')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='限制加载训练图总数 (0=全部)')
    parser.add_argument('--output-dir', default='m5/base/rl/binary')
    parser.add_argument('--resume-checkpoint', default='',
                        help='从指定 checkpoint 恢复训练')
    parser.add_argument('--val-graphs', type=int, default=0,
                        help='验证集图数量 (0=跳过验证)')
    parser.add_argument('--reward-type', choices=['binary', 'tanh', 'asymmetric'], default='binary',
                        help='R 计算方式: binary=0/1正确性, tanh=tanh(1-CE/ln37), asymmetric=正确+1/错误-0.1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载训练图 ----
    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'

    train_indices = pd.read_csv(train_csv, compression='gzip', header=None).values.flatten().tolist()
    if args.max_graphs > 0 and args.max_graphs < len(train_indices):
        train_indices = train_indices[:args.max_graphs]

    print(f"\n=== Loading {len(train_indices)} training graphs ===")
    train_graphs, train_labels = dgl.load_graphs(processed_path, idx_list=train_indices)
    # 预计算自环 + 预加载到 GPU（避免每次 trajectory 重复 CPU→GPU 传输）
    train_graphs = [dgl.add_self_loop(g).to(device) for g in train_graphs]
    print(f"  Loaded {len(train_graphs)} graphs on {device} (self-loops pre-computed)")

    # ---- 2. 加载 val 图（从 valid split，分层抽样） ----
    val_graphs, val_labels, val_indices = None, None, None
    if args.val_graphs > 0:
        val_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'valid.csv.gz'
        all_val_indices = pd.read_csv(val_csv, compression='gzip', header=None).values.flatten().tolist()
        print(f"=== Loading {len(all_val_indices)} validation graphs ===")
        all_graphs, all_labels = dgl.load_graphs(processed_path, idx_list=all_val_indices)
        print(f"  Loaded {len(all_graphs)} graphs")

        if args.val_graphs < len(all_graphs):
            from collections import defaultdict
            class_to_local = defaultdict(list)
            for i, ogb_idx in enumerate(all_val_indices):
                lbl = int(all_labels['labels'][ogb_idx].item())
                class_to_local[lbl].append(i)
            total = len(all_val_indices)
            sampled = []
            for cls in range(K):
                pool = class_to_local.get(cls, [])
                n = max(1, int(args.val_graphs * len(pool) / total))
                n = min(n, len(pool))
                if n > 0:
                    sampled.extend(random.sample(pool, n))
            if len(sampled) > args.val_graphs:
                sampled = random.sample(sampled, args.val_graphs)
            elif len(sampled) < args.val_graphs:
                remaining = [i for i in range(len(all_graphs)) if i not in set(sampled)]
                sampled.extend(random.sample(remaining, min(args.val_graphs - len(sampled), len(remaining))))
            val_graphs = [dgl.add_self_loop(all_graphs[i]).to(device) for i in sampled]
            val_indices = [all_val_indices[i] for i in sampled]
            val_labels = {'labels': all_labels['labels'][[i for i in val_indices]]}
        else:
            val_graphs = [dgl.add_self_loop(g).to(device) for g in all_graphs]
            val_labels, val_indices = all_labels, all_val_indices

        print(f"  Validation set: {len(val_graphs)} graphs on {device} (self-loops pre-computed) "
              f"({len(set(int(val_labels['labels'][idx].item()) for idx in val_indices))}/{K} classes)")

    # ---- 3. 加载 classifier（冻结） ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt_path = repo_root / args.classifier
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 4. 加载/恢复 reader ----
    print("\n=== Loading LearnableReader ===")
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, dropout=args.dropout,
        stochastic=True, temperature=args.temperature,
    ).to(device)

    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    builder = SubgraphBuilder()
    training_log = []
    start_epoch = 0

    if args.resume_checkpoint:
        resume_path = repo_root / args.resume_checkpoint
        print(f"  Resuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        reader_sd, was_patched = patch_old_checkpoint(ckpt['reader_state_dict'])
        reader.load_state_dict(reader_sd)
        if was_patched:
            print("  NOTE: checkpoint architecture patched, using fresh optimizer")
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, RuntimeError):
                print("  WARNING: optimizer state incompatible, using fresh optimizer")
        training_log = ckpt.get('training_log', [])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch + 1}/{args.epochs}")
    else:
        bootstrap_path = repo_root / args.bootstrap_checkpoint
        if bootstrap_path.exists():
            print(f"  Loading bootstrap from {bootstrap_path}")
            reader_ckpt = torch.load(str(bootstrap_path), map_location=device, weights_only=False)
            reader_sd, _ = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
            reader.load_state_dict(reader_sd)
            print(f"  Checkpoint epoch: {reader_ckpt.get('epoch', 'N/A')}, "
                  f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")
        else:
            print(f"  WARNING: {bootstrap_path} not found, training from scratch")

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Reader params: {total_params:,}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Reward type: {args.reward_type}")
    print(f"  Graphs per batch: {args.graphs_per_batch}, Seeds per graph: {args.seeds_per_graph}")
    print(f"  Effective batch size: {args.graphs_per_batch * args.seeds_per_graph} trajectories")

    # ---- 5. Self-Sampled REINFORCE 训练 ----
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        metrics = train_one_epoch(
            reader, classifier, builder, optimizer,
            train_graphs, train_indices, train_labels['labels'],
            args.seeds_per_graph, args.graphs_per_batch, device,
            reward_type=args.reward_type,
        )
        print(f"  Train: loss={metrics['loss']:.4f}  avg_R={metrics['avg_R']:.3f}  acc={metrics['acc']:.3f}")

        log_entry = {'epoch': epoch + 1, 'train_loss': metrics['loss'],
                     'train_avg_R': metrics['avg_R'], 'train_acc': metrics['acc']}

        if val_graphs is not None:
            val_metrics = val_one_epoch(
                reader, classifier, builder,
                val_graphs, val_indices, val_labels['labels'],
                device,
            )
            print(f"  Val:   avg_CE={val_metrics['avg_CE']:.3f}  acc={val_metrics['acc']:.3f}")
            log_entry.update({f'val_{k}': v for k, v in val_metrics.items()})

        training_log.append(log_entry)

        ckpt = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
        }
        ckpt_path = output_dir / f'checkpoint_epoch{epoch + 1}.pt'
        torch.save(ckpt, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nREINFORCE complete. Output: {output_dir}")


if __name__ == '__main__':
    main()

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
  python code/m5/base/train_rl.py --root /mnt/data --temperature 0.2 --reward-type asymmetric --output-dir m5/base/rl/asymmetric --resume-checkpoint m5/base/rl/asymmetric/checkpoint_epoch6.pt
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
    """Self-sampled REINFORCE with cross-graph lockstep.

    All graphs in a batch run simultaneously — batch_size × n_seeds states
    advance step-by-step together through a single lockstep loop.  Graphs
    with smaller budgets drop out early while larger graphs keep running.
    Loss and optimizer.step() fire once per batch over all trajectories.
    """
    reader.train()
    classifier.eval()

    total_R = 0.0
    total_trajs = 0
    total_correct = 0
    total_loss = 0.0

    graph_order = list(range(len(graphs)))
    random.shuffle(graph_order)

    pbar = tqdm(range(0, len(graph_order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(graph_order))
        batch_indices = graph_order[batch_start:batch_end]
        n_graphs = len(batch_indices)

        # ---- Pass 1: create ALL states (n_graphs × n_seeds) ----
        all_states = []
        state_labels = []       # label tensor per state
        per_state_log_probs = []  # list-of-lists: log-prob per step per state

        for gi, local_i in enumerate(batch_indices):
            g_raw = graphs[local_i]
            graph_idx = ogb_indices[local_i]
            label = int(labels_tensor[graph_idx].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()

            budget_ratio = random.choice(BUDGET_OPTIONS)
            B = max(1, int(budget_ratio * n_nodes))

            for seed_i in range(n_seeds):
                seed_node = deterministic_seed(graph_idx, seed_i, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                all_states.append(state)
                state_labels.append(y)
                per_state_log_probs.append([])

        n_total = len(all_states)  # = n_graphs × n_seeds
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)

        # ---- Pass 2: cross-graph lockstep ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break

            batch_results = reader.forward_states(
                [all_states[j] for j in active_idx],
                temperature=reader.temperature,
            )

            for k, j in enumerate(active_idx):
                scores, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue

                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                per_state_log_probs[j].append(dist.log_prob(action_idx))
                reveal_node(all_states[j], frontier_nodes[action_idx.item()])

                # budget exhausted → drop from dgl batch
                if all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        # ---- Pass 3: collect trajectories ----
        traj_subgraphs = []
        traj_labels = []
        traj_log_probs = []

        for j in range(n_total):
            if per_state_log_probs[j]:
                sub_g, _ = builder.build(all_states[j])
                traj_subgraphs.append(sub_g)
                traj_labels.append(state_labels[j])
                traj_log_probs.append(per_state_log_probs[j])

        if not traj_subgraphs:
            continue

        # ---- batch classify + REINFORCE loss (unchanged logic) ----
        batch_g = dgl.batch(traj_subgraphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            all_logits = classifier(batch_g, nids)

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
            else:
                R = 1.0 if correct else 0.0

            total_R += R
            total_trajs += 1

            sum_log_prob = torch.stack(log_probs).sum()
            batch_losses.append(-R * sum_log_prob)

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
    parser.add_argument('--train-split', default='',
                        help='自定义训练集索引 CSV（gzip, 单列 ogb_idx），'
                             '由 filter_by_classifier.py 生成；为空则用官方 train.csv.gz')
    parser.add_argument('--output-dir', default='m5/base/rl/binary')
    parser.add_argument('--resume-checkpoint', default='',
                        help='从指定 checkpoint 恢复训练')
    parser.add_argument('--reward-type', choices=['binary', 'tanh', 'asymmetric'], default='binary',
                        help='R 计算方式: binary=0/1正确性, tanh=tanh(1-CE/ln37), asymmetric=正确+1/错误-0.1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载训练图 ----
    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    if args.train_split:
        split_path = repo_root / args.train_split
        if not split_path.exists():
            raise FileNotFoundError(f"Train split not found: {split_path}")
        train_indices = pd.read_csv(split_path, compression='gzip', header=None
                                    ).values.flatten().tolist()
        split_label = str(split_path)
    else:
        train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'
        train_indices = pd.read_csv(train_csv, compression='gzip', header=None
                                    ).values.flatten().tolist()
        split_label = 'full training set (train.csv.gz)'
    if args.max_graphs > 0 and args.max_graphs < len(train_indices):
        train_indices = train_indices[:args.max_graphs]

    print(f"\n=== Loading {len(train_indices)} training graphs ===")
    train_graphs, train_labels = dgl.load_graphs(processed_path, idx_list=train_indices)
    # 预计算自环，保持在 CPU（78K 图全加载 GPU 占用 ~8 GB，子图构建路径已支持 CPU→GPU 按需传输）
    train_graphs = [dgl.add_self_loop(g) for g in train_graphs]
    print(f"  Loaded {len(train_graphs)} graphs (CPU, self-loops pre-computed)")

    # ---- 2. 加载 classifier（冻结） ----
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

    # ---- 3. 加载/恢复 reader ----
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

    # ---- 3.5. 保存 config.json ----
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(repo_root), text=True,
        ).strip()
    except Exception:
        git_commit = 'unknown'

    config = {
        'git_commit': git_commit,
        'data_split': split_label,
        'classifier_checkpoint': f'{args.classifier} (val_acc={ckpt.get("val_acc", "N/A")})',
        'bootstrap_reader_checkpoint': str(args.bootstrap_checkpoint),
        'output_dir': str(output_dir),
        'reward_type': args.reward_type,
        'temperature': args.temperature,
        'batch_construction': f'{args.graphs_per_batch} graphs x {args.seeds_per_graph} rollouts = {args.graphs_per_batch * args.seeds_per_graph} trajectories per optimizer.step()',
        'lr': args.lr,
        'epochs': args.epochs,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'budget_options': BUDGET_OPTIONS,
        'total_params': total_params,
        'checkpoint_selection_rule': 'max training accuracy across all epochs',
        'seed': args.seed,
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config saved: {config_path}")

    # ---- 4. 训练 ----
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

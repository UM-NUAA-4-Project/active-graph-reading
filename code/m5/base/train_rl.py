"""v0 LearnableReader — REINFORCE 端到端微调。

与 POMDP REINFORCE 的核心差异：reward 来自外部冻结的 GraphClassifier 而非 reader 自身。

REINFORCE: -R · Σ log π_t(a_t) + β · CE(classifier_logits, y)
R = 1[argmax f_class(sub_G_B) == y]
"""

import argparse
import json
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

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node

from training_utils import (
    TEACHER_NAMES, BUDGET_RATIOS,
    merge_trajectories, load_graphs_for_trajectories,
)


def compute_budget(n_nodes, ratio):
    return max(1, int(ratio * n_nodes))


def train_one_epoch(reader, classifier, builder, optimizer, graph_cache,
                    train_trajs, batch_size, device, beta_aux, budgets_rel):
    reader.train()
    classifier.eval()
    total_reward = 0
    total_trajs = 0

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        batch_losses = []
        batch_rewards = []

        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            y = torch.tensor(label, device=device)

            idx = traj.graph_idx ^ traj.seed_node
            br = budgets_rel[idx % len(budgets_rel)]
            B = compute_budget(g.num_nodes(), br)

            reader.reset()
            state = GraphState(g, seed_node=traj.seed_node)
            log_probs = []

            for _step in range(B):
                scores, probs, frontier_nodes, action_idx, log_prob = \
                    reader.forward(state, sample=True)

                if len(frontier_nodes) == 0:
                    break

                log_probs.append(log_prob)
                chosen = frontier_nodes[action_idx.item()]
                reveal_node(state, chosen)

            # 终端 reward: 外部 classifier
            sub_g, _ = builder.build(state)
            sub_g = sub_g.to(device)
            node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = classifier(sub_g, node_ids)
                pred = logits.argmax(dim=1).item()
            R = 1.0 if pred == label else 0.0

            if log_probs:
                rl_loss = -R * torch.stack(log_probs).sum()
                aux_loss = F.cross_entropy(logits, y.unsqueeze(0))
                traj_loss = rl_loss + beta_aux * aux_loss
                batch_losses.append(traj_loss)
                batch_rewards.append(R)

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_reward += sum(batch_rewards)
        total_trajs += len(batch_rewards)
        avg_r = total_reward / total_trajs if total_trajs > 0 else 0
        pbar.set_postfix({'loss': f'{batch_loss.item():.4f}',
                          'avg_R': f'{avg_r:.2f}'})

    return 0.0


def main():
    parser = argparse.ArgumentParser(description='v0 LearnableReader REINFORCE fine-tuning')
    parser.add_argument('--bootstrap-checkpoint', default='m5/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--beta-aux', type=float, default=0.5)
    parser.add_argument('--budgets-rel', nargs='+', type=float,
                        default=[0.05, 0.10, 0.15])
    parser.add_argument('--limit-trajs', type=int, default=0)
    parser.add_argument('--output-dir', default='m5/rl/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载轨迹和图
    data_dir = repo_root / 'm4/data/stage5_filtered'
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir)
    if args.limit_trajs > 0:
        all_trajs = all_trajs[:args.limit_trajs]
        print(f"  Limited to {len(all_trajs)} trajectories")

    print("\n=== Loading graphs ===")
    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    # 2. 加载 classifier（冻结）
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt_path = repo_root / args.classifier
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # 3. 加载 reader
    print("\n=== Loading LearnableReader ===")
    reader = LearnableReader(edge_feat_dim=7, hidden_dim=128, n_layers=2, dropout=0.1).to(device)

    ckpt_path = repo_root / args.bootstrap_checkpoint
    if ckpt_path.exists():
        print(f"  Loading from {ckpt_path}")
        reader_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        reader.load_state_dict(reader_ckpt['reader_state_dict'])
        print(f"  Checkpoint epoch: {reader_ckpt.get('epoch', 'N/A')}, "
              f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")
    else:
        print(f"  WARNING: {ckpt_path} not found, training from scratch")

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Reader params: {total_params:,}")

    optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    builder = SubgraphBuilder()
    training_log = []

    # 4. REINFORCE 训练
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_one_epoch(
            reader, classifier, builder, optimizer, graph_cache, all_trajs,
            args.batch_size, device, args.beta_aux, args.budgets_rel,
        )

        training_log.append({'epoch': epoch + 1})

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

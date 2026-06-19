"""Base Learnable Reader — Behavior Cloning 训练 + 微调。

基于 M4 teacher 轨迹，使用 BC 训练 LearnableReader：
每步 teacher 选择的 frontier node 转为 one-hot target，
与 reader softmax 输出计算 cross-entropy loss。

所有轨迹用于训练，不划分验证集。

支持模式：
  --resume PATH     从 checkpoint 恢复训练（相同超参）
  --finetune PATH   从 checkpoint 微调（更低 LR，自动推断 hidden_dim）
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

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier

from training_utils import (
    TEACHER_NAMES, BUDGET_RATIOS,
    merge_trajectories, load_graphs_for_trajectories,
    train_one_epoch, infer_reader_config,
)


def main():
    parser = argparse.ArgumentParser(description='BC Training / Fine-tuning for LearnableReader')
    parser.add_argument('--data-dir', default='m4/data/stage4_balanced')
    parser.add_argument('--pretrained', default='m3/best_model_ft.pt',
                        help='GraphClassifier checkpoint（仅用于参考，不参与训练）')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-dir', default='m5/base/checkpoint')
    parser.add_argument('--n-layers-reader', type=int, default=2, dest='n_layers_reader')
    parser.add_argument('--dropout-reader', type=float, default=0.25, dest='dropout_reader')
    # 恢复 / 微调
    parser.add_argument('--resume', default='',
                        help='从 checkpoint 恢复训练（相同 LR，相同 hidden_dim）')
    parser.add_argument('--finetune', default='',
                        help='从 checkpoint 微调（自动推断 hidden_dim，推荐 --lr 1e-4）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    is_finetune = bool(args.finetune)
    resume_path = args.finetune if is_finetune else args.resume

    output_dir = repo_root / args.output_dir
    if is_finetune:
        output_dir = _m5_dir / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载所有轨迹 & 图 ----
    data_dir = repo_root / args.data_dir
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir)

    print("\n=== Loading graphs ===")
    needed_graphs = {t.graph_idx for t in all_trajs}
    graph_cache = load_graphs_for_trajectories(args.root, needed_graphs)

    # ---- 2. 加载 classifier（仅参考，不参与训练） ----
    print("\n=== Loading classifier (reference only) ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.pretrained), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 3. 创建 / 恢复 Reader ----
    print("\n=== Creating LearnableReader ===")
    start_epoch = 0
    training_log = []

    if resume_path:
        print(f"  Loading from {resume_path}")
        reader_ckpt = torch.load(str(resume_path), map_location='cpu', weights_only=False)

        if is_finetune:
            hidden_dim, n_layers = infer_reader_config(reader_ckpt['reader_state_dict'])
            print(f"  Inferred: hidden_dim={hidden_dim}, n_layers={n_layers}")
            dropout = args.dropout_reader
        else:
            hidden_dim = args.hidden_dim
            n_layers = args.n_layers_reader
            dropout = args.dropout_reader

        print(f"  Checkpoint epoch={reader_ckpt.get('epoch', 'N/A')}")

        reader = LearnableReader(
            edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
        ).to(device)
        reader.load_state_dict(reader_ckpt['reader_state_dict'])

        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
        try:
            optimizer.load_state_dict(reader_ckpt['optimizer_state_dict'])
        except (KeyError, ValueError) as e:
            print(f"  WARNING: Could not load optimizer state ({e}), using fresh optimizer")

        start_epoch = reader_ckpt.get('epoch', -1) + 1
        training_log = reader_ckpt.get('training_log', [])
    else:
        reader = LearnableReader(
            edge_feat_dim=7, hidden_dim=args.hidden_dim,
            n_layers=args.n_layers_reader, dropout=args.dropout_reader,
        ).to(device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"  Params: {total_params:,}")

    # ---- 4. 训练循环（全部轨迹用于训练，无验证集） ----
    mode = "Fine-tuning" if is_finetune else ("Resuming" if resume_path else "Training")
    print(f"\n=== {mode}: epoch {start_epoch + 1} → {args.epochs}, lr={args.lr} ===")
    print(f"  Train trajectories: {len(all_trajs)}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        avg_loss = train_one_epoch(
            reader, optimizer, graph_cache, all_trajs, args.batch_size, device,
        )
        print(f"  Avg step loss: {avg_loss:.4f}")

        log_entry = {
            'epoch': epoch + 1,
            'avg_step_loss': avg_loss,
        }
        training_log.append(log_entry)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': {
                'hidden_dim': reader.frontier_encoder.hidden_dim,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'n_layers_reader': reader.frontier_encoder.n_layers,
                'dropout_reader': reader.frontier_encoder.dropout,
            },
        }

        prefix = 'reader_ft' if is_finetune else 'reader_checkpoint'
        ckpt_path = output_dir / f'{prefix}_epoch{epoch + 1}.pt'
        torch.save(checkpoint, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_name = 'finetune_log.json' if is_finetune else 'training_log.json'
        log_path = output_dir / log_name
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    # 保存最终模型
    final_name = 'reader_final_ft.pt' if is_finetune else 'reader_final.pt'
    final_path = output_dir / final_name
    checkpoint['epoch'] = args.epochs - 1
    torch.save(checkpoint, str(final_path))
    print(f"\n{mode} complete. Final model: {final_path}")


if __name__ == '__main__':
    main()

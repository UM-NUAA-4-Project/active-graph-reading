"""目标数据集全图 GIN 分类器训练（上界）。

移植 m3/train_full_graph.py，数据来自 m7 数据集注册表；
GINEEncoder 通过 node_feat_dim 支持节点特征（如 CIFAR10 的 5 维）。
数据集不划分验证集（DGL 官方仅 train/test），checkpoint 每轮覆盖、
最后一轮为最终产物，评估统一走 code/m7/eval.py 的 test 划分。

训练准确率在训练循环内按 batch 即时统计（train 模式）；
--eval-every N 每 N 轮在 test 集评估，保留该轮 checkpoint
（eval_epoch{N}.pt）并按 test 准确率维护最佳 checkpoint
（best_<dataset>_classifier.pt）。输出目录同时写 config.json
（完整参数）与 history.json（每轮 loss/acc + 最佳记录）。

用法:
    python code/m7/train_classifier.py --dataset cifar10 --root /mnt/data/superpixels 
    --device cuda:0 --batch-size 128 --epochs 100 --lr 0.001 --hidden-dim 256
  python code/m7/train_classifier.py --dataset cifar10 --root D:\\superpixels --device cuda:0
  python code/m7/train_classifier.py --dataset cifar10 --root D:\\superpixels \
      --device cpu --epochs 2 --max-graphs 200   # 冒烟
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
sys.path.insert(0, str(_repo_root))
# 把脚本目录 code/m7 从 sys.path 移除：否则 dgl→torchdata 内部的
# `import datasets` 会解析到 code/m7/datasets（遮蔽 HuggingFace datasets），
# 引发循环导入（dgl 部分初始化即访问 DGLGraph）。本脚本只需 code/ 在 path 上。
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from tqdm import tqdm

from active_graph_reader.gnn.classifier import GraphClassifier
from m7.datasets.registry import get_dataset


def train_one_epoch(clf, optimizer, graphs, labels, batch_size, device):
    clf.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    order = list(range(len(graphs)))
    random.shuffle(order)
    for start in tqdm(range(0, len(order), batch_size), desc="  Train"):
        idx = order[start:start + batch_size]
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[idx].to(device)
        logits = clf(batch_g, nids)
        loss = nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(idx)
    # 训练准确率：train 模式下按 batch 即时统计（BN 用当前 batch 统计），
    # 不用 eval 模式全量重算——eval 模式受 BN running stats 滞后影响
    return total_loss / max(n_batches, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(clf, graphs, labels, batch_size, device):
    clf.eval()
    correct = total = 0
    for start in range(0, len(graphs), batch_size):
        idx = list(range(start, min(start + batch_size, len(graphs))))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = clf(batch_g, nids)
        pred = logits.argmax(dim=-1).cpu()
        correct += (pred == labels[idx]).sum().item()
        total += len(idx)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description='M7 全图 GIN 上界分类器训练')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录，数据地址唯一来源（如 D:\\superpixels）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='限制训练/评估图数（0=全部）')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='每 N 轮在 test 集评估并保留该轮 checkpoint（0=不评估）')
    parser.add_argument('--output-dir', default='m7/checkpoint')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root)
    spec = ds.spec
    print(f"Dataset {spec.name}: {spec.n_classes} classes, "
          f"node_feat_dim={spec.node_feat_dim}, edge_feat_dim={spec.edge_feat_dim}")

    train_graphs = ds.train_graphs
    test_graphs = ds.test_graphs
    if args.max_graphs > 0:
        train_graphs = train_graphs[:args.max_graphs]
        test_graphs = test_graphs[:args.max_graphs]
    train_labels = ds.train_labels[:len(train_graphs)]
    test_labels = ds.test_labels[:len(test_graphs)]
    print(f"Train graphs: {len(train_graphs)} (数据集不划分验证集)")

    clf = GraphClassifier(
        edge_feat_dim=spec.edge_feat_dim,
        n_classes=spec.n_classes,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        node_feat_dim=spec.node_feat_dim,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in clf.parameters()):,}")

    optimizer = optim.Adam(clf.parameters(), lr=args.lr)
    out_dir = _repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f'{spec.name}_classifier'
    path = out_dir / f'{base_name}.pt'
    best_path = out_dir / f'best_{base_name}.pt'

    # config.json：完整训练参数快照，训练开始即写
    config = {'args': vars(args),
              'dataset_spec': {'name': spec.name, 'n_classes': spec.n_classes,
                               'node_feat_dim': spec.node_feat_dim,
                               'edge_feat_dim': spec.edge_feat_dim,
                               'needs_self_loop': spec.needs_self_loop,
                               'budget_options': spec.budget_options},
              'device': str(device),
              'started_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    history = {'dataset': spec.name, 'epochs': []}
    best_acc = -1.0
    best_epoch = None
    # 无验证集：训练内按 batch 统计 train acc 观察收敛，checkpoint 每轮覆盖
    # （最后一轮即最终 checkpoint）；--eval-every 在 test 集评估并保留该轮
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(clf, optimizer, train_graphs,
                                          train_labels, args.batch_size, device)
        rec = {'epoch': epoch, 'loss': round(loss, 6),
               'train_acc': round(train_acc, 6), 'test_acc': None,
               'eval_checkpoint': None}
        ckpt = {'model_state_dict': clf.state_dict(),
                'config': {'dataset': spec.name, 'n_classes': spec.n_classes,
                           'node_feat_dim': spec.node_feat_dim,
                           'edge_feat_dim': spec.edge_feat_dim,
                           'hidden_dim': args.hidden_dim, 'n_layers': args.n_layers,
                           'dropout': args.dropout}}
        torch.save(ckpt, str(path))
        line = f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}  train_acc={train_acc:.4f}"
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            test_acc = evaluate(clf, test_graphs, test_labels,
                                args.batch_size, device)
            rec['test_acc'] = round(test_acc, 6)
            eval_path = out_dir / f'eval_epoch{epoch}.pt'
            ckpt['epoch'] = epoch
            ckpt['train_acc'] = round(train_acc, 6)
            ckpt['test_acc'] = round(test_acc, 6)
            torch.save(ckpt, str(eval_path))
            rec['eval_checkpoint'] = eval_path.name
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(ckpt, str(best_path))
            line += (f"  test_acc={test_acc:.4f}  "
                     f"best={best_acc:.4f}(ep{best_epoch})")
        history['epochs'].append(rec)
        history['best'] = (None if best_epoch is None
                           else {'epoch': best_epoch,
                                 'test_acc': round(best_acc, 6),
                                 'checkpoint': best_path.name})
        with open(out_dir / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(line + f"  saved → {path}")
    print(f"Done. Final checkpoint: {path}")
    if best_epoch is not None:
        print(f"Best: test_acc={best_acc:.4f} at epoch {best_epoch} → {best_path}")


if __name__ == '__main__':
    main()

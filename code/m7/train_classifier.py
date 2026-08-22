"""目标数据集全图 GIN 分类器训练（上界）。

移植 m3/train_full_graph.py，数据来自 m7 数据集注册表；
GINEEncoder 通过 node_feat_dim 支持节点特征（如 CIFAR10 的 5 维）。

用法:
  python code/m7/train_classifier.py --dataset cifar10 --device cuda:0
  python code/m7/train_classifier.py --dataset cifar10 --device cpu --epochs 2 \
      --max-graphs 200 --valid-graphs 50   # 冒烟
"""

import argparse
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
    order = list(range(len(graphs)))
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
    return total_loss / max(n_batches, 1)


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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='限制训练图数（0=全部）')
    parser.add_argument('--valid-graphs', type=int, default=0,
                        help='限制验证图数（0=全部）')
    parser.add_argument('--output-dir', default='m7/checkpoint')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset)
    spec = ds.spec
    print(f"Dataset {spec.name}: {spec.n_classes} classes, "
          f"node_feat_dim={spec.node_feat_dim}, edge_feat_dim={spec.edge_feat_dim}")

    train_graphs = ds.train_graphs
    if args.max_graphs > 0:
        train_graphs = train_graphs[:args.max_graphs]
    valid_graphs = ds.valid_graphs[:args.valid_graphs] if args.valid_graphs > 0 else ds.valid_graphs
    print(f"Train graphs: {len(train_graphs)}, Valid graphs: {len(valid_graphs)}")

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
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(clf, optimizer, train_graphs,
                               ds.train_labels[:len(train_graphs)],
                               args.batch_size, device)
        val_acc = evaluate(clf, valid_graphs, ds.valid_labels[:len(valid_graphs)],
                           args.batch_size, device)
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            out_dir = _repo_root / args.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {'model_state_dict': clf.state_dict(), 'val_acc': val_acc,
                    'config': {'dataset': spec.name, 'n_classes': spec.n_classes,
                               'node_feat_dim': spec.node_feat_dim,
                               'edge_feat_dim': spec.edge_feat_dim,
                               'hidden_dim': args.hidden_dim, 'n_layers': args.n_layers,
                               'dropout': args.dropout}}
            path = out_dir / f'{spec.name}_classifier.pt'
            torch.save(ckpt, str(path))
            print(f"  Best val_acc={best_acc:.4f} saved → {path}")
    print(f"Done. Best val_acc={best_acc:.4f}")


if __name__ == '__main__':
    main()

"""SAGPool scorer 训练：完整图上标准监督（单图 batch，对齐原实现）。

训练出的 checkpoint 供 SagpoolStrategy 协议合规评测使用：
    python code/train_reader/eval.py ... --strategy sagpool \
        --strategy-ckpt baseline/sagpool_{ds}.pt

超参对齐原实现默认（SAGPool-master/main.py）：hidden 128 / pool_ratio 0.5 /
dropout 0.5 / lr 5e-4 / weight-decay 1e-4。自环归一（remove+add）保证与
SagpoolStrategy.build_visible_graph 的构图约定一致。

用法:
  python baseline/run_sagpool.py --dataset cifar10 --root D:\\data\\superpixels \
      --epochs 10
  # 小规模冒烟
  python baseline/run_sagpool.py --dataset cifar10 --root D:\\data\\superpixels \
      --max-graphs 500 --epochs 2 --output baseline/sagpool_cifar10_smoke.pt
"""

import argparse
import datetime
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _REPO_ROOT / 'code'
for _p in (_CODE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from baseline.sagpool import SAGPoolModel  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import REPO_ROOT  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SAGPool scorer 训练（full graph）')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 图数上限（0=全部；PPA 必配）')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--pool-ratio', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=777, help='对齐原实现默认')
    parser.add_argument('--output', default='',
                        help='checkpoint 输出路径（默认 baseline/sagpool_{ds}.pt）')
    return parser


@torch.no_grad()
def evaluate(model, graphs, labels, device):
    model.eval()
    correct = 0
    for i in range(len(graphs)):
        logits = model(graphs[i].to(device))
        correct += int(logits.argmax(dim=-1).item() == int(labels[i].item()))
    model.train()
    return correct / max(len(graphs), 1)


def main():
    args = build_parser().parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root,
                     splits=['train', 'test'], max_graphs=args.max_graphs)
    spec = ds.spec
    print(f"Dataset {spec.name}: train={len(ds.train_graphs)} test={len(ds.test_graphs)}")

    # 自环归一（与 SagpoolStrategy.build_visible_graph 的构图约定一致）
    train_graphs = [dgl.add_self_loop(dgl.remove_self_loop(g))
                    for g in tqdm(ds.train_graphs, desc='Self-loop train')]
    test_graphs = [dgl.add_self_loop(dgl.remove_self_loop(g))
                   for g in tqdm(ds.test_graphs, desc='Self-loop test')]
    train_labels = ds.train_labels
    test_labels = ds.test_labels

    model = SAGPoolModel(
        node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
        n_classes=spec.n_classes, hidden=args.hidden, n_layers=args.layers,
        pool_ratio=args.pool_ratio, dropout=args.dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    order = list(range(len(train_graphs)))
    best_acc, best_state = -1.0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(order)
        correct = total = 0
        for i in tqdm(order, desc=f'Epoch {epoch}/{args.epochs}'):
            g = train_graphs[i].to(device)
            y = torch.tensor([int(train_labels[i].item())], device=device)
            logits = model(g)
            loss = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += int(logits.argmax(dim=-1).item() == y.item())
            total += 1
        test_acc = evaluate(model, test_graphs, test_labels, device)
        history.append({'epoch': epoch, 'train_acc': round(correct / max(total, 1), 6),
                        'test_acc': round(test_acc, 6)})
        print(f"Epoch {epoch}: train_acc={correct / max(total, 1):.4f} "
              f"test_acc={test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # 中间产物固定在 baseline/artifacts/{ds}/（相对路径按仓库根解析，
    # 不经 tools.paths.resolve——其 code/ 优先规则会误解析）
    out = Path(args.output)
    out_path = out if out.is_absolute() else (
        REPO_ROOT / 'baseline' / 'artifacts' / spec.name / 'sagpool.pt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state if best_state is not None
        else model.state_dict(),
        'config': {'dataset': spec.name, 'node_feat_dim': spec.node_feat_dim,
                   'edge_feat_dim': spec.edge_feat_dim,
                   'n_classes': spec.n_classes, 'hidden': args.hidden,
                   'n_layers': args.layers, 'pool_ratio': args.pool_ratio,
                   'dropout': args.dropout},
        'test_acc': best_acc, 'history': history,
        'trained_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'max_graphs': args.max_graphs, 'epochs': args.epochs,
    }, str(out_path))
    save_json({'dataset': spec.name, 'history': history, 'test_acc': best_acc,
               'config': {'hidden': args.hidden, 'n_layers': args.layers,
                          'pool_ratio': args.pool_ratio,
                          'dropout': args.dropout,
                          'max_graphs': args.max_graphs,
                          'epochs': args.epochs}},
              out_path.with_suffix('.history.json'))
    print(f"Best test_acc={best_acc:.4f} → {out_path}")


if __name__ == '__main__':
    main()

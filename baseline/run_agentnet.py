"""AgentNet scorer 训练：完整图上端到端分类监督（单 agent；--batch-size 控制每步优化图数）。

训练出的 checkpoint 供 AgentnetStrategy 协议合规评测使用：
    python code/train_reader/eval.py ... --strategy agentnet \
        --strategy-ckpt baseline/artifacts/{ds}/agentnet.pt

超参对齐原版 README 的子集（hidden 128 / steps 16 / lr 1e-4 / wd 0.01 /
clip-grad 1.0 / visited-decay 0.9 / gumbel τ=2/3 / seed 777）。适配偏离（详见
agentnet.py 模块 docstring 与 docs/decisions.md 2026-09-19 条目）：单 agent
（原版 K=18~26）、无 time conditioning、无 global_agent_pool、聚合用 sum
（原版 'log'）、MLP 宽度 ×1（原版 ×2）。自环归一（remove+add）保证与
AgentnetStrategy 的构图约定一致。

游走步数：训练/评测 own_predict 均按预算比例现场计算
steps = max(1, int(ratio×n))（ratio 逐样本从预算档采样），与评测预算驱动的
策略游走长度对齐；模型无 time conditioning、读出按 1/(T+1) 归一，天然支持
变步数。batch/配方说明：原版 README 配方为 batch 32 / AdamW+cosine / 350 epochs；
本脚本 batch 经**梯度累积**实现（前向为序列游走含 gumbel 采样，不可
dgl.batch；逐图 loss/块大小反向累加，块末 clip + 统一 step），默认 32。
mlp_width_mult=2 与 reduce='log' 为 2026-09-19 源码审查后恢复的原版机制
（此前未声明丢失，导致分类头训练坍缩——见 docs/decisions.md）。

用法:
  python baseline/run_agentnet.py --dataset cifar10 --root D:\\data\\superpixels \
      --epochs 100
  # 小规模冒烟
  python baseline/run_agentnet.py --dataset cifar10 --root D:\\data\\superpixels \
      --max-graphs 500 --epochs 2 --output baseline/artifacts/cifar10/agentnet_smoke.pt
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

from baseline.agentnet import AgentNetModel, symmetrize_with_feat  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import REPO_ROOT  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='AgentNet scorer 训练（full graph）')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 图数上限（0=全部；PPA 必配）')
    parser.add_argument('--stratified', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='max_graphs 截断用分层采样（按类别比例、固定种子，'
                             'train/test 各自分层）；--no-stratified 退回前缀截断')
    parser.add_argument('--epochs', type=int, default=100,
                        help='原版 350（TU 小图）；cifar10 10k 图 × 100 已充分')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='每步优化的图数（原版 32；>1 时梯度累积：前向保持'
                             '单图，逐图 loss/块大小反向累加，块末 clip + 统一 step）')
    parser.add_argument('--mlp-width-mult', type=int, default=2,
                        help='更新 MLP 隐层宽度倍数（原版 README 2；旧 ckpt 为 1）')
    parser.add_argument('--reduce', choices=['log', 'sum'], default='log',
                        help='邻域聚合归约（原版 README log=sum/log2(deg+1)；'
                             '旧 ckpt 为 sum）')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num-steps', type=int, default=16,
                        help='固定步数（仅在 --budget-ratios explicit 为空列表时使用；'
                             '默认按预算比例逐样本现场计算)')
    parser.add_argument('--budget-ratios', nargs='+', type=float, default=None,
                        help='训练游走步数 = max(1, int(ratio×n))，ratio 逐样本从该列表'
                             '均匀采样（缺省取数据集预算档 0.05/0.10/0.15），与评测'
                             'own_predict 读出游走长度对齐；传空列表回退固定 --num-steps')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--visited-decay', type=float, default=0.9)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=777, help='对齐原实现默认')
    parser.add_argument('--output', default='',
                        help='checkpoint 输出路径（默认 baseline/artifacts/{ds}/agentnet.pt）')
    return parser


@torch.no_grad()
def evaluate(model, graphs, labels, device, budget_ratios):
    model.eval()
    correct = 0
    for i in range(len(graphs)):
        steps = (max(1, int(random.choice(budget_ratios) * graphs[i].num_nodes()))
                 if budget_ratios else None)
        logits = model(graphs[i].to(device), steps=steps)
        correct += int(logits.argmax(dim=-1).item() == int(labels[i].item()))
    model.train()
    return correct / max(len(graphs), 1)


def main():
    args = build_parser().parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root,
                     splits=['train', 'test'], max_graphs=args.max_graphs,
                     stratified=args.stratified)
    spec = ds.spec
    budget_ratios = (list(args.budget_ratios) if args.budget_ratios
                     else list(spec.budget_options))
    if not budget_ratios:
        budget_ratios = []   # 回退固定 num_steps
    print(f"Dataset {spec.name}: train={len(ds.train_graphs)} test={len(ds.test_graphs)}"
          f" | walk steps: budget-driven {budget_ratios if budget_ratios else args.num_steps}")

    # 无向化（保留特征；cifar10 等数据集的 DGL 图为单向边，agent 移动/卷积
    # 按无向语义查边，与 AGR 协议的无向 frontier 语义对齐）+ 自环归一
    # （与 AgentnetStrategy 侧 build_visible_graph 的构图约定一致）
    train_graphs = [dgl.add_self_loop(symmetrize_with_feat(
        dgl.remove_self_loop(g)))
        for g in tqdm(ds.train_graphs, desc='Self-loop train')]
    test_graphs = [dgl.add_self_loop(symmetrize_with_feat(
        dgl.remove_self_loop(g)))
        for g in tqdm(ds.test_graphs, desc='Self-loop test')]
    train_labels = ds.train_labels
    test_labels = ds.test_labels

    model = AgentNetModel(
        node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
        n_classes=spec.n_classes, hidden=args.hidden,
        num_steps=args.num_steps, dropout=args.dropout,
        visited_decay=args.visited_decay, mlp_width_mult=args.mlp_width_mult,
        reduce=args.reduce).to(device)
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
        chunks = [order[i:i + args.batch_size]
                  for i in range(0, len(order), args.batch_size)]
        for chunk in tqdm(chunks, desc=f'Epoch {epoch}/{args.epochs}'):
            optimizer.zero_grad()
            chunk_correct = 0
            for i in chunk:
                g = train_graphs[i].to(device)
                y = torch.tensor([int(train_labels[i].item())], device=device)
                steps = (max(1, int(random.choice(budget_ratios)
                                  * train_graphs[i].num_nodes()))
                         if budget_ratios else None)
                logits = model(g, steps=steps)
                # 均值梯度：按实际块大小缩放（末块不足 batch_size 亦正确）
                (torch.nn.functional.cross_entropy(logits, y) / len(chunk)).backward()
                chunk_correct += int(logits.argmax(dim=-1).item() == y.item())
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip_grad)
            optimizer.step()
            correct += chunk_correct
            total += len(chunk)
        test_acc = evaluate(model, test_graphs, test_labels, device,
                            budget_ratios)
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
    out = Path(args.output) if args.output else None
    out_path = out if (out is not None and out.is_absolute()) else (
        out if out is not None
        else REPO_ROOT / 'baseline' / 'artifacts' / spec.name / 'agentnet.pt')
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state if best_state is not None
        else model.state_dict(),
        'config': {'dataset': spec.name, 'node_feat_dim': spec.node_feat_dim,
                   'edge_feat_dim': spec.edge_feat_dim,
                   'n_classes': spec.n_classes, 'hidden': args.hidden,
                   'num_steps': args.num_steps, 'dropout': args.dropout,
                   'visited_decay': args.visited_decay,
                   'mlp_width_mult': args.mlp_width_mult,
                   'reduce': args.reduce,
                   'batch_size': args.batch_size,
                   'budget_ratios': budget_ratios},
        'test_acc': best_acc, 'history': history,
        'trained_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'max_graphs': args.max_graphs, 'epochs': args.epochs,
    }, str(out_path))
    save_json({'dataset': spec.name, 'history': history, 'test_acc': best_acc,
               'config': {'hidden': args.hidden, 'num_steps': args.num_steps,
                          'dropout': args.dropout,
                          'visited_decay': args.visited_decay,
                          'mlp_width_mult': args.mlp_width_mult,
                          'reduce': args.reduce,
                          'batch_size': args.batch_size,
                          'budget_ratios': budget_ratios,
                          'max_graphs': args.max_graphs,
                          'stratified': args.stratified,
                          'epochs': args.epochs}},
              out_path.with_suffix('.history.json'))
    print(f"Best test_acc={best_acc:.4f} → {out_path}")


if __name__ == '__main__':
    main()

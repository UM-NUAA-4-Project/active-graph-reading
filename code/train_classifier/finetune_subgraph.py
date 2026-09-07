"""子图适应分类器微调（通用，任意注册数据集）。

动机：全图上界分类器只在完整图上训练，对部分揭示子图（度数远低于全图、
规模小）可能分布外塌缩（如 collab 无节点特征时退化为"永远预测最稀有类"），
导致预算内评测在塌缩地板上失去区分度。本脚本用协议一致的子图采样
（固定 seed 公式 + 前沿约束揭示 + spec 预算档）微调分类器，使裁判适应
部分子图分布；混入少量全图样本（--full-frac）防止全图参照能力遗忘。

采样策略混合：random（frontier 均匀）/ random-walk（baseline 深度游走）/
bfs（广度 FIFO），按图轮转保证均衡；每个 epoch 重新采样（不同 seed_i），
验证集用固定 seed 采样以便跨 epoch 可比。

- 结构继承 --base-ckpt 内嵌 config（必填），权重从其初始化（微调而非从零）；
- checkpoint 内嵌完整 config + judge 元信息，下游 load_classifier /
  eval.py --classifier 直接重建；
- 选型指标：验证集各预算档子图准确率均值（best_subgraph_classifier.pt）。

用法:
  python code/train_classifier/finetune_subgraph.py --dataset collab \
      --root D:\\data\\COLLAB \
      --base-ckpt result/collab/classifier/finetune/best_collab_classifier.pt \
      --epochs 6 --lr 1e-4
"""

import argparse
import datetime
import random
import sys
from pathlib import Path

# 标准引导：把 <repo>/code 插到 sys.path，移除脚本自身目录（防包名遮蔽）
_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

from tools.paths import CODE_DIR, REPO_ROOT  # noqa: E402
from train_classifier.class_weight import compute_class_weight  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from classifier import GraphClassifier  # noqa: E402
from classifier.factory import classifier_config_from_ckpt  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import build_experiment_config, load_json, save_json  # noqa: E402
from tools.seeds import deterministic_seed, set_all_seeds  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import reveal_node  # noqa: E402
from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402


def resolve_path(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='子图适应分类器微调（通用）')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多加载的图数（0=全部；PPA 必配）')
    parser.add_argument('--base-ckpt', required=True,
                        help='被微调的全图分类器 checkpoint（结构继承自此）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='微调学习率（默认 1e-4，远小于全图训练的 1e-3）')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--class-weight', default='none',
                        choices=['none', 'balanced'],
                        help='类别加权（cross_entropy weight，同 train_full_graph）：'
                             '多数类退化时用 balanced（默认 none=不加权）')
    # 采样
    parser.add_argument('--policies', nargs='+',
                        default=['random', 'random-walk', 'bfs'],
                        choices=['random', 'random-walk', 'bfs'],
                        help='子图采样策略混合（按图轮转）')
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='训练/验证预算比例（默认取 spec.budget_options）')
    parser.add_argument('--subgraphs-per-graph', type=int, default=1,
                        help='每图每 epoch 采样的子图数')
    parser.add_argument('--full-frac', type=float, default=0.1,
                        help='训练样本中保留全图的比例（防全图能力遗忘）')
    parser.add_argument('--val-max-graphs', type=int, default=1000,
                        help='验证集最多评估的图数（0=全部；控制每轮验证开销）')
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='',
                        help='实验 config JSON（CLI 显式传参覆盖其值）')
    parser.add_argument('--output-dir', default='',
                        help='默认 result/{dataset}/classifier/finetune_subgraph')
    return parser


def _ensure_policies_registered(policy_names):
    if 'random-walk' in policy_names:
        # baseline/ 在仓库根（PEP 420 命名空间包），注册后 create_fixed_reader 可构造
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import baseline.strategies
        baseline.strategies.register_baseline_strategies()


def make_agents(policy_names):
    _ensure_policies_registered(policy_names)
    return {p: create_fixed_reader(p) for p in policy_names}


def rollout_partial(graph, agent, budget_ratio, graph_idx: int, seed_i: int,
                    builder: SubgraphBuilder):
    """按协议采样一个部分子图（与 eval 固定策略 rollout 同逻辑）。

    预算=不含 seed 的揭示节点数；预算检查与 eval_loop.rollout_strategy
    一致：达到预算的最后一次揭示计入轨迹。
    """
    state = GraphState(graph, deterministic_seed(graph_idx, seed_i,
                                                 graph.num_nodes()))
    state._budget_limit = max(1, round(budget_ratio * state.n_nodes_total))
    agent.reset()
    if state._initial_frontier:
        agent.on_new_frontier(list(state._initial_frontier))
    while not state.is_frontier_empty and state.get_budget_used() < state._budget_limit:
        v = agent.select(state)
        if v is None:
            break
        new_frontier = reveal_node(state, v)
        if new_frontier:
            agent.on_new_frontier(new_frontier)
    return builder.build(state)[0]


def train_one_epoch(clf, optimizer, graphs, labels, args, policies, agents,
                    ratios, builder, epoch: int, device, rng,
                    class_weight=None):
    clf.train()
    order = list(range(len(graphs)))
    rng.shuffle(order)
    total_loss = n_batches = correct = total = 0
    for start in tqdm(range(0, len(order), args.batch_size), desc='  Train'):
        idx = order[start:start + args.batch_size]
        sub_graphs = []
        ys = []
        for pos in idx:
            if args.full_frac > 0 and rng.random() < args.full_frac:
                sub_graphs.append(graphs[pos])   # 全图样本
                ys.append(int(labels[pos]))
                continue
            k = rng.randrange(args.subgraphs_per_graph)
            policy = policies[(pos * args.subgraphs_per_graph + k) % len(policies)]
            ratio = ratios[(pos * args.subgraphs_per_graph + k) % len(ratios)]
            sub_g = rollout_partial(graphs[pos], agents[policy], ratio,
                                    graph_idx=pos,
                                    seed_i=epoch * args.subgraphs_per_graph + k,
                                    builder=builder)
            sub_graphs.append(sub_g)
            ys.append(int(labels[pos]))
        batch_g = dgl.batch(sub_graphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        logits = clf(batch_g, nids)
        loss = nn.functional.cross_entropy(logits, y, weight=class_weight,
                                           label_smoothing=args.label_smoothing)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(ys)
    return total_loss / max(n_batches, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_subgraphs(clf, graphs, labels, policies, agents, ratios,
                       builder, device, batch_size=256, val_max=0):
    """验证集子图准确率：每 (graph, ratio) 固定 seed 采一个子图，
    策略按 (graph_pos + ratio_idx) 轮转。返回 (per_ratio dict, mean)。"""
    clf.eval()
    n = len(graphs) if val_max <= 0 else min(val_max, len(graphs))
    per_ratio = {f'{r:.2f}': [0, 0] for r in ratios}
    for r_idx, ratio in enumerate(ratios):
        for start in range(0, n, batch_size):
            pos = list(range(start, min(start + batch_size, n)))
            subs, ys = [], []
            for p in pos:
                policy = policies[(p + r_idx) % len(policies)]
                sub = rollout_partial(graphs[p], agents[policy], ratio,
                                      graph_idx=p, seed_i=10_000 + p * len(ratios) + r_idx,
                                      builder=builder)
                subs.append(sub)
                ys.append(int(labels[p]))
            batch_g = dgl.batch(subs).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            pred = clf(batch_g, nids).argmax(dim=-1).cpu()
            y = torch.tensor(ys, dtype=torch.long)
            per_ratio[f'{ratio:.2f}'][0] += (pred == y).sum().item()
            per_ratio[f'{ratio:.2f}'][1] += len(ys)
    accs = {k: c / max(t, 1) for k, (c, t) in per_ratio.items()}
    return accs, sum(accs.values()) / max(len(accs), 1)


@torch.no_grad()
def evaluate_full(clf, graphs, labels, device, batch_size=256, val_max=0):
    clf.eval()
    n = len(graphs) if val_max <= 0 else min(val_max, len(graphs))
    correct = total = 0
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        pred = clf(batch_g, nids).argmax(dim=-1).cpu()
        correct += (pred == labels[idx]).sum().item()
        total += len(idx)
    return correct / max(total, 1)


def _strip_dgl_id(graphs):
    """TUDataset 原图自带 _ID 帧列，SubgraphBuilder 产物没有；全图样本与
    子图样本混 batch 时 dgl.batch 会因 schema 不一致报错，统一剥掉
    （_ID 是 dgl 内部记账列，非模型输入）。"""
    for g in graphs:
        if '_ID' in g.ndata:
            g.ndata.pop('_ID')
        if '_ID' in g.edata:
            g.edata.pop('_ID')


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        cfg_baseline = load_json(args.config)
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, max_graphs=args.max_graphs)
    spec = ds.spec
    ratios = list(args.budget_ratios) if args.budget_ratios \
        else list(spec.budget_options)
    print(f"Dataset {spec.name}: {spec.n_classes} classes, "
          f"budget_ratios={ratios}, policies={args.policies}")

    train_graphs, train_labels = ds.train_graphs, ds.train_labels
    valid_graphs = ds.valid_graphs
    valid_labels = ds.valid_labels
    if not valid_graphs:
        print("[warn] 数据集无验证集，模型选型退化为训练子图准确率")
        valid_graphs, valid_labels = train_graphs, train_labels
    _strip_dgl_id(train_graphs)
    _strip_dgl_id(valid_graphs)
    print(f"Train: {len(train_graphs)} graphs | Valid: {len(valid_graphs)} graphs")

    class_weight = compute_class_weight(args.class_weight, train_labels,
                                        spec.n_classes, device=device)

    # ---- 模型：结构/权重均来自 base ckpt（微调）----
    base_path = resolve_path(args.base_ckpt)
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    clf_cfg = classifier_config_from_ckpt(base_ckpt)
    clf = GraphClassifier(config=clf_cfg).to(device)
    clf.load_state_dict(base_ckpt['model_state_dict'], strict=True)
    print(f"Loaded base weights from {base_path} | "
          f"params={sum(p.numel() for p in clf.parameters()):,}")

    agents = make_agents(args.policies)
    builder = SubgraphBuilder()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    # ---- 输出目录 + config 落盘 ----
    out_dir = resolve_path(args.output_dir) if args.output_dir \
        else CODE_DIR / 'result' / spec.name / 'classifier' / 'finetune_subgraph'
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / 'subgraph_classifier.pt'
    best_path = out_dir / 'best_subgraph_classifier.pt'

    experiment_cfg = build_experiment_config(args, extra={
        'classifier': clf_cfg.to_dict(),
        'base_checkpoint': str(base_path),
        'budget_ratios': ratios,
        'device': str(device),
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
    })
    save_json(experiment_cfg, out_dir / 'config.json')

    ckpt_config = {'classifier': clf_cfg.to_dict(), 'dataset': spec.name,
                   'n_classes': spec.n_classes,
                   'judge_kind': 'subgraph-finetuned',
                   'base_checkpoint': str(base_path),
                   'policies': list(args.policies),
                   'budget_ratios': ratios,
                   'full_frac': args.full_frac}

    history = {'dataset': spec.name, 'base_checkpoint': str(base_path),
               'budget_ratios': ratios, 'epochs': []}
    best_mean, best_epoch = -1.0, None
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(clf, optimizer, train_graphs,
                                          train_labels, args, args.policies,
                                          agents, ratios, builder, epoch,
                                          device, rng,
                                          class_weight=class_weight)
        rec = {'epoch': epoch, 'loss': round(loss, 6),
               'train_acc': round(train_acc, 6)}
        line = f"Epoch {epoch}/{args.epochs}: loss={loss:.4f} train_acc={train_acc:.4f}"

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            sub_accs, sub_mean = evaluate_subgraphs(
                clf, valid_graphs, valid_labels, args.policies, agents,
                ratios, builder, device, val_max=args.val_max_graphs)
            full_acc = evaluate_full(clf, valid_graphs, valid_labels, device,
                                     val_max=args.val_max_graphs)
            rec.update({'val_sub_acc': sub_accs, 'val_sub_acc_mean': round(sub_mean, 6),
                        'val_full_acc': round(full_acc, 6)})
            line += (f"  val_sub={sub_mean:.4f} "
                     f"({', '.join(f'{k}={v:.3f}' for k, v in sub_accs.items())})  "
                     f"val_full={full_acc:.4f}")
            ckpt = {'model_state_dict': clf.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'config': ckpt_config,
                    'val_acc': round(sub_mean, 6),
                    'val_acc_per_ratio': sub_accs,
                    'val_full_acc': round(full_acc, 6)}
            torch.save(ckpt, str(last_path))
            if sub_mean > best_mean:
                best_mean, best_epoch = sub_mean, epoch
                ckpt['best_acc'], ckpt['best_epoch'] = best_mean, best_epoch
                torch.save(ckpt, str(best_path))
                line += f"  [best] → {best_path.name}"
        history['epochs'].append(rec)
        history['best'] = (None if best_epoch is None
                           else {'epoch': best_epoch, 'val_sub_acc_mean': round(best_mean, 6),
                                 'checkpoint': best_path.name})
        save_json(history, out_dir / 'history.json')
        print(line)

    print(f"Done. Best val_sub_acc={best_mean:.4f} @ epoch {best_epoch} → {best_path}")
    if best_epoch is not None:
        ck = torch.load(str(best_path), map_location='cpu', weights_only=False)
        print(f"  val_acc_per_ratio={ck['val_acc_per_ratio']}  "
              f"val_full_acc={ck['val_full_acc']}")


if __name__ == '__main__':
    main()

"""M7 RL 微调：以迁移后的 PPA checkpoint 为起点，在目标数据集上 REINFORCE 微调。

Loss = mean_i(-R_i · Σ_t log π_t(a_t))，R_i 由目标数据集上界 GIN 分类器给出
（binary/tanh/asymmetric）。base reader 无 belief；attention reader 用
belief lockstep（复用 m6/training_utils）。

用法:
    python code/m7/train_rl.py --dataset cifar10 --root /mnt/data/superpixels --device cuda:0 
    --reader-kind attention --bootstrap code/m6/attention/bc/reader_belief_epoch5.pt --reward asymmetric 
    --graphs-per-batch 64 --seeds-per-graph 4 --output-dir m7/checkpoint/rl/attention-bc 
    --classifier m7\checkpoint\finite\best_cifar10_classifier.pt
  python code/m7/train_rl.py --dataset cifar10 --root D:\\superpixels \
      --reader-kind attention --bootstrap m6/attention/bc/reader_belief_epoch1.pt \
      --classifier m7/checkpoint/cifar10_classifier.pt --device cuda:0
  python code/m7/train_rl.py --dataset cifar10 --root D:\\superpixels \
      --reader-kind base --bootstrap m5/base/checkpoint/best_reader.pt --device cuda:0
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
_project_root = _repo_root.parent                   # 仓库根
sys.path.insert(0, str(_repo_root))
# 把脚本目录 code/m7 从 sys.path 移除：否则 dgl→torchdata 内部的
# `import datasets` 会解析到 code/m7/datasets（遮蔽 HuggingFace datasets），
# 引发循环导入（dgl 部分初始化即访问 DGLGraph）。本脚本只需 code/ 在 path 上。
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader.attention.attention_reader import AttentionReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node

from m7.datasets.registry import get_dataset
from m7.transfer import transfer_weights
from m6.training_utils import (
    deterministic_seed, reveal_and_update_beliefs_batch, strip_classifier_keys,
)


def build_reader(kind, spec, args, device):
    kwargs = dict(
        edge_feat_dim=args.edge_feat_dim or spec.edge_feat_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        stochastic=True,
        temperature=args.temperature,
    )
    if kind == 'base':
        kwargs['node_feat_dim'] = args.node_feat_dim or spec.node_feat_dim
        return LearnableReader(**kwargs).to(device)
    kwargs['m'] = args.m
    kwargs['n_heads'] = args.n_heads
    kwargs['node_feat_dim'] = args.node_feat_dim or spec.node_feat_dim
    return AttentionReader(**kwargs).to(device)


def freeze_shared_layers(reader, spec, args):
    """--freeze：冻结除特征投影层（edge_encoder / node_feat_proj）外的全部参数。"""
    if not args.freeze:
        return
    frozen = 0
    for name, p in reader.named_parameters():
        if 'edge_encoder' not in name and 'node_feat_proj' not in name:
            p.requires_grad = False
            frozen += 1
    print(f"  --freeze: {frozen} 层已冻结（仅特征投影层可训练）")


def train_one_epoch_base(reader, classifier, builder, optimizer,
                         graphs, labels, batch_size, n_seeds,
                         device, reward_type, ln_k, budget_options):
    reader.train()
    classifier.eval()
    total_R = total_trajs = total_correct = total_loss = 0.0
    order = list(range(len(graphs)))
    random.shuffle(order)
    pbar = tqdm(range(0, len(order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_indices = order[batch_start:batch_start + batch_size]
        all_states, state_labels, per_state_log_probs = [], [], []
        for gi, local_i in enumerate(batch_indices):
            g_raw = graphs[local_i]
            label = int(labels[local_i].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()
            budget_ratio = random.choice(budget_options)
            B = max(1, int(budget_ratio * n_nodes))
            for seed_i in range(n_seeds):
                seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                all_states.append(state)
                state_labels.append(y)
                per_state_log_probs.append([])

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break
            batch_results = reader.forward_states(
                [all_states[j] for j in active_idx],
                temperature=reader.temperature)
            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                per_state_log_probs[j].append(dist.log_prob(action_idx))
                reveal_node(all_states[j], frontier_nodes[action_idx.item()])
                if all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        traj_subgraphs, traj_labels, traj_log_probs = [], [], []
        for j in range(n_total):
            if per_state_log_probs[j]:
                sub_g, _ = builder.build(all_states[j])
                traj_subgraphs.append(sub_g)
                traj_labels.append(state_labels[j])
                traj_log_probs.append(per_state_log_probs[j])
        if not traj_subgraphs:
            continue

        batch_g = dgl.batch(traj_subgraphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            all_logits = classifier(batch_g, nids)

        batch_losses = []
        for i, log_probs in enumerate(traj_log_probs):
            logits_i = all_logits[i:i + 1]
            correct = 1.0 if logits_i.argmax(dim=-1).item() == traj_labels[i].item() else 0.0
            if reward_type == 'tanh':
                ce = F.cross_entropy(logits_i, traj_labels[i]).item()
                R = math.tanh(1.0 - ce / ln_k)
            elif reward_type == 'asymmetric':
                R = 1.0 if correct else -0.1
            else:
                R = 1.0 if correct else 0.0
            total_R += R
            total_trajs += 1
            total_correct += correct
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
    return {'loss': total_loss / n, 'avg_R': total_R / n, 'acc': total_correct / n}


def _diag_nan(reader, state, belief, batch_result, batch_start, step, state_idx):
    """NaN 崩溃点诊断：打印各环节数值后终止，用于定位 NaN 来源。"""
    scores, probs, frontier_nodes = batch_result
    print('\n' + '=' * 60)
    print(f'[NaN 诊断] batch={batch_start}, lockstep_step={step}, state_idx={state_idx}')
    print(f'  graph: n_nodes={state.n_nodes_total}, revealed={len(state.V_revealed)}, '
          f'frontier_nodes={len(state.get_frontier_nodes())}, '
          f'frontier_edges={len(state.frontier_edges)}, budget={state._budget_limit}')
    print(f'  scores: finite={torch.isfinite(scores).all().item()} '
          f'max={scores.abs().max().item():.3e}')
    print(f'  probs:  finite={torch.isfinite(probs).all().item()} '
          f'sum={probs.sum().item():.3f} min={probs.min().item():.3e}')
    print(f'  belief: finite={torch.isfinite(belief).all().item()} '
          f'max={belief.abs().max().item():.3e} shape={tuple(belief.shape)}')
    bad = [n for n, p in reader.named_parameters()
           if not torch.isfinite(p).all()]
    print(f'  weight NaN/Inf 层: {bad if bad else "无（权重有限 → 前向路径问题）"}')
    for name, p in reader.named_parameters():
        if 'edge_encoder' in name or 'node_feat_proj' in name:
            print(f'  skipped 层 {name}: max={p.abs().max().item():.3e} '
                  f'finite={torch.isfinite(p).all().item()}')
    print('=' * 60)
    raise RuntimeError('NaN 诊断：probs 非有限（见上方定位）')


def train_one_epoch_attention(reader, classifier, builder, optimizer,
                              graphs, labels, batch_size, n_seeds,
                              device, reward_type, ln_k, budget_options):
    """belief lockstep 版（对齐 m6/attention/train_rl.py）。"""
    reader.train()
    classifier.eval()
    total_R = total_trajs = total_correct = total_loss = 0.0
    order = list(range(len(graphs)))
    random.shuffle(order)
    micro_batch_size = 16
    pbar = tqdm(range(0, len(order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_indices = order[batch_start:batch_start + batch_size]
        optimizer.zero_grad()
        n_micro = (len(batch_indices) + micro_batch_size - 1) // micro_batch_size
        for mb_i in range(0, len(batch_indices), micro_batch_size):
            mb_indices = batch_indices[mb_i:mb_i + micro_batch_size]
            all_states, state_labels, per_state_log_probs = [], [], []
            state_src = []  # state → (图 index, seed_i) 映射（诊断用）
            for local_i in mb_indices:
                g_raw = graphs[local_i]
                label = int(labels[local_i].item())
                y = torch.tensor([label], device=device)
                n_nodes = g_raw.num_nodes()
                budget_ratio = random.choice(budget_options)
                B = max(1, int(budget_ratio * n_nodes))
                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node)
                    state._budget_limit = B
                    all_states.append(state)
                    state_labels.append(y)
                    per_state_log_probs.append([])
                    state_src.append((local_i, seed_i))

            n_total = len(all_states)
            active = [True] * n_total
            max_budget = max(s._budget_limit for s in all_states)
            all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
            all_subgraphs = [None] * n_total

            for _step in range(max_budget):
                active_idx = [j for j in range(n_total)
                              if active[j] and not all_states[j].is_frontier_empty]
                if not active_idx:
                    break
                active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
                active_subgraphs = None
                if all_subgraphs[active_idx[0]] is not None:
                    active_subgraphs = [all_subgraphs[j] for j in active_idx]
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx],
                    temperature=reader.temperature,
                    beliefs=active_beliefs,
                    subgraphs=active_subgraphs)
                batch_states, batch_nodes, batch_local_idx = [], [], []
                for k, j in enumerate(active_idx):
                    _, probs, frontier_nodes = batch_results[k]
                    if not frontier_nodes:
                        active[j] = False
                        continue
                    if not torch.isfinite(probs).all():
                        _diag_nan(reader, all_states[j], all_beliefs[j],
                                  batch_results[k], batch_start, _step, j)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample()
                    per_state_log_probs[j].append(dist.log_prob(action_idx))
                    batch_states.append(all_states[j])
                    batch_nodes.append(frontier_nodes[action_idx.item()])
                    batch_local_idx.append(j)
                if batch_states:
                    batch_beliefs = torch.cat(
                        [all_beliefs[j] for j in batch_local_idx], dim=0)
                    new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                        reader, batch_states, batch_nodes, batch_beliefs, device,
                        state_src=[state_src[j] for j in batch_local_idx])
                    if not torch.isfinite(new_beliefs).all():
                        print('\n' + '=' * 60)
                        print('[NaN 监测] belief 更新后 NaN')
                        print(f'  输入 belief finite={torch.isfinite(batch_beliefs).all().item()}')
                        print(f'  输出 belief finite={torch.isfinite(new_beliefs).all().item()}')
                        print('=' * 60)
                        raise RuntimeError('NaN 监测: belief 更新后 NaN')
                    for k, j in enumerate(batch_local_idx):
                        all_beliefs[j] = new_beliefs[k:k + 1]
                        all_subgraphs[j] = new_subgraphs[k]
                for j in active_idx:
                    if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                        active[j] = False

            final_states, final_log_probs, final_labels = [], [], []
            for j in range(n_total):
                if per_state_log_probs[j]:
                    final_states.append(all_states[j])
                    final_log_probs.append(per_state_log_probs[j])
                    final_labels.append(state_labels[j])
            if not final_states:
                continue

            all_logits = []
            with torch.no_grad():
                for c_start in range(0, len(final_states), 512):
                    chunk = final_states[c_start:c_start + 512]
                    sub_graphs = [builder.build(s)[0] for s in chunk]
                    batch_g = dgl.batch(sub_graphs).to(device)
                    nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
                    all_logits.append(classifier(batch_g, nids))
            logits_all = torch.cat(all_logits, dim=0)

            batch_losses = []
            for k in range(len(final_states)):
                logits_i = logits_all[k:k + 1]
                correct = 1.0 if logits_i.argmax(dim=-1).item() == final_labels[k].item() else 0.0
                if reward_type == 'tanh':
                    ce = F.cross_entropy(logits_i, final_labels[k]).item()
                    R = math.tanh(1.0 - ce / ln_k)
                elif reward_type == 'asymmetric':
                    R = 1.0 if correct else -0.1
                else:
                    R = 1.0 if correct else 0.0
                total_R += R
                total_trajs += 1
                total_correct += correct
                sum_log_prob = torch.stack(final_log_probs[k]).sum()
                batch_losses.append(-R * sum_log_prob)
            rl_loss = torch.stack(batch_losses).mean()
            loss = rl_loss / n_micro
            loss.backward()
            total_loss += loss.item() * len(batch_losses) * n_micro
        optimizer.step()
        n = max(total_trajs, 1)
        pbar.set_postfix({
            'loss': f'{total_loss / n:.4f}',
            'avg_R': f'{total_R / n:.3f}',
            'acc': f'{total_correct / n:.3f}',
        })
    n = max(total_trajs, 1)
    return {'loss': total_loss / n, 'avg_R': total_R / n, 'acc': total_correct / n}


def main():
    parser = argparse.ArgumentParser(description='M7 RL 微调（迁移自 PPA 或从头）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录，数据地址唯一来源（如 D:\\superpixels）')
    parser.add_argument('--reader-kind', choices=['base', 'attention'], default='attention')
    parser.add_argument('--bootstrap', required=True,
                        help='迁移起点 checkpoint（相对 code/ 目录，如 m6/attention/bc/...）')
    parser.add_argument('--classifier', required=True,
                        help='上界 GIN checkpoint（相对 code/ 目录，或绝对路径）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--graphs-per-batch', type=int, default=32)
    parser.add_argument('--seeds-per-graph', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='限制训练图数（0=全部，默认全部；冒烟测试传小值）')
    parser.add_argument('--reward', choices=['binary', 'tanh', 'asymmetric'], default='binary')
    parser.add_argument('--freeze', action='store_true',
                        help='冻结共享层，仅训练特征投影层（冻结消融用）')
    parser.add_argument('--edge-feat-dim', type=int, default=None,
                        help='覆盖数据集边特征维度（默认跟随 DatasetSpec）')
    parser.add_argument('--node-feat-dim', type=int, default=None,
                        help='覆盖数据集节点特征维度（默认跟随 DatasetSpec）')
    parser.add_argument('--output-dir', default='m7/checkpoint/rl/finetune')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None)
    spec = ds.spec
    BUDGET_OPTIONS = spec.budget_options
    ln_k = math.log(spec.n_classes)

    train_graphs = ds.train_graphs[:args.max_graphs] if args.max_graphs > 0 else ds.train_graphs
    print(f"Train graphs: {len(train_graphs)} (of {len(ds.train_graphs)}")
    print(f"Budget options: {BUDGET_OPTIONS}")

    # ---- classifier（冻结，reward 来源） ----
    clf_path = Path(args.classifier)
    if not clf_path.is_absolute():
        clf_path = _repo_root / args.classifier
    cls_ckpt = torch.load(str(clf_path), map_location=device, weights_only=False)
    # 按 checkpoint 自身 config 构建分类器（train_classifier 保存了 config）；
    # 无 config 时回退到上界默认结构（hidden 300 / 5 层）。这样分类器结构
    # 永远与训练时一致（冒烟用小维度分类器也能加载）。
    _cfg = cls_ckpt.get('config', {})
    classifier = GraphClassifier(
        edge_feat_dim=_cfg.get('edge_feat_dim', spec.edge_feat_dim),
        n_classes=_cfg.get('n_classes', spec.n_classes),
        hidden_dim=_cfg.get('hidden_dim', 300),
        n_layers=_cfg.get('n_layers', 5),
        dropout=_cfg.get('dropout', 0.5),
        node_feat_dim=_cfg.get('node_feat_dim', spec.node_feat_dim),
    ).to(device)
    classifier.load_state_dict(cls_ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"Classifier loaded (val_acc={cls_ckpt.get('val_acc', 'N/A')})")

    # ---- reader（迁移初始化） ----
    bootstrap_path = _repo_root / args.bootstrap
    src_ckpt = torch.load(str(bootstrap_path), map_location='cpu', weights_only=False)
    # 源 checkpoint 携带训练超参（train_rl 保存 training_log=config dict；
    # m6 类 checkpoint 的 training_log 是逐轮列表、超参在 config 键下）：
    # CLI 未显式传入（== argparse 默认值）时沿用
    _tl = src_ckpt.get('training_log')
    src_cfg = _tl if isinstance(_tl, dict) else src_ckpt.get('config') or {}
    if not isinstance(src_cfg, dict):
        src_cfg = {}
    reader_args = SimpleNamespace(**vars(args))
    resolved = {}
    _alt_keys = {'n_layers': 'n_layers_reader', 'dropout': 'dropout_reader'}
    for name in ('hidden_dim', 'n_layers', 'm', 'n_heads'):
        v = src_cfg.get(name)
        if v is None:
            v = src_cfg.get(_alt_keys.get(name))
        if v is not None and getattr(args, name) == parser.get_default(name):
            setattr(reader_args, name, v)
            resolved[name] = v
    reader = build_reader(args.reader_kind, spec, reader_args, device)
    print(f"Reader config: hidden_dim={reader_args.hidden_dim}, "
          f"n_layers={reader_args.n_layers}, m={reader_args.m}, "
          f"n_heads={reader_args.n_heads}, dropout={reader_args.dropout}, "
          f"temperature={reader_args.temperature} "
          f"(from checkpoint config: {resolved or 'none'})")
    src_sd = src_ckpt.get('reader_state_dict', src_ckpt)
    src_sd = strip_classifier_keys(src_sd)
    n_transferred, n_padded, skipped = transfer_weights(src_sd, reader)
    print(f"Bootstrap {args.bootstrap}: transferred={n_transferred}, "
          f"padded={n_padded}, skipped={len(skipped)} (新层/维度不匹配)")
    if skipped:
        print(f"  skipped: {skipped[:6]}{'...' if len(skipped) > 6 else ''}")

    freeze_shared_layers(reader, spec, args)
    optimizer = torch.optim.Adam(
        [p for p in reader.parameters() if p.requires_grad], lr=args.lr)
    builder = SubgraphBuilder()
    total_params = sum(p.numel() for p in reader.parameters())
    trainable_params = sum(p.numel() for p in reader.parameters() if p.requires_grad)
    print(f"Reader params: {total_params:,} (trainable {trainable_params:,})")

    output_dir = _repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'dataset': spec.name,
        'reader_kind': args.reader_kind,
        'bootstrap': args.bootstrap,
        'classifier': str(clf_path),
        'n_transferred': n_transferred,
        'n_padded': n_padded,
        'n_skipped': len(skipped),
        'freeze': args.freeze,
        'reward': args.reward,
        'temperature': args.temperature,
        'budget_options': list(BUDGET_OPTIONS),
        'lr': args.lr,
        'epochs': args.epochs,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'm': args.m,
        'n_heads': args.n_heads,
        'max_graphs': args.max_graphs,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'seed': args.seed,
    }
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved: {output_dir / 'config.json'}")

    train_fn = (train_one_epoch_base if args.reader_kind == 'base'
                else train_one_epoch_attention)
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = train_fn(reader, classifier, builder, optimizer,
                           train_graphs, ds.train_labels[:len(train_graphs)],
                           args.graphs_per_batch, args.seeds_per_graph,
                           device, args.reward, ln_k, BUDGET_OPTIONS)
        print(f"  Train: loss={metrics['loss']:.4f}  avg_R={metrics['avg_R']:.3f}  "
              f"acc={metrics['acc']:.3f}")
        ckpt = {'epoch': epoch,
                'reader_state_dict': reader.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': config}
        ckpt_path = output_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(ckpt, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

    print(f"\nREINFORCE complete. Output: {output_dir}")


if __name__ == '__main__':
    main()

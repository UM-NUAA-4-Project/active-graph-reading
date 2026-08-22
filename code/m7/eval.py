"""M7 统一评估：reader checkpoint（零样本/微调后）或固定策略（random/bfs）。

预算档位跟随 DatasetSpec.budget_options，seed_i=0 确定性 seed（PPA 协议），
argmax 选择（默认）；分类器用目标数据集上界 GIN。
输出 JSON 结构对齐 m6 eval_heldout：{split: {'learnable': {'best': {...}}}}。

用法:
  python code/m7/eval.py --dataset cifar10 --strategy random --classifier m7/checkpoint/cifar10_classifier.pt
  python code/m7/eval.py --dataset cifar10 --checkpoint m7/checkpoint/rl/finetune/checkpoint_epoch5.pt \
      --classifier m7/checkpoint/cifar10_classifier.pt
"""

import argparse
import json
import sys
from collections import defaultdict
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
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.reader.fixed import create_fixed_reader

from m7.datasets.registry import get_dataset
from m7.train_rl import build_reader
from m7.transfer import transfer_weights
from m6.training_utils import deterministic_seed, strip_classifier_keys


@torch.no_grad()
def rollout_strategy(agent, state):
    """单图 rollout：agent.select 直到预算耗尽或 frontier 为空。

    有状态策略（如 BFS 的 FIFO 队列）依赖 reset/on_new_frontier 接线：
    每个 rollout 独立，重置内部状态并以初始前沿播种，揭示后把新发现
    的前沿边喂回 agent（对齐 code/m5/base/analysis/eval.py run_one_seed_collect）。
    """
    agent.reset()
    initial_frontier = list(state._initial_frontier)
    if initial_frontier:
        agent.on_new_frontier(initial_frontier)
    while not state.is_frontier_empty and state.get_budget_used() < state._budget_limit:
        v = agent.select(state)
        if v is None:
            break
        new_frontier = reveal_node(state, v)
        if new_frontier:
            agent.on_new_frontier(new_frontier)


def evaluate_reader(reader, classifier, builder, graphs, indices, labels,
                    budget_options, device, argmax=True, chunk_size=32):
    reader.eval()
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    use_belief = hasattr(reader, 'init_beliefs')

    for chunk_start in tqdm(range(0, len(graphs), chunk_size), desc="  Revealing"):
        chunk_end = min(chunk_start + chunk_size, len(graphs))
        all_states, state_info = [], []
        for local_i in range(chunk_start, chunk_end):
            g_raw = graphs[local_i]
            label = int(labels[local_i].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()
            for budget_ratio in budget_options:
                B = max(1, int(budget_ratio * n_nodes))
                seed_node = deterministic_seed(local_i, 0, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                all_states.append(state)
                state_info.append((y, label, budget_ratio))

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)
        if use_belief:
            all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
            all_subgraphs = [None] * n_total

        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break
            if use_belief:
                active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
                active_subgraphs = None
                if all_subgraphs[active_idx[0]] is not None:
                    active_subgraphs = [all_subgraphs[j] for j in active_idx]
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx], temperature=1.0,
                    beliefs=active_beliefs, subgraphs=active_subgraphs)
            else:
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx], temperature=1.0)

            batch_states, batch_nodes, batch_local_idx = [], [], []
            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                chosen = frontier_nodes[probs.argmax().item()] if argmax else \
                    frontier_nodes[torch.distributions.Categorical(probs).sample().item()]
                batch_states.append(all_states[j])
                batch_nodes.append(chosen)
                batch_local_idx.append(j)
            if not use_belief:
                # belief 路径下 reveal 由 reveal_and_update_beliefs_batch 内部完成
                # （其已调用 reveal_node，重复 reveal 会报错）
                for state, node in zip(batch_states, batch_nodes):
                    reveal_node(state, node)
            if use_belief and batch_states:
                from m6.training_utils import reveal_and_update_beliefs_batch
                batch_beliefs = torch.cat([all_beliefs[j] for j in batch_local_idx], dim=0)
                new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                    reader, batch_states, batch_nodes, batch_beliefs, device)
                for k, j in enumerate(batch_local_idx):
                    all_beliefs[j] = new_beliefs[k:k + 1]
                    all_subgraphs[j] = new_subgraphs[k]
            for j in active_idx:
                if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        all_subgraphs_out, all_labels_out, all_budgets_out = [], [], []
        for j in range(n_total):
            sub_g, _ = builder.build(all_states[j])
            all_subgraphs_out.append(sub_g)
            all_labels_out.append(state_info[j][:2])
            all_budgets_out.append(state_info[j][2])

        for i in range(0, len(all_subgraphs_out), 512):
            chunk_sub = all_subgraphs_out[i:i + 512]
            batch_g = dgl.batch(chunk_sub).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(batch_g, nids)
            for j, (y, label) in enumerate(all_labels_out[i:i + 512]):
                budget = all_budgets_out[i + j]
                correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
                per_budget_correct[budget] += correct
                per_budget_total[budget] += 1
                per_budget_outcomes[budget].append(correct)

    results = {}
    for b in budget_options:
        results[f'acc_{int(b * 100)}%'] = per_budget_correct[b] / max(per_budget_total[b], 1)
    results['acc_overall'] = (sum(per_budget_correct.values()) /
                              max(sum(per_budget_total.values()), 1))
    results['n_graphs'] = len(graphs)
    results['outcomes'] = {b: list(per_budget_outcomes[b]) for b in budget_options}
    return results


def evaluate_fixed(strategy_name, classifier, builder, graphs, indices, labels,
                   budget_options, device, argmax=True):
    agent = create_fixed_reader(strategy_name)
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    all_subgraphs, all_labels, all_budgets = [], [], []
    for local_i in tqdm(range(len(graphs)), desc="  Rolling"):
        g_raw = graphs[local_i]
        label = int(labels[local_i].item())
        y = torch.tensor([label], device=device)
        n_nodes = g_raw.num_nodes()
        for budget_ratio in budget_options:
            B = max(1, int(budget_ratio * n_nodes))
            seed_node = deterministic_seed(local_i, 0, n_nodes)
            state = GraphState(g_raw, seed_node=seed_node)
            state._budget_limit = B
            rollout_strategy(agent, state)
            sub_g, _ = builder.build(state)
            all_subgraphs.append(sub_g)
            all_labels.append((y, label))
            all_budgets.append(budget_ratio)
    for i in range(0, len(all_subgraphs), 512):
        batch_g = dgl.batch(all_subgraphs[i:i + 512]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)
        for j, (y, label) in enumerate(all_labels[i:i + 512]):
            budget = all_budgets[i + j]
            correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
            per_budget_correct[budget] += correct
            per_budget_total[budget] += 1
            per_budget_outcomes[budget].append(correct)
    results = {}
    for b in budget_options:
        results[f'acc_{int(b * 100)}%'] = per_budget_correct[b] / max(per_budget_total[b], 1)
    results['acc_overall'] = (sum(per_budget_correct.values()) /
                              max(sum(per_budget_total.values()), 1))
    results['n_graphs'] = len(graphs)
    results['outcomes'] = {b: list(per_budget_outcomes[b]) for b in budget_options}
    return results


def _budget_stats(outcomes):
    arr = np.array(outcomes, dtype=float)
    return {'mean': float(arr.mean()) if len(arr) else 0.0,
            'std': float(arr.std()) if len(arr) else 0.0,
            'n': len(arr)}


def main():
    parser = argparse.ArgumentParser(description='M7 统一评估')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--checkpoint', default='',
                        help='reader checkpoint（相对仓库根目录）；与 --strategy 二选一')
    parser.add_argument('--strategy', default='',
                        choices=['random', 'bfs', 'max-frontier-degree'],
                        help='固定策略名（reader.fixed.registry）')
    parser.add_argument('--classifier', required=True,
                        help='上界 GIN checkpoint（相对 code/ 目录，或绝对路径）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='评估图数限制（0=全部测试集）')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction, default=True,
                        help='argmax 选择（默认开启；--no-argmax 改为按分布采样）')
    parser.add_argument('--edge-feat-dim', type=int, default=None)
    parser.add_argument('--node-feat-dim', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--output-dir', default='m7/results')
    args = parser.parse_args()

    assert bool(args.checkpoint) != bool(args.strategy), \
        "--checkpoint 与 --strategy 必须二选一"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset)
    spec = ds.spec
    budget_options = spec.budget_options
    ln_k = np.log(spec.n_classes)

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

    builder = SubgraphBuilder()
    split = 'test'
    graphs = ds.test_graphs
    if args.max_graphs > 0:
        graphs = graphs[:args.max_graphs]
    labels = ds.test_labels[:len(graphs)]

    if args.strategy:
        print(f"Strategy: {args.strategy}")
        results = evaluate_fixed(args.strategy, classifier, builder,
                                 graphs, ds.test_indices[:len(graphs)], labels,
                                 budget_options, device, args.argmax)
        key = args.strategy
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _project_root / args.checkpoint
            if not ckpt_path.exists():
                # m6 类 bootstrap checkpoint 在 code/ 下（如 m6/attention/bc/...），
                # 与 train_rl.py --bootstrap 的"相对 code/ 目录"约定一致
                ckpt_path = _repo_root / args.checkpoint
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        sd = strip_classifier_keys(ckpt.get('reader_state_dict', ckpt))
        # reader 超参：优先 checkpoint 训练配置（train_rl 保存 training_log=config dict；
        # m6 类 checkpoint 的 training_log 是逐轮列表、超参在 config 键下，
        # 键名 n_layers_reader/dropout_reader）；CLI 显式传入的参数
        # （值 != argparse 默认值）覆盖 checkpoint 配置
        _tl = ckpt.get('training_log')
        ckpt_cfg = _tl if isinstance(_tl, dict) else ckpt.get('config') or {}
        if not isinstance(ckpt_cfg, dict):
            ckpt_cfg = {}
        _alt_keys = {'n_layers': 'n_layers_reader', 'dropout': 'dropout_reader'}
        reader_kind = ckpt_cfg.get('reader_kind') or \
            ('attention' if any('belief_state' in k for k in sd) else 'base')
        reader_args = SimpleNamespace(**vars(args))
        resolved = {}
        for name in ('hidden_dim', 'n_layers', 'dropout', 'temperature', 'm', 'n_heads'):
            v = ckpt_cfg.get(name)
            if v is None:
                v = ckpt_cfg.get(_alt_keys.get(name))
            if v is not None and getattr(args, name) == parser.get_default(name):
                setattr(reader_args, name, v)
                resolved[name] = v
        reader = build_reader(reader_kind, spec, reader_args, device)
        print(f"Reader config: kind={reader_kind}, "
              f"hidden_dim={reader_args.hidden_dim}, n_layers={reader_args.n_layers}, "
              f"dropout={reader_args.dropout}, temperature={reader_args.temperature}, "
              f"m={reader_args.m}, n_heads={reader_args.n_heads} "
              f"(from checkpoint config: {resolved or 'none'})")
        n_transferred, n_padded, skipped = transfer_weights(sd, reader)
        print(f"Checkpoint {args.checkpoint}: transferred={n_transferred}, "
              f"padded={n_padded}, skipped={len(skipped)}")
        print(f"Evaluation (argmax={'on' if args.argmax else 'off'})")
        results = evaluate_reader(reader, classifier, builder,
                                  graphs, ds.test_indices[:len(graphs)], labels,
                                  budget_options, device, args.argmax)
        key = 'learnable'

    for b in budget_options:
        print(f"  Budget {int(b * 100):2d}%: {results[f'acc_{int(b * 100)}%']:.4f}")
    print(f"  Overall: {results['acc_overall']:.4f}")

    outcomes = results['outcomes']
    json_out = {
        split: {key: {'best': {f'{b:.2f}': _budget_stats(outcomes[b])
                               for b in budget_options}}},
        'config': {
            'dataset': spec.name,
            'strategy_or_checkpoint': args.strategy or args.checkpoint,
            'classifier': str(clf_path),
            'budgets': list(budget_options),
            'n_graphs': results['n_graphs'],
            'argmax': args.argmax,
        },
    }
    out_dir = _repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'eval_heldout_{split}_{args.strategy or "reader"}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {out_path}")


if __name__ == '__main__':
    main()

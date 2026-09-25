"""候选级 rollout dump：机制例图与聚合统计的共用数据源。

逐 policy step 记录全部前沿候选的拓扑量（deg / fd / new_fe = deg−fd /
new_fn）与 reader 选择概率，以及揭示前后 reader 证据头、（可选）分类器
的熵与预测——用于两类机制命题：
1. cifar10：选 fd 1~2 的节点换来更多新前沿边特征（new_fe 大）；
2. collab：选 fd 大的节点把揭示区锁进社区核（致密化 + 置信跳变）。

协议与 eval/analyze_choice_behavior 严格同款：argmax 确定性 rollout、
deterministic_seed(graph_idx, seed_i, n)、teleport 同款（tp 节点喂入
证据流）。度量定义：
  deg   = 开邻域大小（不含自环）；
  fd    = state.get_frontier_degree（与已揭示区连接边数）；
  new_fe= deg − fd（揭示后新可见的前沿边数）；
  new_fn= 揭示后新入前沿的节点数；
  clf 熵 = InfoGainTeacher 口径 H(softmax)，G_0（单节点无边）记 null。

用法（仓库根运行）:
  python code/diagnosis/dump_candidate_rollouts.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt \
      --max-graphs 100 --n-seeds 4
  python code/diagnosis/dump_candidate_rollouts.py --dataset collab \
      --root D:\\data\\COLLAB \
      --checkpoint code/result/collab/experiments/evidence1/evidence/rl/checkpoint_epoch6.pt \
      --classifier code/result/collab/classifier/finetune/best_collab_classifier.pt \
      --clf-entropy --max-graphs 100 --n-seeds 1
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import (  # noqa: E402
    reveal_node, teleport_if_stuck)
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset, release_dataset  # noqa: E402
from tools.seeds import deterministic_seed  # noqa: E402
from train_reader.core import (  # noqa: E402
    load_classifier_frozen, load_reader_from_checkpoint)

# 大前沿截断：|F| 超过此值时只存 prob/fd/new_fe 各 top-K 并集 + 被选节点
FRONTIER_FULL_MAX = 40
TOP_KEEP = 30


def entropy_nats(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-8)).sum())


def neighbors_map(g) -> dict:
    """开邻域表（不含自环）。"""
    src, dst = g.edges()
    adj = defaultdict(set)
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def clf_entropy(builder, clf, state, device):
    """分类器对当前揭示子图的 (熵, 预测)；G_0 记 (None, None)。"""
    if len(state.V_revealed) == 1 and len(state.revealed_edge_keys) == 0:
        return None, None
    sub_g, _ = builder.build(state)
    sub_g = sub_g.to(device)
    nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = clf(sub_g, nids)
    probs = torch.softmax(logits, dim=-1)[0]
    return round(entropy_nats(logits), 6), int(probs.argmax().item())


def keep_candidates(cands, chosen):
    """|F| 大时截断：各指标 top-K ∪ 被选节点（保序去重）。"""
    if len(cands) <= FRONTIER_FULL_MAX:
        return cands, False
    keep_ids = {chosen}
    for key in ('prob', 'fd', 'new_fe', 'closed_tri'):
        for c in sorted(cands, key=lambda c: -c[key])[:TOP_KEEP]:
            keep_ids.add(c['node'])
    return [c for c in cands if c['node'] in keep_ids], True


def rollout_dump(reader, g, graph_idx, seed_i, label, budget_ratio, device,
                 adj, clf, builder, want_clf):
    n = g.num_nodes()
    B = max(1, int(budget_ratio * n))
    seed_node = deterministic_seed(graph_idx, seed_i, n)
    state = GraphState(g, seed_node=seed_node)
    state._budget_limit = B
    hc = reader.init_evidence(1, device)
    last_revealed = seed_node
    rng = random.Random(seed_node * 1_000_003 + B)
    steps = []

    while state.get_budget_used() < state._budget_limit:
        if state.is_frontier_empty:
            tp = teleport_if_stuck(state, rng=rng)
            steps.append({'step': len(steps), 'teleport': True, 'node': tp})
            if tp is None:
                break
            last_revealed = tp
            continue

        frontier = state.get_frontier_nodes()
        with torch.no_grad():
            results, hc, ev_logits = reader.forward_states_evidence(
                [state], temperature=1.0, hc_list=hc,
                last_revealed=[last_revealed])
        _, probs, r_frontier = results[0]
        if not r_frontier:
            break

        revealed = set(state.V_revealed)
        fset = set(r_frontier)
        cands = []
        for i, v in enumerate(r_frontier):
            nb = adj[v]
            fd = len(nb & revealed)
            # 闭合三角形：v 的已揭示邻居之间已有的边数——高 fd 且邻居互密
            # = v 坐在社区中心（COLLAB 命题 2 的操作化指标）
            rn = nb & revealed
            closed = sum(len(adj[u] & rn) for u in rn) // 2
            cands.append({'node': int(v), 'deg': len(nb), 'fd': fd,
                          'new_fe': len(nb - revealed),
                          'new_fn': len(nb - revealed - fset),
                          'closed_tri': closed,
                          'prob': round(float(probs[i]), 6)})
        pos = int(probs.argmax().item())
        chosen = int(r_frontier[pos])
        rec = {'step': len(steps), 'teleport': False,
               'revealed_count_pre': len(revealed),
               'revealed_edges_pre': len(state.revealed_edge_keys),
               'n_frontier': len(r_frontier),
               'ev_entropy_pre': round(entropy_nats(ev_logits[0]), 6),
               'ev_pred_pre': int(ev_logits[0].argmax().item()),
               'chosen': chosen, 'chosen_prob': round(float(probs[pos]), 6)}
        if want_clf:
            h, pr = clf_entropy(builder, clf, state, device)
            rec['clf_entropy_pre'], rec['clf_pred_pre'] = h, pr
        cands, truncated = keep_candidates(cands, chosen)
        rec['candidates'] = cands
        rec['frontier_truncated'] = truncated

        reveal_node(state, chosen)
        last_revealed = chosen
        rec['revealed_edges_post'] = len(state.revealed_edge_keys)
        if want_clf:
            h, pr = clf_entropy(builder, clf, state, device)
            rec['clf_entropy_post'], rec['clf_pred_post'] = h, pr
        steps.append(rec)

    with torch.no_grad():
        _, _, ev_final = reader.forward_states_evidence(
            [state], temperature=1.0, hc_list=hc,
            last_revealed=[last_revealed])
    ev_pred = int(ev_final[0].argmax().item())
    return {'graph_idx': graph_idx, 'seed_i': seed_i,
            'seed_node': seed_node, 'label': int(label), 'n_nodes': n,
            'budget': B, 'n_steps': len(steps), 'ev_pred': ev_pred,
            'ev_correct': int(ev_pred == label), 'steps': steps}


def main():
    parser = argparse.ArgumentParser(
        description='候选级 rollout dump（机制例图 + 聚合统计数据源）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--classifier', default=None,
                        help='--clf-entropy 时必填（collab 用子图适应版）')
    parser.add_argument('--clf-entropy', action='store_true',
                        help='记录揭示前后分类器熵/预测')
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--split', default='test',
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--n-seeds', type=int, default=4)
    parser.add_argument('--max-graphs', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    if args.clf_entropy and not args.classifier:
        raise SystemExit('--clf-entropy 需要 --classifier')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt_path = Path(args.checkpoint)
    reader = load_reader_from_checkpoint(ckpt_path, device,
                                         stochastic=False, temperature=1.0)
    reader.eval()
    if getattr(reader.config, 'kind', '') != 'evidence':
        raise SystemExit('本 dump 假定 evidence reader（证据头逐步预测）')
    clf = builder = None
    if args.clf_entropy:
        clf = load_classifier_frozen(args.classifier, device)
        builder = SubgraphBuilder()
        print(f'Classifier: {args.classifier}')
    print(f'Reader: {ckpt_path} (deterministic argmax)')

    ds = get_dataset(args.dataset, root=args.root, splits=(args.split,),
                     max_graphs=args.max_graphs or None)
    graphs = getattr(ds, f'{args.split}_graphs')
    labels = getattr(ds, f'{args.split}_labels')
    print(f'[{args.split}] {len(graphs)} 图 | budget_ratio={args.budget_ratio}'
          f' | n_seeds={args.n_seeds}')
    out_path = (Path(args.output) if args.output else
                ckpt_path.parent / 'behavior-analysis'
                / f'candidate_rollouts_b10_g{len(graphs)}s{args.n_seeds}.json')

    episodes, n_steps_policy = [], 0
    for gi in tqdm(range(len(graphs)), desc='graphs'):
        g = graphs[gi]
        adj = neighbors_map(g)
        for si in range(args.n_seeds):
            ep = rollout_dump(reader, g, gi, si, labels[gi], args.budget_ratio,
                              device, adj, clf, builder, args.clf_entropy)
            n_steps_policy += sum(1 for s in ep['steps'] if not s['teleport'])
            episodes.append(ep)
    release_dataset(ds)

    out = {
        'config': {
            'dataset': args.dataset, 'split': args.split,
            'budget_ratio': args.budget_ratio, 'n_seeds': args.n_seeds,
            'max_graphs': args.max_graphs, 'argmax': True,
            'checkpoint': str(ckpt_path),
            'classifier': args.classifier,
            'clf_entropy': args.clf_entropy,
            'seed_method': 'deterministic_seed(graph_idx, seed_i, n_nodes)',
            'metrics': {
                'deg': '开邻域大小（不含自环）',
                'fd': 'state.get_frontier_degree（与已揭示区连接边数）',
                'new_fe': 'deg − fd（揭示后新可见前沿边数）',
                'new_fn': '揭示后新入前沿节点数',
                'closed_tri': 'v 的已揭示邻居间已有边数（揭示 v 闭合的三角形数，'
                              '社区中心指标）',
                'ev_*': 'reader 证据头逐步预测（pre=揭示前 forward）',
                'clf_*': ('InfoGainTeacher 口径 H(softmax)，G_0 记 null'
                          if args.clf_entropy else '未记录'),
                'frontier_truncated': f'|F|>{FRONTIER_FULL_MAX} 时只存各指标 '
                                      f'top{TOP_KEEP} ∪ 被选节点',
            },
        },
        'n_graphs': len(graphs), 'n_episodes': len(episodes),
        'n_steps_policy': n_steps_policy,
        'ev_acc': round(sum(ep['ev_correct'] for ep in episodes)
                        / max(len(episodes), 1), 4),
        'per_graph': episodes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'saved: {out_path}')
    print(f"ev_acc = {out['ev_acc']} ({len(episodes)} episodes, "
          f'{n_steps_policy} policy steps)')


if __name__ == '__main__':
    main()

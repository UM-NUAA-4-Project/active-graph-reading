"""读者选择行为分析：主结果 reader 在固定预算下每步所选前沿节点的教师指标画像。

在 eval 协议下（argmax 确定性 rollout、确定性种子、teleport 播种与
eval_loop 一致）逐图回放 reader 的阅读过程，对每一步被选中的前沿节点记录：

- frontier_degree：GraphState 口径的前沿度数（该节点与已揭示区的连接边数，
  state.get_frontier_degree）；
- graph_degree：全图度数（StructuralTeacher 口径，g.in_degrees()，含数据
  加载器补的自环）；
- grad_score：梯度归因 teacher（GradientTeacher，全图一次反传的静态节点
  分数）在该节点的分数；
- info_gain：信息增益 = reader 证据头 belief 熵的逐步下降 H(b_t) − H(b_t+1)。
  证据头预测（ev_logits）本就随每步 forward_states_evidence 返回，沿实际
  轨迹记录各步 belief 熵、相邻两步作差，即当前所选节点带来的熵减——与
  eval 终判同一分类头，零额外前向。每步 IG 由下一步前向回填（末步由终局
  判定前向回填，teleport 步同理），全部 policy 步都有定义；
- init_entropy / n_classes：G_0（仅种子、未揭示）时的证据头 belief 熵与
  类别数。IG 面板横坐标 1 的位置记录种子信息增益 ln k − init_entropy
  （相对均匀先验的熵减）；
- degree_rank_frac / degree_is_max：所选节点前沿度数在前沿候选内的中间
  名次分位（并列取半份额，全并列=0.5）与是否并列最优；
- cand_degree_mean / cand_degree_max：该步前沿候选的度数均值/最大值
  （所选值与之对照即"度数偏好"）。

所有排名分位统一用中间名次（并列各取半份额，全并列=0.5）：COLLAB 上
大量候选的 IG 恰好并列，strict 名次会失真；连续分数（cifar10）下中间
名次与 strict 名次数值相同。

同时记录所选节点在每个指标上前沿内的排名分位（rank_frac ∈ [0,1]，1=选了
该指标最高的候选）与是否为最优（is_max），便于回答"读者偏好高梯度/高
信息增益/高度数节点吗"。

用法:
  python code/diagnosis/analyze_choice_behavior.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch3.pt \
      --classifier code/result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt \
      --budget-ratio 0.10 [--n-seeds 1] [--max-graphs 50] [--split test]

输出：<checkpoint 同目录>/choice_behavior_b{10}.json（--output 可覆盖；
--max-graphs 冒烟时自动加 _smokeN 后缀，每 1000 图增量落盘）。
"""

import argparse
import json
import math
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
from trajectories.teachers.gradient import GradientTeacher  # noqa: E402


def _entropy_of_logits(logits) -> float:
    """类别分布的香农熵（nats）。"""
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-8)).sum())


def _edge_key_to_id(g) -> dict:
    """canonical (min,max) key → 首见边 id（与 eval_loop 同款预构建）。"""
    if 'feat' not in g.edata:
        return {}
    src, dst = g.edges()
    out = {}
    for eid in range(g.num_edges()):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in out:
            out[key] = eid
    return out


def _rank_frac(values, idx) -> float:
    """chosen 在候选分数中的排名分位：严格更小的占比；单候选记 1.0。"""
    if len(values) <= 1:
        return 1.0
    v = values[idx]
    return sum(1 for x in values if x < v) / (len(values) - 1)


def _rank_mid_frac(values, idx) -> float:
    """含并列的中间名次分位（度数等离散量用）：并列各取一半份额；
    全并列 → 0.5（等价于在并列中随机选）。"""
    if len(values) <= 1:
        return 1.0
    v = values[idx]
    less = sum(1 for x in values if x < v)
    equal_others = sum(1 for x in values if x == v) - 1
    return (less + 0.5 * equal_others) / (len(values) - 1)


def _is_max(values, idx) -> bool:
    return values[idx] >= max(values)


def _mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return round(m, 6), round(math.sqrt(var), 6), len(xs)


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx == 0 or vy == 0:
        return None
    return round(cov / math.sqrt(vx * vy), 6)


def main():
    parser = argparse.ArgumentParser(description='reader 选择行为分析（教师指标画像）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--classifier', required=True)
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--max-graphs', type=int, default=0)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # 分类器需保留 autograd（梯度归因 teacher 反传）；推理处一律 no_grad
    classifier = load_classifier_frozen(args.classifier, device, requires_grad=True)
    print(f"Classifier: {args.classifier}", flush=True)
    ckpt_path = Path(args.checkpoint)
    reader = load_reader_from_checkpoint(ckpt_path, device,
                                         stochastic=False, temperature=1.0)
    reader.eval()
    use_evidence = getattr(reader.config, 'kind', '') == 'evidence'
    print(f"Reader: kind={reader.config.kind} (deterministic argmax)"
          if args.argmax else "Reader: stochastic sampling", flush=True)

    grad_teacher = GradientTeacher()
    builder = SubgraphBuilder()

    ds = get_dataset(args.dataset, root=args.root, splits=(args.split,),
                     max_graphs=args.max_graphs, stratified=args.stratified)
    graphs = getattr(ds, f'{args.split}_graphs')
    labels = getattr(ds, f'{args.split}_labels')
    print(f"[{args.split}] {len(graphs)} 图 | budget_ratio={args.budget_ratio}"
          f" | n_seeds={args.n_seeds}", flush=True)

    if args.output:
        out_path = Path(args.output)
    else:
        # 规范目录：分析产物一律入 checkpoint 同级的 behavior-analysis/
        out_dir = ckpt_path.parent / 'behavior-analysis'
        suffix = f"_smoke{args.max_graphs}" if args.max_graphs else ""
        out_path = out_dir / f"choice_behavior_b{int(args.budget_ratio * 100)}{suffix}.json"

    config = {
        'dataset': args.dataset,
        'split': args.split,
        'budget_ratio': args.budget_ratio,
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'checkpoint': str(ckpt_path),
        'classifier': str(args.classifier),
        'seed_method': 'deterministic_seed(graph_idx, seed_i, n_nodes)',
        'metrics': {
            'frontier_degree': 'state.get_frontier_degree（与已揭示区的连接边数）',
            'graph_degree': '全图 in_degrees（StructuralTeacher 口径，含自环）',
            'grad_score': 'GradientTeacher 全图静态节点分数（requires_grad 反传）',
            'info_gain': 'H(b_t) − H(b_t+1)：reader 证据头 belief 熵的逐步'
                         '下降（沿实际轨迹相邻前向作差回填，全部 policy 步'
                         '有定义）',
        },
    }

    per_graph_records = []
    done = 0

    def _dump():
        summary = summarize(per_graph_records, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'config': config, 'summary': summary,
                       'n_graphs_done': done, 'per_graph': per_graph_records},
                      f, ensure_ascii=False)

    for local_i in tqdm(range(len(graphs)), desc="  Analyzing"):
        g_raw = graphs[local_i]
        label = int(labels[local_i].item())
        n_nodes = g_raw.num_nodes()
        B = max(1, int(args.budget_ratio * n_nodes))
        ek2id = _edge_key_to_id(g_raw)
        deg = g_raw.in_degrees().float()   # StructuralTeacher 口径（含自环）

        # 梯度归因：全图一次反传的静态节点分数（轨迹管线同款）
        with torch.enable_grad():
            grad_scores = grad_teacher.score_nodes(g_raw, classifier, label)

        for seed_i in range(args.n_seeds):
            seed_node = deterministic_seed(local_i, seed_i, n_nodes)
            state = GraphState(g_raw, seed_node=seed_node, edge_key_to_id=ek2id)
            state._budget_limit = B
            hc = reader.init_evidence(1, device) if use_evidence else None
            last_revealed = seed_node
            rng = random.Random(seed_node * 1_000_003 + B)

            steps = []
            step_idx = 0
            prev_ent = None       # 上一次前向的 belief 熵（当前步揭示前）
            pending_idx = None    # 等待用下一步 belief 熵回填 IG 的 step 下标
            init_ent = None       # G_0（仅种子）的 belief 熵
            n_cls = None
            if use_evidence:
                # G_0 前向：初始 belief 熵（IG 面板 x=1 的种子信息增益
                # ln k − init_entropy 由它算出），并作为首个 IG 差分的起点
                with torch.no_grad():
                    _, _, ev0 = reader.forward_states_evidence(
                        [state], temperature=1.0, hc_list=hc,
                        last_revealed=[last_revealed])
                init_ent = _entropy_of_logits(ev0[0])
                n_cls = int(ev0[0].shape[0])
                prev_ent = init_ent
            while state.get_budget_used() < state._budget_limit:
                if state.is_frontier_empty:
                    tp_node = teleport_if_stuck(state, rng=rng)
                    if tp_node is None:
                        break
                    steps.append({
                        'step': step_idx, 'chosen': int(tp_node),
                        'frontier_degree': 0, 'graph_degree': int(deg[tp_node]),
                        'grad_score': round(float(grad_scores[tp_node]), 6),
                        'info_gain': None, 'n_frontier': 0,
                        'teleport': True,
                    })
                    pending_idx = step_idx
                    step_idx += 1
                    last_revealed = tp_node
                    continue

                frontier = state.get_frontier_nodes()
                deg_vals = [state.get_frontier_degree(v) for v in frontier]

                with torch.no_grad():
                    if use_evidence:
                        results, hc, ev_logits = reader.forward_states_evidence(
                            [state], temperature=1.0, hc_list=hc,
                            last_revealed=[last_revealed])
                    else:
                        results = reader.forward_states([state], temperature=1.0)
                _, probs, r_frontier = results[0]
                if not r_frontier:
                    break

                # 本步揭示前的 belief 熵；与上一道前向的熵作差即回填上一步
                # 所选节点的 info_gain（H(b_t) − H(b_t+1)）
                if use_evidence:
                    cur_ent = _entropy_of_logits(ev_logits[0])
                    if pending_idx is not None and prev_ent is not None:
                        steps[pending_idx]['info_gain'] = round(
                            prev_ent - cur_ent, 6)
                    prev_ent = cur_ent
                pending_idx = None

                pos = (probs.argmax().item() if args.argmax
                       else torch.distributions.Categorical(probs).sample().item())
                chosen = int(r_frontier[pos])

                grad_vals = [float(grad_scores[v]) for v in frontier]
                cpos = frontier.index(chosen)   # r_frontier 与 frontier 同序（已核）
                rec = {
                    'step': step_idx, 'chosen': chosen,
                    'frontier_degree': int(state.get_frontier_degree(chosen)),
                    'graph_degree': int(deg[chosen]),
                    'grad_score': round(float(grad_scores[chosen]), 6),
                    'grad_rank_frac': round(_rank_mid_frac(grad_vals, pos), 4),
                    'grad_is_max': _is_max(grad_vals, pos),
                    'degree_rank_frac': round(_rank_mid_frac(deg_vals, cpos), 4),
                    'degree_is_max': _is_max(deg_vals, cpos),
                    'cand_degree_mean': round(sum(deg_vals) / len(deg_vals), 4),
                    'cand_degree_max': int(max(deg_vals)),
                    'info_gain': None,
                    'n_frontier': len(frontier),
                    'teleport': False,
                }
                steps.append(rec)
                pending_idx = step_idx
                step_idx += 1

                reveal_node(state, chosen)
                last_revealed = chosen

            # ---- 终局判定（eval 同款：证据头补充前向 + 冻结分类器对最终子图）----
            if use_evidence:
                with torch.no_grad():
                    _, _, final_ev = reader.forward_states_evidence(
                        [state], temperature=1.0, hc_list=hc,
                        last_revealed=[last_revealed])
                ev_pred = int(final_ev[0].argmax(dim=-1).item())
                # 末步所选节点的 info_gain：末步揭示前后的 belief 熵差
                if pending_idx is not None and prev_ent is not None:
                    steps[pending_idx]['info_gain'] = round(
                        prev_ent - _entropy_of_logits(final_ev[0]), 6)
            else:
                ev_pred = None
            sub_g, _ = builder.build(state)
            sub_g = sub_g.to(device)
            nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
            with torch.no_grad():
                clf_logits = classifier(sub_g, nids)
            clf_pred = int(clf_logits.argmax(dim=-1).item())

            per_graph_records.append({
                'graph_idx': local_i, 'seed_i': seed_i, 'seed_node': int(seed_node),
                'label': label, 'n_nodes': int(n_nodes), 'budget': B,
                'n_steps': len(steps),
                'init_entropy': (None if init_ent is None
                                 else round(init_ent, 6)),
                'n_classes': n_cls,
                'ev_pred': ev_pred,
                'ev_correct': (None if ev_pred is None else int(ev_pred == label)),
                'clf_pred': clf_pred, 'clf_correct': int(clf_pred == label),
                'steps': steps,
            })

        done = local_i + 1
        if done % 1000 == 0:
            _dump()
            print(f"[progress] {done}/{len(graphs)} 图已分析（增量落盘）", flush=True)

    _dump()
    release_dataset(ds)
    print(f"JSON saved to {out_path}", flush=True)


def summarize(records, config):
    """逐步记录 → 行为统计（所选节点指标分布 / 排名分位 / 与最优重合率）。"""
    step_recs = [r for g in records for r in g['steps']]
    policy_steps = [r for r in step_recs if not r.get('teleport')]
    metrics = ('frontier_degree', 'graph_degree', 'grad_score', 'info_gain')

    out = {
        'n_graphs': len(records),
        'n_steps_total': len(step_recs),
        'n_steps_policy': len(policy_steps),
        'n_steps_teleport': len(step_recs) - len(policy_steps),
        'ev_acc': _mean_std([g['ev_correct'] for g in records
                             if g['ev_correct'] is not None])[0],
        'clf_acc_on_final_subgraph': _mean_std(
            [g['clf_correct'] for g in records])[0],
    }
    for m in metrics:
        vals = [r.get(m) for r in policy_steps]
        m_, s_, n = _mean_std(vals)
        entry = {'chosen_mean': m_, 'chosen_std': s_, 'n': n}
        if m == 'grad_score':
            entry['chosen_is_max_rate'] = round(sum(
                1 for r in policy_steps if r.get('grad_is_max')) / n, 4) if n else None
            entry['mean_rank_frac'] = round(sum(
                r['grad_rank_frac'] for r in policy_steps
                if 'grad_rank_frac' in r)
                / max(sum(1 for r in policy_steps if 'grad_rank_frac' in r), 1), 4)
        if m == 'frontier_degree':
            entry['chosen_is_max_rate'] = round(sum(
                1 for r in policy_steps if r.get('degree_is_max')) / n, 4) if n else None
            entry['mean_rank_frac'] = round(sum(
                r['degree_rank_frac'] for r in policy_steps
                if 'degree_rank_frac' in r)
                / max(sum(1 for r in policy_steps if 'degree_rank_frac' in r), 1), 4)
            entry['cand_degree_mean'] = round(sum(
                r['cand_degree_mean'] for r in policy_steps) / max(n, 1), 4)
        if m == 'info_gain':
            have = [r for r in policy_steps if r.get('info_gain') is not None]
            entry['coverage'] = round(len(have) / max(len(policy_steps), 1), 4)
        out[m] = entry

    grad_ig_corr = _pearson([r.get('grad_score') for r in policy_steps],
                            [r.get('info_gain') for r in policy_steps])
    out['pearson_grad_vs_info_gain'] = grad_ig_corr

    # 种子信息增益（IG 面板 x=1）：ln k − G_0 初始 belief 熵
    have_init = [g for g in records if g.get('init_entropy') is not None]
    out['init_info_gain'] = _mean_std([
        math.log(g['n_classes']) - g['init_entropy'] for g in have_init])[0]

    # 按读入进度分桶（step / budget 五分位）：早期 vs 后期选择偏好
    quintile = defaultdict(list)
    for g in records:
        for r in g['steps']:
            if r.get('teleport') or g['budget'] <= 0:
                continue
            q = min(4, int(5 * r['step'] / g['budget']))
            quintile[q].append(r)
    out['by_progress_quintile'] = {
        f'q{q}': {
            'n': len(qs),
            'frontier_degree': _mean_std([r['frontier_degree'] for r in qs])[0],
            'grad_score': _mean_std([r.get('grad_score') for r in qs])[0],
            'info_gain': _mean_std([r.get('info_gain') for r in qs])[0],
            'grad_rank_frac': round(sum(
                r['grad_rank_frac'] for r in qs if 'grad_rank_frac' in r)
                / max(sum(1 for r in qs if 'grad_rank_frac' in r), 1), 4),
            'degree_rank_frac': round(sum(
                r['degree_rank_frac'] for r in qs if 'degree_rank_frac' in r)
                / max(sum(1 for r in qs if 'degree_rank_frac' in r), 1), 4),
            'cand_degree_mean': round(sum(
                r['cand_degree_mean'] for r in qs) / max(len(qs), 1), 4),
        }
        for q, qs in sorted(quintile.items())
    }

    # 按终判正误分组：读对/读错的图，所选节点指标是否不同
    for val, name in ((1, 'ev_correct'), (0, 'ev_wrong')):
        grp = [r for g in records if g['ev_correct'] == val
               for r in g['steps'] if not r.get('teleport')]
        out[name] = {
            'n_steps': len(grp),
            'grad_score': _mean_std([r.get('grad_score') for r in grp])[0],
            'info_gain': _mean_std([r.get('info_gain') for r in grp])[0],
            'frontier_degree': _mean_std([r['frontier_degree'] for r in grp])[0],
            'degree_rank_frac': round(sum(
                r['degree_rank_frac'] for r in grp if 'degree_rank_frac' in r)
                / max(sum(1 for r in grp if 'degree_rank_frac' in r), 1), 4),
        }
    return out


if __name__ == '__main__':
    main()

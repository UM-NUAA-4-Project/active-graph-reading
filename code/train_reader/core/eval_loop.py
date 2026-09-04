"""评估循环：learnable reader（base/attention/evidence 统一）+ 固定策略。

收敛 m5/base/analysis/eval.py → m6/attention/eval_heldout.py → m7/eval.py
三代实现（以 m7 为基准），JSON schema 与旧版完全兼容：
    {split: {'learnable'|strategy: {'s0': {...}, ..., 'mean': {...}}}, ...}

evidence reader：rollout 走 forward_states_evidence（选点仍与 base 一致），
终判 = 读满预算后补充一次证据前向的 ev_logits argmax（决策-分类一体，
含最后一次 reveal 的证据）；冻结上界分类器对同一批最终子图的判定写为
参照键 acc_clf_*（并落盘 acc_clf_reference）。
"""

import random
from collections import defaultdict

import numpy as np
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck
from tools.seeds import deterministic_seed
from .common import reveal_and_update_beliefs_batch


@torch.no_grad()
def rollout_strategy(agent, state):
    """单图 rollout：agent.select 直到预算耗尽或 frontier 为空。

    有状态策略（如 BFS 的 FIFO 队列）依赖 reset/on_new_frontier 接线：
    每个 rollout 独立，重置内部状态并以初始前沿播种。
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


def evaluate_reader(reader, classifier, builder, graphs, labels,
                    budget_options, device, argmax=True, chunk_size=32,
                    n_seeds=4):
    """learnable reader 评估：跨图 lockstep + 批量分类器 + 多预算/多 seed 统计。

    evidence reader：终判由自身证据头给出（读满后补充前向的 ev_logits
    argmax，决策-分类一体）；冻结上界分类器对同一批最终揭示子图的判定
    作为参照，写入 acc_clf_* 键。base/attention 行为不变（上界分类器即终判）。
    """
    reader.eval()
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_seed_outcomes = {i: defaultdict(list) for i in range(n_seeds)}
    per_seed_per_class = {i: defaultdict(lambda: defaultdict(list))
                          for i in range(n_seeds)}
    per_budget_clf_correct = defaultdict(int)
    per_budget_clf_outcomes = defaultdict(list)
    use_belief = hasattr(reader, 'init_beliefs')
    use_evidence = (getattr(reader, 'config', None) is not None
                    and getattr(reader.config, 'kind', '') == 'evidence')

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
                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node)
                    state._budget_limit = B
                    all_states.append(state)
                    state_info.append((y, label, budget_ratio, seed_i))

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)
        if use_belief:
            all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
            all_subgraphs = [None] * n_total
        if use_evidence:
            all_hc = reader.init_evidence(n_total, device)
            last_revealed = [s.seed_node for s in all_states]

        eval_rng = random.Random(0)  # 跳转确定性可复现
        for _step in range(max_budget):
            for j in range(n_total):
                if active[j] and all_states[j].is_frontier_empty:
                    st = all_states[j]
                    tp_node = teleport_if_stuck(st, rng=eval_rng)
                    if tp_node is not None:
                        if use_belief:
                            all_subgraphs[j] = reader.frontier_builder.build(st)
                        if use_evidence:
                            last_revealed[j] = tp_node
                        if st.get_budget_used() >= st._budget_limit:
                            active[j] = False
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break
            if use_evidence:
                batch_results, active_hc, _ = reader.forward_states_evidence(
                    [all_states[j] for j in active_idx], temperature=1.0,
                    hc_list=[all_hc[j] for j in active_idx],
                    last_revealed=[last_revealed[j] for j in active_idx])
                for k, j in enumerate(active_idx):
                    all_hc[j] = active_hc[k]
            elif use_belief:
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
                for state, node in zip(batch_states, batch_nodes):
                    reveal_node(state, node)
                if use_evidence:
                    for j, node in zip(batch_local_idx, batch_nodes):
                        last_revealed[j] = node
            if use_belief and batch_states:
                batch_beliefs = torch.cat([all_beliefs[j] for j in batch_local_idx], dim=0)
                new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                    reader, batch_states, batch_nodes, batch_beliefs, device)
                for k, j in enumerate(batch_local_idx):
                    all_beliefs[j] = new_beliefs[k:k + 1]
                    all_subgraphs[j] = new_subgraphs[k]
            for j in active_idx:
                if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                    active[j] = False

        # ---- evidence 终局判定：读满后的补充前向（含最后一次 reveal 的证据）----
        if use_evidence:
            _, _, final_ev = reader.forward_states_evidence(
                all_states, temperature=1.0, hc_list=all_hc,
                last_revealed=last_revealed)

        all_subgraphs_out, all_labels_out, all_budgets_out, all_seed_ids_out = \
            [], [], [], []
        for j in range(n_total):
            sub_g, _ = builder.build(all_states[j])
            all_subgraphs_out.append(sub_g)
            all_labels_out.append(state_info[j][:2])
            all_budgets_out.append(state_info[j][2])
            all_seed_ids_out.append(state_info[j][3])

        for i in range(0, len(all_subgraphs_out), 512):
            chunk_sub = all_subgraphs_out[i:i + 512]
            batch_g = dgl.batch(chunk_sub).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(batch_g, nids)
            for j, (y, label) in enumerate(all_labels_out[i:i + 512]):
                budget = all_budgets_out[i + j]
                seed_i = all_seed_ids_out[i + j]
                if use_evidence:
                    # 终判：证据头读满后的预测（决策-分类一体）
                    correct = 1.0 if final_ev[i + j].argmax(dim=-1).item() == label else 0.0
                    clf_correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
                    per_budget_clf_correct[budget] += clf_correct
                    per_budget_clf_outcomes[budget].append(clf_correct)
                else:
                    correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
                per_budget_correct[budget] += correct
                per_budget_total[budget] += 1
                per_budget_outcomes[budget].append(correct)
                per_seed_outcomes[seed_i][budget].append(correct)
                per_seed_per_class[seed_i][budget][label].append(correct)

    return _summarize(budget_options, graphs, per_budget_correct,
                      per_budget_total, per_budget_outcomes,
                      per_seed_outcomes, per_seed_per_class,
                      clf_correct=(per_budget_clf_correct if use_evidence else None),
                      clf_outcomes=(per_budget_clf_outcomes if use_evidence else None))


def evaluate_fixed(strategy_name, classifier, builder, graphs, labels,
                   budget_options, device, argmax=True, n_seeds=4):
    """固定策略（random/bfs/max_degree）评估，统计口径与 evaluate_reader 一致。"""
    from active_graph_reader.reader.fixed import create_fixed_reader
    agent = create_fixed_reader(strategy_name)
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_seed_outcomes = {i: defaultdict(list) for i in range(n_seeds)}
    per_seed_per_class = {i: defaultdict(lambda: defaultdict(list))
                          for i in range(n_seeds)}
    all_subgraphs, all_labels, all_budgets, all_seed_ids = [], [], [], []
    for local_i in tqdm(range(len(graphs)), desc="  Rolling"):
        g_raw = graphs[local_i]
        label = int(labels[local_i].item())
        y = torch.tensor([label], device=device)
        n_nodes = g_raw.num_nodes()
        for budget_ratio in budget_options:
            B = max(1, int(budget_ratio * n_nodes))
            for seed_i in range(n_seeds):
                seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                state = GraphState(g_raw, seed_node=seed_node)
                state._budget_limit = B
                rollout_strategy(agent, state)
                sub_g, _ = builder.build(state)
                all_subgraphs.append(sub_g)
                all_labels.append((y, label))
                all_budgets.append(budget_ratio)
                all_seed_ids.append(seed_i)
    for i in range(0, len(all_subgraphs), 512):
        batch_g = dgl.batch(all_subgraphs[i:i + 512]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, nids)
        for j, (y, label) in enumerate(all_labels[i:i + 512]):
            budget = all_budgets[i + j]
            seed_i = all_seed_ids[i + j]
            correct = 1.0 if logits[j].argmax(dim=-1).item() == label else 0.0
            per_budget_correct[budget] += correct
            per_budget_total[budget] += 1
            per_budget_outcomes[budget].append(correct)
            per_seed_outcomes[seed_i][budget].append(correct)
            per_seed_per_class[seed_i][budget][label].append(correct)

    return _summarize(budget_options, graphs, per_budget_correct,
                      per_budget_total, per_budget_outcomes,
                      per_seed_outcomes, per_seed_per_class)


def _summarize(budget_options, graphs, per_budget_correct, per_budget_total,
               per_budget_outcomes, per_seed_outcomes, per_seed_per_class,
               clf_correct=None, clf_outcomes=None):
    results = {}
    for b in budget_options:
        results[f'acc_{int(b * 100)}%'] = per_budget_correct[b] / max(per_budget_total[b], 1)
    results['acc_overall'] = (sum(per_budget_correct.values()) /
                              max(sum(per_budget_total.values()), 1))
    if clf_correct is not None:
        # 参照：冻结上界分类器对同一批最终揭示子图的判定（仅 evidence）
        for b in budget_options:
            results[f'acc_clf_{int(b * 100)}%'] = clf_correct[b] / max(per_budget_total[b], 1)
        results['acc_clf_overall'] = (sum(clf_correct.values()) /
                                      max(sum(per_budget_total.values()), 1))
        results['clf_outcomes'] = {b: list(clf_outcomes[b]) for b in budget_options}
    results['n_graphs'] = len(graphs)
    results['outcomes'] = {b: list(per_budget_outcomes[b]) for b in budget_options}
    n_seeds = len(per_seed_outcomes)
    results['per_seed_outcomes'] = {
        i: {b: list(per_seed_outcomes[i][b]) for b in budget_options}
        for i in range(n_seeds)
    }
    results['per_seed_per_class'] = {
        i: {b: {lbl: list(v) for lbl, v in per_seed_per_class[i][b].items()}
            for b in budget_options}
        for i in range(len(per_seed_per_class))
    }
    return results


def budget_stats(outcomes):
    """outcomes → {'mean','std','n'}（eval JSON 的叶子统计格式）。"""
    arr = np.array(outcomes, dtype=float)
    return {'mean': float(arr.mean()) if len(arr) else 0.0,
            'std': float(arr.std()) if len(arr) else 0.0,
            'n': len(arr)}


def build_eval_json(results, split: str, key: str, n_seeds: int,
                    budget_options, config: dict) -> dict:
    """评估结果 → 旧版兼容 JSON（{split: {key: {s0..sN, mean}}}）。"""
    per_seed_stats = {
        f's{seed_i}': {f'{b:.2f}': budget_stats(per_seed[b])
                       for b in budget_options}
        for seed_i, per_seed in results['per_seed_outcomes'].items()
    }
    pooled = {b: [] for b in budget_options}
    for seed_i, per_seed in results['per_seed_outcomes'].items():
        for b in budget_options:
            pooled[b] += per_seed[b]
    per_seed_stats['mean'] = {
        f'{b:.2f}': budget_stats(pooled[b]) for b in budget_options
    }
    all_labels = sorted({lbl for i in results['per_seed_per_class']
                         for b in budget_options
                         for lbl in results['per_seed_per_class'][i][b]})
    per_seed_class_stats = {
        f's{seed_i}': {
            str(lbl): {f'{b:.2f}': budget_stats(
                results['per_seed_per_class'][seed_i][b].get(lbl, []))
                for b in budget_options}
            for lbl in all_labels
        }
        for seed_i in range(n_seeds)
    }
    pooled_per_class = {b: defaultdict(list) for b in budget_options}
    for seed_i in range(n_seeds):
        for b in budget_options:
            for lbl in all_labels:
                pooled_per_class[b][lbl] += \
                    results['per_seed_per_class'][seed_i][b].get(lbl, [])
    per_seed_class_stats['mean'] = {
        str(lbl): {f'{b:.2f}': budget_stats(pooled_per_class[b][lbl])
                   for b in budget_options}
        for lbl in all_labels
    }
    out = {
        split: {key: per_seed_stats},
        f'per_class_{split}': {key: per_seed_class_stats},
        'config': config,
    }
    if 'clf_outcomes' in results:
        # evidence 参照指标落盘：冻结分类器对最终揭示子图的判定
        out['acc_clf_reference'] = {
            f'{b:.2f}': budget_stats(results['clf_outcomes'][b])
            for b in budget_options
        }
    return out

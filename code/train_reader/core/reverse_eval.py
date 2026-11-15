"""反向固定 BAGR 预测头评测（rebuttal checklist 2026-11-15 第 2 项）。

与「统一使用冻结分类器」（eval_baselines 的 clf 口径）构成正反双向控制
变量：这里固定 **BAGR-RL 的 evidence 头** 为唯一终判，把每步选点动作替换
为 Random / BFS / Max-Frontier-Degree / SAGPool / AgentNet 的选点——
BAGR 的 LSTM/Attention 证据状态仍按协议逐步经 forward_states_evidence
更新（选点评分被忽略，证据积分路径不受替换影响），读满预算后用证据头
终判。所有选点来源被同一个专为局部序列训练的预测头一致评判。

实现与 evaluate_reader 完全同构（跨图 lockstep + teleport + 终局补充
前向 + 确定性 state RNG），仅第 k 步的 chosen 来自 agent.select(state)：

- agent 返回 None（内部状态耗尽，如 agentnet 导航上限）→ 该 rollout
  提前结束（active=False，与 evaluate_fixed 的 rollout 契约一致），
  终局证据前向仍执行；
- agent 的 reset/on_new_frontier 接线与 rollout_strategy 相同，但 lockstep
  交错推进要求 per-state 独立实例：random/bfs/max-frontier-degree 逐
  state 新建（纯 Python，零成本）；sagpool 的 select 无跨 rollout 状态，
  全部 state 共享同一实例；agentnet 经 clone_walk 共享模型与 per-graph
  静态缓存、独立游走状态（见 baseline/agentnet.py）。

无需冻结分类器、无需终局子图构建（evidence 头即终判）——内存占用 =
reader + 各策略 scorer + 测试集图。
"""

import random
from collections import defaultdict

import torch
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck
from tools.seeds import deterministic_seed
from .eval_loop import _edge_key_to_id, _summarize


@torch.no_grad()
def evaluate_reverse(reader, agent_make, graphs, labels, budget_options,
                     device, chunk_size=32, n_seeds=4):
    """反向评测单策略：BAGR evidence 头终判 + agent 选点。

    reader: 已加载的 evidence reader（stochastic=False，eval 态；冻结，
    本函数不更新任何参数）。
    agent_make: () -> agent 的工厂；每个 (图, 预算, seed) rollout 调一次，
    返回带独立内部状态的策略实例（无状态策略可返回同一实例）。调用方
    负责 reset/on_new_frontier 初始接线（本函数内完成，与 rollout_strategy
    一致）。
    其余参数/返回值口径与 evaluate_reader 相同（acc_* / per_seed_* /
    outcomes；无 clf 参照——终判即证据头，主表 learnable 行即其对照）。
    """
    if getattr(reader.config, 'kind', '') != 'evidence':
        raise ValueError("evaluate_reverse 需要 evidence reader"
                         "（终判 = 自身证据头）")
    reader.eval()
    per_budget_correct = defaultdict(int)
    per_budget_total = defaultdict(int)
    per_budget_outcomes = defaultdict(list)
    per_seed_outcomes = {i: defaultdict(list) for i in range(n_seeds)}
    per_seed_per_class = {i: defaultdict(lambda: defaultdict(list))
                          for i in range(n_seeds)}

    for chunk_start in tqdm(range(0, len(graphs), chunk_size),
                            desc="  Reverse"):
        chunk_end = min(chunk_start + chunk_size, len(graphs))
        all_states, state_info, agents = [], [], []
        for local_i in range(chunk_start, chunk_end):
            g_raw = graphs[local_i]
            label = int(labels[local_i].item())
            y = torch.tensor([label], device=device)
            n_nodes = g_raw.num_nodes()
            ek2id = _edge_key_to_id(g_raw)
            for budget_ratio in budget_options:
                B = max(1, int(budget_ratio * n_nodes))
                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node,
                                       edge_key_to_id=ek2id)
                    state._budget_limit = B
                    agent = agent_make()
                    agent.reset()
                    initial_frontier = list(state._initial_frontier)
                    if initial_frontier:
                        agent.on_new_frontier(initial_frontier)
                    all_states.append(state)
                    state_info.append((y, label, budget_ratio, seed_i))
                    agents.append(agent)

        n_total = len(all_states)
        active = [True] * n_total
        all_hc = reader.init_evidence(n_total, device)
        last_revealed = [s.seed_node for s in all_states]
        state_rng = [random.Random(s.seed_node * 1_000_003 + s._budget_limit)
                     for s in all_states]

        for _step in range(max(s._budget_limit for s in all_states)):
            for j in range(n_total):
                if active[j] and all_states[j].is_frontier_empty:
                    st = all_states[j]
                    tp_node = teleport_if_stuck(st, rng=state_rng[j])
                    if tp_node is not None:
                        last_revealed[j] = tp_node
                        if st.get_budget_used() >= st._budget_limit:
                            active[j] = False
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty]
            if not active_idx:
                break
            # 证据状态照常更新；选点打分跳过（need_scores=False，省 ~40%
            # 前向——动作由 agent 给出，scores/probs 本就被丢弃）
            _, active_hc, _ = reader.forward_states_evidence(
                [all_states[j] for j in active_idx], temperature=1.0,
                hc_list=[all_hc[j] for j in active_idx],
                last_revealed=[last_revealed[j] for j in active_idx],
                need_scores=False)
            for k, j in enumerate(active_idx):
                all_hc[j] = active_hc[k]

            for j in active_idx:
                chosen = agents[j].select(all_states[j])
                if chosen is None:
                    active[j] = False
                    continue
                new_frontier = reveal_node(all_states[j], chosen)
                last_revealed[j] = chosen
                if new_frontier:
                    agents[j].on_new_frontier(new_frontier)
                if all_states[j].get_budget_used() \
                        >= all_states[j]._budget_limit:
                    active[j] = False

        # ---- 终局判定：读满后的补充前向（与 evaluate_reader 同构；打分跳过）----
        _, _, final_ev = reader.forward_states_evidence(
            all_states, temperature=1.0, hc_list=all_hc,
            last_revealed=last_revealed, need_scores=False)
        for j in range(n_total):
            y, label, budget, seed_i = state_info[j]
            correct = 1.0 if final_ev[j].argmax(dim=-1).item() == label else 0.0
            per_budget_correct[budget] += correct
            per_budget_total[budget] += 1
            per_budget_outcomes[budget].append(correct)
            per_seed_outcomes[seed_i][budget].append(correct)
            per_seed_per_class[seed_i][budget][label].append(correct)

    return _summarize(budget_options, graphs, per_budget_correct,
                      per_budget_total, per_budget_outcomes,
                      per_seed_outcomes, per_seed_per_class)

"""RL（REINFORCE）统一训练循环：base 与 attention（belief lockstep）一套实现。

收敛 m5/base/train_rl.py、m6/attention/train_rl.py、m7/train_rl.py 三份拷贝，
保留 m7 的最新改进：teleport_if_stuck、batch 优势归一化 + 负侧降权、
NaN 防护。attention 分支额外：belief 维护、micro-batch 梯度累积。
奖励判定来源可选（reward_source）：冻结分类器（默认）、reader 自身
证据头终判（--reward-source reader，仅 evidence kind），或 both（分类器
与证据头同时判对才计奖，仅 evidence kind）。
"""

import math
import random
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck
from tools.seeds import deterministic_seed
from .common import (compute_reward, normalize_advantages,
                     reveal_and_update_beliefs_batch)

MICRO_BATCH_SIZE = 16   # attention reader 梯度累积的 micro-batch（沿用 m6/m7）


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
    print('=' * 60)
    raise RuntimeError('NaN 诊断：probs 非有限（见上方定位）')


def train_one_epoch(reader, classifier, builder, optimizer,
                    graphs, labels, batch_size: int, n_seeds: int,
                    device, reward_type: str, ln_k: float, budget_options,
                    neg_scale: float, use_belief: bool = None,
                    evidence_lambda: float = 0.0,
                    adv_normalize: bool = True,
                    reward_source: str = 'classifier') -> Dict:
    """单轮 REINFORCE（self-sampled）。

    Loss = mean_i(-A_i · Σ_t log π_t(a_t))；A 为 batch 优势归一化
    （负侧乘 neg_scale）。R 的判定来源由 reward_source 控制：
    - classifier：冻结上界分类器对揭示子图的判定（默认）；
    - reader：reader 自身分类头终判（仅 evidence kind，强制开启证据前向）；
    - both：分类器（primary）与证据头终判（gate）同时判对才计奖
      （仅 evidence kind；tanh 变换在判错时钳到非正）。
    evidence_lambda: >0 且 reader 含证据流时加 λ·L_ev（逐步图类别监督 CE，
    不进策略梯度），总损失 = L_RL + λ·L_ev。
    """
    if reward_source in ('reader', 'both') \
            and not hasattr(reader, 'forward_states_evidence'):
        raise ValueError(
            "reward_source='reader'/'both' 需要 evidence reader"
            "（base/attention 无自身分类头，无法参与联合/自身判定）")
    if use_belief is None:
        use_belief = hasattr(reader, 'init_beliefs')
    use_evidence = (evidence_lambda > 0 or reward_source in ('reader', 'both')) \
        and hasattr(reader, 'forward_states_evidence')
    reader.train()
    if classifier is not None:
        classifier.eval()   # reward_source='reader' 时不加载分类器（None）
    total_R = total_trajs = total_correct = total_loss = 0.0
    ev_correct = ev_total = 0
    ev_final_correct = ev_final_total = 0
    order = list(range(len(graphs)))
    random.shuffle(order)
    micro_batch_size = MICRO_BATCH_SIZE if use_belief else len(order)  # base 无累积
    pbar = tqdm(range(0, len(order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_indices = order[batch_start:batch_start + batch_size]
        if use_belief:
            optimizer.zero_grad()
            # attention：梯度累积的 micro-batch；base：整 batch 一次
            micro = MICRO_BATCH_SIZE
        else:
            micro = len(batch_indices)
        n_micro = (len(batch_indices) + micro - 1) // micro
        for mb_i in range(0, len(batch_indices), micro):
            mb_indices = batch_indices[mb_i:mb_i + micro]
            all_states, state_labels, per_state_log_probs = [], [], []
            state_src = []  # state → (图 index, seed_i)（NaN 诊断用）
            last_revealed = []  # evidence 流：各 state 上一步所揭示节点
            all_hc = []
            for local_i in mb_indices:
                g_raw = graphs[local_i]
                label = int(labels[local_i].item())
                y = torch.tensor([label], device=device)
                n_nodes = g_raw.num_nodes()
                budget_ratio = random.choice(list(budget_options))
                B = max(1, int(budget_ratio * n_nodes))
                for seed_i in range(n_seeds):
                    seed_node = deterministic_seed(local_i, seed_i, n_nodes)
                    state = GraphState(g_raw, seed_node=seed_node)
                    state._budget_limit = B
                    all_states.append(state)
                    state_labels.append(y)
                    per_state_log_probs.append([])
                    state_src.append((local_i, seed_i))
                    last_revealed.append(seed_node)
            if use_evidence:
                all_hc = reader.init_evidence(len(all_states), device)
            ev_losses = []

            n_total = len(all_states)
            active = [True] * n_total
            max_budget = max(s._budget_limit for s in all_states)
            if use_belief:
                all_beliefs = [reader.init_beliefs(1).to(device)
                               for _ in range(n_total)]
                all_subgraphs = [None] * n_total

            for _step in range(max_budget):
                # 预算未耗尽但 frontier 空（seed 落孤立小分量）：
                # 跳转到未揭示节点继续（预算比例 < 1，未揭示节点必然存在）
                for j in range(n_total):
                    if active[j] and all_states[j].is_frontier_empty:
                        st = all_states[j]
                        tp_node = teleport_if_stuck(st)
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
                    batch_results, active_hc, active_ev = reader.forward_states_evidence(
                        [all_states[j] for j in active_idx],
                        temperature=reader.temperature,
                        hc_list=[all_hc[j] for j in active_idx],
                        last_revealed=[last_revealed[j] for j in active_idx])
                    for k, j in enumerate(active_idx):
                        all_hc[j] = active_hc[k]
                        ev_losses.append(F.cross_entropy(
                            active_ev[k].unsqueeze(0), state_labels[j]))
                        ev_correct += int(active_ev[k].argmax().item()
                                          == state_labels[j].item())
                        ev_total += 1
                elif use_belief:
                    active_beliefs = torch.cat(
                        [all_beliefs[j] for j in active_idx], dim=0)
                    active_subgraphs = None
                    if all_subgraphs[active_idx[0]] is not None:
                        active_subgraphs = [all_subgraphs[j] for j in active_idx]
                    batch_results = reader.forward_states(
                        [all_states[j] for j in active_idx],
                        temperature=reader.temperature,
                        beliefs=active_beliefs,
                        subgraphs=active_subgraphs)
                else:
                    batch_results = reader.forward_states(
                        [all_states[j] for j in active_idx],
                        temperature=reader.temperature)

                batch_states, batch_nodes, batch_local_idx = [], [], []
                for k, j in enumerate(active_idx):
                    _, probs, frontier_nodes = batch_results[k]
                    if not frontier_nodes:
                        active[j] = False
                        continue
                    if use_belief and not torch.isfinite(probs).all():
                        _diag_nan(reader, all_states[j], all_beliefs[j],
                                  batch_results[k], batch_start, _step, j)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample()
                    per_state_log_probs[j].append(dist.log_prob(action_idx))
                    if use_belief:
                        # reveal 由批量 belief 更新内部完成
                        batch_states.append(all_states[j])
                        batch_nodes.append(frontier_nodes[action_idx.item()])
                        batch_local_idx.append(j)
                    else:
                        chosen = frontier_nodes[action_idx.item()]
                        reveal_node(all_states[j], chosen)
                        if use_evidence:
                            last_revealed[j] = chosen
                        if all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                            active[j] = False
                if use_belief:
                    if batch_states:
                        batch_beliefs = torch.cat(
                            [all_beliefs[j] for j in batch_local_idx], dim=0)
                        new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                            reader, batch_states, batch_nodes, batch_beliefs,
                            device,
                            state_src=[state_src[j] for j in batch_local_idx])
                        if not torch.isfinite(new_beliefs).all():
                            raise RuntimeError('RL: belief 更新后 NaN')
                        for k, j in enumerate(batch_local_idx):
                            all_beliefs[j] = new_beliefs[k:k + 1]
                            all_subgraphs[j] = new_subgraphs[k]
                    for j in active_idx:
                        if active[j] and all_states[j].get_budget_used() >= all_states[j]._budget_limit:
                            active[j] = False

            # ---- 批量分类器评估 → 奖励 → REINFORCE 损失 ----
            final_states, final_log_probs, final_labels, final_idx = \
                [], [], [], []
            for j in range(n_total):
                if per_state_log_probs[j]:
                    final_idx.append(j)
                    final_states.append(all_states[j])
                    final_log_probs.append(per_state_log_probs[j])
                    final_labels.append(state_labels[j])
            if not final_states:
                continue

            # ---- 终局判定：读满后的补充前向（含最后一次 reveal 的证据）----
            reader_term_logits = None
            if use_evidence:
                _, _, term_ev = reader.forward_states_evidence(
                    final_states, reader.temperature,
                    hc_list=[all_hc[j] for j in final_idx],
                    last_revealed=[last_revealed[j] for j in final_idx])
                reader_term_logits = [t.detach() for t in term_ev]
                for k in range(len(final_states)):
                    ev_losses.append(F.cross_entropy(
                        term_ev[k].unsqueeze(0), final_labels[k]))
                    ev_final_correct += int(term_ev[k].argmax().item()
                                            == final_labels[k].item())
                    ev_final_total += 1

            # ---- 奖励判定来源 ----
            clf_logits = None
            if reward_source in ('classifier', 'both'):
                logits_chunks = []
                with torch.no_grad():
                    for c_start in range(0, len(final_states), 512):
                        chunk = final_states[c_start:c_start + 512]
                        sub_graphs = [builder.build(s)[0] for s in chunk]
                        batch_g = dgl.batch(sub_graphs).to(device)
                        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                                           device=device)
                        logits_chunks.append(classifier(batch_g, nids))
                clf_logits = torch.cat(logits_chunks, dim=0)

            if reward_source == 'reader':
                # reader 判定：奖励来自证据头终判（detached，不进策略梯度之外）
                primary = torch.stack(reader_term_logits, dim=0)
            else:
                primary = clf_logits
            gate = reader_term_logits if reward_source == 'both' else None

            raw_rewards = []
            for k in range(len(final_states)):
                R = compute_reward(
                    primary[k:k + 1], final_labels[k], reward_type, ln_k,
                    gate_logits=(gate[k] if gate is not None else None))
                correct = 1.0 if primary[k].argmax(dim=-1).item() \
                    == final_labels[k].item() else 0.0
                if gate is not None \
                        and gate[k].argmax(dim=-1).item() != final_labels[k].item():
                    correct = 0.0
                raw_rewards.append(R)
                total_R += R
                total_trajs += 1
                total_correct += correct

            advs = normalize_advantages(raw_rewards, neg_scale) \
                if adv_normalize else list(raw_rewards)

            batch_losses = []
            for k in range(len(final_states)):
                sum_log_prob = torch.stack(final_log_probs[k]).sum()
                batch_losses.append(-advs[k] * sum_log_prob)
            if use_belief:
                # 梯度累积：loss/n_micro backward，batch 末统一 step
                rl_loss = torch.stack(batch_losses).mean()
                loss = rl_loss / n_micro
                loss.backward()
                total_loss += loss.item() * len(batch_losses) * n_micro
            else:
                batch_loss = torch.stack(batch_losses).mean()
                if ev_losses:
                    batch_loss = batch_loss + evidence_lambda * torch.stack(ev_losses).mean()
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += batch_loss.item() * len(batch_losses)
        if use_belief:
            optimizer.step()
        n = max(total_trajs, 1)
        postfix = {
            'loss': f'{total_loss / n:.4f}',
            'avg_R': f'{total_R / n:.3f}',
            'acc': f'{total_correct / n:.3f}',
        }
        if ev_total:
            postfix['ev_acc'] = f'{ev_correct / ev_total:.3f}'
        if ev_final_total:
            postfix['ev_final'] = f'{ev_final_correct / ev_final_total:.3f}'
        pbar.set_postfix(postfix)
    n = max(total_trajs, 1)
    out = {'loss': total_loss / n, 'avg_R': total_R / n, 'acc': total_correct / n}
    if ev_total:
        out['ev_acc'] = ev_correct / ev_total
    if ev_final_total:
        out['ev_acc_final'] = ev_final_correct / ev_final_total
    return out

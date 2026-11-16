"""RL（REINFORCE）统一训练循环：base 与 evidence 一套实现。

收敛 m5/base/train_rl.py 与 m7/train_rl.py 两份拷贝，保留 m7 的最新改进：
teleport_if_stuck、batch 优势归一化 + 负侧降权。
（attention 的 belief 维护与 micro-batch 梯度累积随 attention reader 废弃
移除，2026-09-20。）
奖励判定来源可选（reward_source）：冻结分类器（默认）、reader 自身
证据头终判（--reward-source reader，仅 evidence kind）、both（分类器
与证据头同时判对才计奖，仅 evidence kind），或 soft-both（软 both：
R = 两模型对正确类置信度的乘积，∈[0,1]，忽略 --reward 变换）。
"""

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck
from tools.seeds import deterministic_seed
from .common import (compute_reward, compute_soft_both_reward,  # noqa: F401
                     normalize_advantages)


def train_one_epoch(reader, classifier, builder, optimizer,
                    graphs, labels, batch_size: int, n_seeds: int,
                    device, reward_type: str, ln_k: float, budget_options,
                    neg_scale: float,
                    evidence_lambda: float = 0.0,
                    adv_normalize: bool = True,
                    reward_source: str = 'classifier',
                    rl_lambda: float = 1.0) -> Dict:
    """单轮 REINFORCE（self-sampled）。

    Loss = mean_i(-A_i · Σ_t log π_t(a_t))；A 为 batch 优势归一化
    （负侧乘 neg_scale）。R 的判定来源由 reward_source 控制：
    - classifier：冻结上界分类器对揭示子图的判定（默认）；
    - reader：reader 自身分类头终判（仅 evidence kind，强制开启证据前向）；
    - both：分类器（primary）与证据头终判（gate）同时判对才计奖
      （仅 evidence kind；tanh 变换在判错时钳到非正）；
    - soft-both：软 both，R = p_clf(y_true)·p_reader(y_true)
      （仅 evidence kind；忽略 reward_type 变换）。
    evidence_lambda: >0 且 reader 含证据流时加 λ·L_ev（逐步图类别监督 CE，
    不进策略梯度），总损失 = L_RL + λ·L_ev。
    rl_lambda: 策略梯度项权重，总损失 = rl_lambda·L_RL + λ·L_ev。
    rl_lambda=0 即 CE-only 消融（两阶段 RL 受控对照）：自采样轨迹上只用
    交叉熵更新预测头、不给选点头传策略梯度——奖励照常计算（avg_R/acc
    指标可用），但 log π 项权重为零。要求 evidence 流（否则无损失可回传）。
    """
    if rl_lambda == 0 and not (evidence_lambda > 0
                               and hasattr(reader, 'forward_states_evidence')):
        raise ValueError("rl_lambda=0（CE-only）需要 evidence reader 且 "
                         "evidence_lambda > 0（否则无损失可回传）")
    if reward_source in ('reader', 'both', 'soft-both') \
            and not hasattr(reader, 'forward_states_evidence'):
        raise ValueError(
            "reward_source='reader'/'both'/'soft-both' 需要 evidence reader"
            "（base 无自身分类头，无法参与联合/自身判定）")
    use_evidence = (evidence_lambda > 0
                    or reward_source in ('reader', 'both', 'soft-both')) \
        and hasattr(reader, 'forward_states_evidence')
    reader.train()
    if classifier is not None:
        classifier.eval()   # reward_source='reader' 时不加载分类器（None）
    total_R = total_trajs = total_correct = total_loss = 0.0
    ev_correct = ev_total = 0
    ev_final_correct = ev_final_total = 0
    order = list(range(len(graphs)))
    random.shuffle(order)
    pbar = tqdm(range(0, len(order), batch_size), desc="  Training")
    for batch_start in pbar:
        batch_indices = order[batch_start:batch_start + batch_size]
        all_states, state_labels, per_state_log_probs = [], [], []
        last_revealed = []  # evidence 流：各 state 上一步所揭示节点
        for local_i in batch_indices:
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
                last_revealed.append(seed_node)
        if use_evidence:
            all_hc = reader.init_evidence(len(all_states), device)
        ev_losses = []

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(s._budget_limit for s in all_states)

        for _step in range(max_budget):
            # 预算未耗尽但 frontier 空（seed 落孤立小分量）：
            # 跳转到未揭示节点继续（预算比例 < 1，未揭示节点必然存在）
            for j in range(n_total):
                if active[j] and all_states[j].is_frontier_empty:
                    st = all_states[j]
                    tp_node = teleport_if_stuck(st)
                    if tp_node is not None:
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
            else:
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
                chosen = frontier_nodes[action_idx.item()]
                reveal_node(all_states[j], chosen)
                if use_evidence:
                    last_revealed[j] = chosen
                if all_states[j].get_budget_used() >= all_states[j]._budget_limit:
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
        if reward_source != 'reader':
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
        gate = reader_term_logits \
            if reward_source in ('both', 'soft-both') else None

        raw_rewards = []
        for k in range(len(final_states)):
            correct = 1.0 if primary[k].argmax(dim=-1).item() \
                == final_labels[k].item() else 0.0
            if gate is not None \
                    and gate[k].argmax(dim=-1).item() != final_labels[k].item():
                correct = 0.0
            if reward_source == 'soft-both':
                R = compute_soft_both_reward(primary[k], gate[k],
                                             final_labels[k])
            else:
                R = compute_reward(
                    primary[k:k + 1], final_labels[k], reward_type, ln_k,
                    gate_logits=(gate[k] if gate is not None else None))
            raw_rewards.append(R)
            total_R += R
            total_trajs += 1
            total_correct += correct

        advs = normalize_advantages(raw_rewards, neg_scale) \
            if adv_normalize else list(raw_rewards)

        # rl_lambda=0（CE-only）时跳过策略梯度项：log π 仍随采样记录
        # （final_states 的入围判定依赖它），但权重为零不产生梯度
        batch_loss = None
        if rl_lambda != 0:
            batch_losses = []
            for k in range(len(final_states)):
                sum_log_prob = torch.stack(final_log_probs[k]).sum()
                batch_losses.append(-advs[k] * sum_log_prob)
            batch_loss = rl_lambda * torch.stack(batch_losses).mean()
            n_weighted = len(batch_losses)
        else:
            # CE-only：batch_loss 只含 ev 项，按其轨迹数加权
            n_weighted = len(ev_losses) or len(final_states)
        if ev_losses:
            ev_term = evidence_lambda * torch.stack(ev_losses).mean()
            batch_loss = ev_term if batch_loss is None else batch_loss + ev_term
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += batch_loss.item() * n_weighted

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

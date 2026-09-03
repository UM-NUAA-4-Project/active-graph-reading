"""BC（行为克隆）统一训练循环：base 与 attention（belief lockstep）一套实现。

收敛 m5/training_utils.train_one_epoch（base）与
m6/attention/train_bc.py + m7/train_bc.py（attention）三份拷贝。
两版本 lockstep 骨架逐行同构，仅在 belief 维护与 reveal 方式上分支。
"""

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from .common import reveal_and_update_beliefs_batch


def train_one_epoch(reader, optimizer, graph_cache: Dict, train_trajs: List,
                    batch_size: int, device, temperature: float = 1.0,
                    use_belief: bool = None,
                    evidence_lambda: float = 0.0) -> Dict:
    """单轮 BC 训练（跨图锁步批处理）。

    graph_cache: {graph_idx: (graph, label)}；
    use_belief: None=按 reader 类型自动（有 init_beliefs 即 attention）；
    evidence_lambda: >0 且 reader 含证据流时启用 L_ev（逐步图类别 CE），
    总损失 = L_act + λ·L_ev。
    Returns: {'loss': 平均 step loss, 'bc': 同 loss（兼容旧日志字段）,
              'ev_acc': 证据头逐步预测准确率（无证据流时省略）}
    """
    if use_belief is None:
        use_belief = hasattr(reader, 'init_beliefs')
    use_evidence = evidence_lambda > 0 and hasattr(reader, 'forward_states_evidence')
    reader.train()
    total_loss = 0.0
    total_steps = 0
    ev_correct = ev_total = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_trajs = train_trajs[batch_start:batch_start + batch_size]

        # ---- Pass 1: create states (attention 另建 beliefs) ----
        all_states: List[GraphState] = []
        state_labels = []
        reveal_orders = []
        step_counters = []
        for traj in batch_trajs:
            g, label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            all_states.append(state)
            state_labels.append(torch.tensor([int(label)], device=device))
            reveal_orders.append(traj.reveal_order)
            step_counters.append(0)

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(len(ro) for ro in reveal_orders)
        if use_belief:
            all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
            all_subgraphs = [None] * n_total
        if use_evidence:
            all_hc = reader.init_evidence(n_total, device)
            # Δ_t = 上一步所揭示节点的嵌入；step 0 为 seed
            last_revealed = [traj.seed_node for traj in batch_trajs]
        step_losses = []
        ev_losses = []

        # ---- Pass 2: cross-graph lockstep ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty
                          and step_counters[j] < len(reveal_orders[j])]
            if not active_idx:
                break

            if use_evidence:
                batch_results, active_hc, active_ev = reader.forward_states_evidence(
                    [all_states[j] for j in active_idx], temperature,
                    [all_hc[j] for j in active_idx],
                    [last_revealed[j] for j in active_idx])
                for k, j in enumerate(active_idx):
                    all_hc[j] = active_hc[k]
                    ev_losses.append(F.cross_entropy(
                        active_ev[k].unsqueeze(0), state_labels[j]))
                    ev_correct += int(active_ev[k].argmax().item()
                                      == state_labels[j].item())
                    ev_total += 1
            elif use_belief:
                active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
                active_subgraphs = None
                if all_subgraphs[active_idx[0]] is not None:
                    active_subgraphs = [all_subgraphs[j] for j in active_idx]
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx],
                    temperature=temperature,
                    beliefs=active_beliefs,
                    subgraphs=active_subgraphs,
                )
            else:
                batch_results = reader.forward_states(
                    [all_states[j] for j in active_idx], temperature=temperature)

            batch_states, batch_nodes, batch_local_idx = [], [], []
            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue

                teacher_node = reveal_orders[j][step_counters[j]]
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    active[j] = False
                    continue

                step_losses.append(-torch.log(probs[target_idx] + 1e-8))
                step_counters[j] += 1
                if use_belief:
                    # reveal 延迟到批量 belief 更新（内部会 reveal）
                    batch_states.append(all_states[j])
                    batch_nodes.append(teacher_node)
                    batch_local_idx.append(j)
                else:
                    reveal_node(all_states[j], teacher_node)
                    if use_evidence:
                        last_revealed[j] = teacher_node
                if step_counters[j] >= len(reveal_orders[j]):
                    active[j] = False

            if use_belief and batch_states:
                batch_beliefs = torch.cat(
                    [all_beliefs[j] for j in batch_local_idx], dim=0)
                new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                    reader, batch_states, batch_nodes, batch_beliefs, device)
                for k, j in enumerate(batch_local_idx):
                    all_beliefs[j] = new_beliefs[k:k + 1]
                    all_subgraphs[j] = new_subgraphs[k]

        if not step_losses:
            continue

        # ---- Pass 3: L_act + λ·L_ev ----
        batch_loss = torch.stack(step_losses).mean()
        if ev_losses:
            batch_loss = batch_loss + evidence_lambda * torch.stack(ev_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = len(step_losses)
        total_loss += batch_loss.item() * n
        total_steps += n

        avg_step_loss = total_loss / max(total_steps, 1)
        postfix = {
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_step_loss': f'{avg_step_loss:.4f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
        }
        if ev_total:
            postfix['ev_acc'] = f'{ev_correct / ev_total:.3f}'
        pbar.set_postfix(postfix)

    n = max(total_steps, 1)
    out = {'loss': total_loss / n, 'bc': total_loss / n}
    if ev_total:
        out['ev_acc'] = ev_correct / ev_total
    return out

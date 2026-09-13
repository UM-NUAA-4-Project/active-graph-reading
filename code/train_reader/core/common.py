"""reader 训练核心共享件（belief 更新、奖励、优势归一化、冻结）。"""

import math
import random
from typing import List

import torch
import torch.nn.functional as F

from active_graph_reader.engine.reveal_action import reveal_node, teleport_if_stuck


# ---------------------------------------------------------------------------
# belief lockstep 原语（自 m6/training_utils 收编）
# ---------------------------------------------------------------------------

def reveal_and_update_beliefs_batch(reader, states: list, chosen_nodes: list,
                                    beliefs: torch.Tensor, device,
                                    state_src: list = None):
    """Batch：reveal 全部 → 构建子图 → 批量编码 → 批量 belief 更新。

    states: List[GraphState]；chosen_nodes: List[int]；beliefs: (N, m, d)。
    Returns: (new_beliefs (N, m, d), new_subgraphs List[FrontierSubgraph])
    """
    for state, node in zip(states, chosen_nodes):
        reveal_node(state, node)

    new_subgraphs = [reader.frontier_builder.build(s) for s in states]

    new_infos = reader.encode_new_info_batch(
        states, new_subgraphs, chosen_nodes, device, state_src=state_src,
    )  # (N, 1, d)

    new_beliefs = reader.belief_state.update_fn(beliefs, new_infos)
    return new_beliefs, new_subgraphs


# ---------------------------------------------------------------------------
# RL 奖励与优势（自 m5/m6/m7 三份实现收敛）
# ---------------------------------------------------------------------------

def compute_reward(logits_i: torch.Tensor, label: torch.Tensor,
                   reward_type: str, ln_k: float,
                   gate_logits: torch.Tensor = None) -> float:
    """单轨迹奖励：binary / tanh / asymmetric。

    gate_logits 不为 None（reward_source='both'）：主判 logits 与 gate
    logits 的 argmax 同时正确才记判对；tanh 判错时钳到非正
    （"同时判对才给奖励"对连续变换同样成立）。gate_logits=None 时
    行为与旧版逐位一致。
    """
    correct = logits_i.argmax(dim=-1).item() == label.item()
    if gate_logits is not None:
        correct = correct and (gate_logits.argmax(dim=-1).item() == label.item())
    if reward_type == 'tanh':
        ce = F.cross_entropy(logits_i, label).item()
        r = math.tanh(1.0 - ce / ln_k)
        if gate_logits is not None and not correct:
            r = min(r, 0.0)
        return r
    if reward_type == 'asymmetric':
        return 1.0 if correct else -0.1
    return float(correct)  # binary


def compute_soft_both_reward(clf_logits: torch.Tensor, reader_logits: torch.Tensor,
                             label: torch.Tensor) -> float:
    """软 both 奖励（reward_source='soft-both'）：概率乘积代替硬联合判定。

    R = p_clf(y_true) · p_reader(y_true) ∈ [0, 1]——两模型对正确类的
    softmax 置信度同高才接近 1，任一方给正确类的概率低则奖励按比例衰减
    （与硬 both 的区别：部分正确可得部分奖励）。--reward 变换对此源不适用。
    """
    y = label.item()
    p_clf = torch.softmax(clf_logits.float().flatten(), dim=-1)[y]
    p_reader = torch.softmax(reader_logits.float().flatten(), dim=-1)[y]
    return float(p_clf * p_reader)


def normalize_advantages(rewards, neg_scale: float) -> List[float]:
    """Batch 级优势归一化 + 负侧降权。

    A = (R - mean)/(std + eps)；A < 0 时 A *= neg_scale。
    中心化自适应修正 tanh 锚点错误（均匀猜测作零点导致奖励几乎全负），
    neg_scale 显式抑制负样本噪声。
    """
    if len(rewards) < 2:
        return list(rewards)
    rs = torch.tensor(rewards, dtype=torch.float)
    mean = rs.mean()
    std = rs.std()
    A = (rs - mean) / (std + 1e-6)
    A = torch.where(A >= 0, A, A * neg_scale)
    return A.tolist()


# ---------------------------------------------------------------------------
# 其它共享
# ---------------------------------------------------------------------------

def freeze_shared_layers(reader, enabled: bool) -> None:
    """--freeze：冻结除特征投影层（edge_encoder / node_feat_proj）外全部参数。"""
    if not enabled:
        return
    frozen = 0
    for name, p in reader.named_parameters():
        if 'edge_encoder' not in name and 'node_feat_proj' not in name:
            p.requires_grad = False
            frozen += 1
    print(f"  --freeze: {frozen} 层已冻结（仅特征投影层可训练）")


def teleport_or_deactivate(state, active: list, j: int, rng: random.Random = None,
                           reader=None, subgraphs: list = None) -> None:
    """预算未耗尽但 frontier 空（seed 落孤立小分量）时的跳转处理。

    跳转后若预算耗尽则置 inactive；use_belief 时同步重建子图缓存。
    """
    st = state
    if teleport_if_stuck(st, rng=rng) is None:
        return
    if reader is not None and subgraphs is not None:
        subgraphs[j] = reader.frontier_builder.build(st)
    if st.get_budget_used() >= st._budget_limit:
        active[j] = False

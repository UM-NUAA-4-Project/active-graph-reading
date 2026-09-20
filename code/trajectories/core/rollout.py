"""rollout：模型构建（从 checkpoint config）+ BC reader 批量 rollout + 候选验证。

收编自 m7/augment_trajectories.py 与 balance_trajectories.py 的共享函数。
模型构建统一走 classifier.factory / active_graph_reader 的 config 推断，
消灭硬编码维度。
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import dgl

from active_graph_reader import build_reader as _build_reader
from active_graph_reader.config import reader_config_from_checkpoint
from active_graph_reader.engine.reveal_action import reveal_node
from classifier import GraphClassifier, classifier_config_from_ckpt
from tools.ckpt_io import pad_reader_mlp_input, strip_classifier_keys
from .trajectory import Trajectory


def build_classifier(ckpt: dict, device, requires_grad: bool = False) -> GraphClassifier:
    """按 checkpoint 权重/内嵌 config 重建上界分类器并加载。

    requires_grad=False（默认）：冻结权重（eval + 不回传）。
    requires_grad=True：保持 autograd 图（GradientTeacher 反向传播需要）；
    "冻结"语义 = 只加载权重、无优化器更新。
    """
    clf = GraphClassifier(config=classifier_config_from_ckpt(ckpt)).to(device)
    clf.load_state_dict(ckpt['model_state_dict'], strict=True)
    clf.eval()
    if not requires_grad:
        for p in clf.parameters():
            p.requires_grad = False
    return clf


def build_reader(ckpt: dict, device, stochastic: bool = False,
                 temperature: float = None):
    """按 checkpoint 权重/内嵌 config 重建 reader（base/evidence 均可）。

    兼容旧 checkpoint：strip belief classifier 键 + v0/v1 mlp 输入补零。
    """
    sd = ckpt.get('reader_state_dict', ckpt.get('state_dict'))
    sd = strip_classifier_keys(dict(sd))
    cfg = reader_config_from_checkpoint(ckpt, sd)
    if temperature is not None:
        cfg.temperature = float(temperature)
    if stochastic:
        cfg.stochastic = True
    reader = _build_reader(cfg).to(device)
    target_dim = reader.reader_mlp.mlp[0].in_features if cfg.kind == 'base' else None
    if target_dim is not None:
        sd, _ = pad_reader_mlp_input(sd, target_dim)
    reader.load_state_dict(sd, strict=True)
    reader.eval()
    return reader


def build_edge_key_map(g):
    """每图一次的 canonical edge key→edge id 映射（GraphState M4 优化接口）。

    GraphState 不传该 dict 时会在每个 state 内部重建，一轮数万个 state
    会把内存打满；因此每图构建一次、同图全部 state 共享。
    """
    if 'feat' not in g.edata:
        return None
    src, dst = g.edges()
    m = {}
    for eid in range(g.num_edges()):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in m:
            m[key] = eid
    return m


def rollout_batch(reader, states: list, rng: random.Random) -> list:
    """跨图 lockstep 批量 rollout（argmax 贪婪阅读）。

    states: 已设置 seed 与 _budget_limit 的 GraphState 列表。
    返回与 states 等长的 list：(reveal_order, final_state)，无动作的为 ([], state)。
    """
    n_total = len(states)
    orders = [[] for _ in range(n_total)]
    active = [True] * n_total
    max_budget = max(s._budget_limit for s in states)
    for _step in range(max_budget):
        active_idx = [j for j in range(n_total)
                      if active[j] and not states[j].is_frontier_empty]
        if not active_idx:
            break
        batch = reader.forward_states([states[j] for j in active_idx],
                                      temperature=reader.temperature)
        for k, j in enumerate(active_idx):
            _scores, probs, frontier_nodes = batch[k]
            if not frontier_nodes:
                active[j] = False
                continue
            node = frontier_nodes[int(probs.argmax().item())]
            orders[j].append(node)
            reveal_node(states[j], node)
            if states[j].get_budget_used() >= states[j]._budget_limit:
                active[j] = False
    return [(o, s) for o, s in zip(orders, states)]


def validate_candidates(cands, builder, classifier, device, train_labels,
                        batch_size):
    """用上界分类器批量验证候选（argmax == label，同时算 ce）。

    cands: [(graph_idx, ratio, traj, fin_state)]；返回验证通过的
    [(graph_idx, ratio, traj)]，验证后 state 随函数返回立即释放。
    """
    accepted = []
    for c_start in range(0, len(cands), batch_size):
        chunk = cands[c_start:c_start + batch_size]
        sub_graphs = []
        for _gi, _ratio, _traj, state in chunk:
            sub_g, _ = builder.build(state)
            sub_graphs.append(sub_g)
        batch_g = dgl.batch(sub_graphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = classifier(batch_g, nids)
        for k, (gi, ratio, traj, state) in enumerate(chunk):
            logits_i = logits[k:k + 1]
            pred = int(logits_i.argmax(dim=-1).item())
            ce = float(torch.nn.functional.cross_entropy(
                logits_i, torch.tensor([int(train_labels[gi])],
                                       device=device)).item())
            traj.ce = ce
            if pred == int(train_labels[gi]):
                traj.final_pred = pred
                accepted.append((gi, ratio, traj))
    return accepted


def count_labels(trajs: list, train_labels) -> Dict[int, int]:
    """按 label 统计轨迹数（graph_idx → 本地位置 → train_labels）。"""
    cnt = defaultdict(int)
    for t in trajs:
        cnt[int(train_labels[t.graph_idx])] += 1
    return dict(cnt)

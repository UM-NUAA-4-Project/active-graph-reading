"""M5 训练共享工具：轨迹加载、图加载、BC 训练循环、数据划分。

train_bc.py / train_pomdp_bc.py 等脚本共用，避免代码重复。

内存优化：不依赖 DglGraphPropPredDataset（避免加载全量 158k 图）。
使用 raw CSV 读标签 + dgl.load_graphs(idx_list=...) 按需加载。
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node

TEACHER_NAMES = [
    'gradient', 'structural-degree', 'structural-pagerank',
    'hybrid-a0.30', 'hybrid-a0.70', 'v0-reader',
]
BUDGET_RATIOS = ['0_050', '0_100', '0_150', '0_200']


# ---------------------------------------------------------------------------
# 轨迹加载
# ---------------------------------------------------------------------------

def merge_trajectories(data_dir: Path) -> List:
    """加载 stage4_balanced 中所有 .pt 文件，展平为 List[Trajectory]。"""
    from m4.teachers.base import Trajectory

    all_trajs: List[Trajectory] = []
    for ratio_key in BUDGET_RATIOS:
        fname = f'trajectories_balanced_r{ratio_key}.pt'
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        trajs_dict = data['trajectories']
        n_before = len(all_trajs)
        for teacher_name in TEACHER_NAMES:
            if teacher_name not in trajs_dict:
                continue
            for per_graph_list in trajs_dict[teacher_name]:
                all_trajs.extend(per_graph_list)
        print(f"  {fname}: +{len(all_trajs) - n_before} trajectories")
    print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs


# ---------------------------------------------------------------------------
# 图加载
# ---------------------------------------------------------------------------

def load_graphs_for_trajectories(root: str, needed_graphs: set):
    """只加载轨迹引用的训练集图，加自环后缓存。

    使用 dgl.load_graphs(idx_list=...) 仅加载需要的图，不加载全量 158k 数据集。
    """
    data_dir = Path(root) / 'ogbg_ppa'

    # 读训练集 split（不加载图数据）
    train_idx = pd.read_csv(
        data_dir / 'split' / 'species' / 'train.csv.gz',
        compression='gzip', header=None,
    ).values.flatten()
    train_set = set(train_idx.tolist())

    to_load = sorted(needed_graphs & train_set)
    skipped = needed_graphs - train_set
    if skipped:
        print(f"  WARNING: {len(skipped)} graph_idx not in training set, ignored")

    if not to_load:
        return {}

    # 仅加载需要的图
    proc_path = str(data_dir / 'processed' / 'dgl_data_processed')
    print(f"Loading {len(to_load)} graphs (training set subset)...")
    graphs, label_dict = dgl.load_graphs(proc_path, idx_list=to_load)

    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]] = {}
    for i, ogb_idx in enumerate(tqdm(to_load, desc="  Loading graphs")):
        g = dgl.add_self_loop(graphs[i])
        graph_cache[ogb_idx] = (g, int(label_dict['labels'][ogb_idx].item()))
    return graph_cache


# ---------------------------------------------------------------------------
# BC 训练循环
# ---------------------------------------------------------------------------

def train_one_epoch(
    reader: LearnableReader,
    optimizer: torch.optim.Optimizer,
    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]],
    train_trajs: List,
    batch_size: int,
    device: torch.device,
) -> float:
    """单轮 BC 训练，返回平均 step loss。"""
    reader.train()
    total_loss = 0.0
    total_steps = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]
        batch_losses = []
        batch_steps = 0

        for traj in batch_trajs:
            g, _label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            step_losses = []

            for teacher_node in traj.reveal_order:
                scores, probs, frontier_nodes = reader.forward(state)
                n_frontier = len(frontier_nodes)
                if n_frontier == 0:
                    break
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    break
                loss_t = -torch.log(probs[target_idx] + 1e-8)
                step_losses.append(loss_t)
                reveal_node(state, teacher_node)

            if step_losses:
                traj_loss = torch.stack(step_losses).mean()
                batch_losses.append(traj_loss)
                batch_steps += len(step_losses)

        if not batch_losses:
            continue

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item() * len(batch_losses)
        total_steps += batch_steps

        avg_step_loss = total_loss / max(total_steps, 1)
        pbar.set_postfix({
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_step_loss': f'{avg_step_loss:.4f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
        })

    return total_loss / max(total_steps, 1)


# ---------------------------------------------------------------------------
# checkpoint 工具
# ---------------------------------------------------------------------------

def infer_reader_config(state_dict: dict) -> Tuple[int, int]:
    """从 state_dict 推断 hidden_dim 和 n_layers。"""
    hidden_dim = state_dict['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in state_dict:
        if k.startswith('frontier_encoder.convs.'):
            layer_idx = int(k.split('.')[2])
            n_layers = max(n_layers, layer_idx + 1)
    return hidden_dim, n_layers


def edl_loss(evidence, alpha, S, y_onehot, k_evidence, lambda_kl=0.01,
             return_components=False):
    """EDL loss — Bayes risk (期望 MSE) + KL 正则（Sensoy et al., NeurIPS 2018）。

    L = Σ_k (y_k - p_k)²             ← 预测均值与标签的平方误差
      + Σ_k α_k(S-α_k) / [S²(S+1)]   ← Dirichlet 方差，推动增加总证据 S
      + λ_kl · KL(Dir(α̃) || Dir(1))  ← 压制非真类的证据

    其中 p_k = α_k/S（Dirichlet 期望），α̃ = y + (1-y) ⊙ α。

    Parameters
    ----------
    evidence : Tensor (B, K)  非负证据 e
    alpha : Tensor (B, K)     Dirichlet 参数 = 1 + e
    S : Tensor (B, K) or (B,) Dirichlet 强度
    y_onehot : Tensor (B, K)  one-hot 标签
    k_evidence : int          类别数
    lambda_kl : float         KL 正则权重（默认 0.01）
    return_components : bool  若 True，返回 (total, mse, var, kl) 四个标量

    Returns
    -------
    loss : scalar Tensor（return_components=False）
    (loss, mse, var, kl) : tuple（return_components=True）
    """
    if S.dim() == 1:
        S = S.unsqueeze(-1)  # (B, 1)

    p = alpha / S  # (B, K) Dirichlet 期望概率

    # Bayes risk: 期望 MSE = (y - p)² + Var(p)
    mse = ((y_onehot - p) ** 2).sum(dim=-1)                              # (B,)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=-1)          # (B,)

    # KL( Dir(α̃) || Dir(1) ) — 压制非真类证据
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(-1))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        - torch.lgamma(torch.tensor(float(k_evidence), device=alpha.device))
        + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
    )                                                                    # (B,)

    total = (mse + var + lambda_kl * kl).mean()
    if return_components:
        return total, mse.detach().mean(), var.detach().mean(), kl.detach().mean()
    return total


# ---------------------------------------------------------------------------
# 数据划分：按图分层分割
# ---------------------------------------------------------------------------

def split_trajectories(
    all_trajs: List,
    root: str,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List, List]:
    """按标签对图分层划分，再映射为 train/val 轨迹集合。

    从原始 CSV 读取标签和 split，不加载任何图数据。
    同一图的所有轨迹（不同 seed、不同 teacher、不同 budget）一定分到同一集合，
    避免数据泄漏。
    """
    from m4.teachers.base import Trajectory

    data_dir = Path(root) / 'ogbg_ppa'

    # 读全量标签 + 训练集 split（不加载图数据）
    labels_arr = pd.read_csv(
        data_dir / 'raw' / 'graph-label.csv.gz',
        compression='gzip', header=None,
    ).values.flatten()  # (158100,) int64

    train_idx = pd.read_csv(
        data_dir / 'split' / 'species' / 'train.csv.gz',
        compression='gzip', header=None,
    ).values.flatten()
    train_set = set(train_idx.tolist())

    # 1. 收集所有被引用的 graph_idx
    graph_ids = sorted({t.graph_idx for t in all_trajs})

    # 2. 从标签数组查标签（不加载图数据）
    graph_to_label = {}
    missing = 0
    for gid in graph_ids:
        if gid not in train_set:
            missing += 1
            continue
        graph_to_label[gid] = int(labels_arr[gid])
    if missing:
        print(f"  WARNING: {missing} graph_idx not in training set, excluded from split")

    valid_ids = sorted(graph_to_label.keys())
    labels = [graph_to_label[gid] for gid in valid_ids]

    # 3. 分层划分：稀有类（仅1个图）全部放入训练集，其余按标签分层
    from collections import Counter
    label_counts = Counter(labels)
    rare_labels = {lb for lb, cnt in label_counts.items() if cnt < 2}
    if rare_labels:
        forced_train = [gid for gid in valid_ids if graph_to_label[gid] in rare_labels]
        stratify_ids = [gid for gid in valid_ids if graph_to_label[gid] not in rare_labels]
        stratify_labels = [graph_to_label[gid] for gid in stratify_ids]
        print(f"  {len(rare_labels)} rare label(s) with <2 graphs → forced to train ({len(forced_train)} graphs)")
    else:
        forced_train = []
        stratify_ids = valid_ids
        stratify_labels = labels

    if len(stratify_ids) > 0:
        train_ids, val_ids = train_test_split(
            stratify_ids, test_size=val_ratio, stratify=stratify_labels,
            random_state=random_seed,
        )
        train_ids = list(train_ids) + forced_train
    else:
        train_ids = forced_train
        val_ids = []
    train_set_ids = set(train_ids)
    val_set_ids = set(val_ids)

    # 4. 映射轨迹
    train_trajs = [t for t in all_trajs if t.graph_idx in train_set_ids]
    val_trajs = [t for t in all_trajs if t.graph_idx in val_set_ids]

    print(f"Split: {len(train_trajs)} train / {len(val_trajs)} val trajectories "
          f"(from {len(train_ids)} / {len(val_ids)} graphs)")
    return train_trajs, val_trajs


# ---------------------------------------------------------------------------
# BC 验证循环
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_one_epoch(
    reader,
    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]],
    val_trajs: List,
    batch_size: int,
    device: torch.device,
) -> float:
    """单轮 BC 验证（v0 LearnableReader），返回平均 step loss。"""
    reader.eval()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(range(0, len(val_trajs), batch_size), desc="  Validating")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(val_trajs))
        batch_trajs = val_trajs[batch_start:batch_end]

        for traj in batch_trajs:
            g, _label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            step_losses = []

            for teacher_node in traj.reveal_order:
                scores, probs, frontier_nodes = reader.forward(state)
                n_frontier = len(frontier_nodes)
                if n_frontier == 0:
                    break
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    break
                loss_t = -torch.log(probs[target_idx] + 1e-8)
                step_losses.append(loss_t)
                reveal_node(state, teacher_node)

            if step_losses:
                total_loss += torch.stack(step_losses).sum().item()
                total_steps += len(step_losses)

    avg_step_loss = total_loss / max(total_steps, 1)
    return avg_step_loss

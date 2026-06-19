"""M5 训练共享工具：轨迹加载、图加载、BC 训练循环。

train_bc.py / train_pomdp_bc.py 等脚本共用，避免代码重复。
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import dgl
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from ogb.graphproppred import DglGraphPropPredDataset

TEACHER_NAMES = [
    'gradient', 'structural-degree', 'structural-pagerank',
    'hybrid-a0.30', 'hybrid-a0.70',
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
    """只加载轨迹引用的训练集图，加自环后缓存。"""
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
    train_indices = set(dataset.get_idx_split()['train'].tolist())

    to_load = needed_graphs & train_indices
    skipped = needed_graphs - train_indices
    if skipped:
        print(f"  WARNING: {len(skipped)} graph_idx not in training set, ignored")

    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]] = {}
    print(f"Loading {len(to_load)} graphs (training set subset)...")
    for i in tqdm(sorted(to_load), desc="  Loading graphs"):
        g, lbl = dataset[i]
        g = dgl.add_self_loop(g)
        graph_cache[i] = (g, lbl.item())
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

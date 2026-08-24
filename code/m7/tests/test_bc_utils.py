"""train_bc 纯逻辑单测（merge/split/ratio 键，不依赖真实数据与训练循环）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch

from m7.train_bc import ratio_key, merge_trajectories, split_trajectories


def _traj(gid, seed=0):
    from m4.teachers.base import Trajectory
    return Trajectory(graph_idx=gid, seed_node=seed, reveal_ratio=0.1,
                      reveal_order=[seed], final_pred=0, is_correct=True,
                      teacher_type='gradient')


def test_ratio_key():
    assert ratio_key(0.05) == '0_050'
    assert ratio_key(0.10) == '0_100'


def test_merge(tmp_path):
    torch.save({'trajectories': {'gradient': [[_traj(1)], [_traj(2)]]}},
               str(tmp_path / 'trajectories_r0_100.pt'))
    trajs = merge_trajectories(tmp_path, ['gradient'], [0.1])
    assert len(trajs) == 2


def test_split_no_leakage_rare_forced():
    labels = torch.tensor([g % 4 for g in range(20)])
    labels[19] = 99  # 稀有类（仅 1 图）
    trajs = [_traj(g) for g in range(20)]
    tr, va = split_trajectories(trajs, labels, val_ratio=0.2, random_seed=42)
    tr_ids = {t.graph_idx for t in tr}
    va_ids = {t.graph_idx for t in va}
    assert 19 in tr_ids               # 稀有类强制进训练
    assert tr_ids.isdisjoint(va_ids)  # 同一图不跨集合（防泄漏）

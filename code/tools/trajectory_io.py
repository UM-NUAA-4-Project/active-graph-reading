"""轨迹文件 IO：带 pickle 兼容的加载、目录合并、保存。

旧轨迹 .pt（m4/m5/m6/m7 时代）pickle 引用 `m4.teachers.base.Trajectory`；
本模块用自定义 Unpickler 把该路径重映射到 trajectories.core.trajectory.Trajectory，
旧文件在新结构下照常加载（m4 目录删除后依然可用）。

轨迹文件格式（沿袭 m4/m7）：
    {'trajectories': {teacher_name: [[Trajectory, ...] per graph]}, 'config': {...}}
"""

import pickle
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

# m4 时代 Trajectory 所在模块路径（pickle 反序列化重映射用）
LEGACY_PICKLE_MODULE = 'm4.teachers.base'


def _trajectory_cls():
    """惰性导入 Trajectory（避免 tools ↔ trajectories 循环导入）。"""
    from trajectories.core.trajectory import Trajectory
    return Trajectory


class _TrajectoryUnpickler(pickle.Unpickler):
    """把旧模块路径 m4.teachers.base 重映射到新 Trajectory。"""

    def find_class(self, module, name):
        if name == 'Trajectory' and module == LEGACY_PICKLE_MODULE:
            return _trajectory_cls()
        return super().find_class(module, name)


class _compat_pickle_module:
    """torch.load(pickle_module=...) 所需的最小接口。"""

    Unpickler = _TrajectoryUnpickler


def load_trajectory_file(path) -> Dict:
    """加载单个轨迹 .pt（新旧 pickle 路径都兼容）。"""
    return torch.load(str(path), map_location='cpu', weights_only=False,
                      pickle_module=_compat_pickle_module)


def normalize_trajectory(t):
    """任意来源（新类/旧类实例/dict）的轨迹 → 新 Trajectory。"""
    Trajectory = _trajectory_cls()
    if isinstance(t, Trajectory):
        return t
    return Trajectory(
        graph_idx=int(t.graph_idx),
        seed_node=int(t.seed_node),
        reveal_ratio=float(t.reveal_ratio),
        reveal_order=[int(v) for v in t.reveal_order],
        final_pred=int(t.final_pred),
        is_correct=bool(t.is_correct),
        teacher_type=str(t.teacher_type),
        ce=float(getattr(t, 'ce', 0.0) or 0.0),
    )


def ratio_key(ratio: float) -> str:
    """0.05 → '0_050'（文件名用，与 m4/m7 命名兼容）。"""
    return f'0_{round(ratio * 1000):03d}'


def ratio_key_from_name(name: str) -> float:
    """'0_050' → 0.05（ratio_key 的逆）。"""
    m = re.search(r'0_(\d{3})', name)
    if not m:
        raise ValueError(f"无法从文件名解析预算档: {name}")
    return int(m.group(1)) / 1000.0


def save_trajectory_file(path, trajs_by_teacher: Dict[str, list],
                         config: Dict) -> None:
    """保存轨迹文件：{teacher: [per_graph lists]} 平铺为 [[t, ...]] 结构保留。

    trajs_by_teacher: {teacher_name: List[Trajectory]}（已平铺）
    存储结构与 m4/m7 一致：{teacher: [[t...] per graph]} —— 按 graph_idx 分组。
    """
    Trajectory = _trajectory_cls()
    grouped: Dict[str, Dict[int, List]] = {}
    for teacher, trajs in trajs_by_teacher.items():
        per_graph: Dict[int, List[Trajectory]] = {}
        for t in trajs:
            per_graph.setdefault(t.graph_idx, []).append(normalize_trajectory(t))
        grouped[teacher] = [per_graph[k] for k in sorted(per_graph)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'trajectories': grouped, 'config': config}, str(path))


def iter_file_trajectories(data: Dict) -> Iterator:
    """遍历轨迹文件中所有轨迹（展平）。"""
    for per_graph in data.get('trajectories', {}).values():
        for trajs in per_graph:
            for t in trajs:
                yield t


def merge_trajectory_dir(data_dir, pattern: str = 'trajectories_*.pt',
                         verbose: bool = True) -> List:
    """加载目录下全部轨迹文件并展平（收敛 m5 merge_trajectories 与 m7 版本）。

    不再硬编码 teacher 名单与预算档——目录里有什么就合并什么。
    """
    Trajectory = _trajectory_cls()
    data_dir = Path(data_dir)
    all_trajs: List = []
    for fpath in sorted(data_dir.glob(pattern)):
        data = load_trajectory_file(fpath)
        n_before = len(all_trajs)
        for t in iter_file_trajectories(data):
            all_trajs.append(normalize_trajectory(t))
        if verbose:
            print(f"  {fpath.name}: +{len(all_trajs) - n_before} trajectories")
    if verbose:
        print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs

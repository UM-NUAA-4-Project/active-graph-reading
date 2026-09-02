"""teachers 包：T1 gradient / T2 info-gain / T3 structural / T4 hybrid。

build_teacher(name) 工厂统一解析教师名（generate_trajectories.py 用）：
    gradient | structural-degree | structural-pagerank | info-gain | hybrid-a{α}
"""

from .base import Teacher, Trajectory, LEGACY_PICKLE_MODULE  # noqa: F401
from .gradient import GradientTeacher
from .structural import StructuralTeacher
from .hybrid import HybridTeacher
from .info_gain import InfoGainTeacher


def build_teacher(name: str, classifier=None, subgraph_builder=None,
                  device: str = 'cpu') -> Teacher:
    """教师名 → Teacher 实例。info-gain 需要 classifier/builder/device。"""
    if name == 'gradient':
        return GradientTeacher()
    if name.startswith('structural-'):
        return StructuralTeacher(method=name.split('-', 1)[1])
    if name.startswith('hybrid-a'):
        return HybridTeacher(alpha=float(name.split('-')[1].replace('a', '')))
    if name == 'info-gain':
        if classifier is None or subgraph_builder is None:
            raise ValueError("info-gain 教师需要 classifier 与 subgraph_builder")
        return InfoGainTeacher(classifier, subgraph_builder, device=device)
    raise ValueError(f"未知教师 '{name}'，可用: gradient, structural-degree, "
                     "structural-pagerank, info-gain, hybrid-a{α}")

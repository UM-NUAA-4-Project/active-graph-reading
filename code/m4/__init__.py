"""M4: Teacher 轨迹生成模块。"""

from .teachers.base import Teacher, Trajectory
from .teachers.gradient_teacher import GradientTeacher
from .teachers.structural_teacher import StructuralTeacher
from .teachers.hybrid_teacher import HybridTeacher
from .teachers.info_gain_teacher import InfoGainTeacher
from .graph_meta import GraphMeta, precompute_graph_meta
from .trajectory_generator import TrajectoryGenerator

__all__ = [
    'Teacher', 'Trajectory',
    'GradientTeacher', 'StructuralTeacher', 'HybridTeacher', 'InfoGainTeacher',
    'GraphMeta', 'precompute_graph_meta',
    'TrajectoryGenerator',
]

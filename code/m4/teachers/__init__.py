"""M4 Teacher implementations.

Teachers produce per-node importance scores from the full graph.
These scores guide simulated reveal sequences that become training
trajectories for M5 behavior cloning.
"""

from m4.teachers.base import Teacher
from m4.teachers.gradient_teacher import GradientTeacher
from m4.teachers.structural_teacher import StructuralTeacher
from m4.teachers.hybrid_teacher import HybridTeacher
from m4.teachers.info_gain_teacher import InfoGainTeacher

__all__ = [
    'Teacher',
    'GradientTeacher',
    'StructuralTeacher',
    'HybridTeacher',
    'InfoGainTeacher',
]

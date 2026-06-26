"""Teacher implementations."""

from .base import Teacher, Trajectory
from .gradient_teacher import GradientTeacher
from .structural_teacher import StructuralTeacher
from .hybrid_teacher import HybridTeacher
from .info_gain_teacher import InfoGainTeacher

__all__ = [
    'Teacher', 'Trajectory',
    'GradientTeacher', 'StructuralTeacher', 'HybridTeacher', 'InfoGainTeacher',
]

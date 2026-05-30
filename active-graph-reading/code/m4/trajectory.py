"""Trajectory data structures for M4."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Trajectory:
    """A single teacher-guided graph reading trajectory.

    Records the complete reveal sequence produced by a teacher,
    along with per-step classifier feedback (confidence, entropy)
    for downstream scoring and behavior cloning.
    """

    graph_idx: int
    seed_node: int
    budget_used: int
    reveal_order: List[int] = field(default_factory=list)
    scores_per_step: List[float] = field(default_factory=list)
    confidence_per_step: List[float] = field(default_factory=list)
    entropy_per_step: List[float] = field(default_factory=list)
    final_pred: int = -1
    is_correct: bool = False
    teacher_type: str = ""
    teacher_params: Dict[str, Any] = field(default_factory=dict)
    label: int = -1

    @property
    def n_steps(self) -> int:
        return len(self.reveal_order)

    @property
    def final_confidence(self) -> float:
        return self.confidence_per_step[-1] if self.confidence_per_step else 0.0

    @property
    def final_entropy(self) -> float:
        return self.entropy_per_step[-1] if self.entropy_per_step else float('inf')

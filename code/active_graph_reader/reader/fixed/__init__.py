"""Fixed strategies — simple non-learnable readers."""

from typing import Type

from ...core.base_reader import BaseReader
from .random import RandomStrategy
from .bfs import BfsStrategy
from .max_degree import MaxFrontierDegreeStrategy
from .registry import get_strategy, register_strategy, list_strategies

_STRATEGY_CLASSES = {
    'random': RandomStrategy,
    'bfs': BfsStrategy,
    'max-frontier-degree': MaxFrontierDegreeStrategy,
}


def create_fixed_reader(name: str) -> BaseReader:
    """Factory: return a concrete strategy instance by name.

    Usage: create_fixed_reader('random') returns a RandomStrategy instance.
    """
    cls = _STRATEGY_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}', available: {sorted(_STRATEGY_CLASSES)}"
        )
    return cls()


__all__ = [
    'RandomStrategy',
    'BfsStrategy',
    'MaxFrontierDegreeStrategy',
    'get_strategy',
    'register_strategy',
    'list_strategies',
    'create_fixed_reader',
]

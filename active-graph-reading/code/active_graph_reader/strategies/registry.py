"""策略注册表：提供策略的注册、查找和列举功能。"""

from typing import Dict, Type

from ..reader.base_reader import BaseReader
from .random_strategy import RandomStrategy
from .bfs_strategy import BfsStrategy
from .max_frontier_degree_strategy import MaxFrontierDegreeStrategy

_registry: Dict[str, Type[BaseReader]] = {
    'random': RandomStrategy,
    'bfs': BfsStrategy,
    'max-frontier-degree': MaxFrontierDegreeStrategy,
}


def register_strategy(name: str, cls: Type[BaseReader]) -> None:
    _registry[name] = cls


def get_strategy(name: str) -> BaseReader:
    if name not in _registry:
        raise ValueError(f"未知策略 '{name}'，可用: {sorted(_registry)}")
    return _registry[name]()


def list_strategies() -> Dict[str, Type[BaseReader]]:
    return dict(_registry)

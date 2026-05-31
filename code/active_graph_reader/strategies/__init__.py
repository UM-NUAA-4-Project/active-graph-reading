"""前沿边揭示策略模块。提供可插拔的策略接口和三种内置实现。"""

from .random_strategy import RandomStrategy
from .bfs_strategy import BfsStrategy
from .max_frontier_degree_strategy import MaxFrontierDegreeStrategy
from .registry import get_strategy, register_strategy, list_strategies

__all__ = [
    'RandomStrategy',
    'BfsStrategy',
    'MaxFrontierDegreeStrategy',
    'get_strategy',
    'register_strategy',
    'list_strategies',
]

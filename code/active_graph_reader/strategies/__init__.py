"""前沿边揭示策略模块。

提供可插拔的策略接口和三种内置实现。
新增策略只需继承 RevealStrategy 并注册即可。
"""

from .base import RevealStrategy
from .random_strategy import RandomStrategy
from .bfs_strategy import BfsStrategy
from .max_frontier_degree_strategy import (
    MaxFrontierDegreeStrategy,
)
from .registry import (
    get_strategy,
    register_strategy,
    list_strategies,
)

__all__ = [
    # 抽象基类
    'RevealStrategy',
    # 具体策略
    'RandomStrategy',
    'BfsStrategy',
    'MaxFrontierDegreeStrategy',
    # 注册表
    'get_strategy',
    'register_strategy',
    'list_strategies',
]

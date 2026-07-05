"""
策略注册表
==========

提供揭示策略的注册、查找和列举功能。通过名称字符串动态获取策略实例，
实现策略的「可插拔」切换。

内置策略:
- 'random'          → RandomStrategy
- 'bfs'             → BfsStrategy
- 'max-frontier-degree' → MaxFrontierDegreeStrategy

扩展方式（新增策略只需两步，无需修改主流程）:
    1. 继承 RevealStrategy 实现新策略类。
    2. 调用 register_strategy('my-name', MyStrategy)。
    → 之后可通过 get_strategy('my-name') 获取实例。
"""

from typing import Dict, Type

from .base import RevealStrategy
from .random_strategy import RandomStrategy
from .bfs_strategy import BfsStrategy
from .max_frontier_degree_strategy import (
    MaxFrontierDegreeStrategy,
)

# ---- 内置策略注册 -----------------------------------------------------------
_BUILTIN_STRATEGIES: Dict[str, Type[RevealStrategy]] = {
    'random': RandomStrategy,
    'bfs': BfsStrategy,
    'max-frontier-degree': MaxFrontierDegreeStrategy,
}

# 运行时注册表
_registry: Dict[str, Type[RevealStrategy]] = dict(_BUILTIN_STRATEGIES)


def register_strategy(name: str, strategy_cls: Type[RevealStrategy]) -> None:
    """注册一个新的揭示策略类。

    Parameters
    ----------
    name : str
        策略名称（用于配置和命令行指定）。
    strategy_cls : Type[RevealStrategy]
        策略类（须为 RevealStrategy 的子类）。

    Raises
    ------
    TypeError
        如果 strategy_cls 不是 RevealStrategy 的子类。
    """
    if not issubclass(strategy_cls, RevealStrategy):
        raise TypeError(
            f"'{strategy_cls.__name__}' 必须是 RevealStrategy 的子类")
    _registry[name] = strategy_cls


def get_strategy(name: str) -> RevealStrategy:
    """根据名称获取策略实例（每次调用创建新实例）。

    Parameters
    ----------
    name : str
        策略名称。

    Returns
    -------
    RevealStrategy
        策略实例。

    Raises
    ------
    ValueError
        如果指定的策略名称未注册。
    """
    if name not in _registry:
        available = sorted(_registry.keys())
        raise ValueError(
            f"未知策略 '{name}'，可用: {available}")
    return _registry[name]()


def get_strategy_class(name: str) -> Type[RevealStrategy]:
    """根据名称获取策略类（不实例化）。"""
    if name not in _registry:
        available = sorted(_registry.keys())
        raise ValueError(
            f"未知策略 '{name}'，可用: {available}")
    return _registry[name]


def list_strategies() -> Dict[str, Type[RevealStrategy]]:
    """列举所有已注册的策略。

    Returns
    -------
    Dict[str, Type[RevealStrategy]]
    """
    return dict(_registry)

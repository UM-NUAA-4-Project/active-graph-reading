"""Reader module — unified interface for fixed and learnable graph readers."""

from .base_reader import BaseReader
from .learnable_reader import LearnableReader
from .frontier_encoder import FrontierNodesEncoder
from .facade import reveal_one, reveal_batch, evaluate, create_dataloader

# FixedStrategyReader: backward-compat shim (eliminates the old pass-through wrapper)
_FIXED_STRATEGY_MAP = {
    'random': None,
    'bfs': None,
    'max-frontier-degree': None,
}


def _init_fixed_strat():
    """Lazy-import strategy classes to avoid circular imports at module level."""
    from ..strategies.random_strategy import RandomStrategy
    from ..strategies.bfs_strategy import BfsStrategy
    from ..strategies.max_frontier_degree_strategy import MaxFrontierDegreeStrategy
    _FIXED_STRATEGY_MAP.update({
        'random': RandomStrategy,
        'bfs': BfsStrategy,
        'max-frontier-degree': MaxFrontierDegreeStrategy,
    })


class FixedStrategyReader(BaseReader):
    """Backward-compat: returns a concrete strategy instance by name.

    Usage: FixedStrategyReader('random') returns a RandomStrategy instance.
    """

    def __new__(cls, strategy_name: str):
        if _FIXED_STRATEGY_MAP.get(strategy_name) is None:
            _init_fixed_strat()
        cls_or_none = _FIXED_STRATEGY_MAP.get(strategy_name)
        if cls_or_none is None:
            raise ValueError(f"Unknown strategy '{strategy_name}', available: {sorted(_FIXED_STRATEGY_MAP)}")
        return cls_or_none()


__all__ = [
    'BaseReader',
    'FixedStrategyReader',
    'LearnableReader',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
    'evaluate',
    'create_dataloader',
]

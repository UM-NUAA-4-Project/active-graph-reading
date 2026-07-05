"""数据集模块。"""

from .loader import (
    load_ogbg_ppa,
    get_graph_stats,
    PPADataset,
)

__all__ = [
    'load_ogbg_ppa',
    'get_graph_stats',
    'PPADataset',
]

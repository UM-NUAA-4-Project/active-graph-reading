"""m7 数据集注册表。"""

from .registry import (
    DatasetSpec, Dataset, register, get_dataset,
    has_self_loops, DATASETS,
)
from . import cifar10  # noqa: F401  (注册 cifar10 条目)

__all__ = ['DatasetSpec', 'Dataset', 'register', 'get_dataset',
           'has_self_loops', 'DATASETS']

"""数据集注册表包。

注意（遮蔽问题）：顶层包名 datasets 会遮蔽 HuggingFace 的 datasets 包，
而 dgl→torchdata 初始化链会隐式 import datasets。因此：
- registry.py 刻意不依赖 dgl；
- cifar10/ppa 等数据集模块不在此处导入，由 get_dataset() 首次调用时
  懒注册（见 registry._ensure_registered）。

用法：
    from datasets import get_dataset
    ds = get_dataset('cifar10', root='D:/superpixels', splits=('test',))
"""

from .registry import (
    DatasetSpec,
    Dataset,
    DATASETS,
    get_dataset,
    register,
    has_self_loops,
)

__all__ = [
    'DatasetSpec', 'Dataset', 'DATASETS', 'get_dataset', 'register',
    'has_self_loops',
]

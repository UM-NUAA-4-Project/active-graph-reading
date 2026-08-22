"""数据集注册表：名称 → DatasetSpec（加载器 + 元信息）。

新增数据集 = 实现 load 函数并 register 一个 DatasetSpec，
m7 所有脚本通过 --dataset 名称无缝切换。
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import dgl
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSpec:
    name: str
    n_classes: int
    node_feat_dim: int            # 0 = 无节点特征
    edge_feat_dim: int
    needs_self_loop: bool
    budget_options: Tuple[float, ...] = (0.05, 0.10, 0.15)
    load: Optional[Callable] = None   # load(spec) -> Dataset


@dataclass
class Dataset:
    spec: DatasetSpec
    train_graphs: List[dgl.DGLGraph]
    train_labels: torch.Tensor
    valid_graphs: List[dgl.DGLGraph]
    valid_labels: torch.Tensor
    test_graphs: List[dgl.DGLGraph]
    test_labels: torch.Tensor
    train_indices: List[int]
    test_indices: List[int]


DATASETS: dict = {}
_CACHE: dict = {}


class _LazyDataset(Dataset):
    """懒加载包装：get_dataset 只读 spec，首次访问数据字段时才调用 spec.load（结果缓存）。

    注：与 plan 原稿的差异——plan 的 get_dataset 在首次调用时即 spec.load(spec)，
    但 test_datasets.py 要求 get_dataset('cifar10') 只读 spec、不触发下载
    （数据目录 D:\\CIFAR10-Superpixel 可能尚未就绪），故把 load 推迟到
    数据字段首次访问（真·懒加载，语义仍为"懒加载 + 缓存"）。
    """

    _DATA_FIELDS = frozenset({
        'train_graphs', 'train_labels', 'valid_graphs', 'valid_labels',
        'test_graphs', 'test_labels', 'train_indices', 'test_indices',
    })

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self._real: Optional[Dataset] = None

    def _ensure_loaded(self) -> Dataset:
        if self._real is None:
            self._real = self.spec.load(self.spec)
        return self._real

    def __getattr__(self, name: str):
        if name in _LazyDataset._DATA_FIELDS:
            return getattr(self._ensure_loaded(), name)
        raise AttributeError(f"{type(self).__name__!r} 没有属性 {name!r}")


def register(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


def get_dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"未知数据集 '{name}'，可用: {sorted(DATASETS)}")
    if name not in _CACHE:
        spec = DATASETS[name]
        if spec.load is None:
            raise ValueError(f"数据集 '{name}' 未实现 load 函数")
        _CACHE[name] = _LazyDataset(spec)
    return _CACHE[name]


def stratified_split(labels: List[int], test_size: float = 0.1,
                     seed: int = 42) -> Tuple[List[int], List[int]]:
    idx = list(range(len(labels)))
    train_idx, valid_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed)
    return list(train_idx), list(valid_idx)


def has_self_loops(g: dgl.DGLGraph) -> bool:
    u, v = g.edges()
    return bool(torch.any(u == v).item())

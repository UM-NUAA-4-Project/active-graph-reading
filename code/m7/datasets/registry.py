"""数据集注册表：名称 → DatasetSpec（加载器 + 元信息）。

新增数据集 = 实现 load 函数并 register 一个 DatasetSpec，
m7 所有脚本通过 --dataset 名称无缝切换。
"""

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

import torch
import dgl


@dataclass
class DatasetSpec:
    name: str
    n_classes: int
    node_feat_dim: int            # 0 = 无节点特征
    edge_feat_dim: int
    needs_self_loop: bool
    budget_options: Tuple[float, ...] = (0.05, 0.10, 0.15)
    root: Optional[str] = None   # 数据根目录（--root 覆盖）；None = 用加载器默认/环境变量
    splits: Tuple[str, ...] = ('train', 'valid', 'test')  # 需加载的 split（get_dataset 按需覆盖）
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

    def __repr__(self) -> str:
        # 覆写 dataclass 生成的 __repr__：不能访问数据字段
        # （会触发 2.5GB 懒加载），只显示 spec 名与加载状态
        return (f"_LazyDataset(spec={self.spec.name!r}, "
                f"loaded={self._real is not None})")

    def __eq__(self, other) -> bool:
        # 覆写 dataclass 生成的 __eq__：按身份比较，绝不触碰数据字段
        return self is other


def register(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


def get_dataset(
    name: str,
    root: Optional[str] = None,
    splits=('train', 'valid', 'test'),
) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"未知数据集 '{name}'，可用: {sorted(DATASETS)}")
    splits = tuple(splits)
    cache_key = (name, root, splits)
    if cache_key not in _CACHE:
        spec = DATASETS[name]
        if spec.load is None:
            raise ValueError(f"数据集 '{name}' 未实现 load 函数")
        # 副本，避免污染注册表；splits 由加载器读取，未列出的 split 返回空列表
        # （如 eval.py 只加载 test 时不再加载 50000 张 train 图）
        spec = replace(spec, root=root, splits=splits)
        _CACHE[cache_key] = _LazyDataset(spec)
    return _CACHE[cache_key]


def has_self_loops(g: dgl.DGLGraph) -> bool:
    u, v = g.edges()
    return bool(torch.any(u == v).item())

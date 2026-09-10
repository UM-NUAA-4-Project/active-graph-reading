"""数据集注册表：名称 → DatasetSpec（加载器 + 元信息）。

新增数据集 = 实现 load 函数并 register 一个 DatasetSpec，
所有训练/生成/评估脚本通过 --dataset 名称无缝切换。

自 code/m7/datasets/registry.py 迁入，扩展：
- DatasetSpec.max_graphs：按 split 截断加载（PPA 158k 全量加载不可行）
- DatasetSpec.load_graph_subset：按 OGB 索引选择性加载（BC 轨迹引用图）
- Dataset.valid_indices：valid split 的索引列表

注意：本模块刻意不 import dgl —— 顶层包名 datasets 会遮蔽 HuggingFace
datasets，而 dgl→torchdata 初始化链会隐式 import datasets；在这里引入
dgl 会造成循环导入。各数据集模块（cifar10/ppa）由 get_dataset 首次
调用时懒注册。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

import torch

# 分层子采样的固定种子（独立于训练 --seed，保证任何运行选到同一子集）
SUBSET_SEED = 20260909


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
    # 每 split 最多加载的图数（0=不限制）。大图数据集（PPA）必配。
    max_graphs: int = 0
    # 按 OGB train 索引选择性加载：fn(spec, iterable[int]) -> {idx: (graph, label)}
    # 轨迹管线（BC/诊断）用它只加载被引用的图
    load_graph_subset: Optional[Callable] = None
    # max_graphs > 0 时按类别比例分层抽样（False = 前 N 截断，历史默认）
    stratified: bool = False


@dataclass
class Dataset:
    spec: DatasetSpec
    train_graphs: List
    train_labels: torch.Tensor
    valid_graphs: List
    valid_labels: torch.Tensor
    test_graphs: List
    test_labels: torch.Tensor
    train_indices: List[int]
    test_indices: List[int]
    valid_indices: List[int] = field(default_factory=list)


DATASETS: dict = {}
_CACHE: dict = {}
_REGISTERED = False


def _ensure_registered() -> None:
    """首次 get_dataset 时才加载数据集模块（避免 dgl 循环导入）。"""
    global _REGISTERED
    if not _REGISTERED:
        from . import cifar10, collab, ppa  # noqa: F401  导入即注册
        _REGISTERED = True


class _LazyDataset(Dataset):
    """懒加载包装：get_dataset 只读 spec，首次访问数据字段时才调用 spec.load（结果缓存）。

    注：与 plan 原稿的差异——plan 的 get_dataset 在首次调用时即 spec.load(spec)，
    但 test_datasets.py 要求 get_dataset('cifar10') 只读 spec、不触发下载
    （数据目录可能尚未就绪），故把 load 推迟到数据字段首次访问
    （真·懒加载，语义仍为"懒加载 + 缓存"）。
    """

    _DATA_FIELDS = frozenset({
        'train_graphs', 'train_labels', 'valid_graphs', 'valid_labels',
        'test_graphs', 'test_labels', 'train_indices', 'test_indices',
        'valid_indices',
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
        # （会触发全量懒加载），只显示 spec 名与加载状态
        return (f"_LazyDataset(spec={self.spec.name!r}, "
                f"loaded={self._real is not None})")

    def __eq__(self, other) -> bool:
        # 覆写 dataclass 生成的 __eq__：按身份比较，绝不触碰数据字段
        return self is other

    def __hash__(self) -> int:
        return id(self)


def register(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


def stratified_take(indices, label_of, n: int,
                    seed: int = SUBSET_SEED) -> List[int]:
    """按类别比例从 indices 抽 n 个（固定种子，确定性）；返回升序列表。

    indices: 候选 id 列表；label_of: id → 类别（函数或按 id 索引的序列）。
    每类配额用最大余数法（总数恰为 n，小数部分大的类优先补 1）；
    类内 rng 洗牌后取前 k。n<=0 或 n>=len(indices) 原样返回全部（升序）。
    """
    indices = list(indices)
    if n <= 0 or n >= len(indices):
        return sorted(indices)

    def _label(i):
        return label_of(i) if callable(label_of) else label_of[i]

    by_class: dict = {}
    for i in indices:
        by_class.setdefault(_label(i), []).append(i)
    classes = sorted(by_class)
    total = len(indices)
    exact = {c: n * len(by_class[c]) / total for c in classes}
    ks = {c: int(exact[c]) for c in classes}
    leftover = n - sum(ks.values())
    for c in sorted(classes, key=lambda c: (-(exact[c] - ks[c]), c))[:leftover]:
        ks[c] += 1
    rng = random.Random(seed)
    picked: List[int] = []
    for c in classes:
        pool = by_class[c][:]
        rng.shuffle(pool)
        picked += pool[:ks[c]]
    return sorted(picked)


def get_dataset(
    name: str,
    root: Optional[str] = None,
    splits=('train', 'valid', 'test'),
    max_graphs: int = 0,
    stratified: bool = False,
) -> Dataset:
    _ensure_registered()
    if name not in DATASETS:
        raise ValueError(f"未知数据集 '{name}'，可用: {sorted(DATASETS)}")
    splits = tuple(splits)
    cache_key = (name, root, splits, int(max_graphs), bool(stratified))
    if cache_key not in _CACHE:
        spec = DATASETS[name]
        if spec.load is None:
            raise ValueError(f"数据集 '{name}' 未实现 load 函数")
        # 副本，避免污染注册表；splits/max_graphs/stratified 由加载器读取，
        # 未列出的 split 返回空列表（如 eval.py 只加载 test 时不再加载 train 图）
        spec = replace(spec, root=root, splits=splits, max_graphs=int(max_graphs),
                       stratified=bool(stratified))
        _CACHE[cache_key] = _LazyDataset(spec)
    return _CACHE[cache_key]


def has_self_loops(g) -> bool:
    u, v = g.edges()
    return bool(torch.any(u == v).item())


def release_dataset(ds) -> None:
    """移除缓存引用并丢弃已加载数据（eval 逐 split「加载→评估→释放」用）。

    仅从 _CACHE 摘除该实例并清空其 _real；调用方须同时删除自己的局部
    引用（graphs/labels 等），否则对象不会回收。
    """
    for key in [k for k, v in _CACHE.items() if v is ds]:
        del _CACHE[key]
    if isinstance(ds, _LazyDataset):
        ds._real = None

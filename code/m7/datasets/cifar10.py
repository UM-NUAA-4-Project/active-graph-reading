"""CIFAR10-SuperPixel 数据集加载（DGL 内置，use_feature=True）。

节点特征 5 维（位置 2 + 颜色 3），边特征 1 维，10 类。
不划分验证集：训练直接用 DGL 官方 train 划分，valid 恒为空
（评估统一走 test，见 code/m7/eval.py）。
数据目录完全由 DatasetSpec.root（--root 参数）指定，无默认值/环境变量。
"""

from typing import List, Tuple

import torch
import dgl
from dgl.data import CIFAR10SuperPixelDataset

from .registry import DatasetSpec, Dataset, register, has_self_loops


def _load_split(split: str, raw_dir: str) -> Tuple[List[dgl.DGLGraph], List[int]]:
    ds = CIFAR10SuperPixelDataset(split=split, use_feature=True, raw_dir=raw_dir)
    graphs = []
    labels = []
    for g, y in ds:
        graphs.append(g)
        labels.append(int(y))
    return graphs, labels


def load_cifar10(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"数据集 '{spec.name}' 数据目录未指定：必须通过 --root 传入"
            "（如 --root D:\\superpixels）")
    train_graphs, train_labels = _load_split('train', spec.root)
    test_graphs, test_labels = _load_split('test', spec.root)

    # 自环检查：数据集已含自环则不再添加
    if spec.needs_self_loop and not has_self_loops(train_graphs[0]):
        train_graphs = [dgl.add_self_loop(g) for g in train_graphs]
        test_graphs = [dgl.add_self_loop(g) for g in test_graphs]

    return Dataset(
        spec=spec,
        train_graphs=train_graphs,
        train_labels=torch.tensor(train_labels),
        valid_graphs=[],
        valid_labels=torch.tensor([]),
        test_graphs=test_graphs,
        test_labels=torch.tensor(test_labels),
        train_indices=list(range(len(train_graphs))),
        test_indices=list(range(len(test_graphs))),
    )


register(DatasetSpec(
    name='cifar10',
    n_classes=10,
    node_feat_dim=5,
    edge_feat_dim=1,
    needs_self_loop=True,
    budget_options=(0.05, 0.10, 0.15),
    load=load_cifar10,
))

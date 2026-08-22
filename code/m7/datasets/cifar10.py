"""CIFAR10-SuperPixel 数据集加载（DGL 内置，use_feature=True）。

节点特征 5 维（位置 2 + 颜色 3），边特征 1 维，10 类。
DGL 仅提供 train/test 划分；valid 从 train 按类别分层切 10%。
raw_dir 通过环境变量 M7_CIFAR10_RAW 配置（默认 D:\\CIFAR10-Superpixel）。
"""

import os
from typing import List, Tuple

import torch
import dgl
from dgl.data import CIFAR10SuperPixelDataset

from .registry import DatasetSpec, Dataset, register, stratified_split, has_self_loops

_RAW_DIR = os.environ.get('M7_CIFAR10_RAW', r'D:\CIFAR10-Superpixel')


def _load_split(split: str) -> Tuple[List[dgl.DGLGraph], List[int]]:
    ds = CIFAR10SuperPixelDataset(split=split, use_feature=True, raw_dir=_RAW_DIR)
    graphs = []
    labels = []
    for g, y in ds:
        graphs.append(g)
        labels.append(int(y))
    return graphs, labels


def load_cifar10(spec: DatasetSpec) -> Dataset:
    train_graphs, train_labels = _load_split('train')
    test_graphs, test_labels = _load_split('test')

    # 自环检查：数据集已含自环则不再添加
    if spec.needs_self_loop and not has_self_loops(train_graphs[0]):
        train_graphs = [dgl.add_self_loop(g) for g in train_graphs]
        test_graphs = [dgl.add_self_loop(g) for g in test_graphs]

    train_idx, valid_idx = stratified_split(train_labels, test_size=0.1, seed=42)
    valid_graphs = [train_graphs[i] for i in valid_idx]
    valid_labels = torch.tensor([train_labels[i] for i in valid_idx])
    train_graphs_keep = [train_graphs[i] for i in train_idx]
    train_labels_keep = torch.tensor([train_labels[i] for i in train_idx])

    return Dataset(
        spec=spec,
        train_graphs=train_graphs_keep,
        train_labels=train_labels_keep,
        valid_graphs=valid_graphs,
        valid_labels=valid_labels,
        test_graphs=test_graphs,
        test_labels=torch.tensor(test_labels),
        train_indices=list(range(len(train_graphs_keep))),
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

"""CIFAR10-SuperPixel 数据集加载（DGL 内置，use_feature=True）。

节点特征 5 维（位置 2 + 颜色 3），边特征 1 维，10 类。
不划分验证集：训练直接用 DGL 官方 train 划分，valid 恒为空
（评估统一走 test）。数据目录完全由 DatasetSpec.root（--root 参数）指定。

自 code/m7/datasets/cifar10.py 迁入；新增 max_graphs 截断（加载后截取，
与 m7 脚本 --max-graphs 切片语义一致）。
"""

from typing import List, Tuple

import torch
import dgl
from tqdm import tqdm

from .registry import DatasetSpec, Dataset, register, has_self_loops


def _load_split(split: str, raw_dir: str) -> Tuple[List[dgl.DGLGraph], List[int]]:
    # dgl.data 延迟导入：本模块可能在本包遮蔽 HF datasets 的状态下被
    # dgl 初始化链间接加载（见 registry.py 模块注释）
    from dgl.data import CIFAR10SuperPixelDataset
    ds = CIFAR10SuperPixelDataset(split=split, use_feature=True, raw_dir=raw_dir)
    graphs = []
    labels = []
    for g, y in tqdm(ds, desc=f"Loading {split}"):
        graphs.append(g)
        labels.append(int(y))
    return graphs, labels


def load_cifar10(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"数据集 '{spec.name}' 数据目录未指定：必须通过 --root 传入"
            "（如 --root D:\\superpixels）")

    # 只加载 spec.splits 列出的 split，其余返回空列表（如 eval 只需 test、或
    # train_bc/train_rl/generate_trajectories 只需 train，省去另一半加载时间）
    train_graphs, train_labels = [], []
    test_graphs, test_labels = [], []
    if 'train' in spec.splits:
        train_graphs, train_labels = _load_split('train', spec.root)
    if 'test' in spec.splits:
        test_graphs, test_labels = _load_split('test', spec.root)

    # max_graphs 截断（加载后截取，索引即前 N 个，确定性）
    if spec.max_graphs > 0:
        if 'train' in spec.splits:
            train_graphs, train_labels = (train_graphs[:spec.max_graphs],
                                          train_labels[:spec.max_graphs])
        if 'test' in spec.splits:
            test_graphs, test_labels = (test_graphs[:spec.max_graphs],
                                        test_labels[:spec.max_graphs])

    # 自环检查：数据集已含自环则不再添加（按实际加载到的 split 处理）
    if spec.needs_self_loop:
        if 'train' in spec.splits and train_graphs and not has_self_loops(train_graphs[0]):
            train_graphs = [dgl.add_self_loop(g)
                            for g in tqdm(train_graphs, desc="Self-loop train")]
        if 'test' in spec.splits and test_graphs and not has_self_loops(test_graphs[0]):
            test_graphs = [dgl.add_self_loop(g)
                           for g in tqdm(test_graphs, desc="Self-loop test")]

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

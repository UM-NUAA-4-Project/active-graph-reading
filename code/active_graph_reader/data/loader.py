"""
数据加载与预处理模块
====================

封装 OGBG-PPA 数据集的加载、统计和迭代访问。
使用 DGL 的 DglGraphPropPredDataset 作为底层数据源。

对应协议: base-protocol-v0.md 第 6 节。
"""

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple, Any, Iterator

import dgl
from ogb.graphproppred import DglGraphPropPredDataset

logger = logging.getLogger(__name__)


class PPADataset:
    """OGBG-PPA 数据集包装器。

    提供简便的加载、统计和迭代接口。

    Examples
    --------
    >>> ds = PPADataset()
    >>> print(f"总图数: {len(ds)}")
    >>> g, label = ds[0]
    >>> stats = ds.graph_stats(0)
    """

    def __init__(self, root: str = r'D:\ogb-master\dataset'):
        """加载 OGBG-PPA 数据集。

        Parameters
        ----------
        root : str
            OGB 数据集根目录。
        """
        logger.info("加载 OGBG-PPA 数据集 (root=%s)...", root)
        self._dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
        self._split_idx = self._dataset.get_idx_split()
        self.num_classes = self._dataset.num_classes

        logger.info(
            "OGBG-PPA 加载完成: %d 张图, %d 类别, "
            "train=%d, valid=%d, test=%d",
            len(self._dataset), self.num_classes,
            len(self._split_idx['train']),
            len(self._split_idx['valid']),
            len(self._split_idx['test']),
        )

    # ---- 数据访问 ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, int]:
        """获取第 idx 张图及其标签。

        Returns
        -------
        Tuple[dgl.DGLGraph, int]
            (图, 标签类别 ID)。
        """
        g, label = self._dataset[idx]
        return g, label.item()

    def __iter__(self) -> Iterator[Tuple[dgl.DGLGraph, int]]:
        for i in range(len(self)):
            yield self[i]

    # ---- 划分索引 ----------------------------------------------------------

    @property
    def train_indices(self):
        """训练集图索引列表。"""
        return self._split_idx['train'].tolist()

    @property
    def valid_indices(self):
        """验证集图索引列表。"""
        return self._split_idx['valid'].tolist()

    @property
    def test_indices(self):
        """测试集图索引列表。"""
        return self._split_idx['test'].tolist()

    # ---- 图统计 ------------------------------------------------------------

    def graph_stats(self, idx: int) -> Dict[str, Any]:
        """获取第 idx 张图的统计信息。

        Returns
        -------
        dict
            包含 n_nodes, n_edges, has_node_feat, has_edge_feat,
            edge_feat_dim, label 等字段。
        """
        g, label = self[idx]
        return get_graph_stats(g, label)


def load_ogbg_ppa(root: str = r'D:\ogb-master\dataset') -> PPADataset:
    """加载 OGBG-PPA 数据集（便捷函数）。

    Parameters
    ----------
    root : str
        OGB 数据集根目录。

    Returns
    -------
    PPADataset
    """
    return PPADataset(root=root)


def get_graph_stats(g: dgl.DGLGraph, label: int = None) -> Dict[str, Any]:
    """从 DGLGraph 提取统计信息。

    Parameters
    ----------
    g : dgl.DGLGraph
        图。
    label : int, optional
        标签。

    Returns
    -------
    dict
    """
    stats = {
        'n_nodes': g.num_nodes(),
        'n_edges': g.num_edges(),
        'has_node_feat': 'feat' in g.ndata and g.ndata['feat'] is not None,
        'has_edge_feat': 'feat' in g.edata,
    }

    if stats['has_node_feat']:
        stats['node_feat_dim'] = g.ndata['feat'].shape[1]
    if stats['has_edge_feat']:
        stats['edge_feat_dim'] = g.edata['feat'].shape[1]
    if label is not None:
        stats['label'] = label

    return stats


# ------------------------------------------------------------------
# 按索引批量加载
# ------------------------------------------------------------------

def load_graphs_by_index(
    data_root: str,
    graph_indices: Sequence[int],
) -> Tuple[Dict[int, dgl.DGLGraph], Dict[int, int]]:
    """按 OGB 原始索引批量加载全图及其标签。

    Returns
    -------
    graphs : {graph_idx: dgl.DGLGraph}
    labels : {graph_idx: int}
    """
    processed_path = str(
        Path(data_root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed'
    )
    idx_list = sorted(set(int(i) for i in graph_indices))
    logging.getLogger(__name__).info(
        "Loading %d graphs from %s ...", len(idx_list), processed_path,
    )
    graphs, labels_dict = dgl.load_graphs(processed_path, idx_list=idx_list)
    labels_t = labels_dict['labels']
    out_g = {oid: g for oid, g in zip(idx_list, graphs)}
    out_l = {oid: int(labels_t[oid].item()) for oid in idx_list}
    return out_g, out_l

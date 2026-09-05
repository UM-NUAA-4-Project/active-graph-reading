"""分类器配置：逐层 hidden/dropout 可独立调节，序列化到 JSON 供 config 文件与 checkpoint 使用。

序列化约定：所有层相同时折叠为标量（hidden_dim/n_layers/dropout），否则用列表
（hidden_dims/dropouts）。from_dict 两种形式都接受。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _as_list(value, n: Optional[int] = None) -> List:
    """标量 → [v]*n；列表 → 原样。用于 dropout/hidden_dims 的统一化。"""
    if isinstance(value, (list, tuple)):
        return list(value)
    if n is None:
        raise ValueError("标量形式需要提供层数 n")
    return [value] * n


@dataclass
class ClassifierConfig:
    """全图分类器结构配置。

    hidden_dims[i] / dropouts[i] 对应第 i 层；backbone 'gin'（GINEEncoder）
    或 'gat'（GATEncoder，多头注意力含边特征）。n_heads 仅 GAT 使用：
    None → encoder 内默认（每层 4 头）；标量/列表均可，要求
    hidden_dims[i] 能被 n_heads[i] 整除（concat heads）。

    head: 分类头类型，'linear'（单层 Linear，默认）| 'ffn'（两层 MLP）。
    gnn_norm: GNN 层间归一化（注意不含 GINELayer MLP 内部的 BatchNorm），
    None → 各 backbone 原行为（GIN=BatchNorm、GAT=无），
    可显式选 'batchnorm' / 'layernorm' / 'none'。
    readout: 图读出池化（仅 gin backbone），'mean'（默认，AvgPooling）|
    'meanmax'（mean 与 max 拼接，输出维度翻倍；mean/max 均不随节点数缩放，
    全图与揭示子图分布一致）| 'sum'（SumPooling，幅值随节点数变化，
    子图上分布漂移大，慎用）。
    """

    backbone: str = 'gin'
    edge_feat_dim: int = 7
    node_feat_dim: int = 0
    n_classes: int = 37
    hidden_dims: List[int] = field(default_factory=lambda: [300] * 5)
    dropouts: List[float] = field(default_factory=lambda: [0.5] * 5)
    virtual_node: bool = False
    residual: bool = False
    n_heads: Optional[List[int]] = None
    head: str = 'linear'
    gnn_norm: Optional[str] = None
    readout: str = 'mean'

    # ---- 兼容旧标量接口 ----
    @property
    def hidden_dim(self) -> int:
        return self.hidden_dims[0]

    @property
    def n_layers(self) -> int:
        return len(self.hidden_dims)

    @property
    def dropout(self) -> float:
        return self.dropouts[0]

    # ---- 序列化 ----
    def to_dict(self) -> Dict:
        """转 JSON 友好 dict；层间全同时折叠为标量 + n_layers。"""
        d: Dict = {'backbone': self.backbone,
                   'edge_feat_dim': self.edge_feat_dim,
                   'node_feat_dim': self.node_feat_dim,
                   'n_classes': self.n_classes,
                   'virtual_node': self.virtual_node,
                   'residual': self.residual}
        if len(set(self.hidden_dims)) == 1:
            d['hidden_dim'] = self.hidden_dims[0]
            d['n_layers'] = len(self.hidden_dims)
        else:
            d['hidden_dims'] = list(self.hidden_dims)
        if len(set(self.dropouts)) == 1:
            d['dropout'] = self.dropouts[0]
        else:
            d['dropouts'] = list(self.dropouts)
        if self.backbone == 'gat' and self.n_heads is not None:
            if len(set(self.n_heads)) == 1:
                d['n_heads'] = self.n_heads[0]
            else:
                d['n_heads'] = list(self.n_heads)
        if self.head != 'linear':
            d['head'] = self.head
        if self.gnn_norm is not None:
            d['gnn_norm'] = self.gnn_norm
        if self.readout != 'mean':
            d['readout'] = self.readout
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'ClassifierConfig':
        """接受 to_dict 输出或旧版 {hidden_dim, n_layers, dropout} 标量形式。"""
        if 'hidden_dims' in d:
            hidden_dims = list(d['hidden_dims'])
        else:
            hidden_dims = [int(d.get('hidden_dim', 300))] * int(d.get('n_layers', 5))
        if 'dropouts' in d:
            dropouts = list(d['dropouts'])
        else:
            dropouts = [float(d.get('dropout', 0.5))] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError(f"dropouts 长度 {len(dropouts)} != hidden_dims 长度 {len(hidden_dims)}")
        head = d.get('head', 'linear')
        if head not in ('linear', 'ffn'):
            raise ValueError(f"未知分类头 '{head}'，可选: linear, ffn")
        gnn_norm = d.get('gnn_norm')
        if gnn_norm is not None and gnn_norm not in ('batchnorm', 'layernorm', 'none'):
            raise ValueError(
                f"未知层间归一化 '{gnn_norm}'，可选: batchnorm, layernorm, none")
        readout = d.get('readout', 'mean')
        if readout not in ('mean', 'meanmax', 'sum'):
            raise ValueError(f"未知读出池化 '{readout}'，可选: mean, meanmax, sum")
        return ClassifierConfig(
            backbone=d.get('backbone', 'gin'),
            edge_feat_dim=int(d.get('edge_feat_dim', 7)),
            node_feat_dim=int(d.get('node_feat_dim', 0)),
            n_classes=int(d.get('n_classes', 37)),
            hidden_dims=hidden_dims,
            dropouts=dropouts,
            virtual_node=bool(d.get('virtual_node', False)),
            residual=bool(d.get('residual', False)),
            n_heads=(d.get('n_heads') if isinstance(d.get('n_heads'), (list, tuple))
                     else ([int(d['n_heads'])] if 'n_heads' in d else None)),
            head=head,
            gnn_norm=gnn_norm,
            readout=readout,
        )

    @staticmethod
    def from_legacy(edge_feat_dim: int, n_classes: int, hidden_dim: int = 300,
                    n_layers: int = 5, dropout: float = 0.5,
                    virtual_node: bool = False, residual: bool = False,
                    node_feat_dim: int = 0, backbone: str = 'gin',
                    n_heads: Optional[List[int]] = None,
                    head: str = 'linear',
                    gnn_norm: Optional[str] = None,
                    readout: str = 'mean') -> 'ClassifierConfig':
        """旧式标量参数 → 配置（保持 m3 时代调用点一行迁移）。"""
        return ClassifierConfig(
            backbone=backbone,
            edge_feat_dim=edge_feat_dim, n_classes=n_classes,
            node_feat_dim=node_feat_dim, virtual_node=virtual_node,
            residual=residual, n_heads=n_heads, head=head, gnn_norm=gnn_norm,
            readout=readout,
            hidden_dims=[hidden_dim] * n_layers, dropouts=[dropout] * n_layers,
        )

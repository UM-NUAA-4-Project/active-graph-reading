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

    hidden_dims[i] / dropouts[i] 对应第 i 层；backbone 目前只有 'gin'
    （'gat' 为预留骨架）。
    """

    backbone: str = 'gin'
    edge_feat_dim: int = 7
    node_feat_dim: int = 0
    n_classes: int = 37
    hidden_dims: List[int] = field(default_factory=lambda: [300] * 5)
    dropouts: List[float] = field(default_factory=lambda: [0.5] * 5)
    virtual_node: bool = False
    residual: bool = False

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
        return ClassifierConfig(
            backbone=d.get('backbone', 'gin'),
            edge_feat_dim=int(d.get('edge_feat_dim', 7)),
            node_feat_dim=int(d.get('node_feat_dim', 0)),
            n_classes=int(d.get('n_classes', 37)),
            hidden_dims=hidden_dims,
            dropouts=dropouts,
            virtual_node=bool(d.get('virtual_node', False)),
            residual=bool(d.get('residual', False)),
        )

    @staticmethod
    def from_legacy(edge_feat_dim: int, n_classes: int, hidden_dim: int = 300,
                    n_layers: int = 5, dropout: float = 0.5,
                    virtual_node: bool = False, residual: bool = False,
                    node_feat_dim: int = 0) -> 'ClassifierConfig':
        """旧式标量参数 → 配置（保持 m3 时代调用点一行迁移）。"""
        return ClassifierConfig(
            edge_feat_dim=edge_feat_dim, n_classes=n_classes,
            node_feat_dim=node_feat_dim, virtual_node=virtual_node,
            residual=residual,
            hidden_dims=[hidden_dim] * n_layers, dropouts=[dropout] * n_layers,
        )

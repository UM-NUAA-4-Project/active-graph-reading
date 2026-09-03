"""classifier 的 backbone 层：gin（GINEEncoder）、gat（GATEncoder）。"""

from .gin import GINEEncoder, GINELayer
from .gat import GATEncoder, GATLayer

# backbone 名称 → encoder 类；新 backbone 在此注册
BACKBONES = {
    'gin': GINEEncoder,
    'gat': GATEncoder,
}


def get_backbone(name: str):
    if name not in BACKBONES:
        raise ValueError(f"未知 backbone '{name}'，可用: {sorted(BACKBONES)}")
    return BACKBONES[name]


__all__ = ['GINEEncoder', 'GINELayer', 'GATEncoder', 'GATLayer',
           'BACKBONES', 'get_backbone']

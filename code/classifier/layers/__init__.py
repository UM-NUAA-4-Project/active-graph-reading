"""classifier 的 backbone 层：gin（完整）、gat（占位）。"""

from .gin import GINEEncoder, GINELayer

# backbone 名称 → encoder 类；新 backbone 在此注册
BACKBONES = {
    'gin': GINEEncoder,
}


def get_backbone(name: str):
    if name not in BACKBONES:
        raise ValueError(f"未知 backbone '{name}'，可用: {sorted(BACKBONES)}")
    return BACKBONES[name]


__all__ = ['GINEncoder', 'GINELayer', 'BACKBONES', 'get_backbone']

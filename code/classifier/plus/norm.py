"""GNN+ 组件②归一化：层间 norm 工厂（区别于 GINELayer MLP 内部的 BatchNorm）。

'none' 返回 nn.Identity——无参数、无 state_dict 键，因此各 encoder
在默认档位下对旧 checkpoint 的 state_dict 键集零影响。
（自 layers/norm.py 移入 plus/；layers/ 内不再保留副本。）
"""

import torch.nn as nn

NORM_TYPES = ('batchnorm', 'layernorm', 'none')


def make_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == 'batchnorm':
        return nn.BatchNorm1d(dim)
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    if norm_type == 'none':
        return nn.Identity()
    raise ValueError(f"未知归一化类型 '{norm_type}'，可选: {NORM_TYPES}")

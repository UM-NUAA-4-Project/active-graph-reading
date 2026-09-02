"""GAT backbone 骨架（备用扩展点）。

当前未实现前向逻辑；注册在 backbone 注册表中，选择 'gat' 时给出明确报错。
实现时应保持与 GINEEncoder 相同的接口：
    __init__(edge_feat_dim, hidden_dims, dropouts, node_feat_dim, **kw)
    forward(g, node_ids) -> (num_graphs, hidden_dims[-1])
"""

import torch

from .gin import GINEEncoder  # noqa: F401  (GAT 实现时可参考其接口)


class GATEncoder(torch.nn.Module):
    """多头图注意力 encoder（未实现，占位）。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "GAT backbone 尚未实现：请在 classifier/layers/gat.py 中补全前向逻辑，"
            "并保持与 GINEEncoder 相同的构造/前向接口")

    def forward(self, g, node_ids):  # pragma: no cover
        raise NotImplementedError

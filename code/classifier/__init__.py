"""独立分类器包：backbone（GIN/GAT，layers/ 注册表接入）+ GNN+ 积木
（plus/，ICML 2025 arXiv 2502.09263 六组件参数化插拔）+ 分类头 +
config/factory。

与 active_graph_reader 解耦：本包不 import reader 代码；
reader 侧需要 GINELayer 时从 classifier.layers 导入。
"""

from .config import ClassifierConfig
from .graph_classifier import GraphClassifier
from .layers import (BACKBONES, BACKBONE_KWARGS, GATEncoder, GATLayer,
                     GINEEncoder, GINELayer, backbone_init_kwargs,
                     get_backbone)
from .factory import (
    build_classifier,
    load_classifier,
    load_fclass,
    classifier_config_from_ckpt,
)

__all__ = [
    'ClassifierConfig', 'GraphClassifier',
    'GINEEncoder', 'GINELayer', 'GATEncoder', 'GATLayer',
    'BACKBONES', 'BACKBONE_KWARGS', 'backbone_init_kwargs', 'get_backbone',
    'build_classifier', 'load_classifier', 'load_fclass',
    'classifier_config_from_ckpt',
]

"""独立分类器包：backbone（GIN/GAT）+ 分类头 + config/factory。

与 active_graph_reader 解耦：本包不 import reader 代码；
reader 侧需要 GINELayer 时从 classifier.layers 导入。
"""

from .config import ClassifierConfig
from .gin_classifier import GraphClassifier
from .layers import (GINEEncoder, GINELayer, GATEncoder, GATLayer,
                     BACKBONES, get_backbone)
from .factory import (
    build_classifier,
    load_classifier,
    load_fclass,
    classifier_config_from_ckpt,
)

__all__ = [
    'ClassifierConfig', 'GraphClassifier',
    'GINEEncoder', 'GINELayer', 'GATEncoder', 'GATLayer',
    'BACKBONES', 'get_backbone',
    'build_classifier', 'load_classifier', 'load_fclass',
    'classifier_config_from_ckpt',
]

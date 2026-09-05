"""兼容 shim：GraphClassifier 已迁至独立 classifier 包（reader/分类器解耦）。

旧代码 `from active_graph_reader.gnn.classifier import ...` 继续可用；
新代码请直接 `from classifier import GraphClassifier, load_classifier`。
"""

from classifier.graph_classifier import GraphClassifier  # noqa: F401
from classifier.factory import load_fclass, load_classifier  # noqa: F401
from classifier.config import ClassifierConfig  # noqa: F401

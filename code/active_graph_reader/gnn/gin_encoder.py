"""兼容 shim：GINEEncoder/GINELayer 已迁至 classifier 包（reader/分类器解耦）。

旧代码 `from active_graph_reader.gnn.gin_encoder import ...` 继续可用；
新代码请直接 `from classifier.layers.gin import ...`。
"""

from classifier.layers.gin import GINEEncoder, GINELayer  # noqa: F401

"""GNN backbone — GIN encoder, classifier, subgraph builders, and frontier encoding."""

from .subgraph_builder import SubgraphBuilder
from .gin_encoder import GINEEncoder, GINELayer
from .classifier import GraphClassifier
from .frontier import FrontierGraphBuilder, FrontierSubgraph, FrontierNodesEncoder

__all__ = [
    'SubgraphBuilder',
    'GINEEncoder',
    'GINELayer',
    'GraphClassifier',
    'FrontierGraphBuilder',
    'FrontierSubgraph',
    'FrontierNodesEncoder',
]

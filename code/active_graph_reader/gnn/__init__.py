"""GNN backbone — GIN encoder, classifier, subgraph builders, and frontier encoding."""

from .subgraph_builder import SubgraphBuilder
from .gin_encoder import GINEEncoder, GINELayer
from .classifier import GraphClassifier
from .subgraph_encoder import SubgraphEncoder, build_g_phi_from_fclass
from .frontier import FrontierGraphBuilder, FrontierSubgraph, FrontierNodesEncoder

__all__ = [
    'SubgraphBuilder',
    'GINEEncoder',
    'GINELayer',
    'GraphClassifier',
    'SubgraphEncoder',
    'build_g_phi_from_fclass',
    'FrontierGraphBuilder',
    'FrontierSubgraph',
    'FrontierNodesEncoder',
]

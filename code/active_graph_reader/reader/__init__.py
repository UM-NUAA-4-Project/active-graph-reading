"""Reader module — unified interface for fixed and learnable graph readers."""

from ..core.base_reader import BaseReader
from .learnable_reader import LearnableReader
from .prototype_reader import PrototypeReader
from ..gnn.frontier.graph_builder import FrontierGraphBuilder, FrontierSubgraph
from ..gnn.frontier.encoder import FrontierNodesEncoder
from ..engine.reveal import reveal_one, reveal_batch
from .fixed import create_fixed_reader

from .reader_mlp import ReaderMLP
from .gsae import compute_structural_features

# Backward-compat alias
FixedStrategyReader = create_fixed_reader

__all__ = [
    'BaseReader',
    'FixedStrategyReader',
    'LearnableReader',
    'PrototypeReader',
    'FrontierGraphBuilder',
    'FrontierSubgraph',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
    'create_fixed_reader',
    'ReaderMLP',
    'compute_structural_features',
]

"""Reader module — unified interface for fixed and learnable graph readers."""

from ..core.base_reader import BaseReader
from .learnable_reader import LearnableReader
from ..gnn.frontier.graph_builder import FrontierGraphBuilder, FrontierSubgraph
from ..gnn.frontier.encoder import FrontierNodesEncoder
from ..engine.reveal import reveal_one, reveal_batch
from .fixed import create_fixed_reader

from .reader_mlp import ReaderMLP
from .gsae import GSAE, GSAEOutput
from .evidence_network import EvidenceNetwork, EvidenceOutput
from .belief_updater import SimpleDirichletAccumulator
from .final_classifier import FinalClassifier, FinalOutput
from .pomdp_reader import POMDPReader

# Backward-compat alias
FixedStrategyReader = create_fixed_reader

__all__ = [
    'BaseReader',
    'FixedStrategyReader',
    'LearnableReader',
    'FrontierGraphBuilder',
    'FrontierSubgraph',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
    'create_fixed_reader',
    'ReaderMLP',
    'GSAE', 'GSAEOutput',
    'EvidenceNetwork', 'EvidenceOutput',
    'SimpleDirichletAccumulator',
    'FinalClassifier', 'FinalOutput',
    'POMDPReader',
]

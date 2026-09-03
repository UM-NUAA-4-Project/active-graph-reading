"""GraphClassifier：backbone（GINEEncoder 等）+ Linear 分类头。

自 active_graph_reader/gnn/classifier.py 迁出，与 reader 包完全解耦。
保持旧构造签名（hidden_dim/n_layers/dropout 标量）与 forward(g, node_ids)
协议不变；新增 config=ClassifierConfig 支持逐层结构。
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .config import ClassifierConfig
from .layers import get_backbone

logger = logging.getLogger(__name__)


class GraphClassifier(nn.Module):
    """Graph-level classifier = backbone encoder + Linear head.

    两种构造方式：
    - 旧式标量：GraphClassifier(edge_feat_dim, n_classes, hidden_dim, n_layers, dropout, ...)
    - 新式配置：GraphClassifier(config=ClassifierConfig(...))（config 优先）
    """

    def __init__(
        self,
        edge_feat_dim: int = 7,
        n_classes: int = 37,
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
        virtual_node: bool = False,
        residual: bool = False,
        node_feat_dim: int = 0,
        config: Optional[ClassifierConfig] = None,
    ):
        super().__init__()

        if config is not None:
            self.config = config
        else:
            self.config = ClassifierConfig.from_legacy(
                edge_feat_dim=edge_feat_dim, n_classes=n_classes,
                hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
                virtual_node=virtual_node, residual=residual,
                node_feat_dim=node_feat_dim)

        backbone_cls = get_backbone(self.config.backbone)
        backbone_kwargs = dict(
            edge_feat_dim=self.config.edge_feat_dim,
            hidden_dims=self.config.hidden_dims,
            dropouts=self.config.dropouts,
            virtual_node=self.config.virtual_node,
            residual=self.config.residual,
            node_feat_dim=self.config.node_feat_dim,
        )
        if self.config.backbone == 'gat':
            backbone_kwargs['n_heads'] = self.config.n_heads
        self.encoder = backbone_cls(**backbone_kwargs)

        self.head = nn.Linear(self.config.hidden_dims[-1], self.config.n_classes)

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_pretrained(self, path: str, device: str = 'cpu') -> Dict[str, Any]:
        """Load pretrained weights from checkpoint (needs 'model_state_dict').

        Returns the checkpoint metadata dict.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        missing, unexpected = self.load_state_dict(ckpt['model_state_dict'], strict=True)
        if missing:
            logger.warning("Missing keys in pretrained weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in pretrained weights: %s", unexpected)
        logger.info("Loaded pretrained weights from %s (val_acc=%.4f)",
                    path, ckpt.get('val_acc', 0.0))
        return ckpt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, g, node_ids: torch.Tensor) -> torch.Tensor:
        g_emb = self.encoder(g, node_ids)
        return self.head(g_emb)

    def get_embedding(self, g, node_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(g, node_ids)

"""GraphClassifier：backbone（GINEEncoder/GATEncoder，经注册表接入）+
GNN+ 积木插拔 + 分类头（linear/ffn）。

backbone 只需实现 encoder 协议（构造 kwargs 见 layers.BACKBONE_KWARGS
白名单、out_dim 属性、forward(g, node_ids)）即可接入，本类不含任何
backbone 特例分支——积木 kwargs 由 layers.backbone_init_kwargs 按白名单
过滤，配置开启了白名单未声明的积木时大声报错。

保持旧构造签名（hidden_dim/n_layers/dropout 标量）与 forward(g, node_ids)
协议不变；config=ClassifierConfig 优先。
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .config import ClassifierConfig
from .layers import BACKBONE_KWARGS, backbone_init_kwargs, get_backbone

logger = logging.getLogger(__name__)


class GraphClassifier(nn.Module):
    """Graph-level classifier = backbone encoder + readout + Linear/FFN head.

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
        supported = BACKBONE_KWARGS[self.config.backbone]
        # 开启了白名单未声明的积木 → 报错（不静默忽略）
        enabled = (
            ('readout', self.config.readout != 'mean'),
            ('edge_encoder_kind', self.config.edge_encoder != 'linear'),
            ('ffn', self.config.ffn),
            ('pos_enc', self.config.pos_enc != 'none'),
        )
        for block, on in enabled:
            if on and block not in supported:
                raise ValueError(
                    f"backbone '{self.config.backbone}' 不支持 GNN+ 积木 "
                    f"'{block}'（layers.BACKBONE_KWARGS 未声明）")
        self.encoder = backbone_cls(
            **backbone_init_kwargs(self.config.backbone, self.config))

        # meanmax 等读出会改变 encoder 输出维度（encoder.out_dim）；
        # 旧 encoder 无该属性时退回末层宽度
        embed_dim = getattr(self.encoder, 'out_dim', self.config.hidden_dims[-1])
        if self.config.head == 'ffn':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.config.n_classes),
            )
        else:
            self.head = nn.Linear(embed_dim, self.config.n_classes)

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

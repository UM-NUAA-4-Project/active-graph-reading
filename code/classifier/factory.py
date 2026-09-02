"""分类器工厂：config 构建 + 从 checkpoint 重建（消灭下游硬编码维度）。

重建优先级：结构（逐层宽度/边特征维度/类别数/node_feat_dim）一律以
state_dict 权重 shape 为准（m7 曾出现 config 记录错误导致加载失败的
先例，权重 shape 永远是结构事实）；dropout 等无法从权重推断的超参，
若 checkpoint 内嵌 config 有记录则采用，否则用默认值。
"""

import logging
import re
from typing import Dict, Optional, Tuple, Union

import torch

from .config import ClassifierConfig
from .gin_classifier import GraphClassifier

logger = logging.getLogger(__name__)

_CONV_RE = re.compile(r'^encoder\.convs\.(\d+)\.mlp\.0\.weight$')


def infer_classifier_config(state_dict: Dict[str, torch.Tensor]) -> ClassifierConfig:
    """从 state_dict 权重 shape 推断分类器结构（旧 m3 checkpoint 无 config 时用）。"""
    conv_layers = {}
    for key, tensor in state_dict.items():
        m = _CONV_RE.match(key)
        if m:
            # mlp.0: Linear(hidden, hidden*2) → bottleneck = 2 * hidden
            conv_layers[int(m.group(1))] = tensor.shape[0] // 2
    if not conv_layers:
        raise ValueError("state_dict 中未找到 encoder.convs.* 权重，无法推断结构")
    hidden_dims = [conv_layers[i] for i in sorted(conv_layers)]

    if 'encoder.node_proj.weight' in state_dict:
        node_feat_dim = int(state_dict['encoder.node_proj.weight'].shape[1])
    else:
        node_feat_dim = 0

    edge_feat_dim = int(state_dict['encoder.convs.0.edge_encoder.weight'].shape[1])
    n_classes = int(state_dict['head.weight'].shape[0])

    return ClassifierConfig(
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        dropouts=[0.5] * len(hidden_dims),  # dropout 无法从权重推断
    )


def classifier_config_from_ckpt(ckpt: dict, state_dict: Optional[dict] = None
                                ) -> ClassifierConfig:
    """checkpoint → ClassifierConfig：结构取自权重 shape，dropout 取内嵌 config（如有）。"""
    sd = state_dict if state_dict is not None else ckpt.get('model_state_dict', {})
    cfg = infer_classifier_config(sd)
    embedded = ckpt.get('config')
    if isinstance(embedded, dict) and 'dropout' in embedded:
        cfg.dropouts = [float(embedded['dropout'])] * len(cfg.hidden_dims)
    return cfg


def build_classifier(config: ClassifierConfig) -> GraphClassifier:
    """ClassifierConfig → 未训练模型。"""
    return GraphClassifier(config=config)


def load_classifier(
    ckpt_path: str,
    device: Union[str, torch.device] = 'cpu',
    freeze: bool = True,
    strict: bool = True,
) -> GraphClassifier:
    """从 checkpoint 加载分类器。

    freeze=True（默认）：eval 模式 + 参数不回传 —— 冻结奖励/参照分类器 f_class。
    freeze=False：供 train_classifier --init-checkpoint 继续训练。
    兼容无 config 的旧 m3 checkpoint（自动从权重推断结构）。
    """
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict'))
    if sd is None:
        raise KeyError(f"{ckpt_path} 中未找到 model_state_dict/state_dict")

    model = GraphClassifier(config=classifier_config_from_ckpt(ckpt, sd)).to(device)
    model.load_state_dict(sd, strict=strict)
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    logger.info("classifier loaded from %s (val_acc=%s, freeze=%s)",
                ckpt_path, ckpt.get('val_acc', float('nan')), freeze)
    return model


# 兼容旧名（m5/m6/m7 脚本中的 load_fclass）
def load_fclass(ckpt_path: str, device: Union[str, torch.device] = 'cpu',
                **_ignored) -> GraphClassifier:
    return load_classifier(ckpt_path, device=device, freeze=True)

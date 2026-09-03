"""分类器工厂：config 构建 + 从 checkpoint 重建。

重建一律以 checkpoint 内嵌 config 为准（结构与超参同一来源）；
`load_state_dict(strict=True)` 的 shape 校验免费充当 config 与权重的
一致性验证——不匹配即大声报错，不做静默纠正（2026-09-03 起，
废弃旧"从权重 shape 反推结构"路线；遗留无 config 的 m3/m5 checkpoint
用 tools/migrate_legacy_classifier_ckpt.py 一次性补写 config）。
"""

import logging
from typing import Union

import torch

from .config import ClassifierConfig
from .gin_classifier import GraphClassifier

logger = logging.getLogger(__name__)


def classifier_config_from_ckpt(ckpt: dict) -> ClassifierConfig:
    """checkpoint 内嵌 config → ClassifierConfig。

    支持两种内嵌形式：新版 {'classifier': {...}, ...} 与旧版直接
    config dict。无内嵌 config 时报错并指向迁移脚本。
    """
    embedded = ckpt.get('config')
    if isinstance(embedded, dict) and isinstance(embedded.get('classifier'), dict):
        embedded = embedded['classifier']
    if not isinstance(embedded, dict) or not embedded:
        raise ValueError(
            "checkpoint 无内嵌 classifier config，无法重建模型；"
            "旧 checkpoint 请先运行 "
            "python code/tools/migrate_legacy_classifier_ckpt.py <ckpt.pt> --write")
    return ClassifierConfig.from_dict(embedded)


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

    config 与权重不符时（shape 不匹配）抛 RuntimeError——这是对
    "config 记错结构" 的唯一防线，不做任何自动纠正。
    """
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict'))
    if sd is None:
        raise KeyError(f"{ckpt_path} 中未找到 model_state_dict/state_dict")

    model = GraphClassifier(config=classifier_config_from_ckpt(ckpt)).to(device)
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"{ckpt_path}: 权重与内嵌 config 结构不符（{e}）。"
            f"结构以 config 为准；若该 checkpoint 来自旧版本或被手改，"
            f"请重新训练或用 migrate_legacy_classifier_ckpt.py 重建 config") from e
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

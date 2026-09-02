"""模型构建/重建：从 checkpoint 权重与内嵌 config 重建 reader 与分类器。

收敛 m5/m6/m7 各脚本手写的 build_reader/build_classifier/加载块：
- 结构一律以 checkpoint 权重 shape 为准（内嵌 config 只补充非结构超参），
  老 checkpoint（无 config / v0-v1 mlp 宽度 / belief classifier 键）照常加载；
- 分类器加载委托 classifier.factory（消灭 6+ 处硬编码 7/37/300/5/0.5）。
"""

import os
import torch

from active_graph_reader import build_reader as _build_reader
from active_graph_reader.config import reader_config_from_checkpoint
from classifier import GraphClassifier, classifier_config_from_ckpt
from tools.ckpt_io import pad_reader_mlp_input, strip_classifier_keys


def load_classifier_frozen(ckpt_or_path, device, requires_grad: bool = False):
    """加载上界分类器（eval 模式）。

    requires_grad=False：冻结（RL reward / 参照）。
    requires_grad=True：保持 autograd 图（gradient teacher 轨迹生成需要）。
    """
    ckpt = ckpt_or_path
    if isinstance(ckpt_or_path, (str, os.PathLike)):
        ckpt = torch.load(str(ckpt_or_path), map_location=device, weights_only=False)
    clf = GraphClassifier(config=classifier_config_from_ckpt(ckpt)).to(device)
    clf.load_state_dict(ckpt['model_state_dict'], strict=True)
    clf.eval()
    if not requires_grad:
        for p in clf.parameters():
            p.requires_grad = False
    return clf


def load_reader_from_checkpoint(ckpt_or_path, device, stochastic: bool = None,
                                temperature: float = None, strict: bool = True):
    """从 checkpoint 重建 reader（结构：权重 shape > 内嵌 config 推断合并）。

    stochastic/temperature：评估/RL 语义覆盖（None = 保持 config 值）。
    兼容旧 checkpoint：strip belief classifier 键 + mlp 输入维度补零。
    """
    ckpt = ckpt_or_path
    if isinstance(ckpt_or_path, (str, os.PathLike)):
        ckpt = torch.load(str(ckpt_or_path), map_location='cpu', weights_only=False)
    sd = ckpt.get('reader_state_dict', ckpt.get('state_dict'))
    if sd is None:
        raise KeyError("checkpoint 中未找到 reader_state_dict/state_dict")
    sd = strip_classifier_keys(dict(sd))
    cfg = reader_config_from_checkpoint(ckpt, sd)
    if stochastic is not None:
        cfg.stochastic = bool(stochastic)
    if temperature is not None:
        cfg.temperature = float(temperature)
    reader = _build_reader(cfg).to(device)
    if cfg.kind == 'base':
        target_dim = reader.reader_mlp.mlp[0].in_features
        sd, _ = pad_reader_mlp_input(sd, target_dim)
    reader.load_state_dict(sd, strict=strict)
    return reader


def bootstrap_reader(src_ckpt, device, stochastic: bool = True,
                     temperature: float = None):
    """迁移起点：按源 checkpoint 结构建 reader + 部分权重迁移。

    Returns: (reader, (n_transferred, n_padded, skipped))
    """
    from tools.ckpt_io import transfer_weights
    sd = src_ckpt.get('reader_state_dict', src_ckpt.get('state_dict'))
    if sd is None:
        raise KeyError("bootstrap checkpoint 中未找到 reader_state_dict")
    sd = strip_classifier_keys(dict(sd))
    cfg = reader_config_from_checkpoint(src_ckpt, sd)
    if stochastic:
        cfg.stochastic = True
    if temperature is not None:
        cfg.temperature = float(temperature)
    reader = _build_reader(cfg).to(device)
    stats = transfer_weights(sd, reader)
    return reader, stats

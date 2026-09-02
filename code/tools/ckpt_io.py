"""checkpoint IO：统一的每轮保存、训练状态恢复、旧 checkpoint 兼容。

checkpoint 统一格式（所有新实验）：
    {epoch, reader_state_dict, optimizer_state_dict, config, training_log}
  - config：与 <output>/config.json 完全一致的 effective config（含 reader
    逐层结构），resume/eval 据此重建模型；
  - training_log：list[{epoch, ...metrics}]（按 epoch 追加）。

旧格式兼容：
  - m5 RL checkpoint 无 config → reader_config_from_checkpoint 从权重推断；
  - v0/v1 reader_mlp 输入宽度（128/129）→ pad_reader_mlp_input 补零到 132；
  - m6 早期 checkpoint 含 belief_state.classifier.* → strip_classifier_keys。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# reader_mlp 第一层输入宽度的历史版本（v2 = embeds + budget + struct×3）
LEGACY_MLP_INPUT_DIMS = (128, 129)
CURRENT_MLP_INPUT_EXTRA = 4          # budget(1) + struct(3)


def strip_classifier_keys(state_dict: dict) -> dict:
    """去掉早期 belief reader checkpoint 的 belief_state.classifier.* 键。"""
    return {k: v for k, v in state_dict.items()
            if not k.startswith('belief_state.classifier.')}


def pad_reader_mlp_input(state_dict: dict, target_dim: int = 132) -> Tuple[dict, bool]:
    """旧 v0/v1 checkpoint 的 reader_mlp.mlp.0.weight 输入维度补零到 target_dim。

    缺的列零初始化（对应 budget/struct 特征权重从零学习）。
    返回 (state_dict, was_patched)。语义与 m5 patch_old_checkpoint 一致。
    """
    was_patched = False
    key = 'reader_mlp.mlp.0.weight'
    if key in state_dict:
        w = state_dict[key]
        if w.dim() == 2 and w.shape[1] < target_dim:
            gap = target_dim - w.shape[1]
            pad = torch.zeros(w.shape[0], gap, dtype=w.dtype, device=w.device)
            state_dict[key] = torch.cat([w, pad], dim=1)
            was_patched = True
    return state_dict, was_patched


def transfer_weights(source_state_dict: dict,
                     target_reader: nn.Module) -> Tuple[int, int, List[str]]:
    """通用部分权重迁移（自 m7/transfer.py 收编）。

    同键同形 → 复制；仅最后一维更窄 → 零填充后复制；其余 → 保留随机初始化。
    Returns: (n_transferred, n_padded, skipped_keys)
    """
    target_sd = target_reader.state_dict()
    n_transferred = 0
    n_padded = 0
    skipped: List[str] = []

    for k in target_sd:
        if k not in source_state_dict:
            skipped.append(k)
            continue
        src_w = source_state_dict[k]
        tgt_w = target_sd[k]
        if src_w.shape == tgt_w.shape:
            target_sd[k] = src_w.clone()
            n_transferred += 1
        elif (src_w.dim() >= 1
              and src_w.dim() == tgt_w.dim()
              and src_w.shape[:-1] == tgt_w.shape[:-1]
              and tgt_w.shape[-1] > src_w.shape[-1]):
            gap = tgt_w.shape[-1] - src_w.shape[-1]
            pad = torch.zeros(src_w.shape[:-1] + (gap,),
                              dtype=src_w.dtype, device=src_w.device)
            target_sd[k] = torch.cat([src_w, pad], dim=-1)
            n_padded += 1
        else:
            skipped.append(k)

    target_reader.load_state_dict(target_sd, strict=True)
    return n_transferred, n_padded, skipped


def save_epoch_checkpoint(
    path,
    epoch: int,
    reader,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict,
    training_log: List[Dict],
) -> None:
    """统一每轮 checkpoint 保存（收敛此前 6 份复制）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'reader_state_dict': reader.state_dict(),
        'config': config,
        'training_log': training_log,
    }
    if optimizer is not None:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(ckpt, str(path))


def load_training_state(ckpt: dict) -> Tuple[int, List[Dict]]:
    """从 checkpoint 恢复 (start_epoch, training_log)。

    兼容历史坑：部分旧 checkpoint 把超参 dict 存进了 training_log 字段。
    """
    start_epoch = int(ckpt.get('epoch', -1)) + 1
    tl = ckpt.get('training_log', [])
    if isinstance(tl, dict):        # 旧 checkpoint 把 config 存在这里
        tl = []
    return start_epoch, list(tl)


def load_optimizer(optimizer: Optional[torch.optim.Optimizer],
                   ckpt: dict, device: str = 'cpu') -> bool:
    """尽力恢复 optimizer 状态；失败（结构变化/缺键）则保持新初始化。"""
    sd = ckpt.get('optimizer_state_dict')
    if optimizer is None or not sd:
        return False
    try:
        optimizer.load_state_dict(sd)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("optimizer state 恢复失败（使用新初始化）: %s", e)
        return False

"""部分权重迁移：PPA checkpoint → 目标数据集 reader 架构。

机制：遍历 target 的 state_dict 键：
  - 源存在同键且形状相同 → 复制（transferred）；
  - 形状一致但最后一维更窄（如 legacy v0 reader_mlp.mlp.0.weight 128→132，
    缺失的是 budget/struct 列）→ 源沿最后一维零填充到目标宽度再复制（padded，
    对齐 m5/training_utils.patch_old_checkpoint 语义，但按实际宽度差通用适配）；
  - 其余（维度不匹配的 edge_encoder、新增的 node_feat_proj 等）→ 保留随机初始化（skipped）。

零样本评估与 RL 微调共用此函数；BC/RL 起点通过 --bootstrap 指向任意 checkpoint。
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def transfer_weights(source_state_dict: dict,
                     target_reader: nn.Module) -> Tuple[int, int, List[str]]:
    """迁移源 state_dict 到 target reader。

    Returns:
        (n_transferred, n_padded, skipped)：
        n_transferred — 形状完全相同、直接复制的参数数；
        n_padded      — 最后一维零填充后复制的参数数；
        skipped       — 形状不匹配被跳过（保留随机初始化）的参数键列表。
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
            # 仅最后一维更窄：零填充到目标宽度（旧 checkpoint 缺
            # budget/struct 特征列，对应特征权重从零开始学习）
            gap = tgt_w.shape[-1] - src_w.shape[-1]
            pad = torch.zeros(src_w.shape[:-1] + (gap,),
                              dtype=src_w.dtype, device=src_w.device)
            target_sd[k] = torch.cat([src_w, pad], dim=-1)
            n_padded += 1
        else:
            skipped.append(k)

    target_reader.load_state_dict(target_sd, strict=True)
    return n_transferred, n_padded, skipped

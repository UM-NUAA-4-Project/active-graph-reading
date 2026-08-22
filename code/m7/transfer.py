"""部分权重迁移：PPA checkpoint → 目标数据集 reader 架构。

机制：遍历 target 的 state_dict 键，源存在同键且形状相同 → 复制；
否则（维度不匹配的 edge_encoder、新增的 node_feat_proj 等）保留随机初始化。

零样本评估与 RL 微调共用此函数；BC/RL 起点通过 --bootstrap 指向任意 checkpoint。
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def transfer_weights(source_state_dict: dict,
                     target_reader: nn.Module) -> Tuple[int, List[str]]:
    target_sd = target_reader.state_dict()
    n_transferred = 0
    skipped: List[str] = []

    for k in target_sd:
        if k in source_state_dict and \
                source_state_dict[k].shape == target_sd[k].shape:
            target_sd[k] = source_state_dict[k].clone()
            n_transferred += 1
        else:
            skipped.append(k)

    target_reader.load_state_dict(target_sd, strict=True)
    return n_transferred, skipped

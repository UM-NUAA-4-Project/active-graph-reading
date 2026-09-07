"""类别权重计算（分类器训练 --class-weight 开关）。

动机：collab 等数据集多数类占比 >0.5（2080/620/1300），无权重 CE 下
分类器退化为"预测多数类+少量其他类正确"；'balanced' 模式用 sklearn
约定 w_c = N / (K × count_c) 把各类总权重拉平。

mode='none' 返回 None——调用方不向 cross_entropy 传 weight，
与开关引入前的行为逐位一致（默认值）。
"""

from typing import Optional

import torch


CLASS_WEIGHT_MODES = ('none', 'balanced')


def compute_class_weight(mode: str, labels: torch.Tensor, n_classes: int,
                         device=None) -> Optional[torch.Tensor]:
    """按 mode 从训练标签计算每类权重；mode='none' → None（不加权）。

    labels: (N,) 整型类别张量。零样本类权重置 1.0 并警告（无训练样本，
    权重不参与任何 loss 项）。mode != 'none' 时打印 count/权重便于核对。
    """
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(
            f"未知 class_weight 模式 '{mode}'，可选: {CLASS_WEIGHT_MODES}")
    if mode == 'none':
        return None

    counts = torch.bincount(labels.long(), minlength=n_classes)
    weights = torch.ones(n_classes, dtype=torch.float32)
    n_total = int(counts.sum())
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = n_total / (n_classes * counts[c])
        else:
            print(f"  WARNING: 类别 {c} 在训练集内 0 样本，权重置 1.0")
    print(f"  class_weight='balanced': counts={counts.tolist()} → "
          f"weights={[round(w, 4) for w in weights.tolist()]}")
    if device is not None:
        weights = weights.to(device)
    return weights

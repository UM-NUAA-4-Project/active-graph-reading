"""图读出（graph readout）池化积木：encoder 末层节点表示 → 图表示。

- 'mean'：AvgPooling（默认档；键集与旧 checkpoint 一致——池化无参数）；
- 'meanmax'：mean 与 max 拼接，out_dim 翻倍。mean/per-dim max 都不随节点数
  缩放，因此全图训练 → 小揭示子图评估的读出幅值分布一致（子图鲁棒；
  这也是不提供会随节点数线性放大的 'sum' 用于子图场景的原因，'sum'
  仅保留给全图实验）；
- 'sum'：SumPooling。

GNN+ 论文（arXiv 2502.09263）在 CIFAR10-SP/OGB 上用 mean、ZINC 上用
sum、MalNet-Tiny 上用 max——单池化；'meanmax' 是本仓库面向子图鲁棒性
的组合变体。
"""

import dgl.nn as dglnn

READOUT_KINDS = ('mean', 'meanmax', 'sum')
# 各档位读出输出维度相对末层宽度的倍数
READOUT_OUT_MULT = {'mean': 1, 'meanmax': 2, 'sum': 1}


def make_readout(kind: str):
    """kind → (池化模块元组, out_mult)。meanmax 返回 (avg, max) 两个池化。"""
    if kind == 'mean':
        return (dglnn.AvgPooling(),), 1
    if kind == 'meanmax':
        return (dglnn.AvgPooling(), dglnn.MaxPooling()), 2
    if kind == 'sum':
        return (dglnn.SumPooling(),), 1
    raise ValueError(f"未知读出池化 '{kind}'，可选: {READOUT_KINDS}")

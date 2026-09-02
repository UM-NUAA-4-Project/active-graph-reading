"""种子工具：确定性种子公式（唯一实现）+ 全局随机种子设置。

种子公式 (graph_idx*31 + seed_i*17 + 7) % n_nodes 是协议的一部分，
此前散落 8 份拷贝，全部收敛到这里。
"""

import random

import numpy as np
import torch

SEED_METHOD_STR = 'deterministic formula: (idx*31 + seed_i*17 + 7) % n_nodes'


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    """协议规定的确定性种子节点选择。"""
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


def set_all_seeds(seed: int) -> None:
    """python / numpy / torch / cuda 随机种子一次设齐。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

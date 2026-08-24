"""generate_trajectories 纯逻辑单测（不依赖 CIFAR10 下载/GPU/分类器）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch
import dgl

from m7.generate_trajectories import ratio_key, _GraphList


def test_ratio_key():
    assert ratio_key(0.05) == '0_050'
    assert ratio_key(0.10) == '0_100'
    assert ratio_key(0.15) == '0_150'


def test_graph_list_wrapper():
    g = dgl.graph(([0, 1], [1, 0]))
    gl = _GraphList([g, g], torch.tensor([3, 7]))
    assert len(gl) == 2
    gg, lbl = gl[1]
    assert int(lbl) == 7
    assert gg.num_nodes() == 2

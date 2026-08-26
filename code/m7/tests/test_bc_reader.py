"""build_bc_reader 单测（reader 构造分派，无数据依赖）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/
# 移除 code/m7（含 pytest 下 cwd 的空串条目）：否则 dgl→torchdata 的
# `import datasets` 会解析到 code/m7/datasets（CLAUDE.md 记录的循环导入坑）
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(Path(__file__).resolve().parent.parent)]

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader.attention.attention_reader import AttentionReader

from m7.train_bc import build_bc_reader


def _spec():
    return SimpleNamespace(edge_feat_dim=7, node_feat_dim=0)


def _args():
    return SimpleNamespace(hidden_dim=32, n_layers_reader=2, dropout_reader=0.1,
                           m=4, n_heads=4)


def test_build_reader_base():
    reader = build_bc_reader('base', _spec(), _args())
    assert isinstance(reader, LearnableReader)
    assert reader.frontier_encoder.hidden_dim == 32


def test_build_reader_attention():
    reader = build_bc_reader('attention', _spec(), _args())
    assert isinstance(reader, AttentionReader)
    assert reader.frontier_encoder.hidden_dim == 32
    assert reader.belief_state.m == 4

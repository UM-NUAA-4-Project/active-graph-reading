"""transfer_weights 单测：PPA 架构 → CIFAR10 架构的部分权重迁移。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader.attention.attention_reader import AttentionReader
from m7.transfer import transfer_weights


def test_base_reader_transfer():
    # PPA 架构（无节点特征，7 维边特征）
    source = LearnableReader(edge_feat_dim=7, hidden_dim=32, n_layers=2,
                             dropout=0.1, node_feat_dim=0)
    # CIFAR10 架构（5 维节点特征，1 维边特征）
    target = LearnableReader(edge_feat_dim=1, hidden_dim=32, n_layers=2,
                             dropout=0.1, node_feat_dim=5)

    n_transferred, n_padded, skipped = transfer_weights(
        source.state_dict(), target)

    # 形状匹配的共享层复制成功
    assert n_transferred > 0
    assert n_padded == 0
    # 维度不匹配/新层被跳过
    skipped_keys = set(skipped)
    assert any('edge_encoder' in k for k in skipped_keys), skipped
    assert any('node_feat_proj' in k for k in skipped_keys), skipped

    # 抽查：conv MLP 权重一致
    src_sd = source.state_dict()
    tgt_sd = target.state_dict()
    for k in ('frontier_encoder.convs.0.mlp.0.weight',
              'frontier_encoder.bns.0.weight'):
        assert torch.equal(src_sd[k], tgt_sd[k]), k

    # 抽查：edge_encoder 保留随机初始化（与源不同）
    k = 'frontier_encoder.convs.0.edge_encoder.weight'
    assert not torch.equal(src_sd[k], tgt_sd[k]), "edge_encoder 应保留随机初始化"


def test_attention_reader_transfer():
    source = AttentionReader(edge_feat_dim=7, hidden_dim=32, n_layers=2,
                             dropout=0.1, node_feat_dim=0, m=2, n_heads=2)
    target = AttentionReader(edge_feat_dim=1, hidden_dim=32, n_layers=2,
                             dropout=0.1, node_feat_dim=5, m=2, n_heads=2)

    n_transferred, n_padded, skipped = transfer_weights(
        source.state_dict(), target)
    assert n_transferred > 0
    assert n_padded == 0
    skipped_keys = set(skipped)
    assert any('edge_encoder' in k for k in skipped_keys)
    assert any('node_feat_proj' in k for k in skipped_keys)

    src_sd = source.state_dict()
    tgt_sd = target.state_dict()
    assert torch.equal(src_sd['frontier_pointer.input_proj.weight'],
                       tgt_sd['frontier_pointer.input_proj.weight'])
    assert torch.equal(src_sd['belief_state.tokens'],
                       tgt_sd['belief_state.tokens'])


def test_legacy_width_reader_mlp_zero_pad():
    """legacy v0 base reader（reader_mlp.mlp.0.weight 输入宽度少 4 列）→
    完整 CIFAR10 base reader：最后一维零填充复制，其余形状不匹配跳过。"""
    full = LearnableReader(edge_feat_dim=1, hidden_dim=32, n_layers=2,
                           dropout=0.1, node_feat_dim=5)
    legacy = LearnableReader(edge_feat_dim=1, hidden_dim=32, n_layers=2,
                             dropout=0.1, node_feat_dim=5)
    # 模拟 v0 旧 checkpoint：reader_mlp.mlp.0.weight 输入宽度少 4 列
    # （缺 budget/struct 特征列），且 mlp.0.bias 缺失（模拟"无同键"分支）
    legacy_sd = legacy.state_dict()
    k = 'reader_mlp.mlp.0.weight'
    legacy_sd[k] = legacy_sd[k][:, :-4]
    del legacy_sd['reader_mlp.mlp.0.bias']
    full_sd = full.state_dict()

    n_transferred, n_padded, skipped = transfer_weights(legacy_sd, full)

    assert n_padded == 1, (n_padded, skipped)
    assert 'reader_mlp.mlp.0.weight' not in skipped
    out_w = full_sd['reader_mlp.mlp.0.weight']
    assert tuple(out_w.shape) == (32, 32 + 4), out_w.shape
    # 前 32 列与源一致，最后 4 列为零
    assert torch.equal(out_w[:, :-4], legacy_sd[k])
    assert torch.all(out_w[:, -4:] == 0)
    # 目标最后一维比源更窄的键（如 edge_encoder (32,7)→(32,1)，
    # 见 test_base_reader_transfer）与缺失键（bias）均走 skipped 分支
    assert 'reader_mlp.mlp.0.bias' in skipped


if __name__ == '__main__':
    test_base_reader_transfer()
    test_attention_reader_transfer()
    test_legacy_width_reader_mlp_zero_pad()
    print("OK: transfer_weights 部分权重迁移")

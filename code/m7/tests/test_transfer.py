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

    n_transferred, skipped = transfer_weights(source.state_dict(), target)

    # 形状匹配的共享层复制成功
    assert n_transferred > 0
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

    n_transferred, skipped = transfer_weights(source.state_dict(), target)
    assert n_transferred > 0
    skipped_keys = set(skipped)
    assert any('edge_encoder' in k for k in skipped_keys)
    assert any('node_feat_proj' in k for k in skipped_keys)

    src_sd = source.state_dict()
    tgt_sd = target.state_dict()
    assert torch.equal(src_sd['frontier_pointer.input_proj.weight'],
                       tgt_sd['frontier_pointer.input_proj.weight'])
    assert torch.equal(src_sd['belief_state.tokens'],
                       tgt_sd['belief_state.tokens'])


if __name__ == '__main__':
    test_base_reader_transfer()
    test_attention_reader_transfer()
    print("OK: transfer_weights 部分权重迁移")

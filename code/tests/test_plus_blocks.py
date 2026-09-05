"""plus/ 积木包测试：默认档零漂移（金标键集 + 真实 ckpt logits）+ 各积木开启路径。

金标捕获于 2026-09-05 plus/ 重构前的实现（合成 batch：3 图 9 节点，
torch.Generator(42) 生成特征；logits 4 位小数）。默认档任何数值或
键序漂移都会被这里拦截。
"""

import sys
from pathlib import Path

import dgl
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import (ClassifierConfig, GraphClassifier,  # noqa: E402
                        classifier_config_from_ckpt)
from classifier.factory import load_classifier  # noqa: E402
from classifier.plus.pos_enc import random_walk_structural_encoding  # noqa: E402

_CODE_DIR = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# 金标（重构前捕获，禁止改动）
# ------------------------------------------------------------------

GOLDEN_KEYS_NODE_PROJ = [
    'encoder.node_proj.weight', 'encoder.node_proj.bias',
    'encoder.convs.0.eps',
    'encoder.convs.0.mlp.0.weight', 'encoder.convs.0.mlp.0.bias',
    'encoder.convs.0.mlp.1.weight', 'encoder.convs.0.mlp.1.bias',
    'encoder.convs.0.mlp.1.running_mean', 'encoder.convs.0.mlp.1.running_var',
    'encoder.convs.0.mlp.1.num_batches_tracked',
    'encoder.convs.0.mlp.3.weight', 'encoder.convs.0.mlp.3.bias',
    'encoder.convs.0.edge_encoder.weight', 'encoder.convs.0.edge_encoder.bias',
    'encoder.convs.1.eps',
    'encoder.convs.1.mlp.0.weight', 'encoder.convs.1.mlp.0.bias',
    'encoder.convs.1.mlp.1.weight', 'encoder.convs.1.mlp.1.bias',
    'encoder.convs.1.mlp.1.running_mean', 'encoder.convs.1.mlp.1.running_var',
    'encoder.convs.1.mlp.1.num_batches_tracked',
    'encoder.convs.1.mlp.3.weight', 'encoder.convs.1.mlp.3.bias',
    'encoder.convs.1.edge_encoder.weight', 'encoder.convs.1.edge_encoder.bias',
    'encoder.bns.0.weight', 'encoder.bns.0.bias',
    'encoder.bns.0.running_mean', 'encoder.bns.0.running_var',
    'encoder.bns.0.num_batches_tracked',
    'encoder.bns.1.weight', 'encoder.bns.1.bias',
    'encoder.bns.1.running_mean', 'encoder.bns.1.running_var',
    'encoder.bns.1.num_batches_tracked',
    'head.weight', 'head.bias',
]

GOLDEN_LOGITS_GIN_PLUS = [
    [5.0548, -1.7215, -1.0163, -2.1234, -2.9346, -3.3871, 2.3166, 0.1469,
     -2.1872, -2.9948],
    [5.4731, -2.5004, 2.2249, -0.6361, -3.5696, -2.0722, -0.3519, -2.7878,
     -4.0662, -3.7259],
    [0.6797, 2.9403, 6.5018, -1.3364, -5.7381, -4.1381, -3.5164, 0.5229,
     -0.4154, -3.7118],
]

GOLDEN_LOGITS_GIN_RES_VN = [
    [47.3592, -18.6038, 23.1978, -1.3769, -10.6464, -6.7684, 3.2989,
     -17.4711, -8.8023, -23.2414],
    [45.8699, -16.265, 22.4994, -0.9413, -10.9016, -6.7245, 3.8842,
     -18.5764, -8.1661, -22.2461],
    [48.3018, -15.1563, 24.5647, 1.3997, -9.4986, -7.7329, 5.1923,
     -21.3327, -5.636, -23.0273],
]


def _golden_batch():
    """与金标捕获完全一致的合成 batch（3 图 9 节点，node 5 维 edge 1 维）。"""
    g1 = dgl.graph(([0, 1, 1, 2, 2, 0], [1, 2, 0, 0, 1, 2]))
    g2 = dgl.graph(([0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]))
    g3 = dgl.graph(([0, 0], [1, 0]))
    gen = torch.Generator().manual_seed(42)
    for g in (g1, g2, g3):
        g.ndata['feat'] = torch.randn(g.num_nodes(), 5, generator=gen)
        g.edata['feat'] = torch.randn(g.num_edges(), 1, generator=gen)
    return dgl.batch([g1, g2, g3])


def _default_cfg(**kw):
    base = dict(edge_feat_dim=1, node_feat_dim=5, n_classes=10,
                hidden_dim=32, n_layers=2, dropout=0.5)
    base.update(kw)
    return ClassifierConfig.from_legacy(**base)


# ------------------------------------------------------------------
# 默认档零漂移
# ------------------------------------------------------------------

def test_golden_keys_node_proj_branch():
    clf = GraphClassifier(config=_default_cfg())
    assert list(clf.state_dict().keys()) == GOLDEN_KEYS_NODE_PROJ


def test_golden_keys_node_emb_branch():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, node_feat_dim=0,
                                       n_classes=10, hidden_dim=32,
                                       n_layers=2, dropout=0.5, residual=True)
    keys = list(GraphClassifier(config=cfg).state_dict().keys())
    assert keys[0] == 'encoder.node_emb.weight'   # node_emb 分支
    assert keys[-1] == 'head.bias'
    assert keys[1:-2] == GOLDEN_KEYS_NODE_PROJ[2:-2]  # 中段与 proj 分支一致


def test_golden_logits_real_checkpoints():
    """真实 checkpoint strict 加载 + eval 前向 = 重构前逐位一致。"""
    batch = _golden_batch()
    nids = torch.zeros(batch.num_nodes(), dtype=torch.long)
    for name, golden, sub in (
            ('GIN-plus', GOLDEN_LOGITS_GIN_PLUS,
             'result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt'),
            ('GIN-res-vn', GOLDEN_LOGITS_GIN_RES_VN,
             'result/cifar10/classifier/GIN-res-vn/best_cifar10_classifier.pt')):
        ckpt = _CODE_DIR / sub
        if not ckpt.exists():
            pytest.skip(f'{name} checkpoint 不存在: {ckpt}')
        clf = load_classifier(str(ckpt), device='cpu', freeze=True).eval()
        with torch.no_grad():
            logits = clf(batch, nids)
        assert torch.allclose(
            logits, torch.tensor(golden, dtype=logits.dtype), atol=1e-3), name


# ------------------------------------------------------------------
# 积木①边特征整合
# ------------------------------------------------------------------

def test_edge_encoder_linear_default_keys_unchanged():
    clf = GraphClassifier(config=_default_cfg())
    assert 'encoder.convs.0.edge_encoder.weight' in clf.state_dict()
    assert 'encoder.convs.0.edge_encoder.0.weight' not in clf.state_dict()
    assert clf.config.to_dict().get('edge_encoder') is None  # 默认不落盘


def test_edge_encoder_mlp():
    clf = GraphClassifier(config=_default_cfg(edge_encoder='mlp'))
    sd = clf.state_dict()
    assert 'encoder.convs.0.edge_encoder.0.weight' in sd
    assert 'encoder.convs.0.edge_encoder.weight' not in sd
    batch = _golden_batch()
    out = clf(batch, torch.zeros(batch.num_nodes(), dtype=torch.long))
    assert out.shape == (3, 10)
    # config 往返
    d = clf.config.to_dict()
    assert d['edge_encoder'] == 'mlp'
    assert ClassifierConfig.from_dict(d).edge_encoder == 'mlp'


# ------------------------------------------------------------------
# 积木⑤ FFN
# ------------------------------------------------------------------

def test_ffn_off_zero_keys():
    clf = GraphClassifier(config=_default_cfg())
    assert not any(k.startswith('encoder.ffns.') for k in clf.state_dict())


def test_ffn_on_keys_and_forward():
    clf = GraphClassifier(config=_default_cfg(residual=True, ffn=True,
                                              ffn_dropout=0.3))
    sd = clf.state_dict()
    for i in range(2):
        assert f'encoder.ffns.{i}.fc1.weight' in sd
        assert f'encoder.ffns.{i}.fc2.weight' in sd
        assert f'encoder.ffns.{i}.bn.weight' in sd
    batch = _golden_batch()
    clf.eval()
    with torch.no_grad():
        out = clf(batch, torch.zeros(batch.num_nodes(), dtype=torch.long))
    assert out.shape == (3, 10)
    d = clf.config.to_dict()
    assert d['ffn'] is True and d['ffn_dropout'] == 0.3
    back = ClassifierConfig.from_dict(d)
    assert back.ffn and back.ffn_dropout == 0.3


def test_ffn_own_residual_identity_init():
    """FFN 初始近似恒等的一侧保障：h 分支直接进残差+BN，至少 forward 无 NaN。"""
    from classifier.plus.ffn import FFN
    blk = FFN(16, dropout=0.0).eval()
    h = torch.randn(4, 16)
    with torch.no_grad():
        out = blk(h)
    assert out.shape == h.shape and torch.isfinite(out).all()


# ------------------------------------------------------------------
# 积木⑥位置编码 RWSE
# ------------------------------------------------------------------

def test_rwse_known_transition_matrix():
    """单节点自环图：A_hat=I → 所有步对角线=1；两节点对称图：对角线恒=0.5。"""
    g1 = dgl.graph(([0], [0]))
    g1.ndata['feat'] = torch.zeros(1, 1)
    pe1 = random_walk_structural_encoding(g1, 4)
    assert pe1.shape == (1, 4) and torch.allclose(pe1, torch.ones(1, 4))

    g2 = dgl.graph(([0, 1], [1, 0]))  # 双向边
    g2.ndata['feat'] = torch.zeros(2, 1)
    pe2 = random_walk_structural_encoding(g2, 3)
    assert pe2.shape == (2, 3)
    assert torch.allclose(pe2, torch.full((2, 3), 0.5))


def test_rwse_batched_shape_and_injection():
    batch = _golden_batch()
    pe = random_walk_structural_encoding(batch, 8)
    assert pe.shape == (batch.num_nodes(), 8)
    assert (pe >= 0).all() and (pe <= 1 + 1e-6).all()  # 概率对角线

    clf = GraphClassifier(config=_default_cfg(pos_enc='rwse'))
    assert clf.encoder.node_proj.in_features == 5 + 8  # node_feat + 默认 ksteps
    assert clf.encoder.node_proj.out_features == 32
    clf.eval()
    with torch.no_grad():
        out = clf(batch, torch.zeros(batch.num_nodes(), dtype=torch.long))
    assert out.shape == (3, 10)
    d = clf.config.to_dict()
    assert d['pos_enc'] == 'rwse' and 'pos_enc_ksteps' not in d  # 默认 8 不落盘


def test_pos_enc_off_uses_plain_proj():
    clf = GraphClassifier(config=_default_cfg())
    assert clf.encoder.node_proj.in_features == 5


# ------------------------------------------------------------------
# backbone kwargs 白名单
# ------------------------------------------------------------------

def test_whitelist_rejects_unsupported_blocks():
    with pytest.raises(ValueError, match="不支持 GNN\\+ 积木 'ffn'"):
        GraphClassifier(config=_default_cfg(backbone='gat', ffn=True))
    with pytest.raises(ValueError, match="'pos_enc'"):
        GraphClassifier(config=_default_cfg(backbone='gat', pos_enc='rwse'))


def test_gat_readout_meanmax():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, n_classes=10,
                                       hidden_dim=32, n_layers=2, dropout=0.5,
                                       backbone='gat', n_heads=4,
                                       readout='meanmax', head='ffn')
    clf = GraphClassifier(config=cfg)
    assert clf.encoder.out_dim == 64
    assert clf.head[0].in_features == 64
    batch = _golden_batch()
    clf.eval()
    with torch.no_grad():
        out = clf(batch, torch.zeros(batch.num_nodes(), dtype=torch.long))
    assert out.shape == (3, 10)


def test_gat_edge_encoder_mlp_keys():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, n_classes=10,
                                       hidden_dim=32, n_layers=2, dropout=0.5,
                                       backbone='gat', n_heads=4,
                                       edge_encoder='mlp')
    sd = GraphClassifier(config=cfg).state_dict()
    assert 'encoder.convs.0.edge_encoder.0.weight' in sd


def test_gat_default_keys_unchanged():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, n_classes=10,
                                       hidden_dim=32, n_layers=2, dropout=0.5,
                                       backbone='gat', n_heads=4)
    sd = GraphClassifier(config=cfg).state_dict()
    assert 'encoder.convs.0.fc.weight' in sd
    assert not any(k.startswith('encoder.ffns.') for k in sd)
    assert 'encoder.convs.0.edge_encoder.weight' in sd  # linear 默认


# ------------------------------------------------------------------
# config 校验
# ------------------------------------------------------------------

def test_config_validation_and_roundtrip():
    with pytest.raises(ValueError, match='边编码器'):
        ClassifierConfig.from_dict({'edge_encoder': 'dot'})
    with pytest.raises(ValueError, match='位置编码'):
        ClassifierConfig.from_dict({'pos_enc': 'lappe'})
    cfg = ClassifierConfig.from_legacy(
        edge_feat_dim=1, n_classes=10, hidden_dim=32, n_layers=2,
        edge_encoder='mlp', ffn=True, ffn_dropout=0.2,
        pos_enc='rwse', pos_enc_ksteps=12)
    d = cfg.to_dict()
    back = ClassifierConfig.from_dict(d)
    assert back.edge_encoder == 'mlp' and back.ffn and back.ffn_dropout == 0.2
    assert back.pos_enc == 'rwse' and back.pos_enc_ksteps == 12


def test_factory_rebuild_with_blocks(tmp_path):
    """save → load_classifier 全链路：积木 config 经 checkpoint 往返重建。"""
    clf = GraphClassifier(config=_default_cfg(ffn=True, pos_enc='rwse'))
    ckpt = {'model_state_dict': clf.state_dict(),
            'config': {'classifier': clf.config.to_dict()}}
    p = tmp_path / 'm.pt'
    torch.save(ckpt, str(p))
    rebuilt = load_classifier(str(p), device='cpu', freeze=True)
    assert rebuilt.config.ffn and rebuilt.config.pos_enc == 'rwse'
    batch = _golden_batch()
    clf.eval()
    with torch.no_grad():
        a = clf(batch, torch.zeros(batch.num_nodes(), dtype=torch.long))
        b = rebuilt(batch, torch.zeros(batch.num_nodes(), dtype=torch.long))
    assert torch.allclose(a, b)


def test_legacy_ckpt_config_rebuild(tmp_path):
    """无新字段的旧 config dict → 重建后键集与金标一致（向后兼容）。"""
    old = {'backbone': 'gin', 'edge_feat_dim': 1, 'node_feat_dim': 5,
           'n_classes': 10, 'hidden_dim': 32, 'n_layers': 2, 'dropout': 0.5}
    clf = GraphClassifier(config=classifier_config_from_ckpt(
        {'config': {'classifier': old}}))
    assert list(clf.state_dict().keys()) == GOLDEN_KEYS_NODE_PROJ

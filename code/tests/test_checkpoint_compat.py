"""旧 checkpoint 兼容性单测：m5 v0 base / m6 attention / m7 / m3 分类器。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from active_graph_reader import build_reader
from active_graph_reader.config import reader_config_from_checkpoint
from classifier import load_classifier
from tools.ckpt_io import pad_reader_mlp_input, strip_classifier_keys

CODE = Path(__file__).resolve().parents[1]


def _try_load_reader(ckpt_path, need_pad=False):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = strip_classifier_keys(dict(ckpt['reader_state_dict']))
    cfg = reader_config_from_checkpoint(ckpt, sd)
    reader = build_reader(cfg)
    if need_pad:
        target = reader.reader_mlp.mlp[0].in_features
        sd, patched = pad_reader_mlp_input(sd, target)
        assert patched, f'{ckpt_path} 应需要 padding'
    reader.load_state_dict(sd, strict=True)
    return cfg


def test_m5_v0_base_ckpt():
    p = CODE / 'result' / 'ppa' / 'experiments' / 'bc_baseline' / 'base' / 'bc' / 'reader_checkpoint_epoch3.pt'
    if not p.exists():
        print('  SKIP (no m5 ckpt)')
        return
    cfg = _try_load_reader(p, need_pad=True)
    assert cfg.kind == 'base'
    print('  m5 v0 base ckpt strict-load OK')


def test_m6_attention_ckpt():
    p = CODE / 'result' / 'ppa' / 'experiments' / 'bc_baseline' / 'attention' / 'bc' / 'reader_belief_epoch3.pt'
    if not p.exists():
        print('  SKIP (no m6 ckpt)')
        return
    cfg = _try_load_reader(p)
    assert cfg.kind == 'attention'
    print('  m6 attention ckpt strict-load OK')


def test_m7_ckpts():
    for kind in ('base', 'attention'):
        hits = sorted((CODE / 'result' / 'cifar10' / 'experiments').rglob(f'*{kind}*/bc/reader_checkpoint_epoch*.pt'))
        hits = [h for h in hits if 'checkpoint_epoch' in h.name]
        if not hits:
            print(f'  SKIP (no m7 {kind} ckpt)')
            continue
        cfg = _try_load_reader(hits[0])
        assert cfg.kind == kind
        print(f'  m7 {kind} ckpt strict-load OK: {hits[0].name}')


def test_m3_classifier():
    p = CODE / 'result' / 'ppa' / 'classifier' / 'best_model_ft.pt'
    if not p.exists():
        print('  SKIP (no m3 ckpt)')
        return
    clf = load_classifier(str(p))
    assert clf.config.n_classes == 37 and clf.config.hidden_dim == 300
    assert not any(p_.requires_grad for p_ in clf.parameters())
    print('  m3 classifier config 重建+frozen OK')


def run_all():
    test_m5_v0_base_ckpt()
    test_m6_attention_ckpt()
    test_m7_ckpts()
    test_m3_classifier()
    print('test_checkpoint_compat: ALL PASSED')


if __name__ == '__main__':
    run_all()

"""eval JSON 统一格式单测：merge_eval_json 合并/冲突/重命名 + 全图分类器参照。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile

import torch

from train_reader.core import (build_eval_json, evaluate_classifier_full,  # noqa: E402
                               merge_eval_json)


def _cells(value, n=4):
    return {'mean': value, 'std': 0.1, 'n': n}


def _part(split, strategy, value, config=None, clf_ref=None):
    """构造 build_eval_json 形状的最小 part。"""
    out = {
        split: {strategy: {'s0': {'0.05': _cells(value)},
                           'mean': {'0.05': _cells(value)}}},
        f'per_class_{split}': {strategy: {'mean': {'0': {'0.05': _cells(value)}}}},
        'config': dict(config or {}),
    }
    if clf_ref is not None:
        out['acc_clf_reference'] = {'0.05': _cells(clf_ref)}
    return out


def test_merge_multi_split_multi_strategy():
    fixed_test = _part('test', 'random', 0.20, config={'mode': 'fixed'})
    fixed_test2 = _part('test', 'bfs', 0.30, config={'mode': 'fixed'})
    fixed_valid = _part('valid', 'random', 0.25, config={'n_graphs': {'valid': 10}})
    merged = merge_eval_json([fixed_test, fixed_test2, fixed_valid])

    assert set(merged['test']) == {'random', 'bfs'}
    assert set(merged['valid']) == {'random'}
    assert merged['test']['random']['mean']['0.05']['mean'] == 0.20
    assert merged['test']['bfs']['mean']['0.05']['mean'] == 0.30
    # per_class_{split} 同步合并
    assert set(merged['per_class_test']) == {'random', 'bfs'}
    # config dict 字段按键 union，标量后者覆盖
    assert merged['config']['n_graphs'] == {'valid': 10}
    print('  multi-split/multi-strategy merge OK')


def test_merge_conflict_last_wins():
    old = _part('test', 'random', 0.10)
    new = _part('test', 'random', 0.50)
    conflicts = []
    merged = merge_eval_json([old, new], on_conflict=conflicts.append)
    assert merged['test']['random']['mean']['0.05']['mean'] == 0.50
    # 主表与 per_class 表的冲突都会回调
    assert conflicts == ['test.random', 'per_class_test.random']
    print('  conflict last-wins + on_conflict OK')


def test_acc_clf_reference_split_suffix():
    part = _part('valid', 'learnable', 0.40, clf_ref=0.65)
    merged = merge_eval_json([part])
    assert 'acc_clf_reference' not in merged
    assert merged['acc_clf_reference_valid']['0.05']['mean'] == 0.65
    print('  acc_clf_reference → acc_clf_reference_{split} OK')


def test_build_eval_json_integration():
    """两次 build_eval_json（valid+test）合并后可直接被 plot_three_stage 读取。"""
    budgets = (0.05, 0.10, 0.15)
    results = {'acc_5%': 0.3, 'acc_10%': 0.4, 'acc_15%': 0.5, 'acc_overall': 0.4,
               'n_graphs': 8,
               'per_seed_outcomes': {0: {0.05: [1, 0], 0.10: [1, 1], 0.15: [1, 1]},
                                     1: {0.05: [1, 0], 0.10: [1, 0], 0.15: [1, 1]}},
               'per_seed_per_class': {i: {b: {0: [1, 0]} for b in budgets}
                                      for i in range(2)}}
    p_valid = build_eval_json(results, 'valid', 'learnable', 2, budgets, config={})
    p_test = build_eval_json(results, 'test', 'learnable', 2, budgets, config={})
    merged = merge_eval_json([p_valid, p_test])
    merged['config']['full_graph_ref'] = {'valid': 0.64, 'test': 0.66}

    for split in ('valid', 'test'):
        cells = merged[split]['learnable']['mean']
        for b in budgets:
            key = f'{b:.2f}'
            assert key in cells, f'{split}/{key} 缺预算键'
            # plot_three_stage 的读法：cells[f'{b:.2f}']['mean'] + std/sqrt(n)
            _ = cells[key]['mean'], cells[key]['std'] / max(cells[key]['n'], 1) ** 0.5
    assert merged['per_class_valid']['learnable']['mean']['0']['0.05']['n'] == 4
    print('  build_eval_json × merge（plot_three_stage 读法）OK')


def test_evaluate_classifier_full():
    """全图参照评估：mock 分类器按图规模给确定预测，batch 两种切分一致。"""
    import dgl

    class _SizeClf(torch.nn.Module):
        """图节点数 < 15 → 类 0，否则类 1。"""

        def forward(self, g, nids):
            sizes = g.batch_num_nodes()
            logits = torch.zeros(len(sizes), 2)
            logits[:, 1] = (sizes >= 15).float()
            logits[:, 0] = 1.0 - logits[:, 1]
            return logits

    graphs = [dgl.graph(([0, 1], [1, 0]), num_nodes=10),
              dgl.graph(([0, 1], [1, 0]), num_nodes=20)]
    labels = torch.tensor([0, 1])
    clf = _SizeClf()
    assert evaluate_classifier_full(clf, graphs, labels, 'cpu') == 1.0
    assert evaluate_classifier_full(clf, graphs, labels, 'cpu', batch_size=1) == 1.0
    assert evaluate_classifier_full(clf, graphs, torch.tensor([1, 1]), 'cpu') == 0.5
    print('  evaluate_classifier_full OK')


def test_migrate_script_smoke():
    """迁移脚本 classify/parse 逻辑：合成文件树 dry-run 不落盘。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import migrate_eval_json as mig

    fixed = _part('test', 'random', 0.20, config={'dataset': 'syn'})
    fixed['test']['bfs'] = fixed['test']['random']
    reader = _part('test', 'learnable', 0.40, config={'dataset': 'syn'})
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixed_dir = root / 'syn' / 'fixed'
        exp_dir = root / 'syn' / 'experiments' / 'ex1' / 'base' / 'bc'
        (exp_dir / 'evaluate').mkdir(parents=True)
        fixed_dir.mkdir(parents=True)
        mig.save_json(fixed, fixed_dir / 'eval_heldout_test.json')
        mig.save_json(reader, exp_dir / 'evaluate' / 'eval_heldout_test.json')

        kind, target, strategies = mig.classify(
            fixed_dir / 'eval_heldout_test.json', fixed, root)
        assert kind == 'fixed' and strategies == ['bfs', 'random']
        assert target == fixed_dir / 'eval_fixed.json'
        kind, target, _ = mig.classify(
            exp_dir / 'evaluate' / 'eval_heldout_test.json', reader, root)
        assert kind == 'reader' and target == exp_dir / 'eval_learnable.json'

        # canonical 命名文件：位于其规范目标位置 → anchor（合并基底）；
        # 位置不符（如 fixed_eval/ 下的 eval_fixed.json）→ canonical 保留原位
        kind, target, _ = mig.classify(exp_dir / 'eval_learnable.json', reader, root)
        assert kind == 'anchor' and target == exp_dir / 'eval_learnable.json'
        stray = root / 'syn' / 'legacy' / 'eval_learnable.json'
        stray.parent.mkdir(parents=True)
        kind, target, _ = mig.classify(stray, reader, root)
        assert kind == 'anchor' and target == stray
        stray_fixed = root / 'syn' / 'fixed' / 'fixed_eval' / 'eval_fixed.json'
        stray_fixed.parent.mkdir(parents=True)
        kind, _target, _ = mig.classify(stray_fixed, fixed, root)
        assert kind == 'canonical'
    print('  migrate_eval_json classify OK')


def run_all():
    test_merge_multi_split_multi_strategy()
    test_merge_conflict_last_wins()
    test_acc_clf_reference_split_suffix()
    test_build_eval_json_integration()
    test_evaluate_classifier_full()
    test_migrate_script_smoke()
    print('test_eval_json_merge: ALL PASSED')


if __name__ == '__main__':
    run_all()

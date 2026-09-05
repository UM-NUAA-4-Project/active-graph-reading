"""端到端冒烟：全部单测 + 所有 CLI --help + 合成数据评估管线。

不依赖任何真实数据集（合成小图完成评估链路验证）。

用法: cd <repo 根> && /d/anoconda/envs/gnn/python.exe code/tests/smoke_test.py
"""

import random
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
PY = sys.executable

UNIT_TESTS = [
    'test_config_reader.py',
    'test_checkpoint_compat.py',
    'test_trajectory_io.py',
    'test_unified_loops.py',
    'test_self_loop.py',
    'test_evidence_reader.py',
    'test_eval_json_merge.py',
]

CLI_SCRIPTS = [
    'train_classifier/train_full_graph.py',
    'train_classifier/eval_classifier.py',
    'trajectories/generate_trajectories.py',
    'trajectories/filter_trajectories.py',
    'trajectories/balance_trajectories.py',
    'trajectories/augment_trajectories.py',
    'trajectories/analyze_trajectories.py',
    'train_reader/train_bc.py',
    'train_reader/train_rl.py',
    'train_reader/eval.py',
    'train_reader/transfer.py',
    'tools/migrate_eval_json.py',
    'comparison/plot_eval_results.py',
    'comparison/plot_three_stage.py',
    'comparison/qa_three_stage.py',
    'diagnosis/diagnose_checkpoint.py',
    'diagnosis/analyze_trajectory_diversity.py',
    'diagnosis/rank_easy_by_ce.py',
    'diagnosis/visualize_diagnosis.py',
    'diagnosis/frontier_analysis.py',
    'diagnosis/oracle_estimator.py',
    'diagnosis/visualize_rollouts.py',
]


def run_unit_tests():
    print('\n=== Unit tests ===')
    for t in UNIT_TESTS:
        proc = subprocess.run([PY, str(CODE / 'tests' / t)],
                              cwd=str(ROOT), capture_output=True, text=True)
        ok = proc.returncode == 0
        print(f"  [{'OK' if ok else 'FAIL'}] {t}")
        if not ok:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            return False
    return True


def run_cli_help():
    print('\n=== CLI --help checks ===')
    for s in CLI_SCRIPTS:
        if not (CODE / s).exists():
            print(f"  [SKIP] {s} (不存在)")
            continue
        proc = subprocess.run([PY, str(CODE / s), '--help'],
                              cwd=str(ROOT), capture_output=True, text=True)
        ok = proc.returncode == 0
        print(f"  [{'OK' if ok else 'FAIL'}] {s}")
        if not ok:
            print(proc.stderr[-1500:])
            return False
    return True


def run_synthetic_eval():
    """合成小图端到端：evaluate_fixed + evaluate_reader + build_eval_json。"""
    print('\n=== Synthetic eval pipeline ===')
    sys.path.insert(0, str(CODE))
    import dgl
    import torch

    from train_reader.core import (build_eval_json, evaluate_fixed,
                                   evaluate_reader, load_reader_from_checkpoint)
    from active_graph_reader import build_reader, ReaderConfig
    from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder

    class _Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(32, 3)

        def forward(self, g, nids):
            import dgl.function as fn
            with g.local_scope():
                g.ndata['h'] = torch.zeros(g.num_nodes(), 32)
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'H'))
                return self.lin(g.ndata['H'])

    random.seed(0)
    torch.manual_seed(0)
    graphs, labels = [], []
    for i in range(4):
        n = 30
        src = torch.randint(0, n, (120,))
        dst = torch.randint(0, n, (120,))
        g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=n)
        g.edata['feat'] = torch.randn(240, 7)
        graphs.append(dgl.add_self_loop(g))
        labels.append(torch.tensor(i % 3))
    labels_t = torch.stack(labels)
    builder = SubgraphBuilder()
    classifier = _Clf()

    fixed = evaluate_fixed('random', classifier, builder, graphs, labels_t,
                           (0.05, 0.1), 'cpu', n_seeds=2)
    reader = build_reader(ReaderConfig.from_legacy('attention', hidden_dim=32,
                                                   n_layers=2, dropout=0.1))
    learn = evaluate_reader(reader, classifier, builder, graphs, labels_t,
                            (0.05, 0.1), 'cpu', n_seeds=2)
    for name, res in (('fixed', fixed), ('learnable', learn)):
        assert set(res['outcomes']) == {0.05, 0.1}
        assert res['n_graphs'] == 4
    j = build_eval_json(learn, 'test', 'learnable', 2, (0.05, 0.1), config={})
    assert 'test' in j and 'per_class_test' in j and 'mean' in j['test']['learnable']

    # checkpoint 往返：保存后经 load_reader_from_checkpoint 重建并 strict 加载
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / 'reader.pt'
        torch.save({'reader_state_dict': reader.state_dict(),
                    'config': {'reader': reader.config.to_dict()}},
                   str(ckpt_path))
        reader2 = load_reader_from_checkpoint(ckpt_path, 'cpu')
        reader2.load_state_dict(
            torch.load(str(ckpt_path), weights_only=False)['reader_state_dict'],
            strict=True)
    print('  evaluate_fixed / evaluate_reader / build_eval_json / ckpt 往返 OK')
    return True


def main():
    print(f"Python: {PY}")
    print(f"Repo: {ROOT}")
    ok = True
    ok &= run_unit_tests()
    ok &= run_cli_help()
    ok &= run_synthetic_eval()
    print('\n' + ('=== SMOKE PASSED ===' if ok else '=== SMOKE FAILED ==='))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

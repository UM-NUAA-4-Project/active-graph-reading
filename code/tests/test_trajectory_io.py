"""轨迹 IO 单测：新旧 pickle 兼容、ratio_key 往返、保存/加载对称。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile

import torch

from tools.trajectory_io import (iter_file_trajectories, load_trajectory_file,
                                 ratio_key, ratio_key_from_name,
                                 save_trajectory_file)
from trajectories.core.trajectory import Trajectory


def test_ratio_key():
    assert ratio_key(0.05) == '0_050'
    assert ratio_key(0.1) == '0_100'
    assert ratio_key(0.15) == '0_150'
    assert abs(ratio_key_from_name('trajectories_r0_050.pt') - 0.05) < 1e-9


def test_save_load_roundtrip():
    trajs = [Trajectory(graph_idx=3, seed_node=7, reveal_ratio=0.1,
                        reveal_order=[1, 2, 3], final_pred=2, is_correct=True,
                        teacher_type='gradient', ce=0.25)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'trajectories_r0_100.pt'
        save_trajectory_file(path, {'gradient': trajs}, {'dataset': 'x'})
        data = load_trajectory_file(path)
        out = list(iter_file_trajectories(data))
        assert len(out) == 1
        t = out[0]
        assert isinstance(t, Trajectory)
        assert (t.graph_idx, t.seed_node, t.reveal_order, t.ce) == (3, 7, [1, 2, 3], 0.25)
    print('  save/load roundtrip OK')


class OldTrajectory:
    """模拟 m4.teachers.base.Trajectory 的旧类（保留字段参考）。"""

    def __init__(self, graph_idx, seed_node, reveal_ratio, reveal_order,
                 final_pred, is_correct, teacher_type, ce=0.0):
        self.graph_idx = graph_idx
        self.seed_node = seed_node
        self.reveal_ratio = reveal_ratio
        self.reveal_order = reveal_order
        self.final_pred = final_pred
        self.is_correct = is_correct
        self.teacher_type = teacher_type
        self.ce = ce


def test_legacy_pickle_remap():
    """Unpickler 的旧模块路径重映射：m4.teachers.base.Trajectory → 新类。

    （端到端真实旧文件加载由 smoke_test / 手动验证覆盖；
    此处直接测 find_class 的重映射逻辑。）
    """
    import io

    from tools.trajectory_io import _TrajectoryUnpickler
    from trajectories.core.trajectory import Trajectory, LEGACY_PICKLE_MODULE

    u = _TrajectoryUnpickler(io.BytesIO(b''))
    assert u.find_class(LEGACY_PICKLE_MODULE, 'Trajectory') is Trajectory
    assert u.find_class('collections', 'OrderedDict') is not Trajectory
    print('  legacy pickle remap OK')


def run_all():
    test_ratio_key()
    test_save_load_roundtrip()
    test_legacy_pickle_remap()
    print('test_trajectory_io: ALL PASSED')


if __name__ == '__main__':
    run_all()

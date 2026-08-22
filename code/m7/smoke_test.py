"""M7 端到端冒烟：全部单测 + 三个脚本 --help + 固定策略小样本评估。

用法: cd /mnt/c/Users/qq283/Desktop/UM-NUAA && /mnt/d/anoconda/envs/gnn/python.exe code/m7/smoke_test.py
（CIFAR10 数据需已下载；--help 检查不依赖数据）

注意（相对原任务 brief 的两处调整）：
1. 最后一步评估使用的分类器默认指向 Task 5 冒烟产物
   `C:\\tmp\\m7_smoke_clf\\cifar10_classifier.pt`（生产上界分类器
   `m7/checkpoint/cifar10_classifier.pt` 尚未训练）。eval.py 按 checkpoint 自身
   config 构建分类器，因此小维度冒烟分类器可直接加载。可通过 --classifier 覆盖。
2. PY 解释器路径按运行环境自适应：本脚本在 Windows Python 下执行时（brief 用法
   行即是如此），子进程必须用 Windows 路径 `D:\\anaconda\\envs\\gnn\\python.exe`，
   WSL 路径 `/mnt/d/...` 无法被 CreateProcess 解析（WinError 2）；若改由 WSL
   python3 执行则用 WSL 路径。ROOT 由 __file__ 推导，两种环境下均正确。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # 仓库根
if os.name == 'nt':  # Windows Python（conda 环境路径以 sys.executable 为准）
    PY = Path(sys.executable)
else:  # WSL python3（仅用于拉起 Windows Python 子进程）
    PY = r'/mnt/d/anoconda/envs/gnn/python.exe'

def run(cmd, desc):
    print(f"\n=== {desc} ===")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"FAIL: {desc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M7 端到端冒烟')
    parser.add_argument('--classifier', default=r'C:\tmp\m7_smoke_clf\cifar10_classifier.pt',
                        help='评估用分类器 checkpoint（默认 Task 5 冒烟分类器；'
                             '生产上界分类器训练完成后可改为 m7/checkpoint/cifar10_classifier.pt）')
    args = parser.parse_args()

    for test in ['test_gin_node_feat', 'test_reader_node_feat',
                 'test_datasets', 'test_transfer']:
        run([PY, f'code/m7/tests/{test}.py'], test)
    for script in ['train_classifier', 'train_rl', 'eval']:
        run([PY, f'code/m7/{script}.py', '--help'], f'{script} --help')
    run([PY, 'code/m7/eval.py', '--dataset', 'cifar10', '--strategy', 'random',
         '--classifier', args.classifier,
         '--device', 'cpu', '--max-graphs', '10', '--output-dir', 'm7/results'],
        'eval random (10 graphs)')
    print("\nALL SMOKE TESTS PASSED")

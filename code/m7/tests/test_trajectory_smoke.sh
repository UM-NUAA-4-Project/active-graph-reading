#!/bin/bash
# 冒烟：16 图 × cpu 生成教师轨迹（需 CIFAR10 数据已下载；先训小型上界分类器）
set -e
cd /mnt/c/Users/qq283/Desktop/UM-NUAA
PY=/mnt/d/anoconda/envs/gnn/python.exe
$PY code/m7/train_classifier.py --dataset cifar10 --root D:\\superpixels --device cpu \
  --epochs 1 --batch-size 16 --hidden-dim 32 --n-layers 2 \
  --max-graphs 100 --output-dir /tmp/m7_smoke_clf
$PY code/m7/generate_trajectories.py --dataset cifar10 --root D:\\superpixels --device cpu \
  --classifier /tmp/m7_smoke_clf/cifar10_classifier.pt \
  --teachers gradient structural-degree \
  --max-graphs 16 --n-seeds 2 --budget-ratios 0.10 0.15 \
  --output-dir /tmp/m7_smoke_traj
$PY -c "
import sys; sys.path.insert(0, '.')
import torch
d = torch.load('/tmp/m7_smoke_traj/trajectories_r0_100.pt',
               map_location='cpu', weights_only=False)
assert set(d.keys()) == {'trajectories', 'config'}, d.keys()
total = 0
for t in ('gradient', 'structural-degree'):
    n = sum(len(l) for l in d['trajectories'][t])
    total += n
    print(f'{t}: {n} correct trajectories')
assert total >= 1, 'smoke 数据量下至少应产出 1 条正确轨迹'
print('TRAJECTORY STRUCTURE OK')
"
echo "SMOKE OK: teacher trajectories generated"

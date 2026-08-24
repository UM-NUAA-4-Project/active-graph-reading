#!/bin/bash
# 冒烟：小轨迹数据上 BC 2 epoch（需先跑 test_trajectory_smoke.sh 生成 /tmp/m7_smoke_traj）
set -e
cd /mnt/c/Users/qq283/Desktop/UM-NUAA
PY=/mnt/d/anoconda/envs/gnn/python.exe
$PY code/m7/train_bc.py --dataset cifar10 --root D:\\superpixels --device cpu \
  --data-dir /tmp/m7_smoke_traj \
  --teachers gradient structural-degree \
  --epochs 2 --batch-size 8 --hidden-dim 32 --n-layers-reader 2 --dropout-reader 0.1 \
  --output-dir /tmp/m7_smoke_bc
$PY -c "
import sys, json; sys.path.insert(0, '.')
import torch
log = json.load(open('/tmp/m7_smoke_bc/training_log.json'))
assert len(log) == 2, log
ckpt = torch.load('/tmp/m7_smoke_bc/reader_checkpoint_epoch2.pt',
                  map_location='cpu', weights_only=False)
assert 'reader_state_dict' in ckpt and 'config' in ckpt
cfg = ckpt['config']
assert cfg['reader_kind'] == 'base'
print('BC CHECKPOINT OK', {k: cfg[k] for k in ('hidden_dim', 'n_layers_reader',
                                               'dropout_reader')})
"
echo "SMOKE OK: BC trained"

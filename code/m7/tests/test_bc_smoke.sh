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
cfg_file = json.load(open('/tmp/m7_smoke_bc/config.json'))
assert cfg_file == cfg, (cfg_file, cfg)
print('BC CHECKPOINT OK', {k: cfg[k] for k in ('hidden_dim', 'n_layers_reader',
                                               'dropout_reader')})
"

# ---- attention reader 分支 ----
$PY code/m7/train_bc.py --dataset cifar10 --root D:\\superpixels --device cpu \
  --data-dir /tmp/m7_smoke_traj \
  --teachers gradient structural-degree \
  --reader-kind attention \
  --epochs 2 --batch-size 8 --hidden-dim 32 --n-layers-reader 2 --dropout-reader 0.1 \
  --output-dir /tmp/m7_smoke_bc_attention
$PY -c "
import sys, json; sys.path.insert(0, '.')
import torch
log = json.load(open('/tmp/m7_smoke_bc_attention/training_log.json'))
assert len(log) == 2, log
ckpt = torch.load('/tmp/m7_smoke_bc_attention/reader_checkpoint_epoch2.pt',
                  map_location='cpu', weights_only=False)
cfg = ckpt['config']
assert cfg['reader_kind'] == 'attention', cfg
assert cfg['m'] == 4 and cfg['n_heads'] == 4 and cfg['temperature'] == 1.0, cfg
cfg_file = json.load(open('/tmp/m7_smoke_bc_attention/config.json'))
assert cfg_file == cfg, (cfg_file, cfg)
print('BC ATTENTION CHECKPOINT OK', {k: cfg[k] for k in ('reader_kind', 'm', 'n_heads')})
"
# ---- resume 分支：从 epoch2 续训到 epoch3（结构/超参不传，取 checkpoint config） ----
$PY code/m7/train_bc.py --dataset cifar10 --root D:\\superpixels --device cpu \
  --data-dir /tmp/m7_smoke_traj \
  --resume /tmp/m7_smoke_bc/reader_checkpoint_epoch2.pt \
  --epochs 3 --batch-size 8 \
  --output-dir /tmp/m7_smoke_bc
$PY -c "
import sys, json; sys.path.insert(0, '.')
import torch
log = json.load(open('/tmp/m7_smoke_bc/training_log.json'))
assert len(log) == 3, log  # 2 旧 + 1 新
ckpt = torch.load('/tmp/m7_smoke_bc/reader_checkpoint_epoch3.pt',
                  map_location='cpu', weights_only=False)
assert ckpt['epoch'] == 2, ckpt['epoch']  # 0-indexed 已完成轮数
assert ckpt['config']['reader_kind'] == 'base'
assert ckpt['config'] == json.load(open('/tmp/m7_smoke_bc/config.json'))
print('BC RESUME OK', {k: log[-1][k] for k in ('epoch', 'train_loss')})
"
echo "SMOKE OK: BC trained (base + attention + resume)"

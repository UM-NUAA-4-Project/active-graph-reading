#!/bin/bash
# 冒烟：CPU 上 1 epoch、8 图 RL 微调（需 CIFAR10 数据 + 上界分类器已训练）
set -e
cd /mnt/c/Users/qq283/Desktop/UM-NUAA
PY=/mnt/d/anoconda/envs/gnn/python.exe
$PY code/m7/train_classifier.py --dataset cifar10 --device cpu \
  --epochs 1 --batch-size 16 --hidden-dim 32 --n-layers 2 \
  --max-graphs 100 --valid-graphs 20 --output-dir /tmp/m7_smoke_clf
$PY code/m7/train_rl.py --dataset cifar10 --reader-kind attention \
  --bootstrap m6/attention/bc/reader_belief_epoch1.pt \
  --classifier /tmp/m7_smoke_clf/cifar10_classifier.pt \
  --device cpu --epochs 1 --graphs-per-batch 4 --seeds-per-graph 1 \
  --max-graphs 8 --hidden-dim 32 --n-layers 2 --m 2 --n-heads 2 \
  --output-dir /tmp/m7_smoke_rl
echo "SMOKE OK: RL fine-tune ran"

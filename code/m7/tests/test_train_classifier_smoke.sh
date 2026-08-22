#!/bin/bash
# 冒烟：CPU 上 2 epoch、200 图训练上界分类器（需 CIFAR10 数据已下载）
set -e
cd /mnt/c/Users/qq283/Desktop/UM-NUAA
PY=/mnt/d/anoconda/envs/gnn/python.exe
$PY code/m7/train_classifier.py --dataset cifar10 --root D:\\superpixels --device cpu \
  --epochs 2 --batch-size 16 --hidden-dim 32 --n-layers 2 \
  --max-graphs 200 --output-dir /tmp/m7_smoke
echo "SMOKE OK: classifier trained"

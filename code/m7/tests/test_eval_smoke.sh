#!/bin/bash
# 冒烟：固定策略 + reader checkpoint 各评估 10 张图（需上界分类器 + 迁移 checkpoint）
set -e
cd /mnt/c/Users/qq283/Desktop/UM-NUAA
PY=/mnt/d/anoconda/envs/gnn/python.exe
$PY code/m7/eval.py --dataset cifar10 --root D:\\superpixels --strategy random \
  --classifier /tmp/m7_smoke_clf/cifar10_classifier.pt \
  --device cpu --max-graphs 10 --output-dir /tmp/m7_smoke_eval
echo "SMOKE OK: eval random strategy"

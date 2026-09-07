#!/usr/bin/env bash
# collab 单线串行管线：sagpool → 固定基线评估 → RL → RL 评估
# tmux 内运行（tmux attach -t agn）；日志 tee 到 /tmp/pipeline_collab.log
set -uo pipefail
cd /mnt/active-graph-reading
PY=/home/vipuser/miniconda3/envs/gnn/bin/python
CLS=code/result/collab/classifier/cw_balanced/collab_classifier.pt
RL_DIR=code/result/collab/experiments/evidence1-balanced/evidence/rl
ROOT=/mnt/data/COLLAB

echo "=== [1/4] SAGPool scorer ==="
if [ -f baseline/artifacts/collab/sagpool.pt ]; then
  echo "skip: sagpool.pt 已存在"
else
  $PY baseline/run_sagpool.py --dataset collab --root $ROOT || exit 1
fi

echo "=== [2/4] 固定基线评估（5 策略 x valid+test, GPU, ref-batch 16） ==="
if [ -f result/collab/fixed_balanced/eval_fixed.json ]; then
  echo "skip: eval_fixed.json 已存在"
else
  $PY code/train_reader/eval.py --dataset collab --root $ROOT \
    --strategy random bfs max-frontier-degree random-walk sagpool \
    --classifier $CLS --ref-batch-size 16 \
    --output-dir result/collab/fixed_balanced || exit 1
fi

echo "=== [3/4] RL（evidence1-balanced, graphs-per-batch 16） ==="
if [ -f $RL_DIR/checkpoint_epoch10.pt ]; then
  echo "skip: epoch10 checkpoint 已存在"
else
  LAST_CKPT=$(ls -t $RL_DIR/checkpoint_epoch*.pt 2>/dev/null | head -1)
  COMMON="--dataset collab --root $ROOT --reader-kind evidence
    --classifier $CLS --reward binary --epochs 10 --lr 1e-4
    --graphs-per-batch 16 --seeds-per-graph 4
    --evidence-lambda 1.0 --temperature 0.2 --exp evidence1-balanced"
  if [ -n "$LAST_CKPT" ]; then
    echo "resume from $LAST_CKPT"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY code/train_reader/train_rl.py --resume "$LAST_CKPT" $COMMON || exit 1
  else
    echo "bootstrap from evidence1 bc reader_checkpoint_epoch3.pt"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY code/train_reader/train_rl.py \
      --bootstrap code/result/collab/experiments/evidence1/evidence/bc/reader_checkpoint_epoch3.pt \
      $COMMON || exit 1
  fi
fi

echo "=== [4/4] RL 评估（checkpoint_epoch10, GPU, ref-batch 16） ==="
$PY code/train_reader/eval.py --dataset collab --root $ROOT \
  --checkpoint $RL_DIR/checkpoint_epoch10.pt --classifier $CLS \
  --ref-batch-size 16 || exit 1

echo "=== PIPELINE ALL DONE ==="

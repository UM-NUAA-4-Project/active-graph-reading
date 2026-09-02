# result/ 目录结构与迁移索引

每个数据集一级目录，内部结构（四级）：

```
result/{dataset}/
├── classifier/                  # 全图上界分类器 checkpoint（train_full_graph.py 输出）
├── fixed/                       # random/bfs 固定策略评估结果 + 图
├── oracle/                      # 上界 oracle 估计（PPA 专属历史产物）
├── filtered/                    # 过滤后训练划分（rank_easy_by_ce.py 输出）
├── comparison/                  # base vs attention / 三阶段对比图
└── experiments/{exp_name}/
    ├── base/{bc,rl}/            # base reader 的 BC/RL 两阶段
    │   ├── config.json          # 实验完整配置（同 checkpoint 内嵌）
    │   ├── training_log.json    # 逐轮日志
    │   ├── reader_checkpoint_epoch{N}.pt / checkpoint_epoch{N}.pt
    │   └── eval_heldout_*.json  # 评估结果 + 对应 PNG（comparison/plot_eval_results.py 生成）
    └── attention/{bc,rl}/       # attention reader 同构
```

## PPA 数据迁移映射（2026-09-02 重构，自旧 m3~m6 目录 move 而来）

| 旧位置 | 新位置 |
|---|---|
| `m3/*.pt` | `result/ppa/classifier/` |
| `m5/base/rl/{asymmetric,binary,tanh}/ex{N}/` | `result/ppa/experiments/rl_{reward}_{exN}/base/rl/` |
| `m5/base/checkpoint/`（BC 产物） | `result/ppa/experiments/bc_baseline/base/bc/` |
| `m5/base/analysis/eval_results_BC/` | `result/ppa/experiments/bc_baseline/base/bc/` |
| `m5/base/analysis/eavl_fixed/`（拼写如此） | `result/ppa/fixed/fixed_eval/` |
| `m5/base/analysis/three_stage_compare/` | `result/ppa/comparison/three_stage_compare/` |
| `m5/base/filtered/` | `result/ppa/filtered/` |
| `m6/attention/bc/` | `result/ppa/experiments/bc_baseline/attention/bc/` |
| `m6/attention/rl/` | `result/ppa/experiments/rl_baseline/attention/rl/` |
| `m4/oracle/` | `result/ppa/oracle/` |

## CIFAR10 迁移映射（自旧 m7 目录 move 而来）

| 旧位置 | 新位置 |
|---|---|
| `m7/checkpoint/classifier/*` | `result/cifar10/classifier/` |
| `m7/checkpoint/bc/{kind}/{exp}/` | `result/cifar10/experiments/{exp}/{kind}/bc/` |
| `m7/results/bc/{kind}/{exp}/` | 同上（评估结果并入） |
| `m7/checkpoint/rl/{kind}/{exp}/` | `result/cifar10/experiments/{exp}/{kind}/rl/` |
| `m7/results/rl/{kind}/{exp}/` | 同上 |
| `m7/checkpoint/transfer/{kind}-asymmetric/` | `result/cifar10/experiments/transfer_asymmetric/{kind}/init/` |
| `m7/results/transfer/{kind}-asymmetric/` | 同上（评估结果并入 init 目录） |
| `m7/results/fixed/` | `result/cifar10/fixed/` |

注意：旧 m7 结果目录名 `2-teacher`（results）与 `2-teachers`（checkpoint）
不一致，已合并为 `2-teachers`。

⚠️ **历史评估 JSON 视为只读**：对迁移来的 checkpoint 重跑 eval 时，
默认输出目录是 checkpoint 同目录，会覆盖历史结果（git 只保留部分历史）。
请显式传 `--output-dir` 指到新位置，或先备份。

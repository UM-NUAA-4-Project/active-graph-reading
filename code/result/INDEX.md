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

## eval 结果 JSON 统一格式（2026-09-05 起）

自 2026-09-05 起 `train_reader/eval.py` 一次运行评估全部可用 held-out split
（valid+test，cifar10 无验证集自动跳过）并合并输出**单个** JSON：

| 来源 | 输出 | 顶层 strategy 键 |
|---|---|---|
| 固定策略 `--strategy random bfs` | `result/{ds}/fixed/eval_fixed.json` | `random`、`bfs` |
| reader checkpoint | `<checkpoint 同目录>/eval_learnable.json` | `learnable` |

Schema（与 m5 历史参考文件同构，comparison 脚本直接消费）：

```
{valid|test} → {strategy} → {s0..sN, mean} → "<budget:.2f>" → {mean, std, n}
per_class_{split} → {strategy} → {s0..sN, mean} → {class} → ...
acc_clf_reference_{split}   # 仅 evidence reader（冻结分类器参照）
config.full_graph_ref       # 分类器全图 acc（现场评估），plot_three_stage 上界虚线
```

历史三代 eval JSON（m7 `eval_heldout_{split}_*.json`、m6 `evaluate/` 子目录、
旧 fixed 命名）已用 `tools/migrate_eval_json.py` 合并重命名为上述 canonical
文件（dry-run 默认、--write 落盘、--archive 归档旧文件至 `_legacy_eval/`）；
被合并的旧文件默认保留原位，勿再作为绘图输入。已知遗留：collab fixed 评估
用预算档 0.05/0.10/0.15 而 reader 用 0.10/0.15/0.20，三阶段图 `--budgets`
需取交集。

## 跨数据集联合图（2026-09-25 新增，同日分子文件夹重组）

`result/cross-dataset-figures/` 按分析类型分子文件夹收录 cifar10 + COLLAB
联合分析图（单数据集图仍在各自目录）：

- `behavior/choice_behavior_1x4.*`（1×4 单行行为分析合并图，a/b=CIFAR10、
  c/d=COLLAB；CIFAR10 数据自 SVG 提取、口径=epoch3，注意 epoch6 行为反转）
- `reward-curves/rl_reward_curves_1x4.*`（1×4 单行 RL 奖励曲线：a/b=scratch-RL
  对比（图例 BC warm-start vs scratch-RL），c/d=奖励源切换链，矢量拼接）
- `state-attention/attn_dist_*`（state-attention 读出注意力分布重绘版：
  曲线面板改结尾对齐绝对步长坐标（0, …, last-15, …, last-1 + terminal），
  移除 frozen decoder；数据=100 图×4 seeds 重采（2026-09-25），旧
  `state-attn-figures/` 与旧归一化分箱 eval JSON 已按用户指令清理）

生成脚本：`code/comparison/plot_choice_behavior_1x4.py`、
`code/comparison/merge_reward_svgs_1x4.py`、`code/comparison/plot_attn_dist.py`
（评估端 `code/train_reader/eval_state_attention.py` 新增 attn_dist_last）；
细节见目录内 README.md。

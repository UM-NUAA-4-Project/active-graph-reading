# Attention Reader 废弃清理清单（已执行，2026-09-20）

> evidence reader 为现役设计，attention reader（belief-state 架构）废弃。
> 本清单为执行记录：全部项已于 2026-09-20 完成并全量回归通过
> （13 个单测文件 + smoke_test.py 全绿）。决策记录见 `docs/decisions.md`
> 同日条目 "Attention reader removed"。

## 0. 边界声明 —— 未动项（验证过零耦合）
- [x] state-attention 新轨：`train_state_attention.py` / `eval_state_attention.py` /
      `result/**/experiments/state-attn/`（无 reader/attention import）
- [x] evidence reader 全部（`evidence.py` 含 return_states）+ ev*/ablation_* 实验
- [x] base reader `learnable_reader.py`；`reader/common.py`（stable_probs，base/evidence 共用）
- [x] `baseline/` 全部；`classifier/`（曾被 attention 单向 import，反向无依赖）
- [x] `tools/ckpt_io.strip_classifier_keys`（保留，旧 ckpt 兼容无害）
- [x] `docs/design/evidence-reader*.md`、`docs/Summary of results.md` 的 attention 小节
      与 `docs/figures/attention-reader.png`（历史记录，含图引用，保留）

## A. 纯 attention 代码包（已删除，622 行）
- [x] `code/active_graph_reader/reader/attention/{__init__,attention_reader,belief_state,frontier_pointer,new_info_encoder}.py`
- [x] `code/configs/train_bc.example.json`、`train_rl.example.json` → 重写为 evidence 模板

## B. 接线剪除
### B1 必剪（已完成）
- [x] `reader/__init__.py`、`active_graph_reader/__init__.py`：AttentionReader 导出移除
- [x] `active_graph_reader/config.py`：kind 元组 → base/evidence；同宽校验删除；
      belief_state 推断删除（旧 attention ckpt 现被 validate() 干净拒绝）；
      build_reader attention 分支删除；**`m`/`n_heads` 字段及全部序列化管道删除**
- [x] `tools/cli.py`：`--reader-kind` choices → base/evidence；`--m`/`--n-heads` 删除
- [x] `tests/smoke_test.py` 合成评估改用 base reader；
      `test_config_reader.py`（roundtrip 改 evidence、validate 反例、infer 改 evidence）；
      `test_checkpoint_compat.py`（m6 attention 用例删除，m7 循环去 attention）；
      `test_unified_loops.py`（BC/RL 循环 base/attention → base/evidence，coverage 反升）

### B2 死代码剪除（已完成）
- [x] `core/bc.py`：use_belief 全套分支移除（base/evidence 一套实现）
- [x] `core/rl.py`：use_belief 分支 + MICRO_BATCH_SIZE 梯度累积 + `_diag_nan` 移除
- [x] `core/eval_loop.py`：use_belief 分支移除
- [x] `core/common.py`：`reveal_and_update_beliefs_batch` 删除；
      `teleport_or_deactivate` 去 reader/subgraphs 参数；`core/__init__.py` 再导出移除
- [x] diagnosis 三脚本：use_belief 分支移除；`diagnose_checkpoint.py` 的
      `--reader-kind` choices 补 evidence（原只有 base/attention）；
      m/n_heads 打印移除（eval_reader.py 同）
- [x] `train_rl.py` **默认 reader_kind: attention → evidence（行为变化）**；
      train_bc/transfer/eval_reader/train_reader/__init__ docstring 同步
- [x] 过时 docstring：comparison/plot_training_curves.py、plot_bc_gap.py
      （示例路径改指 ev1 / ev-R-read 实际产物）、trajectories/core/rollout.py、
      diagnosis/__init__.py

## C. 实验产物（已删除；5 顶层目录、约 378MB）
- [x] `result/cifar10/experiments/1-teacher/attention/{bc,rl}/`（126M）
- [x] `result/cifar10/experiments/richer-balanced-trajectory/attention/{bc,rl}/`（141M）
- [x] `result/cifar10/experiments/transfer_asymmetric/attention/init/`（63M，全 untracked）
- [x] `result/ppa/experiments/bc_baseline/attention/bc/`（50M，含 evaluate/）
- [x] `result/ppa/experiments/rl_baseline/attention/rl/`（含 evaluate/）
- 组成：30 个 .pt（~375MB，从未进 git，不可恢复——decisions.md 2026-09-19 已判
  attention 评估因 dropout 未关无效，无保留价值）；22 个 json（2026-09-09 已 untrack）；
  15 个 git-tracked png 随目录删除（工作区显示为 deletion，随下次提交落库）

## D. 连带孤儿产物（已删除）
- [x] `result/ppa/comparison/{bc_gap,bc_gap_rl,teacher_gap}.png` + 同名 summary.json
      （数据源为已删 attention eval JSON；three_stage 与 info_ceiling 保留）
- [x] `result/cifar10/diagnosis/rl/diagnosis.json`（reader_kind=attention、bootstrap
      指向早已不存在的 m7 路径）

## E. 数据/评估文件（核对无需动作）
- [x] `code/data/`：teacher 产物，reader 无关，无 attention 路径
- [x] `result/{ds}/fixed/eval_fixed.json` 及 new-fixed/：零 attention 条目
      （PPA 三散文件已于同日合并进 `ppa/fixed/eval_fixed.json`，7 策略全量）

## F. 服务器（26016 / 20928）—— 待手动
- [ ] 服务器为 git clone：代码剪除经 git pull 生效后，服务器本地
      `code/result/**/attention/` 目录不会被自动删除，需手动 rm

## G. 执行核对（已完成）
- [x] 全部单测 + smoke 通过（B1 后、B2 后、终验共三轮）
- [x] `git status`：23 D（5 包文件 + 15 实验 png + 3 gap png）+ 28 M，无预期外条目
- [x] `find code/result -type d -name attention` → 0
- [x] 全仓 grep：`AttentionReader` / `use_belief` / `init_beliefs` /
      `reveal_and_update_beliefs` 仅存于本清单与 decisions.md 历史条目

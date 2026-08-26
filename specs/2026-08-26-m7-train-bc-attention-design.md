# M7 train_bc 增加 attention reader 支持

日期: 2026-08-26
状态: 设计已批准

## 背景

`code/m7/train_bc.py` 目前只支持 base reader（`LearnableReader`，无状态打分）。
AttentionReader（`active_graph_reader/reader/attention/attention_reader.py`）的
`forward_states` 依赖 cross-step belief token（`beliefs` 参数），揭示后需用
`m6.training_utils.reveal_and_update_beliefs_batch` 批量更新 belief 与复用 subgraph，
与 base 的简单锁步循环（`reveal_node` 即止）本质不同，不能共用同一循环。

现有一致参考：`m6/attention/train_bc.py`（belief lockstep 版 BC 训练）与
`m7/train_rl.py`（已有的 `train_one_epoch_base` / `train_one_epoch_attention` 双轨结构）。
`m7/eval.py` 已支持 attention checkpoint：从 checkpoint `config` 读 `reader_kind`
决定 reader 类型，并按 `m`/`n_heads`/`temperature`（及 `n_layers_reader`/`dropout_reader`
别名）重建超参。

## 目标

`train_bc.py` 增加 `--reader-kind {base, attention}`（默认 `base`，向后兼容），
attention 模式产出与 `m7/eval.py` 兼容的 checkpoint。数据加载（m7 registry 轨迹）、
教师合并、epoch/checkpoint 流程均不变。

## 设计

### 1. 新参数

- `--reader-kind` choices=`['base', 'attention']`，默认 `base`
- `--m` int 默认 4（belief token 数，仅 attention 有效）
- `--n-heads` int 默认 4（attention 头数，仅 attention 有效）
- `--temperature` float 默认 1.0（BC 训练 softmax 温度，仅 attention 有效）
- `--output-dir` 默认改为 `None` → 解析为 `m7/checkpoint/bc/{reader_kind}`
  （base 时解析结果与现状 `m7/checkpoint/bc/base` 完全一致）

### 2. 训练循环双轨（对齐 m7/train_rl.py 结构）

- 现有 `train_one_epoch` 改名 `train_one_epoch_base`：逻辑零改动。
- 新增 `train_one_epoch_attention`：移植 `m6/attention/train_bc.py` 的
  belief lockstep——Pass 1 建 state + `init_beliefs`；Pass 2 每步
  `forward_states(states, temperature, beliefs, subgraphs)` 打分 → 收集教师动作与
  BC loss（`-log π(teacher)`）→ `reveal_and_update_beliefs_batch` 批量揭示并更新
  belief/subgraph；Pass 3 纯 BC loss 反传（无分类头、无 CE 项）。
- 教师动作不在 frontier（ValueError）或 frontier 为空的轨迹直接跳过（同现状/m6）。

### 3. Reader 构造

- base：`LearnableReader(edge_feat_dim, hidden_dim, n_layers, dropout, node_feat_dim)`
  （同现状）。
- attention：`AttentionReader(edge_feat_dim, hidden_dim, n_layers, dropout, m, n_heads,
  node_feat_dim)`；edge/node feat dim 均取 `spec.*`（同现状）。
  BC 训练按教师动作 softmax（`stochastic=False`，温度由 `--temperature` 传入
  `forward_states`，对齐 m6）。

### 4. Checkpoint config

现有 config dict 中 `reader_kind` 由硬编码 `'base'` 改为动态 `args.reader_kind`，
并补充 `m`/`n_heads`/`temperature`（键名与 `m7/train_rl.py` 一致）。
eval 侧的 `n_layers_reader`/`dropout_reader` 别名不需要改动。

### 5. 冒烟测试

`code/m7/tests/test_bc_smoke.sh` 追加 attention 分支（`--reader-kind attention`，
1–2 epoch，CPU 小模型），校验 checkpoint config 含 `reader_kind == 'attention'`、
`m`/`n_heads` 键。现有 base 断言保持不动。

## 兼容性

- 默认参数（`--reader-kind base`）下：训练循环、checkpoint 结构、输出路径、
  config 键全部与现状一致，`test_bc_smoke.sh` base 分支不受影响。
- checkpoint → eval：attention checkpoint 的 `config.reader_kind == 'attention'`，
  eval 经 `m7.train_rl.build_reader` 重建 AttentionReader；超参从 config 解析
  （CLI 显式覆盖时以 CLI 为准，逻辑在 eval.py 内，无需改动）。

## 不做的事（YAGNI）

- 不加 m6 式的 `--teacher`/`--finetune`/`--resume`/`--max-trajs` 参数
  （m7 train_bc 设计本来如此）。
- 不改数据加载、轨迹格式、`eval.py`、`train_rl.py`。
- 不动 `test_bc_utils.py` 对 `split_trajectories` 的既有引用（遗留问题，另行处理）。

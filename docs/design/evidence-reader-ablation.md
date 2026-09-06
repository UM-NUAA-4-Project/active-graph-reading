# Evidence Reader 消融方案（v2）

日期：2026-09-06 ｜ 状态：方案定稿（A2–A4 待实现一个 `--evidence-variant` 开关）
配套：`docs/design/evidence-reader.md`（v2 设计）；主对比（base / attention /
evidence 三 reader × 三阶段）不属于本文，见 result/INDEX 与实验计划。

约定：预算档 0.10 / 0.15 / 0.20，`--n-seeds 4`，与主表口径一致；
消融在 **collab + cifar10** 两个数据集上进行（PPA 主训练进行中，不加载）；
同一 (reader, budget) 的读数与主表同源（统一 eval JSON），保证可比。

## 1. 消融矩阵总览

| 档 | 消融对象 | 改动 | 成本 | 回答的问题 |
|---|---|---|---|---|
| A1 | 终判来源：证据头 vs 冻结分类器 | **无** | 零（已有数据） | "决策-分类一体终判"是否优于外部裁判 |
| A2 | 历史引导注意力读出（evidence_pool） | `--evidence-variant no_pool` | BC 重训（轨迹复用） | 积分器是否需要"回头看"已读证据的哪部分 |
| A3 | 递归积分器（evidence_lstm） | `--evidence-variant no_lstm` | BC 重训（轨迹复用） | 跨步证据是否需要时间维度深度积分（设计 §2 核心论证） |
| A4 | 两者全去（= v1 档） | `--evidence-variant v1` | BC 重训（轨迹复用） | 证据流整体 vs 逐步独立判读 |
| A5 | L_ev 关闭（λ=0） | **无**（`--evidence-lambda 0`） | BC+RL 重训 | 监督信号对共享编码器/选点的整形作用 |
| A6 | RL 奖励源 | **无**（`--reward`） | RL 重训 | 奖励形式（binary/tanh/asymmetric）对策略的影响 |

优先级：A1 → A2/A3/A4 → A6 → A5（A5 最低，见 §3 读数限制）。

## 2. 各档定义与读数

### A1 终判来源对比（免费）

每个 evidence checkpoint 的 eval 已同时输出：
- `acc_final`（读满后补充前向的证据头终判）；
- `acc_clf_reference_{split}`（冻结分类器判同一揭示子图）。

读数：两列按 budget 并排 + Δ。若证据头稳定 ≥ 冻结分类器，则"历史积分
终判"成立；若只在低预算档成立，说明证据头的优势集中在信息稀缺区。

### A2 `no_pool`——去历史引导读出

保留 LSTM（h_t 照常递归），读出退化为**非引导**均值池化：

```
h_t = LSTM(Δ_t, h_{t-1})          # 不变
z_t = mean(R_t)                   # 替换 AttnPool(h_t, R_t)
```

回答：h_t 作为 query 引导 K/V 重读的贡献。若 A2 ≈ full，说明积分器本身
已携带足够判读信息，注意力读出是装饰。

### A3 `no_lstm`——去递归积分

保留 AttnPool 机制，积分器退化为**无递归直通**：

```
h_t = Δ_t                         # 替换 LSTM 步（hc 状态不再使用）
z_t = AttnPool(h_t, R_t)          # 机制不变，query 换成最新证据
```

回答：时间维度积分的必要性。A3 退化多少，直接检验设计文档 §2
"信息冗余、计算必要"的论证——尤其 collab 这类长直径图应当对 A3 最敏感。

### A4 `v1`——两者全去（设计文档预留的 v1 档）

```
h_t = Δ_t,  z_t = mean(R_t)
```

即 v1（纯 base + 证据头）。A4 与 full 的差距 = 证据流整体贡献；
A2/A3 与 A4 的差距 = 各组件的**独立**贡献。

### A5 λ=0——L_ev 关闭（可选，最低优先级）

`--evidence-lambda 0`：证据流不前向、不训练。**读数限制**（设计文档 §6.1
已注明）：终判头是随机初始化，`acc_final` 无意义；只能看教师动作一致率
`act_acc`（BC）与下游 RL 的策略效果——用于回答"逐步预测监督是否反过来
整形了共享编码器/选点"。只在 RL 阶段接一个 λ=0 的 BC 起点，对比主 RL。

### A6 RL 奖励源消融

从同一 BC checkpoint bootstrap，分别用三种奖励跑 RL：
binary（默认）、tanh、asymmetric。读数：RL 后 eval_learnable 的
budget 曲线。回答：奖励形式对策略的影响，也是 asymmetric 设计的依据。

## 3. 代码改动需求（仅 A2/A3/A4 需要）

现状核对结论：`EvidenceReader` 的 LSTM / HistoryPool 无开关（写死于
`__init__` 与 `forward_states_evidence`），`ReaderConfig` 无对应字段；
A1/A5/A6 均为现有 CLI 参数，零改动。

A2/A3/A4 需要 **一个 `--evidence-variant` 开关**，改动点：

1. `active_graph_reader/config.py`：`ReaderConfig` 增
   `evidence_variant: str = 'full'`（'full' | 'no_pool' | 'no_lstm' |
   'v1'），`to_dict`/`from_dict` 增键（旧 ckpt 无此键 → 默认 full，
   兼容），`validate()` 增取值与 kind=='evidence' 检查；
2. `reader/evidence.py` `__init__`：variant 非 full 的组件不创建
   （`no_pool` 不建 evidence_pool；`no_lstm`/`v1` 不建 evidence_lstm），
   state_dict 键随之收敛；
3. `reader/evidence.py` `forward_states_evidence`：按 variant 走
   z_t / h_t 分支（约 6 行；`no_lstm` 时 new_hc 原样返回占位零状态）；
4. `train_reader/core/build.py`：`reader_config_from_checkpoint` 的
   embedded merge 已自动携带新键（权重 shape 推断不受缺键影响），
   预计零改动，需加一条回归断言；
5. `tools/cli.py add_reader_args`：增 `--evidence-variant`（仅
   reader_kind=evidence 时生效）；
6. `tests/test_evidence_reader.py`：每 variant 一组前向 + 防毒断言
   （`no_pool` 下扰动 h_t 不得改变 z_t；`no_lstm` 下扰动 (h,c) 不得
   改变输出；`v1` 两者皆然），并注册进 smoke_test。

预计 ~100 行含测试，半天内可完成并保持旧 checkpoint 兼容
（无 evidence_lstm/pool 键的旧权重只能重建 full/对应 variant——加载时
`strict=True` 会按 config 生成空缺模块，天然一致）。

## 4. 命令（从仓库根运行）

A1：直接消费现有 eval JSON，无命令。

A2/A3/A4（BC，教师轨迹复用，每 variant 每 数据集一条；PPA 待主训练
完成后同式追加）：

```bash
# 以 collab 为例（cifar10 换 --root D:\data\superpixels 与 --data-dir）
python code/train_reader/train_bc.py --dataset collab \
    --root D:\\data\\COLLAB \
    --data-dir code/data/collab/trajectories_filtered \
    --reader-kind evidence --evidence-variant no_pool \
    --exp ablation_no_pool --epochs 10
# no_lstm / v1 同式替换 variant 与 exp 名
```

A5（可选）：

```bash
python code/train_reader/train_bc.py --dataset collab \
    --root D:\\data\\COLLAB \
    --data-dir code/data/collab/trajectories_filtered \
    --reader-kind evidence --evidence-lambda 0 \
    --exp ablation_lambda0 --epochs 10
```

A6（RL，`--bootstrap` 用各自数据集的主 BC 产物）：

```bash
python code/train_reader/train_rl.py --dataset collab \
    --root D:\\data\\COLLAB --reader-kind evidence \
    --bootstrap <主BC checkpoint> --classifier <collab 上界分类器> \
    --reward tanh --exp ablation_reward_tanh
# binary / asymmetric 同式替换；binary 为主表默认，可只补 tanh 一档
```

评估（所有消融 checkpoint 统一走主仓 eval，产出同目录
`eval_learnable.json`）：

```bash
python code/train_reader/eval.py --dataset collab \
    --root D:\\data\\COLLAB --checkpoint <消融 checkpoint> \
    --classifier <collab 上界分类器>
```

## 5. 报表格式

每数据集一张表：行 = full / no_pool / no_lstm / v1（A6 另起一张：
binary / tanh / asymmetric），列 = B10/B15/B20（acc_final）+
full-graph 参照；A1 以 Δ(acc_final − acc_clf_reference) 列并入主表。
结论句式固定为"组件 X 的移除在 <预算档> 造成 <Δ> 退化，其中
<数据集> 最敏感（对应其 <结构特性>）"。

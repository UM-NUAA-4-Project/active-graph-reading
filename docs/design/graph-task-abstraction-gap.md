# 图级任务抽象缺口（现状盘点与改进方向）

日期：2026-09-06
状态：设计备忘（记录 gap，不立即实现）
动机：collab 塌缩调查与换数据集评估（peptides-functional / MalNet-Tiny /
REDDIT-MULTI-12K / ogbg-mol 系列）暴露：当前管线对"单标签互斥多分类"
这一种任务形态有大量硬编码假设，接入其它图级任务（多标签、多任务二分类、
回归）需要散落多处的一致性改动，而不是一处扩展。

## 1. 哪些层是任务无关的（已经做对的）

| 层 | 说明 |
|---|---|
| engine（GraphState / reveal / RevealLoop） | 纯结构操作，不接触标签 |
| 固定策略与 baseline（random/bfs/random-walk/SAGPool scorer 选点/SubgraphX 规划结构） | 选点逻辑任务无关（注意 SubgraphX 的**奖励打分**是例外，见 §2） |
| 统一 eval JSON 的叶子格式 `{mean, std, n}` 与 split/strategy 层级 | 数值语义由 config 描述，格式本身任务无关 |
| budget / seed / frontier 协议（P1–P6） | 任务无关 |

## 2. 硬编码了"单标签分类"的位置（gap 清单）

按数据流顺序：

1. **`datasets/registry.DatasetSpec`**：只有 `n_classes` 一个任务字段，
   隐含"每图一个互斥整数标签"。`Dataset.*_labels` 是 `(N,)` 整型
   Tensor。没有 task 类型（single-label / multi-label / multi-task /
   regression）与标签形状声明。→ 接 peptides-functional 这类
   multi-hot 数据集时，任务定义只能在 loader 里手工"削"成单标签
   （one-hot 子集 / primary label），或放弃。

2. **`train_classifier/train_full_graph.py`**：`train_one_epoch` 硬编码
   `cross_entropy`；`evaluate` 硬编码 argmax accuracy。损失与指标没有
   factory，选任务即改代码。

3. **`train_reader/core/eval_loop.py`**：`evaluate_classifier_full` /
   `evaluate_fixed` / `evaluate_reader` 的正确性判定全部是
   `argmax == label`；`build_eval_json` 的 `per_class_{split}` 假设
   互斥类（按类别聚合 recall）。

4. **RL 奖励（`train_reader/train_rl.py` 及 reward 实现）**：奖励来源是
   冻结分类器对揭示子图的 argmax 正确性（asymmetric reward 等）。
   多标签/回归下"正确性"未定义——奖励需要由任务指标派生
   （micro-F1 / R² 增量等），目前没有这个间接层。

5. **baseline 的奖励/终判**：SubgraphX 的 MCTS 奖励取 `max_c prob`
   （类别无关但仍是互斥类假设）；evidence reader 的证据头同样以
   CE/argmax 为中心。更换任务形态时这些是第二批次改动点。

6. **`classifier`（GraphClassifier / head）**：输出维度 `n_classes`
   隐含分类头；回归需要独立 head（输出维 = 目标数）与数值目标形状。

7. **comparison/plot 脚本**：纵轴语义固定为 accuracy；`qa_three_stage`
   等报表同样。

## 3. 改进方向（提案，按最小侵入排序）

### 阶段 1：TaskSpec + loss/metric factory（推荐先做）

- `DatasetSpec` 增加字段：
  `task: 'single_label' | 'multi_label' | 'multi_task' | 'regression'`
  （默认 `'single_label'`，全部现有数据集零改动）、
  `label_dim: int`（multi-label 的标签数 / regression 的目标数）。
- 新建 `code/datasets/tasks.py`（或 `code/tasks/`）：
  - `loss_fn(task)`：single_label→CE；multi_label→BCEWithLogits；
    regression→MSE；
  - `predict(task, logits)`：argmax / sigmoid>0.5 / 原值；
  - `score(task, pred, label)`：accuracy / micro-F1 / R²，返回**每图
    0..1 标量**——这是奖励与评估共用的唯一接口。
- `train_full_graph` / `eval_loop` / RL reward 全部改为经此接口取
  loss/predict/score，单标签路径行为不变（默认值兜底）。

### 阶段 2：eval JSON 语义扩展

- config 增加 `task` 与 `metric` 字段；叶子数值含义由其解释。
- `per_class_{split}` 在 multi-label 下改名/改形为 `per_label_{split}`
  （每标签 F1/recall）；plot 脚本按 config 分派。schema 层级不动。

### 阶段 3（后置）：奖励结构化

- RL 奖励从"正确性"泛化为 `score(revealed) - score(baseline)` 型
  函数族（现在的 asymmetric reward 是其特例）；SubgraphX 的 MCTS
  奖励复用同一 `score` 接口（其 max_c prob 是 single_label 特化）。

## 4. 决策记录（2026-09-06）

- 短期不实现阶段 1；选数据集优先迁就现有单标签管线
  （cifar10 / PPA / ogbg-mol 单标签成员 / peptides-functional
  one-hot 子集均是可行解），把"接入新任务形态"留作管线级改造。
- 本文档替代在 loader 里做一次性 task hack 的隐性方案：任何新数据集
  若需要非单标签任务定义，先在此登记 gap，再决定走阶段 1 还是削任务。

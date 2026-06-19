# M4 阶段总结报告：Teacher 轨迹生成与蒸馏可行性分析

**报告类型**: 阶段总结（M4 完成）
**日期**: 2026-06-11
**周期**: 2026-05-24（M4 设计）→ 2026-06-11（上界分析完成）
**关联文档**: `docs/progress/daily/2026-5-24 M4-implementation.md`（设计）、`docs/roadmap.md`（总体路线）

---

## 一、M4 在整体路线中的角色

```
M3: 固定策略 + 基准评测  → 回答"策略选择有空间吗？" → PASS
M4: 轨迹收集 + 质量评分  → 回答"好的阅读轨迹长什么样？可蒸馏吗？" ← 本阶段
M5: 策略模仿学习          → 回答"能不能学会像好轨迹那样读？" → 进行中
```

M4 的核心产出不是模型，而是：

1. 一批高质量 teacher 阅读轨迹（用于 M5 BC 训练）
2. Oracle 性能上界估计（用于设定 M5 目标）
3. 轨迹蒸馏可行性判断（决定 M5 走 BC 还是端到端）

---

## 二、完成的工作

### 2.1 Teacher 实现（5 种）


| Teacher                 | 信号来源                     | 文件                                | 状态 |
| ----------------------- | ---------------------------- | ----------------------------------- | ---- |
| T1: Gradient            | 分类器梯度归因分数           | `m4/teachers/gradient_teacher.py`   | 完成 |
| T3: Structural-Degree   | 全图度中心性                 | `m4/teachers/structural_teacher.py` | 完成 |
| T3: Structural-PageRank | 全图 PageRank                | `m4/teachers/structural_teacher.py` | 完成 |
| T4: Hybrid α=0.30      | 30% gradient + 70% structure | `m4/teachers/hybrid_teacher.py`     | 完成 |
| T4: Hybrid α=0.70      | 70% gradient + 30% structure | `m4/teachers/hybrid_teacher.py`     | 完成 |

**未使用的 teacher**：

- **T2: Info-Gain** — 设计文档中列为"备选"。原因：每步需对每个 frontier candidate 做一次前向传播，PPA 图平均 frontier 20-50，B=15 时每图需 ~450 次前向。在训练集 158K 图上计算成本过高。当前数据规模下优先级低，但保留设计作为验证集质量基准的选项。

### 2.2 四阶段轨迹管线

```
Stage 1: Raw Generation
  generate_trajectories.py → data/raw/
  每图生成 multiple-teacher × single-seed 轨迹

Stage 2: Filter & Merge
  过滤错误轨迹（Acc=0 丢弃）→ data/stage1_merged/
  合并多 teacher 轨迹 → data/stage2_filtered/

Stage 3: Trim
  按最小轨迹数裁剪，确保每图轨迹数一致 → data/stage3_trimmed/

Stage 4: Class-Balance
  按 label 分布平衡采样 → data/stage4_balanced/
  产出: trajectories_balanced_r0_050.pt ~ r0_200.pt
```

### 2.3 最终轨迹数据统计


| Budget       | 轨迹总数 | 缺失类别数                                               | 文件大小 |
| ------------ | -------- | -------------------------------------------------------- | -------- |
| 5% (r0_050)  | 16,806   | 13                                                       | 4.8 MB   |
| 10% (r0_100) | 22,124   | 5                                                        | 6.3 MB   |
| 15% (r0_150) | 23,801   | 4                                                        | 7.1 MB   |
| 20% (r0_200) | 5,138    | 12（剪裁了很多冗余高预算轨迹，只保留高预算分类难的轨迹） | 3.5 MB   |

**各 teacher 轨迹数（以 r0_100 为例）**:


| Teacher             | 轨迹数 | 占比  |
| ------------------- | ------ | ----- |
| gradient            | 3,651  | 16.5% |
| structural-degree   | 6,764  | 30.6% |
| structural-pagerank | 3,369  | 15.2% |
| hybrid-a0.30        | 4,899  | 22.1% |
| hybrid-a0.70        | 3,441  | 15.6% |

**类别覆盖问题**：class 0, 5, 7,8,9,10, 11, 13, 14, 15,16, 19, 20, 22, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35。原因：低 budget 下某些类别难以从单个随机 seed 获得正确预测，同时存在分类器自身 难分类/无法分类 的类别。

**CE 分析亮点**：

- class 17 在几乎所有 teacher 下的平均 CE 接近 0（~1e-6），说明 teacher 对此类别的判断高度一致且确定
- class 18（最大类别之一）的 CE 在 0.17–0.32 范围，teacher 间存在分歧
- structural-pagerank teacher 的 CE 普遍高于其他 teacher，说明 PageRank 信号与分类目标的相关性较弱

### 2.4 Oracle 上界估计

**实验规模**：9,250 张训练图（k=250/class），2 种 seed 来源，4 种 reader，3 个 budget

**核心结果**（gradient seed）：


| Budget | Oracle (gradient/Struct-Deg) | BFS   | Random | Oracle-BFS Δ |
| ------ | ---------------------------- | ----- | ------ | ------------- |
| 5%     | 0.067/0.063                  | 0.051 | 0.036  | +0.016/+0.012 |
| 10%    | 0.095/0.104                  | 0.068 | 0.051  | +0.027/+0.036 |
| 15%    | 0.109/0.130                  | 0.084 | 0.064  | +0.026/+0.047 |

详细分析见：`docs/progress/daily/2026-06-11-m4-oracle-upper-bound-analysis.md`

### 2.5 分析与可视化产出

`code/m4/analysis/` 目录包含 16 张分析图表：

- 4 个 budget × CE heatmap（各 teacher × label 的 CE 分布）
- 4 个 budget × label 分布图（轨迹的类别覆盖）
- 4 个 budget × teacher label 分布图（各 teacher 的类别偏好）
- CE trend across budgets（CE 随 budget 变化趋势）
- Cross-budget CE comparison（跨 budget 对比）
- Zero-trajectory heatmap（缺失类别热力图）

`code/m4/oracle/` 目录包含：

- `oracle_report.json`（完整 oracle 评估数据）
- `oracle_accuracy.png`（accuracy 分组柱状图）
- `oracle_variance.png`（标准差分组柱状图）

## 三、M5 就绪状态

### 3.1 数据就绪

- `m4/data/stage4_balanced/` 包含 4 个 budget 的类别平衡轨迹（~67K trajectories total）
- 轨迹格式：`Trajectory(graph_idx, seed_node, reveal_ratio, reveal_order, teacher_type, ...)`
- 可直接被 `m5/train_bc.py` 的 `merge_trajectories()` 加载

### 3.2 代码就绪

- `m5/train_bc.py`：BC 训练主脚本（batch 训练、按 graph_idx split、多 budget 验证、checkpoint 保存）
- `m5/smoke_test.py`：单轨迹 forward→loss→backward→update 管线验证（已通过）
- `active_graph_reader/reader/learnable_reader.py`：LearnableReader（FrontierEncoder + ReaderMLP）

### 3.3 训练命令

```bash
python m5/train_bc.py \
  --pretrained m3/best_model_ft.pt \
  --root D:\ogb-master\dataset \
  --device cuda:0 \
  --epochs 10 \
  --lr 1e-3 \
  --batch-size 16
```

## 四、下一步


| 优先级 | 任务                                                         | 状态     |
| ------ | ------------------------------------------------------------ | -------- |
| P0     | 服务器上运行`m5/train_bc.py`，获取首轮 BC val acc            | 待执行   |
| P1     | BC val acc vs BFS baseline 对比，决定是否需要 Gumbel-Softmax | 依赖 P0  |
| P2     | 如 BC ≈ BFS，实现 Gumbel-Softmax 端到端微调                 | 待评估   |
| P3     | 如 BC < BFS，排查 reader 架构或训练超参                      | 待评估   |
| —     | 考虑使用 all 158K training graphs 而非仅轨迹覆盖的子集       | 可选增强 |

**M4 阶段完成。M5 BC 训练为当前最高优先级。**

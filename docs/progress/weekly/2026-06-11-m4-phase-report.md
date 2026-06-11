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

| Teacher | 信号来源 | 文件 | 状态 |
|---------|---------|------|------|
| T1: Gradient | 分类器梯度归因分数 | `m4/teachers/gradient_teacher.py` | 完成 |
| T3: Structural-Degree | 全图度中心性 | `m4/teachers/structural_teacher.py` | 完成 |
| T3: Structural-PageRank | 全图 PageRank | `m4/teachers/structural_teacher.py` | 完成 |
| T4: Hybrid α=0.30 | 30% gradient + 70% structure | `m4/teachers/hybrid_teacher.py` | 完成 |
| T4: Hybrid α=0.70 | 70% gradient + 30% structure | `m4/teachers/hybrid_teacher.py` | 完成 |

**未实现的 teacher**：
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

| Budget | 轨迹总数 | 缺失类别数 | 文件大小 |
|--------|---------|-----------|---------|
| 5% (r0_050) | 16,806 | 13 | 4.8 MB |
| 10% (r0_100) | 22,124 | 5 | 6.3 MB |
| 15% (r0_150) | 23,801 | 4 | 7.1 MB |
| 20% (r0_200) | 5,138 | 12 | 3.5 MB |

**各 teacher 轨迹数（以 r0_100 为例）**:

| Teacher | 轨迹数 | 占比 |
|---------|--------|------|
| gradient | 3,651 | 16.5% |
| structural-degree | 6,764 | 30.6% |
| structural-pagerank | 3,369 | 15.2% |
| hybrid-a0.30 | 4,899 | 22.1% |
| hybrid-a0.70 | 3,441 | 15.6% |

**类别覆盖问题**：预算 5% 时 13 个类别无轨迹（class 0, 5, 10, 11, 13, 14, 16, 19, 20, 27, 28, 32, 35），预算 20% 时 12 个类别缺失。原因：低 budget 下某些类别难以从单个随机 seed 获得正确预测。

**CE 分析亮点**：
- class 17 在几乎所有 teacher 下的平均 CE 接近 0（~1e-6），说明 teacher 对此类别的判断高度一致且确定
- class 18（最大类别之一）的 CE 在 0.17–0.32 范围，teacher 间存在分歧
- structural-pagerank teacher 的 CE 普遍高于其他 teacher，说明 PageRank 信号与分类目标的相关性较弱

### 2.4 Oracle 上界估计

**实验规模**：9,250 张训练图（k=250/class），2 种 seed 来源，4 种 reader，3 个 budget

**核心结果**（gradient seed）：

| Budget | Oracle (gradient) | BFS | Random | Oracle-BFS Δ |
|--------|-------------------|-----|--------|-------------|
| 5% | 0.067 | 0.051 | 0.036 | +0.016 |
| 10% | 0.095 | 0.068 | 0.051 | +0.027 |
| 15% | 0.109 | 0.084 | 0.064 | +0.026 |

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

---

## 三、关键决策与偏离

### 3.1 偏离设计文档的部分

| 设计文档（2026-05-24） | 实际执行 | 原因 |
|---|---|---|
| T2 Info-Gain teacher 完整实现 | 未实现，保留设计 | 计算成本过高（见 §2.1） |
| 轨迹评分四维度（BE + IG + Stability + DFI） | 简化：仅 Acc + CE 过滤 | CE 已包含分类置信度信息；BE/Stability 在 batch BC 训练中隐式处理 |
| 每图保留 top-K 条轨迹 + 多样性保证 | 类别平衡采样替代 | 优先保证类别覆盖，多样性由 5 种 teacher 自然提供 |
| Oracle Type I（标签感知绝对上界） | 仅实现 Type II | Type II 是"可学习上界"，对 M5 更有参考价值 |
| DFI 指数计算 | 定性分析替代 | oracle headroom 的数据直接支撑判断，无需复合指数 |

### 3.2 新增超出设计文档的部分

- **四阶段管线**（raw → merge → filter → balance）：设计文档仅描述了单个轨迹生成器，实际管线逐步演化以处理数据质量、类别不平衡等问题
- **详细 CE 分析**：per-teacher × per-label CE 是设计文档未提及的，但对理解 teacher 质量和 BC 训练信号极有价值
- **Oracle estimator 的批量实现**：GPU 批量构建子图 + 分类，比设计文档的逐图方案高效得多
- **M5 BC 训练脚本**：`m5/train_bc.py` 和 `m5/smoke_test.py` 在 M4 阶段后期提前完成

---

## 四、M5 就绪状态

### 4.1 数据就绪

- `m4/data/stage4_balanced/` 包含 4 个 budget 的类别平衡轨迹（~67K trajectories total）
- 轨迹格式：`Trajectory(graph_idx, seed_node, reveal_ratio, reveal_order, teacher_type, ...)`
- 可直接被 `m5/train_bc.py` 的 `merge_trajectories()` 加载

### 4.2 代码就绪

- `m5/train_bc.py`：BC 训练主脚本（batch 训练、按 graph_idx split、多 budget 验证、checkpoint 保存）
- `m5/smoke_test.py`：单轨迹 forward→loss→backward→update 管线验证（已通过）
- `active_graph_reader/reader/learnable_reader.py`：LearnableReader（FrontierEncoder + ReaderMLP）

### 4.3 训练命令

```bash
python m5/train_bc.py \
  --pretrained m3/best_model_ft.pt \
  --root D:\ogb-master\dataset \
  --device cuda:0 \
  --epochs 10 \
  --lr 1e-3 \
  --batch-size 16
```

---

## 五、经验与教训

1. **大规模实验需要分阶段管线**。直接生成全部轨迹 + 一次性过滤的 waterfall 模式导致迭代慢。四阶段管线（每阶段独立保存中间产物）允许在任一阶段检查和修复问题，避免从头重跑。

2. **类别平衡不是事后补救**。Stage 4 的类别平衡显著改变了轨迹分布（某些预算下 r0_200 仅 5,138 条）。如果在 Stage 1 就按类别分层抽样，可以减少后期裁剪量。

3. **Teacher 多样性不等于 teacher 质量**。structural-pagerank teacher 的 CE 系统性地高于其他 teacher，在 BC 训练中可能贡献噪声而非信号。M5 BC 训练时需监控各 teacher 的 per-step loss 贡献。

4. **Oracle 实验设计的关键参数**：k=250/class 提供了每类足够的统计样本，但 9,250 图中大部分图的 oracle accuracy 为 0（because accuracy is 0/1 per graph at this low budget）。均值 0.05–0.11 掩盖了二值分布的本质——只有少数图在低 budget 下能被正确分类。

5. **v1 协议下的 baseline 显著低于 v0**。M3 benchmark 显示 BFS@5% = 0.147，M4 oracle 显示 BFS@5% = 0.051。差异来自 seed 策略和协议细节（M3 v0 可能有信息泄露）。这提醒我们：**跨阶段对比 accuracy 时务必确认协议版本一致**。

---

## 六、下一步

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | 服务器上运行 `m5/train_bc.py`，获取首轮 BC val acc | 待执行 |
| P1 | BC val acc vs BFS baseline 对比，决定是否需要 Gumbel-Softmax | 依赖 P0 |
| P2 | 如 BC ≈ BFS，实现 Gumbel-Softmax 端到端微调 | 待评估 |
| P3 | 如 BC < BFS，排查 reader 架构或训练超参 | 待评估 |
| — | 考虑使用 all 158K training graphs 而非仅轨迹覆盖的子集 | 可选增强 |

**M4 阶段完成。M5 BC 训练为当前最高优先级。**

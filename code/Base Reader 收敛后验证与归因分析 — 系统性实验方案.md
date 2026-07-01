# Base Reader 收敛后验证与归因分析 — 系统性实验方案

> 约束：本文件仅做方案设计，不动代码。所有实验标注所需数据、分析方法、预期输出及对 base reader / POMDP reader 的启示。

---

## 零、现状摘要与方案目标

### 0.1 已有事实


| 维度                 | 数据                                                                  |
| -------------------- | --------------------------------------------------------------------- |
| 全图分类器上限       | valid 70.66%，test 72.60%（5-layer GINE, hidden=300）                 |
| BFS@10% budget       | 15.14%（仅达上限 20.8%）→ 策略选择有巨大空间                         |
| Base reader BC 训练  | 5 epochs，val_acc: r0.05≈0.39, r0.10≈0.48, r0.15≈0.53, r0.20≈0.54 |
| Base reader RL 训练  | 1 epoch，平均 R > 0.3（加入 budget_ratio + struct 后）                |
| 训练集子图分类正确率 | 0.65+（尚未在 val/test 上验证）                                       |

### 0.2 架构回顾

```
LearnableReader 管线:
  GraphState → FrontierGraphBuilder → FrontierSubgraph
    → FrontierNodesEncoder (2-layer GINE, hidden=128)
      节点初始特征: type_embedding (revealed=0, frontier=1)
      边特征: 7-dim 原图边特征
    → frontier_embeds (N_f × 128)
    → 拼接 [frontier_embeds ‖ budget_ratio(1) ‖ struct_v(3)] = N_f × 132
    → ReaderMLP (132→128→128→1)
    → softmax → select node → reveal_node → 循环

评估管线:
  Reader 探索 B 步 → SubgraphBuilder 提取 V_revealed 子图
    → 冻结 GraphClassifier (5-layer GINE, hidden=300) → logits → argmax
```

### 0.3 两次关键改动


| 改动       | 内容                                                                                       | 效果                                 |
| ---------- | ------------------------------------------------------------------------------------------ | ------------------------------------ |
| 损失与奖励 | loss = 每步损失按时序权重 w_t=softmax(2t/B) 求和；R = tanh(1-CE/ln(K))                     | 初始平均 R < 0（不如随机），改后解决 |
| 特征增强   | (1) 加入剩余预算比例 budget_ratio（1维）→ R 提升到 0.3+；(2) 加入 3维结构特征 → 略微提升 |                                      |

### 0.4 方案目标

1. **验证泛化**：base reader 在 val/test 上的真实表现，是否过拟合
2. **归因分析**：哪些因素（特征、架构、训练策略）对任务起关键作用
3. **迁移启示**：为 base reader 改进和 POMDP reader 设计提供经验证据

---

## 一、Phase 1：泛化验证 — val/test 全量评测

### 1.1 实验目标

确立 base reader 在未见数据上的真实性能基线，回答"训练集 0.65+ 正确率是否能泛化"。

### 1.2 实验设计

**实验 1.2.1：全量 val/test 评测**

利用现有 `m5/base/eval.py`，在 valid 和 test split 上评测：


| 变量       | 取值                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| 评测集     | valid (全部), test (全部)                                                                                |
| 策略       | learnable, random, bfs                                                                                   |
| 预算       | 5%, 10%, 15%                                                                                             |
| Seed       | best (max-degree), worst (min-degree)                                                                    |
| Checkpoint | `m5/base/checkpoint/best_reader.pt`（BC）和 `m5/base/rl/checkpoint_epoch1.pt`（RL，如有完整 checkpoint） |

**需要收集的数据：**

- 每个配置下的 accuracy (mean, std, n)
- per-class accuracy（37 类逐类准确率）
- best-seed vs worst-seed 的准确率差异

**分析方法：**

1. 绘制 accuracy vs budget 曲线（3 策略 × 2 seed × 2 split = 12 条线）
2. 计算 learnable 相对 random/bfs 的提升幅度（绝对 pp 和相对百分比）
3. 计算子图准确率占全图准确率的比例（如 learnable@10% / full_graph@100%）
4. best-seed vs worst-seed 差异 → 量化 seed 敏感性
5. per-class 热力图：识别哪些类在低预算下已可分、哪些类始终困难

**预期输出：**

- `eval_results.json`（已有格式）
- `eval_plot.png`（已有格式）
- `per_class_*.png`（已有格式）
- 汇总表格：learnable vs random vs bfs，6 配置 × 2 split

### 1.3 关键判据


| 指标                     | 判据       | 含义               |
| ------------------------ | ---------- | ------------------ |
| learnable vs random      | 提升 > 5pp | 策略学习有效       |
| learnable vs bfs         | 提升 > 3pp | 优于简单启发式     |
| val vs train 差距        | < 10pp     | 无严重过拟合       |
| best vs worst seed       | < 3pp      | seed 鲁棒性好      |
| @10% budget / full-graph | > 50%      | 子图策略有实用价值 |

### 1.4 对 POMDP reader 的启示

- 若 base reader 泛化良好 → POMDP reader 可在此基础上增加信念模块，风险可控
- 若 base reader 严重过拟合 → POMDP reader 需更强正则化或更少参数
- seed 敏感性大 → POMDP 的信念初始化（b_0 从极小观测导出）是关键改进点

---

## 二、Phase 2：特征消融归因 — 量化各输入特征的贡献

### 2.1 实验目标

精确量化 budget_ratio、struct_v（3维结构特征）、edge_feat（7维边特征）各自对决策质量和分类性能的贡献，回答"改动的哪些部分真正起了作用"。

### 2.2 实验设计

由于约束为"只做方案不动代码"，本阶段通过**推理时干预**（inference-time intervention）实现消融，而非修改模型：

**实验 2.2.1：特征置零消融**

在 val 集上，对收敛后的 reader 做推理时特征置零，观察决策变化和最终分类变化：


| 消融配置               | 操作                                      | 对照  |
| ---------------------- | ----------------------------------------- | ----- |
| A0: 完整               | 正常推理（132维输入）                     | 基线  |
| A1: 去 budget_ratio    | 将 budget_ratio 维置为常数 0.5（中性值）  | ΔR_1 |
| A2: 去 struct_v        | 将 3维结构特征置零                        | ΔR_2 |
| A3: 去 budget + struct | 两者同时置零/常数                         | ΔR_3 |
| A4: 去 edge_feat       | FrontierGraphBuilder 构建子图时边特征置零 | ΔR_4 |

**实现方式（不改模型代码，改推理脚本）：**

- A1-A3：在 `reader.forward()` 返回后，对 `frontier_embeds` 的对应列做 `fill_` 操作
- A4：在 `FrontierGraphBuilder.build()` 后，对 `subgraph.g.edata['feat']` 做 `zero_()`

**需要收集的数据：**

每组消融配置：

1. 平均 R = tanh(1 - CE/ln(K))（与训练时一致的奖励指标）
2. 分类准确率（argmax logits == y）
3. **决策一致性**：消融后 reader 选择的节点与完整版的 overlap ratio
4. **逐步决策分布变化**：每步 softmax 分布的 KL 散度（消融 vs 完整）

**分析方法：**

1. 绘制特征贡献瀑布图：A4(去edge) → A2(去struct) → A1(去budget) → A0(完整)
2. 逐步分析：将 B 步分为前 1/3（探索期）、中 1/3（积累期）、后 1/3（利用期），看各特征在不同阶段的贡献
3. 决策一致性分析：消融后有多少比例的决策发生改变？改变的决策是否导致更差的结果？

### 2.3 预期贡献排序假设

根据训练日志中的经验：

```
edge_feat >> budget_ratio > struct_v
```

- edge_feat 是 PPA 数据集上区分 frontier 节点的唯一特征通道（无节点特征），预期贡献最大
- budget_ratio 提供时域信号，解决了"前期探索 vs 后期利用"的策略切换，贡献显著
- struct_v 提供局部拓扑信息，贡献较小但非零

### 2.4 对 POMDP reader 的启示


| 消融结果            | POMDP 设计启示                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| budget_ratio 贡献大 | POMDP 文档 v5 去除 budget_ratio、改由 LSTM 隐式承载时域 → 需验证 LSTM 是否真能学到时域信号；若 base reader 中 budget_ratio 贡献大，POMDP 去除它有风险 |
| struct_v 贡献小     | POMDP 中结构特征可简化或移除，降低复杂度                                                                                                               |
| edge_feat 贡献极大  | 确认边特征是核心信号通道，POMDP 的 GSAE 必须保留前沿边特征输入                                                                                         |
| 某特征在后期贡献大  | 对应 POMDP 中该特征应条件化于信念状态（后期信念更确定）                                                                                                |

---

## 三、Phase 3：决策行为分析 — Reader 实际在做什么

### 3.1 实验目标

理解 reader 的探索策略：它选择什么样的节点？是否在探索 vs 利用间切换？决策是否多样化？

### 3.2 实验设计

**实验 3.2.1：被选节点的结构画像**

在 val 集上，收集 reader 每步选择的节点及该步的图状态：

**收集的逐步数据（per-step, per-graph）：**


| 字段                        | 说明                                              |
| --------------------------- | ------------------------------------------------- |
| step t                      | 当前步数 / 总预算 B                               |
| chosen_node_degree          | 被选节点在原图中的度                              |
| chosen_node_frontier_degree | 被选节点的前沿度（连接的已揭示节点数）            |
| chosen_node_clustering      | 被选节点的局部聚类系数                            |
| chosen_node_struct_v        | 被选节点的 3 维结构特征                           |
| n_frontier_at_step          | 该步的前沿节点总数                                |
| chosen_rank_by_score        | 被选节点在 softmax 排序中的名次（通常=1，argmax） |
| top1_top2_score_gap         | 最高分与次高分之差（决策确定性）                  |
| softmax_entropy             | softmax 分布的熵（决策不确定性）                  |
| budget_ratio_remaining      | 该步的剩余预算比                                  |
| frontier_size_growth        | 该步揭示后前沿净变化（新发现 - 消耗）             |

**分析方法：**

1. **节点选择分布**：绘制 chosen_node_degree 的分布直方图，对比 frontier 中所有候选节点的 degree 分布 → reader 是否偏好高度节点？
2. **时序行为切换**：按 step/B 分箱，绘制各指标的时序曲线：
   - softmax_entropy vs t/B → 决策是否在后期变得更确定？
   - chosen_node_degree vs t/B → 是否前期选高度节点（探索）、后期选低度节点（利用）？
   - frontier_size_growth vs t/B → 探索效率是否随时间下降？
3. **决策确定性分析**：top1_top2_score_gap 的分布 → 多少决策是"勉强"的（gap < 0.1）？
4. **与固定策略对比**：reader 选择的节点序列与 BFS / max-degree 的 overlap ratio → reader 是否学到了超越简单启发式的策略？

**实验 3.2.2：轨迹多样性分析**

对同一张图、同一 seed，用 stochastic=True 模式采样 N=10 条轨迹：

**需要收集的数据：**

- 10 条轨迹的节点序列
- 两两轨迹的 Jaccard 相似度（节点集合交集/并集）
- 最终子图分类结果的一致性

**分析方法：**

1. 平均 Jaccard 相似度 → 低 = 高多样性（探索性强），高 = 低多样性（策略确定）
2. 分类一致性 vs Jaccard 相似度 → 多样性是否带来分类稳定性？
3. 不同 budget 下的多样性变化

### 3.3 对 POMDP reader 的启示


| 观察结果                      | POMDP 设计启示                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------- |
| reader 有明确的探索→利用切换 | POMDP 的 LSTM 信念应能隐式编码这种切换；若不能，需保留显式时域信号               |
| reader 偏好高度节点           | 与 max-degree teacher 的 overlap 应高；POMDP 策略可从结构特征初始化              |
| softmax 熵在后期下降          | 信念 b_t 应在后期变尖锐 → 验证 POMDP 文档中"早期 b_t 平坦、晚期 b_t 尖峰"的假设 |
| 轨迹多样性低                  | argmax 策略缺乏探索 → POMDP 需要探索机制（文档中已诊断为此缺口）                |
| 决策确定性高但分类错误        | reader 自信但错误 → POMDP 的不确定度 u_t 可用于 abstention                      |

---

## 四、Phase 4：分类质量与信息积累分析

### 4.1 实验目标

理解信息是如何随揭示步骤积累的，分类信心何时"锁定"，哪些类容易/困难。

### 4.2 实验设计

**实验 4.2.1：逐步分类信心曲线**

在 val 集上，对每张图记录每一步揭示后的分类器输出：

**收集的逐步数据（per-step, per-graph）：**


| 字段            | 说明                              |
| --------------- | --------------------------------- |
| step t          | 当前揭示步数（1 到 B）            |
| CE_t            | t 步后子图的分类交叉熵            |
| R_t             | t 步的即时奖励 tanh(1-CE_t/ln(K)) |
| pred_correct_t  | t 步后分类是否正确（0/1）         |
| top1_prob_t     | t 步后分类器 top-1 概率           |
| top2_prob_t     | t 步后分类器 top-2 概率           |
| top1_top2_gap_t | top1 - top2 概率差（分类确定性）  |
| pred_label_t    | t 步后预测的类别                  |
| true_label      | 真实类别                          |

**分析方法：**

1. **信息积累曲线**：绘制平均 CE_t / R_t / top1_prob_t vs t/B

   - 预期：单调下降（CE）或单调上升（R, top1_prob）
   - 关键观察：曲线在哪个 t/B 比例开始"平台化"（边际信息递减）？
2. **分类锁定点分析**：对每张图，定义"锁定点" t* 为首个 pred_correct_t=1 且之后不再翻错的步

   - 统计 t*/B 的分布 → 多少图在前 30% 预算就锁定了？
   - 锁定点早 vs 晚的图有什么特征差异？
3. **预测翻转分析**：统计 pred_label_t 变化的次数和方向

   - 高翻转 = 探索过程中信息不断修正认知
   - 翻转后变正确 vs 变错误的比例
4. **信息增益递减分析**：ΔCE_t = CE_{t-1} - CE_t 的分布

   - 哪些步的信息增益最大？（通常前几步）
   - 平均信息增益 vs t/B 曲线 → 确定最优预算的"拐点"

**实验 4.2.2：per-class 分类质量**

**需要收集的数据：**

- 37 个类别在 3 个 budget 下的准确率、平均 CE、平均 top1_prob
- 类间混淆矩阵（37×37）

**分析方法：**

1. 类别准确率排序图 → 识别"容易类"（>80%）、"中等类"（40-80%）、"困难类"（<40%）
2. 混淆矩阵热力图 → 哪些类对容易被混淆？是否有语义相似性？
3. 困难类 vs 容易类的图属性差异（节点数、边数、密度、平均度）
4. 困难类是否在更高 budget 下显著改善？（如果是 → 问题是预算不足；如果否 → 问题是分类器能力上限）

### 4.3 对 POMDP reader 的启示


| 观察结果                     | POMDP 设计启示                                                                |
| ---------------------------- | ----------------------------------------------------------------------------- |
| 信息积累在 50% 预算后平台化  | POMDP 的 Submit-Early 机制（Section 9）有实证支撑                             |
| 锁定点分布集中在早期         | 多数图不需要耗满预算 → 自适应预算有实际价值                                  |
| 预测频繁翻转                 | 说明信念更新剧烈 → POMDP 的 LSTM 信念更新需要处理跳变（文档否决了 L_smooth） |
| 困难类不随 budget 改善       | 问题在分类器上限而非 reader 策略 → POMDP reader 无法解决，需改进分类器       |
| top1-top2 gap 与正确性正相关 | 分类确定性可作为 abstention 信号 → POMDP 的 u_t 有实际意义                   |

---

## 五、Phase 5：失败模式分析

### 5.1 实验目标

理解 reader 在什么情况下失败，失败原因是策略问题、预算问题还是分类器上限问题。

### 5.2 实验设计

**实验 5.2.1：图属性 vs 分类成功**

**需要收集的数据（per-graph）：**


| 字段            | 说明                             |
| --------------- | -------------------------------- |
| graph_idx       | 图索引                           |
| n_nodes         | 节点数                           |
| n_edges         | 边数                             |
| density         | 2*E / (V*(V-1))                  |
| avg_degree      | 平均度                           |
| true_label      | 真实类别                         |
| seed_node       | 种子节点                         |
| seed_degree     | 种子节点的度                     |
| budget_abs      | 实际预算                         |
| budget_ratio    | 预算比例                         |
| final_correct   | 最终分类是否正确                 |
| final_CE        | 最终分类交叉熵                   |
| final_top1_prob | 最终 top-1 概率                  |
| n_steps_to_lock | 锁定点步数（首次正确后不再翻错） |
| n_flips         | 预测翻转次数                     |
| avg_R           | 平均逐步奖励                     |

**分析方法：**

1. **成功 vs 失败的图属性对比**：

   - 按 final_correct 分组，对比 n_nodes / n_edges / density / avg_degree 的分布
   - t 检验或 KS 检验：两组分布是否有显著差异
   - 预期：大图/稀疏图更难（信息更分散）
2. **seed 质量影响**：

   - seed_degree vs final_correct 的相关性
   - best-seed 准确率 vs worst-seed 准确率（已有 eval.py 数据）
   - seed_degree 分布中，低度 seed 是否系统性导致失败？
3. **budget 敏感性**：

   - 同一图在 5% / 10% / 15% budget 下的分类一致性
   - 从 5%→10% 准确率提升幅度 vs 10%→15% 提升幅度 → 边际收益递减点
   - 哪些图在 15% 仍然失败？（hard cases 的特征画像）
4. **错误类型分类**：

   - 类型 A：低 top1_prob + 错误 → 分类器不确定，信息不足
   - 类型 B：高 top1_prob + 错误 → 分类器自信但错误，可能分类器本身有盲区
   - 类型 C：频繁翻转 + 最终错误 → 探索路径不佳
   - 统计三种类型的比例 → 指明改进方向

**实验 5.2.2：困难案例深检**

选取 20-30 个"困难案例"（所有 budget 都失败），做逐案例分析：

**需要收集的数据：**

- 完整 reveal_history（节点序列）
- 逐步 CE / pred_label 曲线
- reader 每步的 softmax 分布（决策信心）
- 与 teacher 轨迹的对比（如有同图的 teacher 轨迹）

**分析方法：**

1. 困难案例的探索路径可视化（子图增长动画或快照）
2. reader 路径 vs teacher 路径的差异 → 是策略问题还是信息上限问题？
3. 困难案例是否集中在特定类别？（与 Phase 4 的 per-class 分析交叉）

### 5.3 对 POMDP reader 的启示


| 失败类型      | 比例预期 | POMDP 改进方向                                              |
| ------------- | -------- | ----------------------------------------------------------- |
| A: 信息不足   | ~40%     | POMDP 的信念更新应更高效地利用每步证据；LSTM 时序整合可改善 |
| B: 分类器盲区 | ~30%     | POMDP 的 Reader 自主裁判（不再依赖冻结 f_class）可部分解决  |
| C: 策略不佳   | ~30%     | POMDP 的信念条件化策略 + 探索机制可改善                     |
| seed 敏感     | 显著     | POMDP 信念初始化 b_0 需处理极小观测；多 seed 聚合策略       |
| 大图困难      | 显著     | POMDP 的 GSAE 状态压缩需适应大图的稀疏子图                  |

---

## 六、Phase 6：Teacher-Reader 关系分析

### 6.1 实验目标

量化 reader 从 teacher 轨迹中学到了什么，在哪里超越了 teacher，在哪里不及 teacher。

### 6.2 实验设计

**实验 6.2.1：决策一致性分析**

在 val 集上（如有 teacher 轨迹覆盖 val 图），对比 reader 与 teacher 的逐步决策：

**需要收集的数据（per-step, per-graph）：**


| 字段               | 说明                                  |
| ------------------ | ------------------------------------- |
| teacher_node       | teacher 在该步选择的节点              |
| reader_node        | reader 在该步选择的节点               |
| agreement          | 两者是否相同（0/1）                   |
| teacher_node_score | reader 给 teacher 节点的 softmax 概率 |
| reader_node_score  | reader 给自己选择节点的 softmax 概率  |
| teacher_node_rank  | teacher 节点在 reader 排序中的名次    |
| both_correct       | 两种选择最终都能正确分类？            |

**分析方法：**

1. **整体一致率**：agreement 的平均比例 → reader 多大程度在模仿 teacher？
2. **逐步一致率**：agreement vs t/B → 一致性是否随步数变化？（前期一致高=学习通用策略，后期分歧=reader 形成自主判断）
3. **分歧案例的结局**：
   - reader 对 teacher 错 → reader 的自主决策是否有害？
   - reader 对 teacher 对 → teacher 是否更优？
   - 两者都对 → 决策不敏感区域
   - 两者都错 → 信息上限问题
4. **teacher 节点的 reader 排名分布**：teacher 选的节点在 reader 的 softmax 中排第几？
   - 如果 teacher 节点常排 2-3 名 → reader 学到了 teacher 的"方向"但具体选择有差异

**实验 6.2.2：不同 teacher 类型的影响**

训练数据包含 6 种 teacher（gradient, structural-degree, structural-pagerank, hybrid-a0.30, hybrid-a0.70, v0-reader）。

**分析方法：**

1. 按 teacher 类型分组统计 reader 的 BC loss → reader 对哪种 teacher 模仿得最好？
2. 不同 teacher 覆盖的图上，reader 的最终分类准确率是否有差异？
3. reader 是否形成了超越所有 teacher 的"混合策略"？

### 6.3 对 POMDP reader 的启示


| 观察结果                  | POMDP 设计启示                                                |
| ------------------------- | ------------------------------------------------------------- |
| reader 大幅超越 teacher   | BC 预训练有效，POMDP 的 Phase 2 BC 可沿用                     |
| reader 不及 teacher       | BC 质量受限，POMDP 需更强 RL 微调（Phase 3 REINFORCE）        |
| reader 在后期偏离 teacher | reader 形成了时序依赖判断 → POMDP 的 LSTM 信念更新有需求基础 |
| teacher 类型影响显著      | POMDP 训练应使用高质量 teacher（gradient/hybrid）             |

---

## 七、Phase 7：RL 训练效果验证

### 7.1 实验目标

BC 预训练后仅做了 1 epoch RL，需验证 RL 方向是否正确，奖励设计是否有效。

### 7.2 实验设计

**实验 7.2.1：BC vs RL checkpoint 对比**

对比 `best_reader.pt`（BC epoch 1）和 `rl/checkpoint_epoch1.pt`（RL epoch 1）：

**需要收集的数据：**

- 两个 checkpoint 在 val 集上的 accuracy / avg_R
- 两个 checkpoint 的决策一致性（同一图同一 seed 的节点序列 overlap）
- RL 训练日志中的 avg_R 曲线（目前仅 1 epoch，需继续训练获得趋势）

**实验 7.2.2：奖励信号诊断**

分析 RL 训练中 R 的分布：

**需要收集的数据（per-trajectory, per-epoch）：**

- R 的分布直方图（正/负/零的比例）
- R 与图属性的相关性（大图 R 更低？困难类 R 更低？）
- R 的方差 → REINFORCE 梯度方差估计

**分析方法：**

1. R < 0 的比例 → 多少图"不如随机"？这些图有什么特征？
2. R 的方差大小 → 是否需要 value baseline 降方差？（POMDP 文档已诊断此缺口）
3. 时序权重 w_t=softmax(2t/B) 的效果 → 后期步的梯度是否主导了更新？

### 7.3 对 POMDP reader 的启示


| 观察结果           | POMDP 设计启示                                            |
| ------------------ | --------------------------------------------------------- |
| RL 提升 BC         | REINFORCE 方向正确，POMDP Phase 3 可沿用                  |
| RL 无提升或退化    | 奖励设计或学习率有问题，POMDP 需重新设计 RL 策略          |
| R 方差极大         | 需要 value baseline（POMDP 文档已诊断此缺口，需优先实现） |
| R < 0 集中在困难类 | 需要分类器上限分析，或 abstention 机制                    |

---

## 八、Phase 8：跨 Budget 泛化分析

### 8.1 实验目标

验证 reader 在训练 budget 和非训练 budget 上的表现是否一致，回答"reader 是否学到了 budget-invariant 策略"。

### 8.2 实验设计

**实验 8.2.1：跨 budget 评测**

训练数据覆盖3个 budget ratio（5%, 10%, 15%,），reader 的 budget_ratio 特征提供了 budget 感知。


| 评测 budget | 训练覆盖 | 目的                   |
| ----------- | -------- | ---------------------- |
| 5%          | 是       | 训练分布内             |
| 10%         | 是       | 训练分布内             |
| 15%         | 是       | 训练分布内             |
| 3%          | 否       | 外推：更低预算         |
| 8%          | 否       | 插值：训练 budget 之间 |
| 25%         | 否       | 外推：更高预算         |

**需要收集的数据：**

- 各 budget 下的 accuracy / avg_R
- budget_ratio 特征值的实际分布（训练时见过的范围 vs 评测时的范围）

**分析方法：**

1. accuracy vs budget 曲线 → 是否平滑？外推点（3%, 25%）是否偏离趋势？
2. 插值点（8%）是否与训练点一致？→ budget_ratio 特征是否泛化
3. 边际收益曲线：Δaccuracy / Δbudget → 确定最优 budget 配置

### 8.3 对 POMDP reader 的启示


| 观察结果              | POMDP 设计启示                                                |
| --------------------- | ------------------------------------------------------------- |
| 跨 budget 泛化好      | budget_ratio 特征有效，POMDP 可考虑保留（与 v5 去除决策矛盾） |
| 外推差                | budget_ratio 仅在训练范围内有效，POMDP 用 LSTM 隐式编码更鲁棒 |
| 边际收益在 15% 后递减 | POMDP 的 Submit-Early 有实证支撑，最优 budget < 最大 budget   |

---

## 九、数据收集汇总与优先级

### 9.1 数据收集矩阵


| Phase | 核心数据                                                | 来源                                       | 优先级 |
| ----- | ------------------------------------------------------- | ------------------------------------------ | ------ |
| 1     | val/test accuracy (3策略 × 3budget × 2seed × 2split) | eval.py（已有）                            | P0     |
| 2     | 消融后 accuracy / R / 决策一致性                        | 修改推理脚本（特征置零）                   | P0     |
| 3     | 逐步决策数据（节点属性、softmax、entropy）              | 新建分析脚本                               | P1     |
| 4     | 逐步分类数据（CE、pred_label、top1_prob）               | 新建分析脚本                               | P1     |
| 5     | 图属性 vs 分类结果                                      | 新建分析脚本                               | P1     |
| 6     | teacher-reader 决策对比                                 | 新建分析脚本（需 teacher 轨迹覆盖 val 图） | P2     |
| 7     | RL checkpoint 对比 / R 分布                             | 已有 checkpoint + RL 日志                  | P2     |
| 8     | 跨 budget 评测                                          | eval.py 加参数                             | P2     |

### 9.2 推荐执行顺序

```
P0（必须先做，建立基线）:
  Phase 1 → Phase 2

P1（核心归因，依赖 P0 的评测框架）:
  Phase 3 → Phase 4 → Phase 5

P2（补充分析，依赖 P1 的数据收集框架）:
  Phase 6 → Phase 7 → Phase 8
```

### 9.3 输出文件规划

```
m5/base/analysis/
├── phase1_eval_summary.md          # 泛化验证汇总
├── phase2_ablation_results.md      # 特征消融结果
├── phase3_decision_analysis.md     # 决策行为分析
├── phase4_info_accumulation.md     # 信息积累分析
├── phase5_failure_analysis.md      # 失败模式分析
├── phase6_teacher_comparison.md    # Teacher-Reader 对比
├── phase7_rl_diagnosis.md          # RL 效果诊断
├── phase8_cross_budget.md          # 跨 Budget 泛化
├── raw_data/                       # 原始数据 JSON
│   ├── phase1_eval.json
│   ├── phase2_ablation.json
│   ├── phase3_step_data.json
│   ├── phase4_step_classify.json
│   ├── phase5_graph_properties.json
│   └── ...
├── figures/                        # 可视化图像
│   ├── accuracy_vs_budget.png
│   ├── feature_waterfall.png
│   ├── info_accumulation_curve.png
│   ├── confusion_matrix.png
│   └── ...
└── final_report.md                 # 汇总报告：归因结论 + POMDP 启示
```

---

## 十、核心归因问题与预期结论框架

### 10.1 五个核心归因问题


| #  | 问题                        | 对应 Phase | 关键指标                                     |
| -- | --------------------------- | ---------- | -------------------------------------------- |
| Q1 | base reader 是否泛化？      | Phase 1    | val/test acc vs train acc                    |
| Q2 | 哪些特征对决策最关键？      | Phase 2    | 消融后 ΔR                                   |
| Q3 | reader 的探索策略是否合理？ | Phase 3    | 节点选择分布、时序行为切换                   |
| Q4 | 信息积累的效率如何？        | Phase 4    | CE_t 曲线、锁定点分布                        |
| Q5 | 失败的主因是什么？          | Phase 5    | 失败类型比例（信息不足/分类器盲区/策略不佳） |

### 10.2 预期结论框架

**对 base reader 的改进建议（基于归因结果）：**

1. 若 budget_ratio 贡献大 → 保留并考虑增加更丰富的时域特征（如已揭示比例、前沿收缩率）
2. 若 struct_v 贡献小 → 可移除以降低参数量和计算开销
3. 若信息在 50% 预算后平台化 → 可引入提前终止机制（类似 POMDP 的 Submit-Early）
4. 若失败主因是分类器盲区 → 需联合微调分类器或使用 Reader 自主裁判
5. 若 RL 有正向效果 → 继续 RL 训练更多 epoch，并引入 value baseline

**对 POMDP reader 的设计验证：**


| POMDP 设计决策      | 需验证的假设                                | 验证 Phase                  |
| ------------------- | ------------------------------------------- | --------------------------- |
| LSTM 信念更新       | reader 决策有时序依赖（后期行为不同于前期） | Phase 3                     |
| 去除 budget_ratio   | LSTM 能隐式编码时域信号                     | Phase 2 (A1消融)            |
| 结构特征 struct_v   | 结构特征贡献小可简化                        | Phase 2 (A2消融)            |
| Reader 自主裁判     | 失败主因是分类器盲区（非策略问题）          | Phase 5                     |
| Submit-Early        | 信息积累在 50% 预算后平台化                 | Phase 4                     |
| 探索机制            | argmax 策略缺乏探索导致部分失败             | Phase 3, 5                  |
| Value baseline      | RL 梯度方差大                               | Phase 7                     |
| EDL 损失（替代 CE） | 弱证据时 CE 退化为均匀分布                  | Phase 4 (低 top1_prob 比例) |

---

## 十一、资源估算


| Phase    | 需加载图数                     | 预计耗时（单 GPU） | 是否需新脚本              |
| -------- | ------------------------------ | ------------------ | ------------------------- |
| 1        | val + test 全量                | 2-4 小时           | 否（eval.py 已有）        |
| 2        | val 子集 (500图)               | 1-2 小时           | 是（消融推理脚本）        |
| 3        | val 子集 (500图)               | 1-2 小时           | 是（逐步数据收集脚本）    |
| 4        | val 子集 (500图)               | 1-2 小时           | 与 Phase 3 共用脚本       |
| 5        | val 全量                       | 2-3 小时           | 与 Phase 1 共用数据       |
| 6        | val 子集 (需 teacher 轨迹覆盖) | 1-2 小时           | 是（teacher 对比脚本）    |
| 7        | val 子集 (500图)               | 0.5-1 小时         | 否（对比已有 checkpoint） |
| 8        | val 全量 × 7 budget           | 4-6 小时           | 否（eval.py 加参数）      |
| **总计** | —                             | **~15-25 小时**    | 3-4 个新分析脚本          |

---

## 十二、注意事项

1. **val/test 图加载**：PPA 数据集约 158k 图，需按 split 逐次加载（eval.py 已实现），避免内存溢出
2. **teacher 轨迹覆盖**：Phase 6 需要 val 图的 teacher 轨迹，需确认 M4 轨迹是否覆盖 val 集（目前仅训练集图有轨迹）
3. **RL checkpoint 完整性**：当前 RL 仅 1 epoch 且日志不完整，Phase 7 需先补充 RL 训练
4. **统计显著性**：val 集约 17k 图，子集分析建议 ≥500 图以保证统计可靠性
5. **可复现性**：所有实验固定 seed=42，记录完整配置（checkpoint 路径、budget、split）

---

*本方案为纯设计文档，不涉及代码修改。执行时按 Phase 顺序推进，每个 Phase 完成后输出分析报告并更新 `final_report.md`。*

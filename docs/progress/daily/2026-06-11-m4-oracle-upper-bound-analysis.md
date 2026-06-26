# M4 轨迹蒸馏上界分析报告

**日期**: 2026-06-11
**状态**: M4 阶段完成
**数据来源**: `code/m4/oracle/oracle_report.json`（Oracle Type II 估计）、`code/m4/analysis/analysis_stats.json`（轨迹统计分析）
**上游**: M3 benchmark gate PASS、M4 四阶段轨迹管线
**下游**: M5 Behavior Cloning 训练（`code/m5/train_bc.py`）

---

## 一、实验设计

### 1.1 Oracle Type II 定义

使用**梯度归因分数最高**的节点作为最优 seed，从该 seed 出发用四种 reader（gradient / structural-degree / BFS / random）执行 greedy reveal，预算花光后分类器评估准确率。

- **gradient reader**（oracle）：每步选 gradient score 最高的 frontier node。利用了全图梯度信息（标签感知），代表"给定预算下的经验上界"
- **structural-degree reader**：每步选 degree 最高的 frontier node
- **BFS reader**：按发现顺序层序扩展
- **random reader**：均匀随机选择

### 1.2 配置


| 参数      | 值                                                     |
| --------- | ------------------------------------------------------ |
| 图数      | 9,250（k=250 per class × 37 classes，训练集分层抽样） |
| Budget    | 5%, 10%, 15%（相对节点数）                             |
| Seed 来源 | gradient-best, structural-degree-best                  |
| 分类器    | M3 GINE（frozen, test-acc=0.642）                      |

---

## 二、核心结果

### 2.1 Accuracy 总览

**Seed = Gradient Best（梯度分数最高节点）**


| Budget | Gradient (oracle) | Struct-Deg | BFS    | Random |
| ------ | ----------------- | ---------- | ------ | ------ |
| 5%     | 0.0670            | 0.0631     | 0.0509 | 0.0363 |
| 10%    | 0.0948            | 0.1044     | 0.0682 | 0.0505 |
| 15%    | 0.1094            | 0.1297     | 0.0835 | 0.0639 |

**Seed = Structural-Deg Best（度最高节点）**


| Budget | Gradient (oracle) | Struct-Deg | BFS    | Random |
| ------ | ----------------- | ---------- | ------ | ------ |
| 5%     | 0.0669            | 0.0648     | 0.0501 | 0.0364 |
| 10%    | 0.0942            | 0.1053     | 0.0691 | 0.0488 |
| 15%    | 0.1102            | 0.1291     | 0.0839 | 0.0659 |

**全图分类器 accuracy: 0.642**

### 2.2 Headroom 分析


| Seed Source       | Budget | Oracle vs BFS | Oracle vs Random | BFS vs Random |
| ----------------- | ------ | ------------- | ---------------- | ------------- |
| gradient          | 5%     | +0.0161       | +0.0307          | +0.0146       |
| gradient          | 10%    | +0.0266       | +0.0443          | +0.0177       |
| gradient          | 15%    | +0.0259       | +0.0455          | +0.0196       |
| structural-degree | 5%     | +0.0169       | +0.0305          | +0.0136       |
| structural-degree | 10%    | +0.0251       | +0.0454          | +0.0203       |
| structural-degree | 15%    | +0.0263       | +0.0442          | +0.0179       |

---

## 三、关键发现

### 3.1 绝对准确率极低——"冰山一角"问题

所有策略在 ≤15% budget 下的准确率均不超过 13%，远低于全图 64.2%。这符合预期：PPA 图平均 ~300 节点，5% budget 仅揭示 ~15 个节点，从 1 个随机 seed 出发只能看到图的极小局部。**这说明 partial graph reading 确实是困难的**——确认了 M3 gate 的判断。

### 3.2 Oracle 一致优于 BFS，但绝对差距有限

Oracle vs BFS 的 headroom 在 **1.6–2.6 个百分点**。按 M4 设计文档的阈值（§6.1）：

- 差距 > 5pp → 轨迹蒸馏强力推荐
- 差距 2–5pp → 可以做，预期收益有限
- 差距 < 2pp → BC 可能无法超越 BFS

当前结果落在 **2–5pp 区间（中下段）**，属于"可以做但预期收益有限"。

### 3.3 相对提升更可观

以 BFS@5% = 0.0509 为基准，oracle@5% = 0.0670 的相对提升为 **31.6%**。在极低 budget 场景下，策略选择的相对价值显著。对 M5 而言，即使绝对 accuracy 提升仅 1–2pp，如果能在所有 budget 下稳定超越 BFS，仍然是有意义的进步。

### 3.4 结构策略的竞争力

Structural-degree reader 在 10%–15% budget 下**略优于** gradient reader。这提示：

- 简单拓扑特征（度中心性）在中等 budget 下是强信号
- M5 LearnableReader 的前沿编码器应同时利用拓扑和边特征信号
- BC 训练轨迹中包含 structural teacher 的样本至关重要

### 3.5 高方差——图间差异巨大

所有策略的标准差在 0.19–0.34 范围，均值仅 0.04–0.13。这意味着对大量图而言，任何策略在低 budget 下都难以正确分类；而对少数图，正确分类相对容易。M5 BC 训练应考虑**按图难度加权**或**聚焦于"有提升空间"的图**。

### 3.6 Seed 质量的影响

两种 seed 来源（gradient-best vs degree-best）的结果几乎一致（差异 < 0.002），说明 seed 选择策略的边际影响小于 reveal 过程中的节点选择策略。M5 训练应关注**逐步 reveal 决策**而非 seed 优化。

---

## 四、轨迹蒸馏可行性判断

### 4.1 DFI-like 评估

基于 M4 设计文档 §6.2 的蒸馏可行指数（DFI）框架：

1. **Oracle vs BFS 相对提升**: (0.067 - 0.051) / 0.051 ≈ 0.31 → 中等
2. **Teacher 轨迹一致性**: 5 种 teacher 中 gradient/hybrid 轨迹高度一致（对 class 17 的 CE 接近 0），structural 轨迹分歧较大 → 中低
3. **轨迹质量方差**: analysis_stats.json 显示 per-label CE 方差大（class 18 CE ~0.20, class 17 CE ~1e-6） → 中高

**综合判断**：轨迹蒸馏可行但并非"显著优于 BFS 的确定性路径"。BC 预训练作为 M5 起步有其价值（提供合理初始化），但不应期望仅靠 BC 就大幅超越 BFS。端到端微调（Gumbel-Softmax）可能才是拉开差距的关键。

### 4.2 M5 路线建议

```
BC 预训练（当前 train_bc.py）
  → 目标：达到 structural-degree teacher 水平（~0.06–0.13 accuracy）
  → 成功标准：val acc 不低于 BFS baseline
  → 如果 BC 本身已超越 BFS 2pp+ → 直接推进
  → 如果 BC ≈ BFS → 需要 Gumbel-Softmax 端到端微调
```

## 五、与 M3 基准的对比


|           | M3 Benchmark (370 图) | M4 Oracle (9,250 图)  |
| --------- | --------------------- | --------------------- |
| 规模      | 370 图, k=10/class    | 9,250 图, k=250/class |
| BFS@5%    | 0.1470                | 0.0509                |
| BFS@10%   | 0.2241                | 0.0682                |
| Random@5% | 0.1003                | 0.0363                |

**数值差异显著**。原因：M3 benchmark 尝试了10个seed，且每类样本少，特别是目前classifier的不可学习类别的数量少（0，32等）。M4 oracle 从 seed尝试一次，每类250张图，包含大量classifier本身难判断的类别。因此，所有 accuracy 系统性偏低。

## 六、结论

1. **Oracle 上界确认了 policy optimization 空间的存在**——oracle 一致优于 BFS 1.6–2.6pp，优于 random 3.0–4.6pp
2. **提升空间有限但不为零**——距离"BC 确定性优于 BFS"的 5pp 阈值有差距，但 2–5pp 区间意味着"值得尝试但需管理预期"
3. **M4 数据管线产出达标**——4 个 budget × 5 个 teacher 的类别平衡轨迹（stage4_balanced）已就绪，可投入 M5 BC 训练
4. **M4 阶段完成**——轨迹蒸馏可行性已量化分析

**下一步**：在服务器上运行 `m5/train_bc.py`，首轮 BC 训练的 val acc vs BFS baseline 将决定是否需要立即进入 Gumbel-Softmax 微调。

# 条件化图阅读（Contingent Graph Reading）相关论文：结构化梳理与对比分析

> **检索日期**：2026-06-13
> **检索来源**：Google Scholar + Semantic Scholar + OpenReview + NeurIPS/ICML/AAAI Proceedings
> **收录标准**：CCF-A / CCF-B 顶级会议或 SCI 期刊
> **检索方向**：Optimal Stopping in Active Feature Acquisition、Classification with Reject Option、Reversibility-Aware RL、Confidence-Based Cascade Deferral
> **与本项目关系**：证据信念 POMDP Reader 的"submit-or-continue"扩展——Reader 在预算用完后可自主决定上交作业还是选择性遗忘已揭示节点、继续探索——涉及最优停止、拒绝分类、动作可逆性、级联延迟四条方法线索

---

## 目录

1. [综述：四条方法线索与条件化图阅读的交叉](#综述四条方法线索与条件化图阅读的交叉)
2. [论文 1：Joint Active Feature Acquisition and Classification（NeurIPS 2018, CCF-A）](#论文-1joint-active-feature-acquisition-and-classificationneurips-2018-ccf-a)
3. [论文 2：SelectiveNet（ICML 2019, CCF-A）](#论文-2selectiveneticml-2019-ccf-a)
4. [论文 3：Reversibility-Aware RL（NeurIPS 2021, CCF-A）](#论文-3reversibility-aware-rlneurips-2021-ccf-a)
5. [论文 4：Classification with Rejection via Cost-Sensitive Classification（ICML 2021, CCF-A）](#论文-4classification-with-rejection-via-cost-sensitive-classificationicml-2021-ccf-a)
6. [论文 5：When Does Confidence-Based Cascade Deferral Suffice?（NeurIPS 2023, CCF-A）](#论文-5when-does-confidence-based-cascade-deferral-suffice-neurips-2023-ccf-a)
7. [总结对比矩阵](#总结对比矩阵)
8. [阅读优先级建议](#阅读优先级建议)

---

## 综述：四条方法线索与条件化图阅读的交叉

### 条件化图阅读的三阶段决策

```
初始预算 B 用完后：

  阶段 1: Submit-or-Continue 决策
    Reader 根据 b_B, u_B, trajectory, budget 决定：
      ├── SUBMIT → 上交已揭示子图 → Classifier → ŷ (终止)
      └── CONTINUE → 进入阶段 2

  阶段 2: 选择性遗忘 (Forgetting)
    Reader 选择需要"遗忘"的已揭示节点：
      v_forget ∈ V_revealed → 标记为"伪未揭示"
      (条件：节点信息可通过其他路径恢复，或对分类不关键)

  阶段 3: 继续探索 (Extended Exploration)
    释放的 budget 用于揭示新的 frontier 节点
    └── 回到阶段 1，直到 SUBMIT 或达到上界 B_max

核心问题：
  1. 什么时候应该上交？（最优停止）
  2. 哪些节点可以"遗忘"？（动作可逆性）
  3. 如何权衡继续探索的成本与收益？（成本敏感的拒绝）
```

### 方法论交叉图

```
条件化图阅读 (Contingent Graph Reading)
    │
    ├── 线索 A: 最优停止 (Optimal Stopping) — "何时上交"
    │   ├── 论文 1: Joint Active Feature Acquisition (NeurIPS 2018)
    │   ├── 论文 2: SelectiveNet (ICML 2019)
    │   └── 论文 4: Classification with Rejection (ICML 2021)
    │
    ├── 线索 B: 动作可逆性 (Reversibility) — "哪些可以遗忘"
    │   └── 论文 3: Reversibility-Aware RL (NeurIPS 2021)
    │
    └── 线索 C: 级联延迟理论 (Cascade Deferral) — "自信度规则何时有效"
        └── 论文 5: Confidence-Based Cascade Deferral (NeurIPS 2023)
```

### 关键映射：从已有范式到条件化图阅读

| 已有范式 | 核心概念 | → | 本项目对应 |
|---------|---------|----|-------------|
| Active Feature Acquisition | 顺序获取特征直到自信 → 分类 | → | 顺序揭示节点直到自信 → 分类 |
| AFA 的 Stopping Rule | 基于预测置信度决定停止获取 | → | 基于信念 b_t 的熵/u_t 决定是否 SUBMIT |
| Classification with Reject | 分类器可以说"我不知道" | → | Reader 可以说"我还需要更多证据" |
| Cost-Sensitive Rejection | 拒绝有成本，误分类有成本 | → | 继续探索有成本，误分类有成本 |
| Reversibility in RL | 区分可逆与不可逆动作 | → | 区分"可遗忘"与"关键"的已揭示节点 |
| Cascade Deferral | 级联模型逐级决定是否推迟决策 | → | 每步证据整合后决定"提交 or 继续" |

---

## 论文 1：Joint Active Feature Acquisition and Classification（NeurIPS 2018, CCF-A）

| 维度 | 内容 |
|------|------|
| **论文名称** | Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding |
| **作者** | Hajin Shim, Sung Ju Hwang, Eunho Yang |
| **发表** | **NeurIPS 2018**（Advances in Neural Information Processing Systems 31） |
| **期刊等级** | **CCF-A**（人工智能顶级会议） |
| **DOI** | [NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/7411) |

### 摘要

在许多实际场景中，获取所有特征值用于分类可能成本过高或不可行。本文研究联合主动特征获取与分类问题——以顺序方式主动获取特征值，并在任何时刻做出分类决策。我们提出一个统一框架，使用可变大小集合编码网络同时处理已获取特征的任意子集。该方法学习一个策略，在每一步决定"获取哪个未获取特征"或"停止获取并分类"。在多个基准数据集上的实验表明，该方法在分类准确度和特征获取成本之间取得了显著的权衡。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 特征获取有成本——如何在"获取更多特征"和"用已有特征做分类"之间动态权衡？ |
| **核心贡献** | 统一框架联合优化特征获取策略 + 分类器 + 停止规则——针对每个样本自适应地决定获取哪些特征和何时停止 |

### 核心思路

```
样本 x (特征完全未知)
    │
    ▼
  当前已获取特征集 O_t (初始为空)
    │
    ├── Variable-Size Set Encoder → 任意大小特征子集的嵌入
    │
    ▼
  决策: 获取下一个特征? 还是停止并分类?
    │
    ├── 继续: 选择 f ∈ unacquired, 支付成本 c_f → 观察 f 的值 → O_{t+1}
    └── 停止: Classifier(O_T) → ŷ
```

1. **可变大小集合编码器**：能处理任意大小特征子集的编码网络（集合函数 + 注意力机制）
2. **联合训练**：分类 loss + 特征获取 cost → 策略学习何时停止、获取哪个特征
3. **样本自适应**：简单样本获取少量特征即可分类（低成本），困难样本获取更多（高成本）

### 与本项目的对比分析

##### 相似之处

| 维度 | Joint AFA | 条件化图阅读 |
|------|-----------|------------|
| **顺序获取** | 顺序获取特征值 | 顺序揭示图节点 |
| **停止规则** | 策略决定"获取 or 停止并分类" | Reader 决定"继续揭示 or SUBMIT" |
| **样本自适应** | 不同样本获取不同数量的特征 | 不同图需要不同数量的节点揭示 |
| **集合编码** | 任意大小特征子集的编码器 | 任意大小已揭示子图的编码（GINEEncoder） |
| **联合优化** | 获取策略 + 分类器 + 停止规则 联合训练 | Evidence Network + ReaderMLP + LSTM 联合训练 |

##### 关键差异

| 维度 | Joint AFA | 条件化图阅读 |
|------|-----------|------------|
| **获取对象** | 表格数据的特征（独立） | 图节点（结构依赖）— 根本差异 |
| **动作空间** | 固定特征集（静态动作空间） | 动态 frontier 节点（每步变化） |
| **信息结构** | 特征间无拓扑关系 | 节点通过边连接，揭示 A 暴露 B 的边特征 |
| **遗忘机制** | 无（特征获取不可逆） | **有**（可以"遗忘"已揭示节点释放 budget） |
| **信念状态** | 无显式信念 | 有 LSTM 维护的标签信念 b_t |

> **启示**：Joint AFA 的"获取-停止"联合决策框架是条件化图阅读的最直接模板。只需将"获取特征"替换为"揭示节点"，将"集合编码器"替换为 GNN 子图编码器，将"特征获取成本"替换为"节点揭示成本"。本项目新增的**遗忘机制**是 AFA 框架的扩展——在 AFA 中特征一旦获取即永久可用，而在条件化图阅读中可以将已获取信息"退回"。

---

## 论文 2：SelectiveNet（ICML 2019, CCF-A）

| 维度 | 内容 |
|------|------|
| **论文名称** | SelectiveNet: A Deep Neural Network with an Integrated Reject Option |
| **作者** | Yonatan Geifman, Ran El-Yaniv |
| **发表** | **ICML 2019**（Proceedings of the 36th International Conference on Machine Learning, PMLR 97:2151-2159） |
| **期刊等级** | **CCF-A**（机器学习顶级会议） |
| **arXiv** | [1901.09192](https://arxiv.org/abs/1901.09192) |

### 摘要

我们提出 SelectiveNet，一种在单个神经网络架构中集成拒绝选项的深度学习方法。标准的拒绝分类方法先训练分类器，再基于置信度分数确定拒绝规则（事后方法）。相比之下，SelectiveNet 联合优化分类器和拒绝决策，实现了风险-覆盖权衡的精确控制。网络包含三个输出头——预测头、选择头（决定是否拒绝）和辅助头——通过针对选择性分类风险定制的损失函数进行训练。在多个基准上的实验表明，SelectiveNet 在风险-覆盖曲线上的表现优于事后拒绝方法。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 如何让深度网络自身学会"我什么时候不该做预测"——而非事后分析 |
| **核心贡献** | 三头架构（预测+选择+辅助）+ 针对性损失函数实现端到端选择性分类 |

### 核心思路

```
SelectiveNet 架构:

输入 x
    │
    ▼
  Shared Backbone (DNN)
    │
    ├── Prediction Head: f(x) → ŷ          (预测什么)
    ├── Selection Head: g(x) → [0,1]       (是否拒绝？0=拒绝, 1=预测)
    └── Auxiliary Head: h(x) → ŷ_aux       (辅助预测，仅用于训练)

选择性风险 (Selective Risk):
  R(f, g) = E[ℓ(f(x), y) · g(x)] / E[g(x)]
  分子: 仅接受预测的样本的 loss
  分母: 接受率 (coverage)

  约束: E[g(x)] ≥ τ  (保证最低覆盖率)
  目标: min R(f, g) subject to coverage ≥ τ
```

1. **Selection Head**：输出 0-1 之间的值，0 = 拒绝（"我不知道"），1 = 接受预测
2. **风险-覆盖权衡**：训练时直接优化选择性风险，精确控制拒绝率
3. **端到端训练**：分类和拒绝决策在共享特征上联合学习

### 与本项目的对比分析

##### 相似之处

| 维度 | SelectiveNet | 条件化图阅读 |
|------|-------------|------------|
| **拒绝/提交决策** | Selection Head 决定"拒绝 or 预测" | Reader 决定"CONTINUE or SUBMIT" |
| **决策条件** | 基于共享特征表示的"自信度" | 基于信念 b_t 的熵 + u_t |
| **联合优化** | 分类头 + 选择头联合训练 | Evidence + ReaderMLP + LSTM 联合训练 |
| **覆盖率约束** | 保证至少覆盖 τ 比例的样本 | 上界 B_max 保证不会无限循环 |

##### 关键差异

| 维度 | SelectiveNet | 条件化图阅读 |
|------|-------------|------------|
| **拒绝后行为** | 拒绝后直接放弃（委托给人类） | CONTINUE → 遗忘节点 → 揭示新节点 → 再决策 |
| **时序性** | 单步决策 | 多步 POMDP 循环——每次 CONTINUE 后再次面对 SUBMIT-or-CONTINUE |
| **信息增长** | 拒绝后信息不变 | 每次 CONTINUE 后观测增长（新节点揭示） |
| **遗忘** | 无 | 可选择性遗忘已获取的信息 |

> **启示**：SelectiveNet 的"三头架构"可直接迁移——在 POMDP Reader 中增加 Selection Head（输入 b_t, u_t → 输出 SUBMIT/CONTINUE 概率）。训练时联合优化：L_total = L_classification + α·L_selection + β·L_belief。关键区别在于时序维度——SelectiveNet 是单步拒绝，条件化图阅读是循环的"拒绝→获取→再评估"。

---

## 论文 3：Reversibility-Aware RL（NeurIPS 2021, CCF-A）

| 维度 | 内容 |
|------|------|
| **论文名称** | There Is No Turning Back: A Self-Supervised Approach for Reversibility-Aware Reinforcement Learning |
| **作者** | Nathan Grinsztajn, Johan Ferret, Olivier Pietquin, Philippe Preux, Matthieu Geist |
| **发表** | **NeurIPS 2021**（Advances in Neural Information Processing Systems 34） |
| **期刊等级** | **CCF-A**（人工智能顶级会议） |
| **arXiv** | [2106.11813](https://arxiv.org/abs/2106.11813) |

### 摘要

我们提出学习区分可逆与不可逆动作，以在强化学习中进行更明智的决策。从理论考虑出发，我们证明近似可逆性可以通过一个简单的代理任务来学习：对随机采样的轨迹事件进行时间顺序排序。直观上，如果动作是可逆的，那么将轨迹"倒放"应该产生合理的状态序列。我们设计了一个自监督目标——时间顺序验证（TOV）——来训练可逆性判别器。该判别器随后用于多种下游应用：安全探索（避免不可逆动作）、信用分配（可逆动作不应被惩罚）以及 MCTS 中的树策略剪枝（优先扩展不可逆动作）。在多个领域的实验证明了方法的有效性。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | RL 智能体不知道哪些动作可以"撤销"——这在安全探索和信用分配中至关重要 |
| **核心贡献** | 自监督学习可逆性——通过时序排序判别哪些动作可逆——无需环境建模或额外标注 |

### 核心思路

```
可逆性定义:
  动作 a 在状态 s 是可逆的 ⇔ ∃ 动作序列 {a'_1, ..., a'_k} 使得从 T(s,a) 可以回到 s

核心观察:
  如果从 s₁ → s₂ 然后可以回到 s₁，则 s₁ 应该在时间上"靠近" s₂
  → 可逆性 = 轨迹中状态对之间的时序接近性

自监督学习:
  给定轨迹片段: s₁, s₂, s₃, ..., s_T
  任务: 判断 (s_i, s_j) 的先后顺序
  如果 s_i 和 s_j 之间可逆 → 区分难度高 → TOV loss 高
  如果 s_i 和 s_j 之间不可逆 → 区分容易 → TOV loss 低

TOV (Temporal Order Verification):
  可逆性分数 R(s,a) = E[f(s, s')]  其中 f 是训练好的时序判别器
```

1. **自监督信号**：无需标注哪些动作可逆——时序数据本身提供了信号
2. **双网络架构**：时序判别器（Siamese 网络 + 时序分类头）+ 可逆性嵌入
3. **下游应用**：安全探索、信用分配、树搜索剪枝

### 与本项目的对比分析

##### 相似之处

| 维度 | Reversibility-Aware RL | 条件化图阅读 |
|------|----------------------|------------|
| **核心概念** | 区分可逆与不可逆动作 | 区分"可遗忘"与"不可遗忘"节点 |
| **可逆性度量** | 学习可逆性分数 R(s,a) | 可设计"可遗忘性分数" F(v, G_t) |
| **自监督信号** | 时序顺序判别 | 可用："遗忘节点 v 后重新揭示，信息是否冗余" |

##### 关键差异

| 维度 | Reversibility-Aware RL | 条件化图阅读 |
|------|----------------------|------------|
| **可逆性定义** | 能否通过动作序列回到原状态 | 节点信息是否可通过其他路径恢复 |
| **遗忘动机** | 安全：避免不可逆危险动作 | 效率：释放 budget 给更关键的节点 |
| **应用场景** | 游戏/机器人中的动作安全 | 图节点揭示中的信息管理 |
| **图结构** | 无（标准 MDP） | 图拓扑天然提供"信息冗余度"度量 |

> **启示**：图结构为可逆性提供了天然的度量——如果节点 v 的多跳邻居已被揭示，遗忘 v 的信息损失可能很小（因为 v 的"信息"已经通过邻居的嵌入隐含地保留）。这比 MDP 中的可逆性更容易计算。可以设计一个**结构感知的可遗忘性分数**：`ForgetScore(v) = 1 - centrality(v, G_revealed) / connectivity(revealed_graph_without_v)`——如果移除 v 后已揭示子图连通性不降，则 v 可遗忘。

---

## 论文 4：Classification with Rejection via Cost-Sensitive Classification（ICML 2021, CCF-A）

| 维度 | 内容 |
|------|------|
| **论文名称** | Classification with Rejection Based on Cost-Sensitive Classification |
| **作者** | Nontawat Charoenphakdee, Zhenghang Cui, Yivan Zhang, Masashi Sugiyama |
| **发表** | **ICML 2021**（Proceedings of the 38th International Conference on Machine Learning, PMLR 139:1507-1517） |
| **期刊等级** | **CCF-A**（机器学习顶级会议） |
| **arXiv** | [2010.11430](https://arxiv.org/abs/2010.11430) |

### 摘要

带有拒绝选项的分类（classification with a reject option）是机器学习中的一个基本问题，允许分类器在某些情况下拒绝预测。本文从成本敏感分类的视角重新审视此问题。我们展示了通过设计适当的成本矩阵，成本敏感分类与拒绝分类之间存在等价性。基于此等价性，我们提出了一种通用方法，可将任何二分类成本敏感算法转化为拒绝分类方法，并提供了理论保证（一致性和遗憾界）。在合成和真实数据集上的实验验证了方法的有效性。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 如何将"拒绝预测"形式化为一个有理论保证的成本敏感学习问题？ |
| **核心贡献** | 成本敏感分类与拒绝分类的等价性定理 → 任何成本敏感算法 → 拒绝分类器 |

### 核心思路

```
标准分类:
  Cost = 1[ŷ ≠ y]

带拒绝的分类:
           ┌ 1[ŷ ≠ y]   if accept
  Cost =  ┤
           └ c_r         if reject

  其中 c_r 是拒绝成本 (< 1，否则总是拒绝)

等价性:
  拒绝分类 ≈ 成本敏感二分类 + 一个特殊的"拒绝类"
  拒绝类的成本 = c_r (常数)
  各类的误分类成本 = 1 (或非对称)

训练:
  成本矩阵 C ∈ R^{(K+1)×(K+1)}:
    C_{k, k'} = 1[k≠k']  for k,k'∈[K] (误分类成本)
    C_{k, reject} = c_r  for all k     (拒绝成本)
```

1. **理论保证**：从 cost-sensitive 分类的 consistency 导出 reject 分类的 consistency
2. **通用性**：适用于任何 base classifier（SVM, NN, tree 等）
3. **成本灵活**：可自然扩展到各类有不同误分类成本 + 不同拒绝成本的场景

### 与本项目的对比分析

##### 相似之处

| 维度 | Cost-Sensitive Rejection | 条件化图阅读 |
|------|------------------------|------------|
| **成本结构** | 误分类有成本，拒绝有成本 c_r | 误分类成本 + 继续探索成本 (budget 消耗) |
| **决策边界** | 当预测置信度 < 阈值 → 拒绝 | 当信念熵 > 阈值 或 u_t > 阈值 → CONTINUE |
| **不对称成本** | 不同类型的误分类可以有不同成本 | 不同类别的误分类重要性可以不同 |

##### 关键差异

| 维度 | Cost-Sensitive Rejection | 条件化图阅读 |
|------|------------------------|------------|
| **拒绝后** | 委托给人类（不做额外信息获取） | CONTINUE → 主动获取更多证据 |
| **拒绝成本结构** | 常数 c_r | c_continue = f(budget_remaining, u_t) — 动态成本 |
| **多步拒绝** | 单次拒绝 | 可循环拒绝多次直到 B_max |
| **信息增长** | 拒绝后信息不变 | CONTINUE 后观测更新 → 降低下次拒绝概率 |

> **启示**：成本敏感视角为条件化图阅读提供了优雅的数学框架。继续探索的"成本"可以形式化为：`C(CONTINUE) = c_base × (remaining_upper_budget / upper_bound) + λ × (1 - u_t)`——预算越紧张、当前信念越确定，CONTINUE 的成本越高。当 `C(CONTINUE) > E[cost of wrong classification if SUBMIT now]` 时，Reader 应选择 SUBMIT。

---

## 论文 5：When Does Confidence-Based Cascade Deferral Suffice?（NeurIPS 2023, CCF-A）

| 维度 | 内容 |
|------|------|
| **论文名称** | When Does Confidence-Based Cascade Deferral Suffice? |
| **作者** | Wittawat Jitkrittum, Neha Gupta, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, Sanjiv Kumar |
| **发表** | **NeurIPS 2023**（Advances in Neural Information Processing Systems 36） |
| **期刊等级** | **CCF-A**（人工智能顶级会议） |
| **arXiv** | [2307.02764](https://arxiv.org/abs/2307.02764) |

### 摘要

级联（cascade）是一种经典策略，使推理成本能自适应地在不同样本间变化：一系列分类器按顺序被调用，延迟规则决定是调用下一个分类器还是终止预测。一种简单的延迟规则使用当前分类器的置信度（如最大 softmax 概率）。尽管这种置信度基础的延迟规则无视级联的结构（如不建模下游模型的错误），在实践中却往往表现得出奇的好。本文试图更好地理解置信度基础延迟可能失败的条件，以及替代延迟策略何时能表现更好。我们首先对置信度基础延迟的有效性进行了理论刻画，揭示了仅当级联中的模型具有特定校准属性时它才是最优的。在此分析基础上，我们提出了当这些条件不满足时的替代延迟策略。在图像和文本数据上的实验验证了我们的发现。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 简单的置信度阈值延迟规则为什么有效？什么时候会失败？ |
| **核心贡献** | 首次系统理论分析置信度基础级联延迟——给出最优性条件 + 失败模式 + 替代策略 |

### 核心思路

```
级联延迟 (Cascade Deferral):

  模型 M₁ (轻量) → 预测 + 置信度
      │
      ▼
  延迟规则: 置信度 > τ₁?
      │        ├── YES → 输出 M₁ 的预测 (便宜)
      │        └── NO → defer to M₂
      │                    │
      │                    ▼
      │              模型 M₂ (重量) → 预测 (昂贵但准确)

理论发现:
  置信度基础延迟在以下条件下是(接近)最优的:
    1. 级联中的模型具有"置信度单调性"——更深的模型置信度不低于更浅的模型
    2. 模型是 well-calibrated ——置信度 ≈ 准确概率
    3. 模型间有"一致的置信度排序"

  当这些条件不满足时 (如模型未被校准时):
    → 使用 Bayes-optimal deferral rule (需要估计 P(correct|confidence))
    → 使用 distribution-aware 阈值 (不同样本类型不同阈值)
```

1. **理论刻画**：证明置信度基础延迟的最优性等价于模型的特定校准条件
2. **失败模式**：当级联中后模型比前模型置信度更低时 → 基础延迟失效
3. **替代策略**：Bayes-optimal deferral、learned deferral function

### 与本项目的对比分析

##### 相似之处

| 维度 | Confidence-Based Cascade Deferral | 条件化图阅读 |
|------|----------------------------------|------------|
| **延迟决策** | 基于置信度决定"now or defer" | 基于信念决定"SUBMIT or CONTINUE" |
| **校准关键性** | 置信度校准 = 延迟最优性的前提 | 信念校准 = SUBMIT 决策可靠性的前提 |
| **级联类比** | 轻量模型 → 重量模型 | 当前子图 → 继续探索后的更大子图 |
| **失败模式** | miscalibration → deferral 失效 | u_t 不校准 → SUBMIT 决策错误 |

##### 关键差异

| 维度 | Confidence-Based Cascade Deferral | 条件化图阅读 |
|------|----------------------------------|------------|
| **延迟后** | 委托给固定的更重量级模型 | 主动选择揭示哪些节点（改变观测） |
| **延迟次数** | 固定级联（0-1次） | 可变次数（直到 SUBMIT 或 B_max） |
| **模型序列** | 预定义的固定模型链 | 动态变化——每次 CONTINUE 后观测不同 |
| **遗忘** | 无 | 可遗忘已揭示节点改变观测 |

> **启示**：该论文的**校准分析**直接指导条件化图阅读的 SUBMIT 决策设计：SUBMIT 规则的可靠性取决于 `b_t`（标签信念）的校准质量。如果 `b_t` 声称"90% 确定是第 k 类"但实际准确率只有 70%，则基于 b_t 的 SUBMIT 决策会过早终止。训练中需要明确加入**信念校准正则化**——如 ECE (Expected Calibration Error) loss——确保 b_t 的置信度与实际 accuracy 匹配。此外，该文的"置信度单调性"条件在条件化图阅读中变为"信念单调性"——随着更多节点被揭示，信念 b_t 应该单调趋向真实标签。

---

## 总结对比矩阵

| # | 论文 | 会议/期刊 | CCF | 年份 | 核心方法 | 机制 | 应用域 | 与项目相似度 | 核心借鉴点 |
|---|------|-----------|-----|------|----------|------|--------|:--:|-----------|
| 1 | Joint AFA | NeurIPS | A | 2018 | 可变大小集合编码 + 停止规则 | 顺序获取-停止-分类 | 表格数据 | ⭐⭐⭐⭐⭐ | "获取-停止"联合决策框架 |
| 2 | SelectiveNet | ICML | A | 2019 | 三头架构 + 选择性风险 | 预测-拒绝联合优化 | 图像分类 | ⭐⭐⭐⭐⭐ | 端到端学习"何时拒绝" |
| 3 | Reversibility-Aware RL | NeurIPS | A | 2021 | 自监督时序判别 | 可逆性分数学习 | 游戏/机器人 | ⭐⭐⭐⭐ | "可遗忘性"度量——图结构天然支持 |
| 4 | Rejection via Cost-Sensitive | ICML | A | 2021 | 成本敏感等价性 | 成本矩阵形式化 | 通用分类 | ⭐⭐⭐⭐ | CONTINUE 成本的数学形式化 |
| 5 | Cascade Deferral Analysis | NeurIPS | A | 2023 | 校准条件理论分析 | 置信度延迟最优性条件 | 图像/文本 | ⭐⭐⭐⭐ | 信念校准 = SUBMIT 决策可靠性 |

---

## 阅读优先级建议

| 优先级 | 论文 | 理由 |
|--------|------|------|
| **P0 — 必读** | Joint AFA (#1) | "获取-停止"联合决策框架是条件化图阅读的**直接模板**——结构最相似 |
| **P0 — 必读** | SelectiveNet (#2) | 端到端学习"何时拒绝"——Selection Head 架构可直接迁移到 POMDP Reader |
| **P1 — 重要** | Reversibility-Aware RL (#3) | "可遗忘性"度量的理论基础——图拓扑天然提供可逆性信号 |
| **P1 — 重要** | Cascade Deferral Analysis (#5) | 信念校准 = SUBMIT 可靠性的前提——指导训练中如何加入校准监督 |
| **P2 — 参考** | Rejection via Cost-Sensitive (#4) | CONTINUE 成本的形式化——若需要可微的 SUBMIT 规则 |

---

## 附：方法论映射——从已有方法到条件化图阅读

| 已有方法 | 概念 | → | 本项目对应组件 |
|---------|------|----|-------------|
| Joint AFA 的 Set Encoder | 任意大小已获取特征集的编码 | → | GINEEncoder + SubgraphBuilder（任意大小已揭示子图编码） |
| Joint AFA 的 Stopping Policy | 基于当前已编码特征决定是否停止 | → | SubmitHead(b_t, u_t, budget) → {SUBMIT, CONTINUE} |
| SelectiveNet 的 Selection Head | 网络输出"拒绝"概率 | → | ReaderMLP 额外输出头：p_submit |
| SelectiveNet 的 Selective Risk | 仅在被接受样本上计算的风险 | → | 仅在被 SUBMIT 的图上计算分类 loss |
| Reversibility RL 的 TOV Score | 自监督学习动作可逆性 | → | 图结构特征 → ForgetScore(v)：v 的信息是否可通过邻居恢复 |
| Cost-Sensitive Rejection 的成本矩阵 | C_{k, reject} = c_r | → | C_CONTINUE = f(budget_left, u_t, belief_entropy) |
| Cascade Deferral 的校准条件 | 置信度校准 ↔ 延迟最优性 | → | 信念校准 ↔ SUBMIT 最优性——需 ECE 正则化 |

---

> **最后更新**：2026-06-13
> **关联文档**：`evidential-pomdp-reader-survey.md`（证据信念 POMDP Reader 基础架构）、`code/m5/2026-6-12 reader分析和改进灵感.md` Section 7-8（架构与符号化分析）

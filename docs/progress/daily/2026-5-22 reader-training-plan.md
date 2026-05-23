# Reader 培训计划

> **定位**：本文档定义 Learned Reader 策略网络的完整培训方案，包括训练方法、反馈循环机制、执行路线和评估指标。
>
> **上游依赖**：M3 基线尺子（Encoder-Classifier）、v0 协议（`base-protocol-v0.md`）
> **下游输出**：可部署的 Reader 策略网络 $\pi_\theta$
> **编制日期**：2026-05-23

---

## 一、培训目标与受众

### 1.1 核心问题

当前 M3 阶段的三种 Reader（random / BFS / max-frontier-degree）是**固定规则策略**——它们只看图拓扑，不看任务信号。Reader 培训的目标是：

> **训练一个能根据已揭示拓扑和特征，自适应选择下一个揭示节点的 Reader 策略网络 $\pi_\theta$。**

具体而言，给定部分观测图 $G_t$（含已揭示节点集、已揭示边特征、frontier 集合），Reader 需输出 frontier 上的概率分布，决定下一个揭示哪个节点。

### 1.2 关键概念


| 概念                                     | 定义                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| **Reader（$\pi_\theta$）**               | 执行`reveal_node(v)` 动作序列的决策组件。输入为 $G_t$，输出为 frontier 上的概率分布  |
| **Encoder-Classifier（$\phi$，"尺子"）** | 消费最终揭示子图$G_B$，输出图级预测 $\hat{y}$。在 Reader 培训中担任**评估者/信号源** |
| **Frontier**                             | ID 已知但尚未揭示的节点集合——Reader 的选择空间                                     |
| **端到端训练**                           | Reader 的策略网络和 Encoder-Classifier 通过同一损失函数联合优化                      |
| **反馈循环**                             | 分类信号（loss / reward）从尺子回流到 Reader，指导策略参数更新                       |

### 1.3 与 Policy-Learn 的差异

本项目与 Policy-Learn（ICLR 2024，最接近的已发表工作）的核心差异决定了培训方案的特殊设计：


| 维度     | Policy-Learn           | 本项目                              |
| -------- | ---------------------- | ----------------------------------- |
| 图可见性 | 选择网络看到完整图     | Reader 只看到部分观测$G_t$          |
| 候选池   | 固定 n 个（全图节点）  | 动态 frontier（仅已揭示节点的邻居） |
| 候选表示 | 每个候选有完整子图特征 | frontier 节点（PPA 无节点特征）     |
| 初始状态 | 已有完整图             | $G_0$ 仅知种子节点 ID               |

**核心挑战**：本项目中 Reader 面对的是部分可观测马尔可夫决策过程（POMDP），frontier 节点在被 reveal 之前几乎没有任何特征信息，决策信号需通过已揭示邻居间接推断。

---

## 二、培训内容体系

Reader 培训包含三大类训练方法，按学习信号类型划分。三种方法并非互斥——推荐以"BC 预训练 → 端到端微调"的序贯方式组合使用。

### 模块一：行为克隆（Behavior Cloning, BC）

**思路**：用能访问完整图信息的 teacher（oracle）生成高质量阅读轨迹，以监督学习方式训练 student Reader 模仿这些轨迹。

**适用场景**：首选方案。训练信号最可靠（标准交叉熵），风险最低。即使端到端方法不收敛，BC Reader 也构成最小可交付成果。

#### Teacher（Oracle）设计


| Teacher 方案         | 原理                                                                            | 计算开销                        | 推荐度     |
| -------------------- | ------------------------------------------------------------------------------- | ------------------------------- | ---------- |
| **梯度归因 Teacher** | 在全图上训练 GNN → 反向求$\|\partial \hat{y} / \partial h_v\|$ → 按重要性排序 | 低（全图一次前向+反向）         | 首选       |
| **信息增益 Teacher** | 贪心前瞻：对每个 frontier 节点模拟揭示 → 选置信度最高者                        | 高（每步$\|frontier\|$ 次前向） | 备选       |
| **结构 Teacher**     | 度中心性 / PageRank                                                             | 极低                            | 多样性补充 |

**Teacher 多样性增强**：混合多种 teacher（如梯度归因 + 度中心性 + PageRank），每张图生成多条不同 teacher 的轨迹，增加训练数据多样性。

#### 轨迹质量评分


| 指标              | 定义                                          | 用途                           |
| ----------------- | --------------------------------------------- | ------------------------------ |
| Final Accuracy    | $\mathbb{1}(\hat{y}_{\text{final}} = y)$      | 必要条件——错误轨迹不用于训练 |
| Budget Efficiency | $\text{acc}_{\text{final}} / B_{\text{used}}$ | 保留高效轨迹                   |
| Information Gain  | $\sum_t [H(\hat{y}_{t-1}) - H(\hat{y}_t)]$    | 衡量每步信息增量               |

**过滤策略**：仅保留 Final Accuracy = 正确 + Budget Efficiency > 中位数的轨迹。每张图保留 top-3 条。

#### BC 训练要点

- **网络结构**：SubgraphGNN（GIN, $d_{\text{hidden}}=128$, $L=2$）+ FrontierScorer
- **损失函数**：$\mathcal{L}_{\text{BC}} = -\frac{1}{B}\sum_{t} \log \pi_\theta(a_t^{\text{teacher}} | s_t)$
- **Label Smoothing**：$\epsilon = 0.1$（teacher 选择并非唯一最优）
- **训练周期**：50~100 epochs，早停基于 val accuracy

### 模块二：端到端训练（Gumbel-Softmax）

**思路**：将 Reader 和冻结尺子组成端到端可微管线，分类 loss 通过 Straight-Through Gumbel-Softmax 回传到 Reader。

**适用场景**：BC 预训练之后的微调阶段。低方差的直通梯度做精细优化，突破 teacher 的性能天花板。

#### 架构

```
G_t → SubgraphGNN → H_t → FrontierScorer → scores
       → GumbelSoftmax(scores, τ) → v_t → reveal_node(v_t)
       → ... (repeat B times) ... → G_B
       → φ_frozen(G_B) → ŷ → L_task(ŷ, y)
       → gradient through ST estimator → update π_θ
```

#### 关键技术点


| 要素     | 设计                                                                                       |
| -------- | ------------------------------------------------------------------------------------------ |
| 选择网络 | SubgraphGNN（$d_{\text{hidden}}=128$, $L=2$）+ FrontierScorer（邻居聚合 → MLP → scalar） |
| 采样方式 | Straight-Through Gumbel-Softmax（前向 hard one-hot，反向 soft 梯度）                       |
| 温度调度 | $\tau$ 从 1.0 退火到 0.1（epoch 级别：$\tau \leftarrow \max(0.1, \tau \times 0.995)$）     |
| 探索机制 | 概率分布上 dropout（$p \in \{0, 0.3, 0.5\}$）；$B \geq 5$ 时 mask 已选节点                 |
| 梯度稳定 | 梯度裁剪 max_norm=1.0；BatchNorm（track_running_stats=False）                              |
| 推理时   | argmax 替代采样（确定性选择）                                                              |

#### Frontier Scorer 设计

```
for each v in frontier:
    revealed_neighbors = {u | (u,v) ∈ E_known, u ∈ V_revealed}
    agg_v = mean(H_t[u] for u in revealed_neighbors)     # 邻居嵌入聚合
    score_v = MLP(concat(agg_v, deg_known_embed))         # d→d→1
return stack([score_v for v in frontier])
```

**信息瓶颈**：frontier 节点揭示前无节点特征，仅能通过已揭示邻居的嵌入间接推断价值。

缓解方案：允许 Reader 在决策前获取 frontier 边特征（`reveal_edge_f` 动作，已在 v0 协议附录 B 预留）。

### 模块三：强化学习训练（REINFORCE / Actor-Critic）

**思路**：将图阅读建模为有限视野 MDP，用策略梯度方法训练 Reader，避免 Gumbel-Softmax 的偏差。

**适用场景**：Gumbel-Softmax 训练不稳定或出现性能平台时的备选方案。也可与 Gumbel-Softmax 混合使用（互补偏差-方差特性）。

#### MDP 形式化


| 元素      | 定义                                                                                    |
| --------- | --------------------------------------------------------------------------------------- |
| 状态$s_t$ | $G_t = (V_{\text{revealed}}^{(t)}, E_{\text{revealed\_feat}}^{(t)}, \text{frontier}_t)$ |
| 动作$a_t$ | 选择$v \in \text{frontier}_t$ 并执行 `reveal_node(v)`                                   |
| 动作空间  | $\text{frontier}_t$，大小动态变化                                                       |
| 状态转移  | 确定性：$s_{t+1} = \psi(s_t, a_t)$                                                      |
| 奖励$R$   | 延迟奖励：$\mathbb{1}(\hat{y} = y)$ 或预测置信度                                        |
| 视野      | 固定预算$B$（5~15 步）                                                                  |

#### REINFORCE 核心公式

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{B-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (R - b) \right]
$$

其中 $b$ 为 baseline（rollout 均值或 Critic 输出），用于降方差。

#### 与 Gumbel-Softmax 对比


| 维度       | Gumbel-Softmax       | REINFORCE                   |
| ---------- | -------------------- | --------------------------- |
| 梯度偏差   | 有偏（ST 近似）      | 无偏（蒙特卡洛）            |
| 梯度方差   | 低                   | 高（需多 rollout）          |
| 信用分配   | 自动（梯度自然分配） | 需显式设计（baseline / TD） |
| 前向开销   | 1×                  | K×（K=5~10 rollout）       |
| 实现复杂度 | 中                   | 低                          |
| 超参敏感度 | 温度退火 schedule    | baseline 选择、K 大小       |

### 三方法对比总结


| 维度        | BC（模块一）                 | Gumbel-Softmax（模块二）  | REINFORCE（模块三）   |
| ----------- | ---------------------------- | ------------------------- | --------------------- |
| 训练范式    | 监督学习                     | 端到端（可微采样）        | 强化学习（策略梯度）  |
| 训练信号    | CrossEntropy(teacher action) | 分类 loss 经 ST estimator | log-prob × advantage |
| 训练稳定性  | **高**                       | 中                        | 中                    |
| 性能上限    | 受限于 teacher 质量          | 理论上界                  | 理论上界              |
| 依赖 oracle | 是（teacher 轨迹）           | 否                        | 否                    |
| 推荐阶段    | 预训练                       | 微调                      | 精修/备选             |

---

## 三、反馈循环机制

### 3.1 整体架构

反馈循环是 Reader 培训的核心基础设施——它定义了训练信号如何从分类结果回流到 Reader 参数。

```
┌──────────────────────────────────────────────────────────────┐
│                    Active Graph Reading                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Reader (策略网络 π_θ)               │   │
│  │                                                      │   │
│  │  G_t = (V_rev, E_rev_feat, frontier)                │   │
│  │    │                                                 │   │
│  │    ▼                                                 │   │
│  │  ┌──────────────┐                                    │   │
│  │  │ Subgraph GNN │ ← 仅对已揭示子图做消息传递         │   │
│  │  │ (2层 GIN)    │                                    │   │
│  │  └──────┬───────┘                                    │   │
│  │         │ node embeddings H_t                        │   │
│  │         ▼                                            │   │
│  │  ┌──────────────────────────┐                        │   │
│  │  │ Frontier Scorer          │                        │   │
│  │  │ for v ∈ frontier:        │                        │   │
│  │  │   score_v = MLP(agg(H_t  │                        │   │
│  │  │     [N(v) ∩ V_rev]))     │                        │   │
│  │  └──────────┬───────────────┘                        │   │
│  │             │ scores ∈ R^{|frontier|}                │   │
│  │             ▼                                        │   │
│  │  采样: GumbelSoftmax(p_t) 或 Categorical(p_t)        │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │ reveal_node(v_t)                       │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Encoder-Classifier (冻结/"尺子")           │   │
│  │  G_B → GINEEncoder(hidden=300, L=5) → y_hat          │   │
│  └──────────────────────────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│          L_task(ŷ, y)  ──→  梯度回传至 Reader               │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 训练信号的三种回流路径


| 路径                 | 原理                                                                               | 对应模块 |
| -------------------- | ---------------------------------------------------------------------------------- | -------- |
| **路径 1：直通梯度** | $L_{\text{task}}$ 经 Gumbel-Softmax ST estimator → Frontier Scorer → SubgraphGNN | 模块二   |
| **路径 2：策略梯度** | $R \cdot \nabla_\theta \log \pi_\theta(a \mid s)$ → Reader 参数                   | 模块三   |
| **路径 3：模仿梯度** | $\text{CE}(\pi_\theta(s_t), a_t^{\text{teacher}})$ → Reader 参数                  | 模块一   |

### 3.3 损失函数

**主损失**（分类任务）：

$$
\mathcal{L}_{\text{task}} = -\sum_{c=1}^{C} y_c \log \phi_{\text{frozen}}(G_B(\pi_\theta))_c
$$

**辅助损失**：


| 损失       | 公式                                                                                          | 目的                                   |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------- |
| 熵正则     | $\mathcal{L}_{\text{ent}} = -\frac{1}{B}\sum_t \sum_{v} p_t(v) \log p_t(v)$                   | 鼓励探索（$\lambda_{\text{ent}} < 0$） |
| 多样性正则 | $\mathcal{L}_{\text{div}} = \sum_t \sum_{s<t} \text{sim}(h_{v_t}, h_{v_s})$                   | 防止重复选择同类节点                   |
| BC 辅助    | $\mathcal{L}_{\text{BC}} = -\frac{1}{B}\sum_t \log \pi_\theta(a_t^{\text{teacher}} \mid s_t)$ | 保留 teacher 知识                      |

### 3.4 训练阶段划分

```
Phase A: 预训练 Encoder-Classifier（M3 尺子）
─────────────────────────────────────────────
  用 random 策略生成 G_B → 训练 φ → 冻结为 φ_frozen

Phase B: 训练 Reader 策略网络
─────────────────────────────────────────────
  冻结 φ_frozen
  可选方案:
    B1: 纯 BC（模块一）
    B2: BC + Gumbel-Softmax 联合（推荐）
    B3: Gumbel-Softmax 从头训练
    B4: REINFORCE（模块三）

Phase C: 交替微调（可选）
─────────────────────────────────────────────
  用 Reader 生成的子图微调 φ → 冻结 φ → 继续训练 π_θ
  目的：修复 domain shift（Reader 子图分布 ≠ random 子图分布）
```

### 3.5 梯度流分析

```
L_task (CrossEntropy)
  │
  ▼
φ_frozen（冻结 → 梯度穿过但不更新参数）
  │
  ▼
最终子图 G_B 的节点嵌入（来自 SubgraphGNN）
  │
  ▼
SubgraphGNN 的每一层 GINEConv ← 梯度在 B 步调用上累积（参数共享）
  │
  ▼
FrontierScorer MLP（通过 Gumbel-Softmax straight-through）
  │
  ▼
每步的 scores[t] ← 仅选中节点的位置有非零梯度
```

**潜在问题与缓解**：


| 问题                                | 缓解                                      |
| ----------------------------------- | ----------------------------------------- |
| B 步展开导致梯度消失/爆炸           | B ≤ 15；BatchNorm；梯度裁剪 max_norm=1.0 |
| 冻结尺子对 Reader 子图 domain shift | Phase C 交替微调                          |
| 梯度通过 ST estimator 有偏差        | 结合 REINFORCE 做无偏校正（见 §4.2）     |

---

## 四、组合方案与执行路线

### 4.1 推荐执行策略：三阶段分层推进

```
╔═══════════════════════════════════════════════════════════════╗
║          Reader 学习策略：推荐执行顺序                         ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Stage 1 (M3)  ──→ 拿到基线：random / BFS / maxdeg           ║
║       │            Kill criterion 检查 → headroom 确认         ║
║       │                                                       ║
║       ▼                                                       ║
║  Stage 2 (M4)  ──→ Teacher 轨迹生成 + 评分                   ║
║       │            （回答"好轨迹长什么样"）                  ║
║       │                                                       ║
║       ├──────────→ Stage 3a (M5a)：BC 预训练                 ║
║       │            （最稳定信号，最小可交付成果）              ║
║       │                                                       ║
║       └──────────→ Stage 3b (M5b)：端到端微调               ║
║                    C+A 联合训练（BC + Gumbel-Softmax）        ║
║                    （从 BC 权重出发，追求更高性能）            ║
║                    （若不稳定 → REINFORCE 备选）              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### 4.2 四种推荐结合方案

#### 思路

A：Policy-Learn 思想——Gumbel-Softmax：将 Reader 和冻结尺子组成端到端可微管线，分类 loss 通过 Straight-Through Gumbel-Softmax 回传到 Reader。

B：REINFORCE with Classifier Reward——将图阅读过程建模为有限视野马尔可夫决策过程（finite-horizon MDP）

C：Teacher-Student Trajectory Distillation；（1）使用能访问完整图信息的 teacher（oracle / strong heuristic）生成高质量阅读轨迹；（2）用行为克隆（Behavior Cloning）训练一个只能看到部分观测的 student Reader 模仿这些轨迹。这避免了端到端的梯度传播难题，将训练转化为标准的监督学习。

#### 方案一（首选）：C+A 联合训练 —— BC + Gumbel-Softmax

**核心**：同一训练 step 接受 BC 损失 + 任务损失两个信号，加权组合。

$$
\mathcal{L}_{\text{joint}} = \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}} + \lambda_{\text{task}} \cdot \mathcal{L}_{\text{task}} + \lambda_{\text{ent}} \cdot \mathcal{L}_{\text{ent}}
$$

**权重调度**：

- 训练早期（epochs 0~20）：$\lambda_{\text{BC}}$ 主导（1.0），$\lambda_{\text{task}}$ 从 0 线性增长
- 训练后期：$\lambda_{\text{BC}}$ 衰减到 0.1，$\lambda_{\text{task}}$ 主导（1.0）

**设计直觉**：先学会模仿，再学会超越。

**收益**：BC 提供安全网保证早期收敛，任务损失提供超越 teacher 的梯度信号。

#### 方案二：A+B 混合梯度 —— Gumbel-Softmax + REINFORCE 互补

**核心**：以 Gumbel-Softmax 梯度为主（低方差），REINFORCE 梯度做无偏校正项。

**简化实现**：交替梯度步——奇数步 Gumbel-Softmax 更新，偶数步 REINFORCE 更新。

**收益**：修正 Gumbel-Softmax 偏差造成的次优收敛。

**适用条件**：纯 Gumbel-Softmax 训练后期 val accuracy 停滞 > 10 epochs 且未达 teacher 上界。

#### 方案三：C+A+B 三阶段管线

**核心**：BC 预训练（初始化）→ Gumbel-Softmax 微调（精细优化）→ REINFORCE 精修（无偏校正）。

**阶段转换触发条件**：

- BC → Gumbel-Softmax：BC val accuracy 10 epochs 不提升
- Gumbel-Softmax → REINFORCE：GS val accuracy 停滞 + 未达 teacher 95%
- REINFORCE 精修为可选项，仅在需要时触发

**投入产出**：三阶段性能上限最高，但实现复杂度也最高。建议先用 C+A 两阶段，REINFORCE 精修留到后续版本。

#### 方案四：C+B 联合 —— BC 作为 REINFORCE 的探索先验

**核心**：在 REINFORCE 损失上添加 KL 散度约束，防止策略偏离 BC 预训练策略太远。

$$
\mathcal{L}_{\text{C+B}} = \mathcal{L}_{\text{REINFORCE}} + \beta \cdot \text{KL}(\pi_{\text{BC}} \| \pi_\theta)
$$

$\beta$ 调度：初始 0.1 → 每 epoch 衰减 $\times 0.99$

**适用场景**：Gumbel-Softmax 训练失败时的备选方案；或预算 $B > 20$ 时（长序列端到端梯度传播困难）。

### 4.3 方案决策树

```
M3 基线和尺子就绪
      │
      ▼
Kill Criterion 检查 ─── headroom 不足 ──→ 换数据集 / 硬化 G_0
      │
      │ headroom 充足
      ▼
M4: Teacher 轨迹生成
      │
      ▼
M5a: BC 预训练（模块一）← 始终执行
      │
      ▼
┌─────────────────────────────────────────┐
│ 推荐一：C+A 联合训练（首选）              │
│ BC + Gumbel-Softmax 联合优化              │
└────────────────┬────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │ 收敛良好             │ 收敛不佳 / 停滞
      ▼                     ▼
  直接使用 π_joint    ┌──────────────────────┐
                      │ 推荐二：A+B 混合梯度   │
                      │ 或 推荐四：C+B 联合    │
                      └──────────┬───────────┘
                                 │
                                 ▼
                       ┌──────────────────────┐
                       │ 推荐三：REINFORCE 精修│
                       │ （可选，论文 rebuttal）│
                       └──────────────────────┘
```

### 4.4 执行时间线


| 阶段 | 产出                                     | 工作量         | 训练时间（估算）    |
| ---- | ---------------------------------------- | -------------- | ------------------- |
| M3   | baseline scores + 冻结尺子               | 已完成         | —                  |
| M4   | curated trajectories（预计 50k~100k 条） | 1~2 周         | 取决于 teacher 方案 |
| M5a  | BC Reader（性能 ≥ BFS baseline）        | 1~2 周         | ~100 epochs         |
| M5b  | C+A 联合训练 Reader                      | 2~3 周         | ~150 epochs         |
| M5c  | 备选方案微调/精修                        | 1~2 周（按需） | ~50 epochs          |

### 4.5 关键风险与前置检测


| 风险                                  | 严重度 | 前置检测方法                                    |
| ------------------------------------- | ------ | ----------------------------------------------- |
| Teacher 轨迹质量不足，BC 效果差       | 高     | M4 阶段检查 teacher vs BFS 性能差距             |
| Gumbel-Softmax 梯度消失（B 步展开后） | 中     | 监控每步梯度范数                                |
| BC 与 Task 梯度方向冲突               | 中     | 监控梯度余弦相似度                              |
| REINFORCE 方差过大                    | 中     | 监控 advantage 标准差                           |
| 冻结尺子对 Reader 子图 domain shift   | 中     | 用 Reader 子图评估尺子 accuracy vs random 子图  |
| Reader 过拟合训练图拓扑               | 低     | OGBG-PPA 有 species split → 天然跨物种泛化检测 |

---

## 五、角色与职责

### 5.1 核心组件


| 角色                                     | 定义                         | 在培训中的职责                                                       |
| ---------------------------------------- | ---------------------------- | -------------------------------------------------------------------- |
| **Reader（$\pi_\theta$）**               | SubgraphGNN + FrontierScorer | 接收$G_t$，输出 frontier 上的概率分布，决定揭示哪个节点              |
| **Encoder-Classifier（$\phi$，"尺子"）** | GINEEncoder + MLP head       | 消费$G_B$ 输出 $\hat{y}$；培训中提供梯度信号或奖励；可冻结或交替微调 |
| **Teacher（Oracle）**                    | 全图 GNN / 信息增益计算器    | 仅 M4/M5a 阶段使用：生成高质量阅读轨迹供 BC 训练                     |
| **SubgraphBuilder**                      | 图构建工具                   | 从 state 构造 DGLGraph（仅含已揭示节点和边）                         |

### 5.2 SubgraphGNN（Reader 共用编码器）

```
SubgraphGNN:
  node_emb:   nn.Embedding(n_nodes_total, d_hidden)   # 可学习节点初始嵌入
  edge_enc:   nn.Linear(7, d_hidden)                   # 边特征映射（PPA 为 7 维）
  convs:      ModuleList[GINEConv × L]                 # L=2~3
  bns:        ModuleList[BatchNorm1d × L]
  dropout:    0.3
```

### 5.3 FrontierScorer（Reader 打分器）

```
FrontierScorer:
  输入: H_t（已揭示节点嵌入）, frontier_set, edge_known
  对每个 v ∈ frontier:
    1. 找到已揭示邻居: N(v) ∩ V_revealed
    2. 聚合邻居嵌入: mean(H_t[u] for u in revealed_neighbors)
    3. 拼接辅助特征: concat(agg, deg_known_embed)
    4. MLP 打分: Linear→ReLU→Linear→scalar
  输出: scores ∈ R^{|frontier|}
```

### 5.4 各阶段角色参与


| 阶段            | Reader                        | Encoder-Classifier   | Teacher          | SubgraphBuilder |
| --------------- | ----------------------------- | -------------------- | ---------------- | --------------- |
| M3              | 固定规则（random/BFS/maxdeg） | 训练 + 冻结          | 无               | 使用            |
| M4              | 无                            | 冻结（评分轨迹）     | **生成轨迹**     | 使用            |
| M5a（BC）       | **训练**                      | 冻结（评分轨迹）     | **提供标签**     | 使用            |
| M5b（端到端）   | **训练**                      | **冻结（提供梯度）** | 可选（辅助监督） | 使用            |
| M5c（交替微调） | **训练**                      | **交替解冻微调**     | 无               | 使用            |

---

## 六、评估指标

### 6.1 Reader 策略质量指标


| 指标                     | 定义                                                 | 用途                             |
| ------------------------ | ---------------------------------------------------- | -------------------------------- |
| **Final Accuracy**       | 使用 Reader 选出的$G_B$ 在冻结尺子上的分类准确率     | 主要指标，对比 random/BFS/maxdeg |
| **Accuracy vs Budget**   | 在不同$B$ 值下的准确率曲线                           | 衡量预算效率                     |
| **Headroom over Random** | $\text{Acc}(\pi_\theta) - \text{Acc}(\text{random})$ | 证明学习策略的增益               |
| **Headroom over BFS**    | $\text{Acc}(\pi_\theta) - \text{Acc}(\text{BFS})$    | 证明超越结构启发式的增益         |

### 6.2 培训过程监控指标


| 指标                              | 定义                                | 警戒线                                 |
| --------------------------------- | ----------------------------------- | -------------------------------------- |
| **梯度范数**                      | 每步梯度的 L2 范数                  | < 1e-6 可能梯度消失；> 10 可能梯度爆炸 |
| **策略熵**                        | $-\sum_v p_t(v) \log p_t(v)$        | 持续 < 0.1 表示过早收敛                |
| **选中节点的度分布**              | Reader 选择节点的 degree 直方图     | 与 teacher/BFS 对比，检测系统性偏差    |
| **BC-Task 梯度余弦相似度**        | BC 损失梯度与 Task 损失梯度的余弦值 | 持续 < -0.5 表示信号冲突               |
| **Advantage 标准差**（REINFORCE） | rollout 间 advantage 的标准差       | > 1.0 表示方差过大                     |

### 6.3 培训阶段验证门


| 阶段       | 通过条件                                  | 失败时的行动                        |
| ---------- | ----------------------------------------- | ----------------------------------- |
| M3 → M4   | random vs BFS 有显著 gap（headroom 存在） | 换数据集或硬化$G_0$                 |
| M4 → M5a  | teacher 轨迹 Final Accuracy > BFS 基线    | 改进 teacher 设计                   |
| M5a → M5b | BC Reader Accuracy > BFS 基线             | 增加 teacher 多样性，扩大 BC 数据量 |
| M5b 完成   | C+A Reader Accuracy > BC Reader           | 检查梯度冲突，调整$\lambda$ 调度    |
| M5c 触发   | GS val accuracy 停滞 > 10 epochs          | 启动 REINFORCE 精修                 |

---

## 附录 A：关键超参速查


| 超参                     | 值                                                     | 适用模块  |
| ------------------------ | ------------------------------------------------------ | --------- |
| $d_{\text{hidden}}$      | 128                                                    | 全部      |
| $L$（GNN 层数）          | 2                                                      | 全部      |
| $B$（预算）              | 5~15（端到端），10~20（BC）                            | 全部      |
| Gumbel$\tau$ 初始        | 1.0                                                    | 模块二    |
| Gumbel$\tau$ 退火        | $\tau \leftarrow \max(0.1, \tau \times 0.995)$ / epoch | 模块二    |
| BC 学习率                | 1e-3                                                   | 模块一    |
| 端到端学习率             | 1e-3（Reader 从头），1e-4（微调）                      | 模块二    |
| REINFORCE 学习率         | 1e-3                                                   | 模块三    |
| 梯度裁剪                 | max_norm=1.0                                           | 全部      |
| 每图 rollout 数 K        | 2~3（Gumbel-Softmax），5~10（REINFORCE）               | 模块二/三 |
| Label smoothing          | 0.1                                                    | 模块一    |
| Entropy bonus$\lambda$   | -0.01                                                  | 全部      |
| BC → Task warmup epochs | 20                                                     | C+A 联合  |

## 附录 B：文档依据索引


| 引用内容                                | 来源文档                            |
| --------------------------------------- | ----------------------------------- |
| 问题定义、动机、研究目标                | `proposal.md`                       |
| M0-M5 里程碑规划                        | `roadmap.md`                        |
| 开放问题（联合训练 vs 分阶段）          | `open-questions.md`                 |
| v0 协议：$G_0$、frontier、reveal_node   | `base-protocol-v0.md`               |
| PPA 数据集特征（无节点特征、7维边特征） | `ppa_analysis_report.txt`           |
| M3 尺子设计、反馈循环图                 | `reader-encoder-classifier-plan.md` |
| M4+ 动作扩展框架                        | `base-protocol-v0.md` 附录 B        |
| Policy-Learn 架构、端到端训练、对比分析 | `policy-learn-deep-read.md`         |

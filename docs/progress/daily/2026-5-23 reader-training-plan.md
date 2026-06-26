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
    1. 找到已揭示邻居: N(v) ∩ V_revealed
       (至少 1 个——正是这个邻居的 reveal 将 v 加入 frontier)
    2. 聚合邻居嵌入: agg_v = mean(H_t[u] for u in revealed_neighbors)
    3. 获取边特征:    edge_v = mean(edge_feat(u,v) for u in revealed_neighbors)← 改进：frontier 边特征已在 reveal 时暴露
    4. 拼接:          z_v = concat(agg_v, edge_v, deg_known_embed)
       deg_known:     v 在已揭示子图中的已知度（1~|N(v)∩V_rev|）
    5. MLP 打分:      score_v = Linear→ReLU→Linear→scalar

输出: scores ∈ R^{|frontier|}
```

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
| 视野      | 固定预算$B$                                                                             |

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

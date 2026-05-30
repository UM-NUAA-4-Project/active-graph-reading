# Policy Optimization for Learned Reader — 方法概述与路线

**日期**: 2026-05-27
**背景**: 为 M5 learned Reader 准备——理解三种可用的策略优化方法的原理、优缺点及在本项目中的使用顺序。
**目标**: 10-15 分钟报告，能画训练流程图，能回答"梯度/学习信号从哪里来"。

---

## 一、问题形式化

### 1.1 Reader 的决策问题

在 budget 约束下，Reader 逐步选择 frontier node 揭示，目标是最大化下游分类器的准确率：

```
State:  G_t = (V_revealed^(t), E_known^(t), X_E^(t), X_V^(t))
        即 GraphState 对象——已揭示子图 + 前沿边 + 边特征,通过前沿边知晓前沿节点的存在性

Action: a_t = v ∈ frontier_nodes
        从前沿节点中选择一个揭示

Transition: G_{t+1} = reveal_node(G_t, v)
        确定性转移——v 揭示 → 新前沿边出现、新边特征已知

Reward:  R = Accuracy(classifier(sub_G))  （episode-level, 整个 trajectory 结束后才有）
        或 R = ΔAccuracy(classifier(sub_G_{t+1})) - ΔAccuracy(classifier(sub_G_t))  （step-level）
```

### 1.2 为什么不能直接 backprop？

```python
# 不可微的决策过程：
frontier_scores = reader_network(G_t)          # [num_frontier_nodes] — 可微
next_node = argmax(frontier_scores)            # int — 不可微！
# ↑ 梯度在此断开——argmax 的导数为 0 或 undefined
```

这是一个**离散序列决策问题**。对于 PPA（平均 |V| ~ 300），在 5% budget 下约 15 步决策，每步从 ~10-50 个候选 frontier node 中选择。动作空间是离散的、每步不同的。

### 1.3 可微训练通用结构：Reader 前向管线（三种方法共享）

```
                        G_t = GraphState
                 (V_revealed, E_known, X_E, X_V)
                              │
                              ▼
               ┌──────────────────────────────┐
               │   frontier_nodes_encoder      │
               │   (GINE / MLP)                │
               │   输入: G_t                    │
               │   输出: frontier_embeds        │
               │   看见: frontier_nodes         │
               │         + 已揭示结点和边        │
               │         + frontier_edges_feat  │
               └──────────────┬───────────────┘
                              │
                              ▼
               frontier_embeds ∈ R^(N_frontier × d)
                              │
                              ▼
               ┌──────────────────────────────┐
               │   reader_mlp                  │
               │   (Linear → ReLU → Linear)    │
               │   输入: frontier_embeds        │
               │   输出: scores                │
               └──────────────┬───────────────┘
                              │
                              ▼
               scores ∈ R^(N_frontier)
                              │
                              │    ★ 决策点：如何从 scores 选出 action？
                              │    ★ 梯度如何从 reward/loss 回到 scores？
                              │
          ╔═══════════════════╪═══════════════════╗
          ║                   │                   ║
          ║                   │                   ║
          ▼                   ▼                   ▼
     Section 二           Section 三          Section 四
   Behavior Cloning    Gumbel-Softmax        REINFORCE
```

### 1.4 三方法全景对比图

```
                        G_t (GraphState)
                              │
                              ▼
               ┌──────────────────────────────┐
               │   frontier_nodes_encoder      │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   reader_mlp                  │
               └──────────────┬───────────────┘
                              │
                              ▼
                         scores
                              │
         ╔════════════════════╪════════════════════╗
         ║                    │                    ║
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────────┐      ┌──────────┐
   │ softmax   │        │ +gumbel_noise │      │ softmax   │
   │ → probs   │        │ /temperature │      │ → probs   │
   └────┬─────┘        └──────┬───────┘      └────┬─────┘
        │                     │                   │
        ▼                     ▼                   ▼
   ┌──────────┐        ┌──────────────┐      ┌──────────────┐
   │ CE loss   │        │ soft_sample   │      │ Categorical   │
   │ (scores,  │        │ @ embeds      │      │ .sample()     │
   │  a*_t)    │        │ → repr        │      │ → action      │
   └────┬─────┘        └──────┬───────┘      └──────┬───────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌──────────┐        ┌──────────────┐      ┌──────────────┐
   │ 梯度来源: │        │ 梯度来源:     │      │ 梯度来源:     │
   │ teacher   │        │ classifier   │      │ reward R     │
   │ label     │        │ loss → STE   │      │ × log_prob   │
   │ a*_t      │        │ → softmax    │      │              │
   └──────────┘        └──────────────┘      └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌──────────┐        ┌──────────────┐      ┌──────────────┐
   │ 可微      │        │ 近似可微      │      │ score func   │
   │ 低方差    │        │ 低方差/有偏   │      │ 无偏/高方差   │
   │ 预训练    │        │ fine-tune    │      │ backup       │
   └──────────┘        └──────────────┘      └──────────────┘
```

核心问题：**从 accuracy/reward 到 Reader 参数 θ 的梯度信号如何传递？**

下面三种方法给出不同答案。

---

## 二、Behavior Cloning（行为克隆）

### 2.1 核心思想

不直接优化 reward。先让 teacher 产生"最优轨迹"，再让 student (Reader) 模仿 teacher 的选择。

### 2.2 Teacher 轨迹如何生成

要求：

* 轨迹经过classifier后必须输出正确类别。
* 一张图需要top-k条尽量不同的途径学习，从而保持策略学习的多样性，以保证泛化能力。

评价不同的轨迹：输入classifier后，计算交叉熵 ，CrossEntropy越低越好。

困难：

1. 一幅图有300个节点，2200+条边，图很稠密，路径多；
2. 启发式生成轨迹在同一个点生产的轨迹是相同的；
3. 区分同一副图不同途径的不同程度？---维护|V|维bool数组异或，计算1的数量
4. 同一张图有k条优秀路径，如何计算损失？

#### 2.2.1 启发式生成轨迹

一幅图从n个seed开始生成轨迹，从中依靠交叉熵选出top-k条路径

T1 (gradient): 选使 classifier loss 下降最快的 frontier node；生成轨迹t11, t12, t13, ……, t1n

T2 (info-gain): 选使 classifier entropy 下降最多的 frontier node;生成轨迹t21, t22, t23, ……, t2n

T3 (structural): 选 degree/pagerank 最高的 frontier node; 生成轨迹t31, t32, t33, ……, t3n

```python
# M4 teacher: 对每张图，启发式地找出"最佳"reveal 序列
for graph in training_graphs:
    # T1 (gradient): 选使 classifier loss 下降最快的 frontier node
    # T2 (info-gain): 选使 classifier entropy 下降最多的 frontier node
    # T3 (structural): 选 degree/pagerank 最高的 frontier node
    teacher_trajectory = [a*_0,  a*_1, ..., a*_T]
```

### 2.3 Student 学什么 & Loss 怎么写

#### 2.3.1 通用打分流程

Student Reader : 输入 GraphState 的表征，输出每个 frontier node 的 score

frontier_embeds = frontier_nodes_encoder(state)        # (N_frontier, d)

注：为frontier_nodes打分时，看见：frontier_nodes + 已揭示结点和边 + 所有frontier_edges_feature

scores = reader_mlp(frontier_embeds).squeeze(-1)      # (N_frontier,)
probs = softmax(scores)                                # (N_frontier,)

梯度流：`loss → scores → reader_mlp → frontier_nodes_encoder`。全程可微。

#### 2.3.2 问题：计算CrossEntropy的时机

1. 揭示完整路径后，计算交叉熵：student、teacher轨迹 → 稀疏vector:(|V|,)；
2. 走一步，计算一次交叉熵：Teacher 选择 a*_t → vector:(N_frontier,),只有a*_t对应位置是1。
3. 是一条一条的算CrossEntropy，还是一张图的一起算CrossEntropy？

假设我们采取第一种方式，揭示完整路径后，计算交叉熵。从不同seed开始。

一条一条的算CrossEntropy：本轮揭示一条路径 → 算CrossEntropy

student轨迹 → 稀疏matrix:(K，|V|)；teacher轨迹 → 稀疏matrix:(K，|V|)；得到一个CrossEntropy Vector：(K,)

### 2.4 流程图

```plaintext
                         scores
                           │
                           ▼
                         softmax
                           │
                           ▼
                    probs ∈ R^(N_frontier)
                           │
                           │    Teacher 提供 a*_t (one-hot, 不参与计算图)
                           ▼
         ┌─────────────────────────────────────┐
         │  loss = CrossEntropy(scores, a*_t)  │
         │  ∇ = probs - one_hot(a*_t)          │
         └─────────────────┬───────────────────┘
                           │
                           │  ═══════ 梯度流 ═══════
                           ▼
         ┌─────────────────────────────────────────┐
         │  loss ──► scores ──► reader_mlp          │
         │                ──► frontier_nodes_encoder │
         │                                          │
         │  全程可微 · 无离散采样 · 标准分类 loss     │
         │  学习信号: teacher label a*_t             │
         └──────────────────────────────────────────┘
```

### 2.5 学习信号从哪里来？

**从 teacher 的 label `a*_t` 来。** 本质是监督学习——teacher 告诉 Reader "这步应该选哪个 frontier node"。

### 2.6 优缺点


| 优点                                                | 缺点                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 训练稳定——标准分类 loss，无 RL 方差问题           | 受 teacher 质量上限限制——student 不会超过 teacher                                  |
| 不需要 episode rollout——每个 (s_t, a*_t) 独立训练 | Teacher 本身是启发式的，不一定真正最优                                               |
| 代码简单——就是`nn.CrossEntropyLoss`               | 分布偏移 (distribution shift): student 的 s_t 来自自己的 policy，偏离 teacher 的 s_t |
| 适合作为预训练——先让 Reader 学到合理的行为        | 无法发现 teacher 不知道的新策略                                                      |

### 2.7 在本项目中的角色

**首选第一步。** M4 teacher 生成轨迹 → M5 BC 预训练 Reader。BC 本身不是终点，但提供了一个不错的初始化，后续可以用 Gumbel-Softmax 微调。

---

## 三、Gumbel-Softmax / Straight-Through Estimator

### 3.1 解决什么问题

让离散的 `argmax` / `sample` 操作近似可微，从而能端到端训练 Reader + Classifier。

### 3.2 Forward 和 Backward 分别发生什么

```python
frontier_embeds = frontier_nodes_encoder(state)        # (N_frontier, d)

注：为frontier_nodes打分时，看见：frontier_nodes + 已揭示结点和边 + 所有frontier_edges_feature
scores = reader_mlp(frontier_embeds).squeeze(-1)  # (N_frontier,)

# ---- Forward: 采样（离散） ----
gumbel_noise = -log(-log(uniform(0, 1) + eps) + eps)
soft_sample = softmax((scores + gumbel_noise) / temperature)  # (N_frontier,)
# temperature → 0: soft_sample → one-hot (退化为 argmax)
# temperature → ∞: soft_sample → uniform

# ---- Forward: "选中的" frontier node 的表征 ----
selected_repr = soft_sample @ frontier_embeds  # weighted sum ≈ 选中的 node

# ---- Backward: 梯度穿过 softmax ----
# 即使 forward 用了 argmax，backward 当作 softmax 处理：
# dL/d(scores) = STE(dL/d(soft_sample))
# 梯度从 selected_repr → soft_sample → scores → reader_mlp
```

**Straight-Through Estimator (STE)** 的关键变体：

```python
# Forward: argmax (真正的离散选择)
hard_choice = one_hot(argmax(scores))

# Backward: 用 softmax 的梯度替代
hard_choice_grad = softmax(scores / temperature)
hard_choice = hard_choice.detach() + hard_choice_grad - hard_choice_grad.detach()
# 数值上 = hard_choice, 但梯度 = softmax 的梯度
```

### 3.3 Temperature 的作用

```
high τ (e.g., 5.0):  梯度平滑，训练稳定，但近似差 → 收敛到 mediocre policy
low τ (e.g., 0.1):  接近离散，近似好，但梯度稀疏/噪声大 → 训练不稳定

标准做法: annealing — 从 τ=2.0 开始，逐步降到 τ=0.1
```

### 3.4 学习信号从哪里来？

**从 classifier 的 loss/accuracy 来，通过 STE 反向传播穿过离散选择。**

```
Classifier_Loss → classifier_head → GINEEncoder → selected_node_repr → STE(soft_sample) 
→ scores → reader_mlp → frontier_nodes_encoder
```

整个链路上的梯度最终到达 Reader 参数 θ。关键假设：**soft sample 近似 hard sample，梯度方向大致正确。**

### 3.5 流程图

```plaintext
                        scores
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  gumbel_noise                        │
         │  = -log(-log(uniform(0,1) + ε) + ε) │
         └─────────────────┬───────────────────┘
                           │
                           ▼
               raw = (scores + gumbel_noise) / temperature
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  soft_sample = softmax(raw)                 │
         │  ∈ R^(N_frontier),  ∑ = 1                   │
         │                                             │
         │  τ → 0: 退化为 one-hot(argmax)               │
         │  τ → ∞: 退化为 uniform                       │
         └─────────────────┬───────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌──────────────────┐    ┌──────────────────────┐
    │  Forward (hard):  │    │  Backward (STE):      │
    │  argmax(scores)   │    │  梯度穿过 softmax     │
    │  → one-hot 选择   │    │  ∂L/∂(scores) =       │
    │  真正离散动作      │    │  ∂L/∂(soft_sample)   │
    └────────┬─────────┘    │  × ∂(soft_sample)/∂scores │
             │              └──────────────────────┘
             │                         ▲
             │                         │
             ▼                         │
   selected_node =                    │
   one_hot(argmax(scores))            │
             │                         │
             ▼                         │
   ┌─────────────────────┐            │
   │  selected_repr =     │            │
   │  hard_choice @       │            │
   │  frontier_embeds     │            │
   └─────────┬───────────┘            │
             │                         │
             ▼                         │
   ┌─────────────────────┐            │
   │  GINEEncoder +       │            │
   │  ClassifierHead      │            │
   │  → logits            │            │
   └─────────┬───────────┘            │
             │                         │
             ▼                         │
   ┌─────────────────────┐            │
   │  Classifier_Loss =   │            │
   │  CrossEntropy(       │            │
   │    logits, label)    │────────────┘
   └─────────────────────┘   STE 梯度回传
```

### 3.6 优缺点


| 优点                                      | 缺点                                                            |
| ----------------------------------------- | --------------------------------------------------------------- |
| 低方差——整个 chain 是可微的（近似）     | 梯度有偏——softmax 梯度 ≠ argmax 真实梯度                     |
| 端到端——Reader 和 Classifier 可联合微调 | Temperature schedule 敏感——调参困难                           |
| 适合 BC 预训练后的 fine-tune              | 在长 trajectory（>10 步）上梯度可能弥散                         |
| 不需要 reward 函数设计                    | 假设 soft selection 近似 hard selection，在稀疏图阶段可能不成立 |

### 3.7 在本项目中的角色

**BC 预训练之后的 fine-tune 首选。** 用 BC 训练的 Reader 作初始化，Gumbel-Softmax 端到端微调。因为 BC 已经让 Reader 输出合理的 score 分布，Gumbel-Softmax 只需要在局部优化——降低了 temperature schedule 的敏感性。

---

## 四、REINFORCE / Policy Gradient

### 4.1 核心思想

不尝试让离散选择可微。直接在 policy gradient 框架下，用采样轨迹的 reward 来估计梯度。

### 4.2 基本公式

$∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τ) ]$

$π_θ(a_t | s_t) = softmax(reader_{mlp}(frontier-nodes-encoder(s_t)))$

$R(τ) = accuracy(classifier(subgraph_{final}))$(episode reward)

直观解释：

- 如果选了 a_t 后最终 reward 高 → 增大 log π(a_t|s_t) → 以后更可能选 a_t
- 如果选了 a_t 后最终 reward 低 → 减小 log π(a_t|s_t) → 以后不太选 a_t

### 4.3 为什么可以处理不可微的离散采样

**不需要 backprop 穿过采样。** 梯度来自 `log π_θ` 的 score function，乘以 reward 作为权重：

```python
# Pseudocode for one training step:
trajectory = sample_trajectory(reader)          # 离散采样，无梯度
reward = evaluate(classifier, trajectory)        # scalar，无梯度 → Reader

# Policy gradient:
for t, (s_t, a_t) in enumerate(trajectory):
    log_prob = log(softmax(reader(s_t))[a_t])    # 可微！
    loss = -log_prob * reward                     # reward 是常数（对 Reader 参数）
    loss.backward()                                # 梯度 → reader 参数
```

梯度方向：如果 reward 为正，`∇(-log_prob * reward)` 指向增大 `log_prob` 的方向 → 以后更可能选 a_t。

### 4.4 Baseline / Advantage 为什么能降方差

```python
# 原始 REINFORCE: 梯度 = log_prob'(a_t) * R(τ)
# 问题: R(τ) 可能总是正的（如 accuracy 总在 [0.1, 0.5]），所有动作都被鼓励

# 引入 baseline:
advantage = R(τ) - baseline(s_t)    # 相对评价
# baseline 可以是:
#   - 移动平均 reward
#   - 值函数 V(s_t) 预测的期望 reward
#   - 同一图的多个 trajectory 的平均 reward

# 现在：高于 baseline 的动作被鼓励，低于的被抑制
loss = -log_prob * advantage.detach()
```

直观：不是问"这个 trajectory 好不好"，而是问"这个 trajectory 比预期好多少"。

### 4.5 流程图

```plaintext
                          scores
                           │
                           ▼
                         softmax
                           │
                           ▼
                    probs ∈ R^(N_frontier)
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Categorical(probs).sample()         │
         │  → action a_t ∈ {0, ..., N_frontier-1}│
         │                                      │
         │  ⚠ 此处梯度断开——sample() 不可微     │
         └─────────────────┬───────────────────┘
                           │
                           │  no gradient
                           ▼
         ┌─────────────────────────────────────┐
         │  state_{t+1} = reveal_node(state_t, a_t) │
         │  ... repeat T steps ...              │
         │  → final subgraph                    │
         └─────────────────┬───────────────────┘
                           │
                           │  no gradient
                           ▼
         ┌─────────────────────────────────────┐
         │  classifier(subgraph) → accuracy     │
         │  reward R = accuracy (scalar)        │
         │                                      │
         │  ⚠ R 对 Reader 参数 θ 不可微         │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  advantage = R - baseline(s_t)       │
         │  baseline: 移动平均 / V(s_t) / 同图  │
         │            多 trajectory 平均        │
         └─────────────────┬───────────────────┘
                           │
                           │  advantage.detach() — 常数（对 θ）
                           ▼
         ┌──────────────────────────────────────────────┐
         │  loss = -log(softmax(scores)[a_t]) * advantage │
         │                                               │
         │  ═══════ 梯度流 ═══════                        │
         │  loss ──► log_prob[a_t] ──► scores            │
         │            ──► reader_mlp                      │
         │            ──► frontier_nodes_encoder          │
         │                                               │
         │  score function 梯度:                          │
         │  ∇_θ J = E[ ∇_θ log π_θ(a_t|s_t) · R(τ) ]    │
         │  无偏 · 高方差 · 不需要可微的 reveal 过程       │
         └──────────────────────────────────────────────┘
```

### 4.6 优缺点


| 优点                                   | 缺点                                                                 |
| -------------------------------------- | -------------------------------------------------------------------- |
| 理论上无偏——不需要可微的 reveal 过程 | 方差大——尤其当 trajectory 长、reward 稀疏时                        |
| 不需要 teacher——Reader 自己探索      | 训练慢——需要多次 rollout 采样                                      |
| 可以学习 teacher 不知道的策略          | 需要 baseline/advantage 降方差，增加复杂度                           |
| 适用于任何离散 action space            | Credit assignment 困难——第 1 步的 action 影响最终 reward，但信号弱 |

### 4.7 在本项目中的角色

**Backup 方案，不作为第一选择。** 如果 BC + Gumbel-Softmax 的效果不理想（例如 STE 梯度有偏导致不收敛），则回退到 REINFORCE。主要原因：

1. PPA 的 trajectory 长度（~10 步）虽然不算长，但 reward 只有最终 accuracy 一个标量，credit assignment 困难
2. 20 seeds/graph × multi-graph batch 可以部分缓解方差——但相比 BC 的稳定性还是差很多
3. REINFORCE 调参（学习率、baseline 设计、rollout 次数）成本高

---

## 五、在本项目中的使用顺序

### 5.1 推荐路线

```
M3 Benchmark Gate
       │
       │  (PASS: policy choice 有空间)
       ▼
M4 Teacher Trajectories
  - T1 gradient / T2 info-gain / T3 structural / T4 hybrid
  - 对 training set 生成 (s_t, a*_t) 轨迹
       │
       ▼
M5 Behavior Cloning (BC)  ← 第一步：稳定预训练
  - loss = CrossEntropy(reader(s_t), a*_t)
  - 快速收敛到 teacher-level performance
  - 不涉及离散采样，训练稳定
       │
       ├── 效果够好？ → Done
       │
       ▼ (不够好)
M5 Gumbel-Softmax Fine-tune  ← 第二步：端到端微调
  - 加载 BC 权重作初始化
  - 小 temperature + annealing
  - 端到端优化 Reader + Classifier（frozen GIN? or fine-tune?）
       │
       ├── 有效？ → Done
       │
       ▼ (STE 梯度有偏 / 不收敛)
M5 REINFORCE  ← Backup
  - 多 trajectory rollout + baseline
  - 无偏梯度估计
  - 训练成本高
```

### 5.2 何时不需要进入这些复杂训练？

在以下情况下不推进 M5：

1. **M3 gate KILL** — BFS 在 ≤10% budget 已接近 full-graph accuracy → 说明 policy choice 不重要，learned Reader 没意义
2. **M3 结果显示三种启发式策略差异不显著** — currently observed at small sample size — 需要扩大样本确认，如果仍然不显著 → 策略空间太小，learned Reader 难以超越
3. **M4 teacher trajectory 质量差** — 所有 teacher 的 accuracy 上限都不比 random 高 → 没有好 teacher，BC 无意义

### 5.3 关键前提

**进入 M5 的前提是 M3 确认 policy choice 有显著空间。** 当前 quick benchmark（37 图）显示 random/BFS/max-frontier-degree 在小 budget 下差异不显著。需要扩大样本（k=5-10 per class）再判断。

---

## 六、三种方法对比总结


|                  | Behavior Cloning   | Gumbel-Softmax           | REINFORCE                |
| ---------------- | ------------------ | ------------------------ | ------------------------ |
| 需不需要 teacher | 需要（M4）         | 不需要（但可从 BC init） | 不需要                   |
| 梯度信号来源     | teacher label a*_t | STE 近似梯度             | reward-weighted log_prob |
| 是否离散可微     | 是（标准 CE loss） | 近似可微                 | 是（score function）     |
| 方差             | 低                 | 低                       | **高**                   |
| 梯度是否有偏     | N/A                | **有偏**                 | 无偏                     |
| 训练稳定性       | 高                 | 中（需调 temperature）   | **低**（需调 baseline）  |
| 能否超越 teacher | 否（上限受限）     | 是                       | 是                       |
| 调参复杂度       | 低                 | 中                       | **高**                   |
| 本项目角色       | **第一步：预训练** | **第二步：fine-tune**    | **Backup**               |

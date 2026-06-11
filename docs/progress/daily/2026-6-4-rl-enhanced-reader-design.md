# RL-Enhanced Reader 架构设计方案

> 日期：2026-06-04
> 状态：设计文档，代码尚未实现
> 前提：Reader BC 架构已完成（FrontierGraphBuilder + FrontierNodesEncoder + ReaderMLP），M4 teacher 轨迹已生成

---

## Context

当前 reader（`LearnableReader` + `FrontierNodesEncoder` + `ReaderMLP`）只完成了 BC 的准备工作：架构能跑通，M4 teacher 轨迹已生成，但存在三个结构性缺失：

1. **ReaderMLP 逐节点独立打分** — 无法建模 frontier 节点间的竞争/互补关系
2. **无图级上下文** — Reader 不知道"已经看了多少"、"还剩多少预算"
3. **无 RL 训练模式** — 只有 `select()` (argmax)，没有可微采样、没有 value estimation、没有端到端 loss 回路

目标：借鉴 RL 方法重新设计 reader 架构，使其在 BC 预训练后可通过"模式开关"进入端到端优化，不再依赖 teacher 轨迹，直接 read → classify → loss → backward。

---

## 一、RL ↔ Reader 概念映射

### 1.1 系统映射表


| RL 概念               | Reader 对应                                | 状态         | 缺口                             |
| --------------------- | ------------------------------------------ | ------------ | -------------------------------- |
| State s_t             | `GraphState` G_t                           | 已实现       | —                               |
| State encoder φ(s)   | `FrontierNodesEncoder(G_t)`                | 已实现       | 只输出 frontier 嵌入，缺图级汇总 |
| Action space A(s)     | `frontier_nodes` (~10-50, 动态)            | 已实现       | —                               |
| Policy π(a\|s)       | `softmax(ReaderMLP(φ(s)_a))`              | 已实现       | 逐节点独立，缺上下文条件化       |
| Transition P(s'\|s,a) | `reveal_node(s, a)` 确定性                 | 已实现       | —                               |
| Reward r(s,a)         | Δclassifier_confidence / episode accuracy | **未形式化** | 核心缺口                         |
| Value V(s)            | 期望 classifier accuracy from s            | **未实现**   | 核心缺口                         |
| Advantage A(s,a)      | Q(s,a) − V(s)                             | **未实现**   | 依赖 V(s)                        |
| Trajectory τ         | reveal_order                               | M4 已生成    | —                               |
| 探索机制              | ε-greedy / Thompson / UCB                 | **无**     | 纯 argmax                        |

### 1.2 梯度流映射：RL vs BC vs 端到端

```
BC 模式的梯度流:
  CE_loss → scores → ReaderMLP → frontier_embeds → FrontierNodesEncoder
  梯度来源: teacher label a*_t
  断点: 无 — 全程可微（标准监督学习）

Gumbel-Softmax 模式的梯度流:
  ClassifierLoss → logits → GINEEncoder → subgraph_repr → [STE] → scores → ReaderMLP → Encoder
  梯度来源: classifier loss 通过 STE 跨越 argmax
  断点: argmax（用 STE 近似桥接）
  ← 类比 RL 的: reward → policy gradient, 但这里 reward 被 classifier loss 替代

REINFORCE 模式的梯度流:
  reward × ∇log π(a_t|s_t) → scores → ReaderMLP → Encoder
  梯度来源: episode reward (accuracy) 加权 log_prob
  断点: 无 — score function 不经过 action 采样
  ← 标准 policy gradient
```

### 1.3 问题重述：Online Graph Exploration as Bandit with Side Observations

每步决策的本质是 **带有图结构侧信息的多臂老虎机**：

- **臂 (arms)**：frontier nodes，每步 ~10-50 个
- **侧信息 (side info)**：frontier edge features + 已揭示子图拓扑 → GNN 编码为 arm embeddings
- **臂非独立**：揭示 node A 会改变 node B 的状态（frontier degree 变化、新边特征暴露）→ 需要在 arm 嵌入上建模交互
- **探索-利用权衡**：预算有限 → 必须在"深入已知方向"和"开辟新方向"之间动态平衡
- **自然奖励信号**：信息增益（鼓励探索）+ 分类是否正确（鼓励利用）

与标准 multi-armed bandit 的关键区别：

- 每次 pull 后，臂集合发生变化（新节点加入 frontier，被选节点移除）
- 侧信息是图结构的（边特征），不是简单的特征向量
- 有明确的 horizon（budget 耗尽）

---

## 二、架构重设计

### 2.1 数据流对比

**当前架构（BC-only）：**

```
GraphState → FrontierGraphBuilder → extended subgraph
                                         ↓
                                   FrontierNodesEncoder (GNN)
                                         ↓
                                   frontier_embeds (N_f, d)
                                         ↓
                                   ReaderMLP (per-node)
                                         ↓
                                   scores → softmax → argmax
```

**增强架构（RL-ready）：**

```
GraphState → FrontierGraphBuilder → extended subgraph
                                         ↓
                                   FrontierNodesEncoder (GNN)
                                         ↓
                                   h_all (N_total, d)
                                         ↓
                          ┌──────────────┼──────────────┐
                          ↓              ↓              ↓
                   frontier_embeds  graph_summary  (revealed 部分)
                    = h_all[N_r:]   = mean(h_all)
                          ↓              ↓
                   ┌──────┴──────┐   value_head
                   ↓             ↓        ↓
            self_attn(Q,K,V)  budget_vec  V(s)
                   ↓             ↓
              attended (N_f,d)  (from budget_encoder)
                   ↓             ↓
              ┌────┴─────┬──────┘
              ↓          ↓
        [frontier || attended || graph_summary || budget_vec]
              ↓
        ReaderMLP (context-aware) → scores → softmax → probs
              ↓
        ┌─ mode='bc':         argmax → action
        ├─ mode='gumbel-softmax': Gumbel-Softmax STE → action
        └─ mode='reinforce':     Categorical sample + log_prob → action
```

### 2.2 新增组件设计意图

#### A. graph_summary — 图级状态编码

- **来源**：`FrontierNodesEncoder` GNN 输出 `h_all`（所有已揭示+前沿节点嵌入）做 mean pooling
- **维度**：`(hidden_dim,)` — 一个固定长度的稠密向量
- **RL 等价**：state representation φ(s) — 从部分观测图到低维摘要
- **为什么需要**：ReaderMLP 目前只看到单个 frontier node 的嵌入，不知道"整个已揭示子图长什么样"——密度如何、规模多大、是否已经覆盖了图的主要部分
- **给谁用**：
  - ReaderMLP 的上下文输入（感知全局进度）
  - value_head 的输入（预测期望分类准确率）

#### B. budget_encoder — 剩余预算感知

- **来源**：`budget_remaining / total_budget`（标量）→ 小型 MLP → `budget_vec`
- **维度**：`(hidden_dim // 4,)`
- **RL 等价**：time-to-go signal in finite-horizon MDP — 策略在 horizon 不同阶段应有不同行为
- **为什么需要**：预算充裕时（还剩 80%）应倾向探索不确定区域；预算紧张时（只剩 5%）应倾向利用已有信息选出最有利于分类的节点

#### C. frontier self-attention — 动作空间交互建模

- **结构**：标准 MultiheadAttention，Q=K=V=frontier_embeds
- **输入**：`frontier_embeds (N_f, d)` → **输出**：`attended (N_f, d)`
- **RL 等价**：action representation learning — 不是独立给每个动作打分，而是建模动作间的关系
- **解决什么问题**：
  - **竞争**：两个 frontier node 连接到同一个未揭示区域 → attention 应降低至少一个的分数（避免冗余揭示）
  - **互补**：两个 frontier node 分别连接不同区域 → attention 保持各自分数（都值得探索）
  - **无交互的独立打分无法捕捉这些关系**

#### D. value_head — 状态价值估计

- **结构**：`graph_summary → MLP → V(s)`（标量）
- **RL 等价**：critic in actor-critic — 估计从当前状态出发的期望 return
- **为什么需要**：
  - REINFORCE 模式：作为 baseline 降方差（`advantage = reward - V(s)`）
  - Gumbel-Softmax 模式：可作为辅助 loss 监督信号
  - BC 模式：可作为额外特征辅助打分（可选项）
- **训练目标**：预测"从当前已揭示子图出发，最终分类是否正确"（二元分类或回归 accuracy）

#### E. Context-aware ReaderMLP

- **输入**：`[frontier_embeds_i || attended_i || graph_summary || budget_vec]` — 拼接四部分
- **与当前 ReaderMLP 的区别**：当前只看 `frontier_embeds_i` 一个向量，增强版看四个信息源
- **RL 等价**：policy head with full state + action context — 类似 dueling DQN 中结合 state value 和 action advantage

### 2.3 BC 兼容性

增强架构 **向后兼容 BC 训练**：

- `graph_summary`、`budget_vec`、`attended` 在 BC 模式下同样有效——提供更丰富的上下文给 ReaderMLP
- `value_head` 在 BC 模式下可选择性使用（辅助 loss）或不使用
- `forward()` 返回 `(scores, probs, frontier_nodes, state_value)` — BC 训练忽略 `state_value` 即可
- `select()` 在未指定 `explore=True` 时行为不变（argmax）

---

## 三、模式切换机制

### 3.1 三模式定义


| 模式             | 用途               | 数据来源        | 梯度来源                         | select() 行为                         |
| ---------------- | ------------------ | --------------- | -------------------------------- | ------------------------------------- |
| `bc`             | 预训练             | M4 teacher 轨迹 | CE(scores, a*_t)                 | argmax (确定性)                       |
| `gumbel-softmax` | 端到端微调         | 无（自主探索）  | classifier loss → STE → scores | Gumbel-Softmax reparameterized sample |
| `reinforce`      | 端到端微调（兜底） | 无（自主探索）  | reward × ∇log π               | Categorical sample + log_prob         |

### 3.2 切换流程

```
1. 创建 reader（默认 mode='bc'）
2. 加载 M4 teacher 轨迹，训练 BC
3. 保存 BC checkpoint
4. reader.mode = 'gumbel-softmax'  ← "开关"
5. 加载 BC checkpoint
6. 设置 temperature schedule（2.0 → 0.1）
7. 端到端训练：read → classify → loss → backward
   （不再使用 teacher 轨迹）
8. 若 GS 不收敛 → reader.mode = 'reinforce' → 回退
```

### 3.3 关键设计决策：select() 的训练/推理双模式

- **推理时**（`select(state)`）：始终 argmax，确定性。模式间无差异。
- **训练时**（`select(state, explore=True)`）：行为由 `mode` 决定：
  - `bc`：argmax → 与 teacher label 比较 CE
  - `gumbel-softmax`：Gumbel-Softmax 采样 → STE 梯度桥接 argmax
  - `reinforce`：Categorical 采样 → 记录 log_prob 用于 policy gradient

---

## 四、RL 训练循环设计（概念级）

### 4.1 Gumbel-Softmax 模式

核心思想：让 argmax 近似可微，classifier loss 的梯度一路回传到 reader 参数。

```
对一个训练样本 (graph, label)：
  1. state = GraphState(graph, seed)
  2. for t in 1..budget:
       a. 构建扩展子图 → Encoder → frontier_embeds + graph_summary
       b. self-attn → attended
       c. context-aware scoring → probs
       d. Gumbel-Softmax STE → 选出一个 frontier node（可微）
       e. reveal_node(state, chosen)
  3. 构建最终子图 → classifier → logits
  4. loss = CE(logits, label) + λ_v * MSE(V(s₀), is_correct)
  5. loss.backward()  ← 梯度穿过 STE → ReaderMLP → Encoder
```

**关键点**：

- 不需要 teacher — reader 自己探索
- classifier loss 直接作为 reader 的监督信号
- STE 假设 soft selection ≈ hard selection（temperature 低时近似好）
- Classifier 可冻结也可选择性微调（Layer 0-1 冻结，Layer 2-4 + head 可训练）

### 4.2 REINFORCE 模式

核心思想：不尝试让离散选择可微，直接在 policy gradient 框架下用采样轨迹的 reward 估计梯度。

```
对一个训练样本 (graph, label)：
  1. state = GraphState(graph, seed)
  2. for t in 1..budget:
       a. 构建扩展子图 → Encoder → scores → probs
       b. Categorical(probs).sample() → action + log_prob
       c. 记录 (log_prob, V(s_t))
       d. reveal_node(state, action)
  3. 构建最终子图 → classifier → accuracy
  4. reward = accuracy (0 或 1)
  5. advantage = reward - V(s₀)  ← value_head 作 baseline
  6. policy_loss = -Σ log_prob × advantage
  7. value_loss = MSE(V(s₀), reward)
  8. (policy_loss + λ_v × value_loss).backward()
```

**关键点**：

- 梯度不经过采样 — `log_prob` 本身可微
- reward 作为常数权重乘在 log_prob 上
- value_head 提供 baseline 降方差
- 可以用 step-level reward shaping 替代 episode reward：
  - `r_t = confidence_after_reveal - confidence_before_reveal`
  - 提供更密集的学习信号

---

## 五、与现有代码的衔接

### 5.1 保持不变的模块


| 模块                                        | 说明                 |
| ------------------------------------------- | -------------------- |
| `FrontierGraphBuilder`                      | 扩展子图构建逻辑不变 |
| `GraphState` / `reveal_node` / `RevealLoop` | 协议引擎不变         |
| `SubgraphBuilder`                           | 最终子图构建不变     |
| `GraphClassifier` / `GINEEncoder`           | 分类器不变           |

### 5.2 需要修改的模块


| 文件                         | 改动程度 | 说明                                                                                              |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| `reader/frontier_encoder.py` | 小       | `forward()` 新增返回 `graph_summary`（mean pool over h_all）                                      |
| `reader/learnable_reader.py` | 大       | 新增 budget_encoder、frontier_attn、value_head、context-aware ReaderMLP、mode 属性、select 双模式 |
| `reader/rl_trainer.py`       | **新建** | Gumbel-Softmax 和 REINFORCE 训练循环封装                                                          |
| `reader/__init__.py`         | 小       | 导出 RLTrainer                                                                                    |

### 5.3 参数量变化

- **当前 ReaderMLP 输入**：`frontier_embeds[i]` (d,)
- **增强后 ReaderMLP 输入**：`[frontier_embeds || attended || graph_summary || budget_vec]` (d + d + d + d/4 ≈ 3.25d)
- 第一层 Linear 增加约 `(2.25 × d²)` 参数（d=128 时约 +37K）
- 总体 ~200K → ~237K，增幅约 18%

---

## 六、风险与缓解


| 风险                                        | 影响                    | 缓解措施                                                                          |
| ------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| STE 梯度有偏导致 GS 不收敛                  | GS 模式失败             | Temperature annealing + 从 BC checkpoint 初始化（非随机初始化）                   |
| Value head 预测不准                         | REINFORCE baseline 无效 | 先用 BC 数据预训练 value head（已知最终 accuracy 作 label）                       |
| Self-attention 在 frontier 很小（<5）时退化 | 额外计算无收益          | 设置 min_frontier 阈值，低于阈值跳过 attention                                    |
| 增强后参数增大导致小样本过拟合              | BC 性能下降             | 保持 dropout + BC 阶段冻结 value_head                                             |
| 端到端训练显存超限                          | 无法训练                | GS 模式每步 reveal 不保留中间图（detach），REINFORCE 模式用 gradient accumulation |

---

## 七、推荐实施顺序

1. **Phase 1**：修改 `frontier_encoder.py`（新增 graph_summary 输出）→ 验证 shape
2. **Phase 2**：修改 `learnable_reader.py`（新增 4 组件 + mode 开关）→ 验证 BC 兼容性
3. **Phase 3**：新建 `rl_trainer.py`（训练循环）→ 验证 GS/REINFORCE 梯度流
4. **Phase 4**：端到端训练（BC 预训练 → GS 微调 → REINFORCE 兜底）

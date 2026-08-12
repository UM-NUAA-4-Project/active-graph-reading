# Phase 2: Transformer Reader with Belief State

## 1. 问题定义

Budgeted graph reading: 给定图 G，预算 B（可揭示节点数，不含 seed），
从 seed 出发逐步揭示节点，预算耗尽后基于已揭示子图分类。

- **State**: `G_t = (V_revealed, E_known, frontier_nodes, frontier_edges)`
- **Action**: `a_t = reveal_node(v)`, v ∈ frontier_nodes
- **Transition**: 确定性，reveal_node 后 v 加入 V_revealed，新邻居加入 frontier
- **Reward**: 分类正确性（episode-level）

### v1 协议关键规则

| 规则               | 说明                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| Budget 不含 seed   | 揭示 B 个节点，seed 不计入                                                                                 |
| Seed 确定性        | `(graph_idx × 31 + seed_i × 17 + 7) % n_nodes`                                                         |
| Frontier edge 可见 | 一端已揭示即可查询边特征，**不消耗 budget**                                                          |
| 可见 ≠ 揭示       | `state.get_frontier_edge_feat(u, v)` 免费，区分候选的唯一特征通道                                        |
| Budget 检查        | 每步 reveal 后检查 `get_budget_used() >= _budget_limit`，最后一步**包含在轨迹中**                  |
| 自环               | 全图加载后 `dgl.add_self_loop(g)`，SubgraphBuilder/FrontierGraphBuilder 内部已加，**禁止外层重复** |

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                       LearnableReader                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Belief Tokens h_t (m×d)                │   │
│  │  每个 token 可分化出不同专长（密度、边类型、边界、分类）    │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│           ┌─────────────┤                                        │
│           │             │                                        │
│           ▼             ▼                                        │
│  ┌──────────────┐ ┌──────────────────┐                          │
│  │ Frontier     │ │ Classification   │                          │
│  │ Pointer      │ │ Head             │                          │
│  │ SelfAttn →   │ │ flatten → MLP    │──→ logits_t (K,)         │
│  │ CrossAttn    │ └──────────────────┘                          │
│  │ (belief KV)  │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ score_proj   │──→ scores (N_f,) → argmax → v                │
│  └──────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ reveal_node  │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ FrontierGraphBuilder │ 构建全子图 S_{t+1}                    │
│  │ (复用, 不重复构建)    │                                       │
│  └──────┬───────────────┘                                       │
│         │                                                        │
│         ├──────────────────────────┐                             │
│         │ (存 cache_subgraphs,     │ (送入 NewInfoEncoder)      │
│         │  下一步 forward_states   │                             │
│         │  直接使用)               ▼                             │
│         │                 ┌──────────────────┐                   │
│         │                 │ NewInfoEncoder   │                   │
│         │                 │ 2层 GIN + 残差池化│                   │
│         │                 │ (局部+全局)       │                   │
│         │                 └──────┬───────────┘                   │
│         │                        │                               │
│         │                        ▼                               │
│         │                 ┌──────────────┐                       │
│         │                 │ BeliefUpdate │                       │
│         └────────────────→│ SelfAttn →   │                       │
│                           │ CrossAttn →  │──→ h_{t+1}           │
│                           │ FFN          │                       │
│                           └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### 梯度隔离

| 路径                                               | 参数       | 目标            |
| -------------------------------------------------- | ---------- | --------------- |
| FrontierNodesEncoder + FrontierPointer             | θ_scoring | 选节点          |
| NewInfoEncoder + BeliefUpdate + ClassificationHead | θ_belief  | 分类 + 信念更新 |

两路参数独立，无梯度冲突。`forward_states` 同时返回 `(scores, probs, frontier_nodes, logits)`，
训练时 RL 项更新 θ_scoring，CE 项更新 θ_belief。

---

## 3. 模块详解

### 3.1 FrontierGraphBuilder

**位置**: `active_graph_reader.gnn.frontier.graph_builder`

将 GraphState 转为扩展子图（供 FrontierNodesEncoder 编码）：

| 属性         | 内容                                                   |
| ------------ | ------------------------------------------------------ |
| 节点         | V_revealed ∪ frontier_nodes                           |
| 边           | revealed↔revealed（双向）+ revealed→frontier（单向） |
| 自环         | 内部已加                                               |
| 边特征       | 原图边特征（frontier edge 已可见）                     |
| 节点类型标记 | revealed=0, frontier=1（通过 `node_type_emb` 编码）  |

### 3.2 FrontierNodesEncoder

**位置**: `active_graph_reader.gnn.frontier.encoder`

GNN 编码扩展子图，输出 frontier 节点的嵌入。2 层 GINEConv，消息通过边特征传播。

输出: `(N_f, hidden_dim)`，每个 frontier 节点一个嵌入。

### 3.3 FrontierPointer

**位置**: `belief/frontier_pointer.py`

```
frontier_embeds (N_f, d_in) → Linear(d_in→d_model) → SelfAttn → CrossAttn(belief KV) → scores (N_f,)
```

**d_in 构成**: `hidden_dim + 1 (budget_ratio_remaining) + 3 (structural features: degree, centrality, etc.)`

**两步计算**:

1. **SelfAttn**: frontier 节点互相对比——"谁在局部结构中更突出？"
2. **CrossAttn**: Q=frontier, KV=belief tokens——"每个候选节点对信念的各个问题能回答多少？"

**设计选择**: 只保留 SelfAttn，无 FFN。frontier 数量动态变化（5~100+），轻量 SelfAttn 足够做结构对比。

### 3.4 BeliefState

**位置**: `belief/belief_state.py`

由三部分组成：

| 子模块                 | 功能                                                     | 参数量 |
| ---------------------- | -------------------------------------------------------- | ------ |
| `tokens`             | (m, d) 可学习初始信念                                    | m × d |
| `BeliefUpdate`       | 1 层 Transformer: SelfAttn → CrossAttn(new_info) → FFN | ~100K  |
| `ClassificationHead` | Flatten(m·d) → MLP → K 类 logits                      | ~70K   |

#### BeliefUpdate 三步计算

```
Step A: SelfAttn(h_old) → 信念内部对齐——"我们已经知道了什么"
Step B: CrossAttn(h_mid, new_info) → 吸收新信息——"新走廊告诉了什么"
Step C: FFN(h_absorb) → 非线性整合
```

**残差链**（三条直通路径并行存在）:

```
h_old ─────────────────────────────────────→ h_new     ← 纯残差，信息不丢失
h_old → SelfAttn ──────────────────────────→ h_new     ← 内部整合增量
h_old → SelfAttn → CrossAttn(new_info) ────→ h_new     ← 外部吸收增量
```

#### 分类头

```
Flatten(h_t): (m, d) → (m·d,)
MLP: Linear(m·d → d) → GELU → LayerNorm → Dropout → Linear(d → K)
```

**为什么 Flatten 而非 Mean Pool？** Mean Pool 压缩 token 分化后的组合信号，
Flatten 让分类头学会"两个特定 token 的特定判断同时成立时才输出某个 class"。

### 3.5 NewInfoEncoder

**位置**: `belief/new_info_encoder.py`

揭示节点 v 后，编码完整 frontier 子图 → (1, d) 供 belief 吸收。

```
h₁ = GINELayer₁(g, h, edge_feats)
pool₁ = concat(h₁[v_idx], mean(h₁[frontier_mask]))  ← (2d,)
residual = Linear→GELU→Dropout(pool₁)                ← (d,)

h₂ = GINELayer₂(g, h₁, edge_feats)
pool₂ = concat(h₂[v_idx], mean(h₂[frontier_mask]))  ← (2d,)
output = Linear→GELU→Dropout→Linear(pool₂)           ← (d,)

new_info = output + residual                          ← (d,)
```

**为什么用全子图而非 star？** 全子图的 2-hop message-passing 同时提供局部（v 的位置和邻居）
和全局（所有已揭示节点之间的连接模式）信息。

**为什么残差池化？** 第 1 层池化结果跳过第 2 层 GIN 直通输出，浅层局部模式不丢失。

---

## 4. 完整计算循环

### 4.1 单步伪代码

```
h = init_belief()                              # (1, m, d)
subgraph = None                                # 第一步内部构建

for t in 0 .. B-1:
    if subgraph is None:
        subgraph = FrontierGraphBuilder(G_t)

    f_emb = FrontierEncoder(subgraph)          # (N_f, d_in)
    scores = FrontierPointer(f_emb, KV=h)      # SelfAttn → CrossAttn
    v = argmax(softmax(scores))                # 选出要揭示的节点

    reveal_node(G_t, v)                        # 推开门
    subgraph = FrontierGraphBuilder(G_{t+1})   # 构建新子图
    new_info = NewInfoEncoder(subgraph, v)     # 2层GIN → (1, d)
    h = BeliefUpdate(h, new_info)              # 信念吸收

logits = ClassificationHead(h)                 # 预算耗尽后，用最终信念分类 → (K,)
```

### 4.2 锁步循环（多图并行）

```
创建 N 个状态，每个状态配备 belief
active = [True] * N
max_budget = max(s._budget_limit for s in all_states)

for step in range(max_budget):
    active_idx = [j for j where active[j] and frontier not empty]

    # 收集 active 状态的 beliefs 和缓存子图
    active_beliefs = stack([all_beliefs[j] for j in active_idx])
    active_subgraphs = [cache_subgraphs[j] for j in active_idx]  # 复用

    # 一次 batch forward
    results = reader.forward_states(states, beliefs=active_beliefs, subgraphs=active_subgraphs)

    for each state:
        sample action, collect log_prob
        batch_states.append(state)
        batch_nodes.append(chosen)

    # batch reveal + encode + update
    new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(reader, batch_states, batch_nodes, batch_beliefs)

    # 更新缓存
    for each:
        all_beliefs[j] = new_beliefs[k]
        cache_subgraphs[j] = new_subgraphs[k]

    # budget 检查（必须）
    for j in active_idx:
        if active[j] and state.get_budget_used() >= state._budget_limit:
            active[j] = False
```

---

## 5. 训练

### 5.1 Stage 1: BC (Behavior Cloning)

**数据**: M4 teacher trajectories，覆盖约 20% 训练集

```
Loss = α · L_BC + (1-α) · CE(y, reader.classify(belief_final))

L_BC = -Σ_t log π_t(teacher_node_t)    ← 模仿 teacher 的节点选择
CE   = cross_entropy(reader_logits, y)  ← 分类头学习分类
```

**α 默认 0.7**，BC 主导，分类头作为辅助。

### 5.2 Stage 2: REINFORCE + CE

**数据**: 全量训练集（自动采样轨迹）

```
Loss = α · (-R_i · Σ_t log π_t(a_t)) + (1-α) · CE(y_i, f(h_i))

π_t = Categorical(softmax(score(frontier_embeds, belief_t)))
h_i = final belief after B_i steps
f   = ClassificationHead(flatten(h_i))
```

**Loss 设计要点**:

| 设计         | 理由                                                             |
| ------------ | ---------------------------------------------------------------- |
| CE 独立于 R  | 不论轨迹好坏，分类头始终降低 CE                                  |
| CE 不被 R 乘 | `R·(RL+CE)` 在 R<0 时会反转 CE 梯度方向——**错误设计** |
| α 默认 0.5  | 但 RL_loss(~60-90) 量级远大于 CE_loss(~0.5-2)，需调小 α         |

**Reward 类型**:

| 类型       | 正确时 R         | 错误时 R         |
| ---------- | ---------------- | ---------------- |
| binary     | +1               | 0                |
| asymmetric | +1               | -0.1             |
| tanh       | tanh(1-CE/ln(K)) | tanh(1-CE/ln(K)) |


### 5.3 分类头训练

REINFORCE 阶段 `compute_reward` 中 `R = ... .item()` 断图，**RL 不更新分类头**。
分类头只能通过 CE loss 项接收梯度。因此 CE 项是分类头在 RL 阶段唯一的学习信号。

---

## 6. 文件结构

```
belief/
├── common.py             共享常量 (K=37, BUDGET_OPTIONS) + 路径设置
├── belief_state.py       BeliefState: tokens + update_fn + classifier
├── frontier_pointer.py   FrontierPointer: SelfAttn → CrossAttn(belief) → scores
├── new_info_encoder.py   NewInfoEncoder: 2层GIN + 残差池化
├── learnable_reader.py   Reader: forward_states, encode_new_info, select, reset
├── training_utils.py     共享: deterministic_seed, reveal_and_update_beliefs_batch
├── train_bc.py           BC 训练: L = α·L_BC + (1-α)·CE
├── train_rl.py           REINFORCE + CE: L = α·(-R·Σlogπ) + (1-α)·CE
├── eval_heldout.py       Held-out 评估（含 budget 检查 + 子图复用）
├── smoke_test.py         冒烟测试（belief 模块 → forward → 完整锁步）
└── docs/
    ├── phase2-transformer-reader.md   ← 本文档
    └── working-experience.md          训练经验记录
```

---

## 7. 参数规模

| 模块                                                | 参数量          |
| --------------------------------------------------- | --------------- |
| FrontierEncoder（Phase 1 继承）                     | ~200K           |
| Belief Tokens embedding (m×d)                      | 512             |
| BeliefUpdate (1 层 Transformer)                     | ~100K           |
| FrontierPointer (SelfAttn + CrossAttn + score_proj) | ~100K           |
| Classification Head (Flatten + 2 层 MLP)            | ~70K            |
| NewInfoEncoder (2 层 GIN + 残差池化)                | ~270K           |
| **总计**                                      | **~840K** |

---

## 8. 对比

|               | MLP        | Belief                           |
| ------------- | ---------- | -------------------------------- |
| Frontier 编码 | 独立 MLP   | SelfAttn → CrossAttn(belief KV) |
| 跨步记忆      | 无         | m token × d dim Belief          |
| Query 来源    | 无 query   | h_t（每步演化）                  |
| NewInfo 编码  | —         | 2层GIN + 残差池化                |
| 分类方式      | 外部分类器 | 内部 ClassificationHead          |
| 训练方法      | BC         | BC → REINFORCE + CE             |
| 参数量        | ~180K      | ~840K                            |

---

## 9. 已知工作经验和修复

详见 `belief/docs/working-experience.md`，包括：

- BC 准确率高 → RL 掉到 1/5 的原因链
- Loss 量级差异和 α 调参
- 稀疏奖励分析
- eval_heldout.py 变量覆盖 bug
- 自环规则（历史 bug 记录）
- 文献参考（UNREAL, Gradient Similarity, Delightful PG 等）

# M3 Base 方案：Active Graph Reading v1 协议

> 基于 v0 协议（2026-5-19）和 2026-05-26 导师反馈修订。
> v1 核心变更：**frontier edge feature 在选择 frontier node 之前对 Reader 可见**。
> 代码实现对应 `active_graph_reader/` 的当前状态。

---

## 变更记录（v0 → v1）

| 条目 | v0（2026-5-19） | v1（2026-5-27） |
|------|-----------------|-----------------|
| Frontier edge feature | 仅两端点均已揭示时才可见 | **在 reveal 之前可见**（一端已揭示即可） |
| 查询接口 | 无显式接口 | `state.get_frontier_edge_feat(u, v)` |
| 可见性约束 | 无守卫 | 至少一个端点在 V_revealed 中，否则 ValueError |
| G_0 验证 | 无要求 | 必须通过测试：不消耗 budget 即可读取 seed-to-frontier edge feat |
| 策略 API | 隐式 | 显式基类：`select()` / `on_new_frontier()` / `reset()` |
| 子图构造 | 未定义 | 明确定义 SubgraphBuilder 行为 |
| 种子确定性 | 未规定 | 确定性种子公式：`(idx*31 + seed_i*17 + 7) % n_nodes` |
| 多图评测 | 简单说明 | 分层抽样 + 跨图批处理推理 |

---

## 一、问题定义

### 1.1 自然语言描述

我们有一个图 G 和一个标签 y。开始时只能看到一个随机选取的种子节点及其关联边列表（邻居 ID），如同从一本合上的书中随机翻开一页——只看到该页的标题和它引用的其他页码编号，但不知道被引用页的内容。

**关键变化（v1）**：Reader 可以看到连接已知页面和未知页面的"引用关系"的边特征（例如引用强度、关系类型），即使被引用页本身尚未翻开。这为策略提供了真实决策信号：哪个邻居更值得优先探索？

我们有预算 B（允许揭示的节点数），可以执行一系列"揭示动作"——翻开一个新节点 v：看清 v 的内容，获取 v 的关联边列表作为下一步探索的线索。v 的邻居节点虽然 ID 已知，但其特征保持隐藏直到被独立揭示。

预算耗尽后，基于看到的部分信息做分类判断。目标是：同样预算下，比随机翻看得更准。

### 1.2 数学定义

给定图 $G = (V, E, X_E)$，$|V| = n$，标签 $y \in \{0, 1, \ldots, C-1\}$。初始观测 $G_0$（见第 2 节），预算 $B \in \mathbb{N}^+$，剩余预算 $B_{\text{current}} \in \mathbb{N}^+$。

**揭示动作**：$\mathcal{A} = \{\text{reveal\_node}(v) \mid v \in \text{frontier}(G_t)\}$

单位成本：$C(a) = 1, \quad \forall a \in \mathcal{A}$

**决策函数**：$\pi: (G_t, B_{\text{current}}) \to \mathcal{A} \cup \{\varnothing\}$

**更新函数**：$\psi: (G_t, a) \to G_{t+1}$

**分类器**：$\phi: G_{\text{final}} \to \hat{y}$

**决策过程**（$t = 1, 2, 3, \ldots$）：

$$
\begin{aligned}
B_{\text{current}}&\leftarrow\ B \\
\text{while } B_{\text{current}} > 0: & \\
&\quad a_t = \pi(G_{t-1}, B_{\text{current}}) \\
&\quad \text{if } a_t = \varnothing: \text{break} \\
&\quad B_{\text{current}} \leftarrow B_{\text{current}} - C(a_t) \\
&\quad G_t = \psi(G_{t-1}, a_t)
\end{aligned}
$$

**最终预测**：$\hat{y} = \phi(G_{\text{final}})$

**优化目标**：

$$
\max_{\pi, \phi} \; \mathbb{E}_{(G,y) \sim \mathcal{D}}\Big[\mathbb{1}(\hat{y} = y)\Big] \quad \text{s.t.} \quad \sum_{t} C(a_t) \leq B
$$

---

## 二、G_0 初始观测（方案 D：k-Seed）

### 2.1 G_0 定义

```
G_0: 随机选取 k 个种子节点（v1 中 k = 1），对每个种子节点：
     ✓ 节点 ID 已知
     ✓ 关联边列表 / 邻居 ID 已知（边存在性）
     ✓ 边特征：种子节点与 frontier 邻居之间的边特征可见 ← v1 关键变更
     ✗ 种子节点特征隐藏（OGBG-PPA 无节点特征）
     ✗ 非邻居节点 ID 未知
     ✗ 全局图规模（|V|）未知
```

### 2.2 形式化

令 $S = \{s_1, \ldots, s_k\} \subset V$ 为随机选取的种子节点集，$k \ll n$（v1 中 $k = 1$）。

初始状态 $G_0$ 揭示的信息：

| 揭示内容 | 符号 | 说明 |
|----------|------|------|
| 种子节点 ID | $V_{\text{revealed}}^{(0)} = S$ | k 个已知节点 |
| 关联边列表 | $E_{\text{known}}^{(0)} = \{(u, w) \mid u \in S, (u,w) \in E\}$ | 种子节点的所有邻居 ID 已知 |
| **Frontier 边特征** ← v1 | $X_E^{\text{frontier},(0)} = \{x_{su} \mid u \in \text{frontier}(G_0)\}$ | **seed→frontier 边特征可见，不消耗 budget** |
| 已揭示节点间边特征 | $X_E^{\text{revealed},(0)} = \varnothing$ | k=1 时无双端点均已揭示的边 |
| 节点特征 | $X_V^{(0)} = \varnothing$ | OGBG-PPA 无节点特征 |

### 2.3 Frontier Edge Feature 可见性（v1 核心变更）

**v0 行为**：边特征在边的两个端点均已揭示后才可见——意味着 Reader 在选择第一个 frontier node 时，不知道任何边的特征信息，frontier 候选节点在信息上完全等价。

**v1 行为**：只要边的一个端点已揭示，该边的特征就对 Reader 可见。具体来说：

- G_0 阶段：seed→所有 frontier neighbor 的边特征可查询，budget 保持为 0
- 每步 reveal 后：新揭示节点→其所有邻居（包括已有 frontier 和新发现 frontier）的边特征立即可查询
- **可见 ≠ 揭示**：frontier 边特征可查询不消耗 budget，不改变揭示状态

**API 设计**（对应代码 `GraphState.get_frontier_edge_feat`）：

```python
state.get_frontier_edge_feat(u, v)  # → torch.Tensor, shape=(edge_feat_dim,)
```

**可见性守卫**：
- 至少一个端点在 `V_revealed` 中 → 允许查询
- 两个端点都不在 `V_revealed` 中 → 抛出 `ValueError`（协议不允许）
- 图无边特征 → 抛出 `ValueError`

**设计理由**：如果 Reader 只知道 frontier nodes 的 ID 存在，但看不到任何边特征，那么这些候选节点在信息上几乎等价，只能随机选择——后续 learned Reader 很难有真实决策信号。frontier edge feature 是连接"已观测子图"和"未知区域"的唯一特征通道。

### 2.4 G_0 验证测试

v1 要求通过以下最小测试（代码 `smoke_test.py:test_g0_frontier_edge_feat`）：

```
Given: synthetic graph, edge_feat_dim=7, seed_node=0
When:  GraphState(g, seed_node=0) initialized
Then:  |V_revealed| == 1
And:   budget_used == 0
And:   for each frontier edge (seed, neighbor): get_frontier_edge_feat returns shape (7,)
And:   budget_used remains 0 after reading features
```

### 2.5 代码实现细节

`GraphState._init_g0()` 的具体行为（`engine/graph_state.py:100-127`）：

1. 种子节点 s 加入 `V_revealed`，追加到 `reveal_history`
2. 遍历 s 的邻居（去重、按节点 ID 升序排列以保证确定性）
3. 每个邻居 nbr：添加前沿边 `(s, nbr)` 到 `frontier_edges`，更新 `_frontier_degree[nbr]`
4. 返回 `_initial_frontier` 列表（供 `RevealLoop` 按序通知策略）

`frontier_edges` 的存储格式：`Set[Tuple[int, int]]`，每条为 `(revealed, unrevealed)`。

---

## 三、核心揭示动作

### 3.1 动作定义

**唯一动作类型**：`reveal_node(v)`

**前置条件**：$v \in \text{frontier}(G_t)$，即存在前沿边 (revealed, v)。

**揭示内容**：

| # | 揭示项 | 说明 |
|---|--------|------|
| ① | v 成为已揭示节点 | $V_{\text{revealed}}^{(t+1)} = V_{\text{revealed}}^{(t)} \cup \{v\}$ |
| ② | v 与已揭示节点之间的边特征 | 收集所有指向 v 的前沿边特征，存入 `revealed_edge_feats` |
| ③ | v 的关联边列表 / 邻居 ID | 新邻居 ID 加入 frontier；已揭示邻居间的边特征补录 |
| ④ | **v 与新发现 frontier 邻居的边特征立即可见** ← v1 | v 揭示后，其→新邻居的 frontier edge feat 可查询 |

**不揭示的内容**：
- ✗ 新发现邻居的节点特征（保持隐藏，直到被独立 `reveal_node`）
- ✗ v 与尚未揭示节点之间的边特征……在 v1 中这些**已经是可见的**（作为 frontier edge feat）
- ✗ 图的其他部分

### 3.2 代码实现细节

`reveal_node()` 的具体行为（`engine/reveal_action.py:23-110`）：

**Step ② — 收集前沿边特征**：
- O(|frontier|) 扫描找出所有终点为 v 的前沿边
- 对每条边：计算 `key = (min(revealed, v), max(revealed, v))`，若不在 `revealed_edge_feats` 中则通过 `_get_edge_feat` 查询真值并存入
- 从 `frontier_edges` 中移除这些边

**Step ① — 标记已揭示**：
- v 加入 `V_revealed`，追加到 `reveal_history`
- `_budget_used += 1`
- 清理 v 的 `_frontier_degree` 条目

**Step ③ — 扩展前沿**：
- 遍历 v 的所有邻居（去重、按 ID 升序）
- 邻居在 `V_revealed` 中 → 防御性代码：补录边特征（v0 协议下为死代码，为未来有向图/逐边揭示预留）
- 邻居不在 `V_revealed` 中 → 添加前沿边 `(v, nbr)`，`_frontier_degree[nbr] += 1`
- 返回新发现的前沿边列表（供策略钩子）

**边特征查询**（`_get_edge_feat`）：
- 优先尝试 u→v 方向的 edge ID
- 若不存在则尝试 v→u 方向
- 返回 squeeze 后的 1D tensor，shape = (feat_dim,)

### 3.3 形式化状态转移

$$
\begin{aligned}
G_t &= (V_{\text{revealed}}^{(t)},\; E_{\text{known}}^{(t)},\; X_E^{\text{revealed},(t)},\; X_E^{\text{frontier},(t)},\; X_V^{(t)}) \\[6pt]
G_{t+1} &= \psi(G_t, \text{reveal\_node}(v)) \quad \text{其中：} \\[4pt]
V_{\text{revealed}}^{(t+1)} &= V_{\text{revealed}}^{(t)} \cup \{v\} \\[4pt]
E_{\text{known}}^{(t+1)} &= E_{\text{known}}^{(t)} \cup \{(v, w) \mid (v,w) \in E\} \\[4pt]
X_E^{\text{revealed},(t+1)} &= X_E^{\text{revealed},(t)} \cup \{x_{vu} \mid u \in V_{\text{revealed}}^{(t)}, (v,u) \in E\} \\[4pt]
X_E^{\text{frontier},(t+1)} &= X_E^{\text{frontier},(t)} \cup \{x_{vw} \mid w \notin V_{\text{revealed}}^{(t+1)}, (v,w) \in E\} \\[4pt]
X_V^{(t+1)} &= X_V^{(t)} \quad (\text{OGBG-PPA 无节点特征})
\end{aligned}
$$

**frontier 更新**：

$$
\text{frontier}(G_{t+1}) = \left(\text{frontier}(G_t) \setminus \{(u,v) \mid u \in V_{\text{revealed}}^{(t)}\}\right) \cup \{(v, w) \mid (v,w) \in E,\; w \notin V_{\text{revealed}}^{(t+1)}\}
$$

### 3.4 成本

**单位成本**：$C(\text{reveal\_node}(v)) = 1$

**累计成本**：$\sum_t C(a_t) = \text{已揭示节点数} \leq B$

预算 B 的含义：**允许揭示的最大节点数**（不含种子节点）。

---

## 四、策略 API

### 4.1 抽象基类

v1 定义了显式的策略基类 `RevealStrategy`（`strategies/base.py`）：

```python
class RevealStrategy(ABC):
    @abstractmethod
    def select(self, graph_state: GraphState) -> Optional[int]:
        """从当前前沿中选择下一个要揭示的节点。返回 None 表示无可用选择。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（如 'random', 'bfs'）。"""
        ...

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        """前沿边扩展通知（可选钩子）。BFS 等有状态策略需重写。"""
        pass

    def reset(self) -> None:
        """重置内部状态（可选钩子）。跨图复用时调用。"""
        pass
```

**select 方法可访问的 GraphState 查询接口**：
- `graph_state.frontier_edges` → `Set[Tuple[int, int]]`，前沿边集合
- `graph_state.get_frontier_degree(node)` → `int`，节点与 V_revealed 的局部连接度
- `graph_state.get_frontier_edge_feat(u, v)` → `Tensor`，frontier 边特征（v1 核心）
- `graph_state.is_frontier_empty` → `bool`
- `graph_state.V_revealed` → `Set[int]`
- `graph_state.get_budget_used()` → `int`

### 4.2 三种基准策略

| 编号 | 策略名称 | 选择规则 | 有状态 | 说明 |
|------|----------|----------|--------|------|
| P1 | **Uniform Random** | `random.choice(list(frontier_edges))[1]` | 无 | 从前沿边中均匀随机选择一条，返回其未揭示端点。注意：多已揭示邻居的节点通过更多前沿边被自然加权 |
| P2 | **BFS Order** | FIFO 队列出队，跳过已揭示端点 | 有（deque） | `on_new_frontier` 将新边追加到队尾；`reset()` 清空队列；邻居排序保证确定性 |
| P3 | **Max Known-Frontier Degree** | $\arg\max_{u \in \text{frontier}} \text{frontier\_degree}(u)$ | 无 | 选择与已揭示子图连接最紧密的节点。frontier_degree 由 `GraphState._frontier_degree` 维护，O(1) 查询 |

**P3 补充说明**：
- 平局时取 frontier_edges 集合迭代到的第一个（非确定性但对等）
- 该度数仅统计节点与 V_revealed 的边数，非真实全图度
- 度数由 reveal_node 在扩展前沿时增量更新（`_frontier_degree[nbr] += 1`），无需每次扫描

### 4.3 策略注册机制

通过 `strategies/registry.py` 实现可插拔策略：

```python
get_strategy('bfs')                    # → BfsStrategy 实例
register_strategy('my-strategy', MyClass)  # 扩展新策略无需修改主流程
list_strategies()                      # → {'random': ..., 'bfs': ..., ...}
```

---

## 五、揭示循环（RevealLoop）

### 5.1 协调逻辑

`RevealLoop`（`engine/reveal_loop.py`）是策略、成本模型、揭示动作之间的协调者：

```
RevealLoop.__init__(state, strategy, cost_model):
    1. strategy.reset()
    2. 通知 strategy.on_new_frontier(state._initial_frontier)  ← G_0 前沿边按邻居 ID 升序传入

RevealLoop.run(budget / max_ratio):
    while budget_used < budget_limit:
        if frontier 为空 → break
        next_node = strategy.select(state)
        if next_node is None → break
        cost = cost_model.compute(next_node, state)
        if cost > remaining → break
        new_frontier = reveal_node(state, next_node)
        if new_frontier → strategy.on_new_frontier(new_frontier)
    return state.stats()
```

### 5.2 预算解析

- 若 `budget` 和 `max_ratio` 均未指定 → 揭示所有可达节点
- 若同时指定 → 取两者中较小的上限
- 最小预算 = 1（即使 `max_ratio * n_nodes = 0`，也允许揭示至少 1 个节点）

### 5.3 终止条件

1. **前沿为空**：所有已发现邻居均已揭示，无法继续探索
2. **策略返回 None**：策略内部逻辑判定无需继续（如 BFS 队列耗尽）
3. **预算不足**：下一个动作成本 > 剩余预算

---

## 六、子图构造（SubgraphBuilder）

### 6.1 构造规则

`SubgraphBuilder.build(state)` 将 GraphState 转换为下游 GNN 可消费的 DGLGraph。

**节点**：
- 仅包含 `V_revealed` 中的节点
- 原始 ID → 子图连续 ID 映射：`original_ids = sorted(V_revealed)`，`old_to_new[old] = i`
- 子图节点数 = |V_revealed|

**边**：
- 仅包含 `revealed_edge_feats` 中的边（即双方均已揭示的边）
- 前端边（一端未揭示）不加入子图
- 每条边构建两个方向（无向图 → GNN 消息双向传递）
- 边特征从 `revealed_edge_feats[key]` 取，key = (min(u,v), max(u,v))

**特殊情况**：
- `V_revealed` 为空 → 抛出 ValueError
- 无边（budget=0 或仅种子揭示） → 创建空边图 `dgl.graph(([], []), num_nodes=1)`
- 无边特征 → `sub_g.edata` 为空，下游编码器用零填充

**附加数据**：
- `sub_g.edata['feat']`：堆叠的边特征 tensor，shape = (2 × |E_revealed|, edge_feat_dim)

### 6.2 与编码器的接口

```python
sub_g, original_ids = builder.build(state)
# 编码器不需要 original_ids——PPA 无节点特征，
# 所有节点共享 Embedding(1, hidden_dim)，通过 node_ids=zeros 索引。
node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long)
graph_embedding = encoder(sub_g, node_ids)
```

---

## 七、编码器与分类器

### 7.1 GINEEncoder

对齐 OGB 官方 GIN 架构（`encoder/gin_encoder.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 300 | 隐藏维度 |
| n_layers | 5 | GIN 卷积层数 |
| dropout | 0.5 | Dropout 概率 |
| edge_feat_dim | 7 | PPA 边特征维度 |
| virtual_node | False | 虚拟节点（默认关闭） |
| residual | False | 残差连接（默认关闭） |

**GINELayer**（单层 GIN with edge features）：
```
edge_emb = Linear(edge_feat_dim, hidden_dim)(edge_feat)
message  = ReLU(src_h + edge_emb)
aggregate = sum(messages)
h_new    = BatchNorm(MLP((1+eps)*h + aggregate))
```

**节点嵌入**：`nn.Embedding(1, hidden_dim)` — PPA 无节点特征，所有节点共享可学习嵌入，差异来自拓扑位置和边特征。

**微调支持**：
- 逐层 freeze 控制（`freeze_layers`）
- Bottleneck Adapter（Houlsby et al., ICML 2019）：`down → ReLU → up`，近零初始化
- 支持 Virtual Node 和 Residual 连接（默认关闭）

### 7.2 GraphClassifier

`GraphClassifier = GINEEncoder + nn.Linear(hidden_dim, n_classes)`

全图训练使用标准 batch training + self-loop 预处理 + OGB Evaluator。

---

## 八、预算与评价指标

### 8.1 预算设置

| 表示方式 | 说明 |
|----------|------|
| 绝对计数 $B$ | 揭示节点数（不含种子） |
| 相对比例 $B/n$ | 揭示比例，跨图可比 |

推荐预算序列：$B/n \in \{1\%, 2\%, 5\%, 10\%, 20\%, 50\%\}$（向上取整，至少 1 节点）。

### 8.2 多种子评测协议

**确定性种子生成**：
```python
seed_node = (graph_index * 31 + seed_i * 17 + 7) % n_nodes
```
保证同一 (graph, seed_i) 组合可复现。

**评测流程**（`benchmark_m3.py:evaluate_strategy`）：
1. 对每张图的每个 seed_i，按所有 budget 从小到大跑 reveal loop（增量 snapshot）
2. 用 SubgraphBuilder 构造子图
3. 跨图批量 GPU 推理（`graph_batch_size` 控制）
4. 按 (graph, budget) 汇总 per-graph mean accuracy

**分层抽样**：从验证集每类随机选 k 张图，保证类别覆盖。

### 8.3 核心指标

1. **预算-准确率曲线**：$B/n$ vs. 分类准确率，每种策略一条曲线（error bar: ±1 std）
2. **全图上界**：全图 GIN 在测试集上的准确率（虚线标注）
3. **曲线下面积（AUC-B）**：预算-准确率曲线的积分
4. **BFS-to-upper ratio**：BFS 在 ≤10% reveal 下达到全图准确率的比例

### 8.4 淘汰标准（Kill Criterion）

若 BFS-from-seed 在 ≤10% 节点 reveal 下达到全图准确率的 ≥90%，则数据集对本问题过易——应切换到更困难的数据集或更困难的 G_0。

---

## 九、数据集：OGBG-PPA

### 9.1 基本信息

| 项目 | 状态 |
|------|------|
| 图数量 | 158,100（train/valid/test 按 OGB 标准划分） |
| 类别数 | 37 |
| 节点特征 | 无（所有图 node_feat 为 None） |
| 边特征 | 7 维向量 |
| 图规模 | 数百节点、数千边/图 |
| 类别不平衡 | max/min ratio ≈ 14.55 |

### 9.2 预处理

- `dgl.add_self_loop(g)` — 对齐 OGB 官方预处理
- 数据加载：`DglGraphPropPredDataset(name='ogbg-ppa', root=...)`
- 训练使用标准 `DataLoader` + `collate_dgl`

---

## 十、代码结构总览

```
active_graph_reader/
├── engine/
│   ├── graph_state.py       # G_t 部分可观测状态 + get_frontier_edge_feat
│   ├── reveal_action.py     # reveal_node(v) 状态转移
│   └── reveal_loop.py       # 主循环：策略→动作→状态协调
├── strategies/
│   ├── base.py              # RevealStrategy 抽象基类
│   ├── random_strategy.py   # P1: Uniform Random
│   ├── bfs_strategy.py      # P2: BFS Order (FIFO)
│   ├── max_frontier_degree_strategy.py  # P3: Max Known-Frontier Degree
│   └── registry.py          # 策略注册表
├── encoder/
│   ├── gin_encoder.py       # GINEEncoder (OGB 官方架构)
│   ├── classifier.py        # GraphClassifier = Encoder + Head
│   ├── adapter.py           # BottleneckAdapter (微调用)
│   └── subgraph_builder.py  # GraphState → DGLGraph 转换
├── cost/
│   ├── base.py              # CostModel 抽象基类
│   ├── unit_cost.py         # UnitCostModel (cost=1)
│   └── registry.py          # 成本模型注册表
├── dataloader/loader.py     # PPADataset 封装
├── config.py                # RevealConfig dataclass
└── main.py                  # CLI 入口 + 程序化 API

m3/
├── train_full_graph.py      # 全图 GIN 训练 → best_model.pt
├── M3_finetune_reader.py    # Reader 子图微调（策略 A/B1/A+B1）
├── benchmark_m3.py          # 多 seed 评测 + 预算曲线 + kill criterion
└── smoke_test.py            # 冒烟测试（合成图 + 可选真实数据）
```

---

## 十一、v1 协议验证清单

以下测试必须在声称"v1 协议实现正确"前通过：

| # | 测试项 | 对应测试 |
|---|--------|----------|
| 1 | G_0 初始化后 |V_revealed|=1, budget=0 | `test_g0_frontier_edge_feat` |
| 2 | G_0 阶段可读取 seed→frontier edge feat | `test_g0_frontier_edge_feat` |
| 3 | 读 frontier edge feat 不消耗 budget | `test_g0_frontier_edge_feat` |
| 4 | reveal_node 后 budget += 1 | `test_reveal_protocol` |
| 5 | 新 frontier edge feat 在 reveal 后立即可读 | `test_reveal_protocol` |
| 6 | 三种策略均能在有限 budget 下运行到终止 | `test_all_strategies` |
| 7 | SubgraphBuilder → GINEEncoder → GraphClassifier pipeline | `test_encoder_classifier` |
| 8 | 跨图批处理推理 shape 正确 | `test_multiple_graphs_batch` |
| 9 | 两端点均未揭示时 get_frontier_edge_feat 抛 ValueError | `test_frontier_edge_feat_guards` |
| 10 | 无边特征图调用 get_frontier_edge_feat 抛 ValueError | `test_frontier_edge_feat_guards` |

---

## 附录 A：v0 与 v1 完整对照

| 条目 | v0（2026-5-19） | v1（2026-5-27） |
|------|-----------------|-----------------|
| G_0 | 方案 D：k-seed, k=1 | 同 v0 |
| Frontier edge feature | 仅两端点揭示后可见 | **一端揭示即可见**，不消耗 budget |
| 查询 API | 无 | `get_frontier_edge_feat(u, v)` + 可见性守卫 |
| 动作体系 | 单一动作 reveal_node(v) | 同 v0 |
| 成本模型 | UnitCostModel (cost=1) | 同 v0 |
| 策略 | random / BFS / max-frontier-degree | 同 v0 + 显式基类 API |
| 策略有状态管理 | 隐式 | `on_new_frontier()` / `reset()` 钩子 |
| 子图构造 | 未定义 | SubgraphBuilder：V_revealed + revealed_edge_feats |
| 种子确定性 | 未规定 | `(idx*31 + seed_i*17 + 7) % n_nodes` |
| 评测抽样 | 简单说明 | 分层抽样 k-per-class + 批处理推理 |
| 编码器 | 标准 GIN | GINEEncoder (OGB 官方) + Adapter 微调支持 |
| Kill criterion | BFS@≤10% vs 90% full-graph | 同 v0 |
| M4+ 脚手架 | 附录 B 保留 | 同 v0，未展开 |

## 附录 B：M4+ 设计脚手架（冻结不展开）

同 v0 附录 B。以下内容保留为 M4 及之后阶段的参考框架：
- 四个原子动作及成本：reveal_node (c_n=1), reveal_edge (c_e=1), reveal_node_f (c_nf=2), reveal_edge_f (c_ef=2)
- 复合动作分类体系
- 方案 A/B/C/D 的跨方案扩展

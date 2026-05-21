# M3 Reader and Encoder-Classifier：度量 Reader 的尺子

> 状态：设计方案（修订版）
> 前提：Reader（reveal 协议 + 三种简单策略）已实现
> 核心思路：Reader 在有限预算下读出关键子图 → 编码器编码 → 分类器预测，以小开销获高回报

---

## 一、当前状态与真实问题

### 1.1 已实现：Reader（阅读器）

```
  ┌──────────┐      reveal_node(v)       ┌──────────────┐
  │  G₀      │ ──────────────────────→  │   G_B        │
  │ k=1 seed │   策略: random/bfs/maxdeg  │ V_revealed   │
  │ frontier │   cost=1 per node         │ edge_feats   │
  └──────────┘                           └──────────────┘
                                              │
                                              │  缺失：编码 → 分类
                                              ▼
                                         ┌──────────────┐
                                         │   ŷ = ?      │
                                         └──────────────┘
```

v0 协议已给出三种最简单的 Reader 策略——从一个随机种子出发，按拓扑规则探索邻域。**这不是终点，是测量基线。**

### 1.2 两个真正的核心问题

**问题 1：效果对初始条件高度敏感**

当前基于简单拓扑规则的 Reader，效果完全取决于：

- **初始种子落在哪里**——落在关键区域 vs. 无关区域，分类结果天差地别
- **预算给多少**——预算越大方差越小（读得越多越接近全图），但预算本身就是要省的东西

需要回答：**多大的预算能平衡方差与性能？不同策略在同等预算下表现差异多大？**

**问题 2：如何让 Reader 学会探索？**

当前三种策略（random / BFS / max-degree）都是**盲目的**——它们不知道"哪里的信息对分类更有用"。它们只看图拓扑，不看任务信号。

真正的目标（M4/M5）：**让 Reader 学会："哪些节点值得揭示？"**——基于已读到的内容，判断下一步该往哪走。

**M3 阶段的目标不是解决这两个问题，而是建立度量它们的基础设施。**

### 1.3 编码器-分类器的定位

```
                  度量 Reader 质量的尺子
                 ════════════════════════

  Reader ──→ G_B ──→ 编码器 ──→ graph_emb ──→ 分类器 ──→ ŷ ──→ accuracy
    ↑                                                              │
    │                                                              │
    └──────────── 反馈信号：策略 A vs B 谁更好？ ──────────────────┘
```

编码器-分类器是**一根尺子**——它不参与阅读决策，只负责评估：

- "在策略 X、预算 Y 下读出的子图，能分对多少？"
- 比较不同策略在同一预算下的分类准确率 → 衡量 Reader 质量

**尺子本身不应该是研究的重点**——用标准 GIN/GCN 即可，不需要花哨设计。

---

## 二、缺失的三块工程拼图


| #     | 拼图              | 作用                                             | 复杂度            |
| ----- | ----------------- | ------------------------------------------------ | ----------------- |
| **A** | **子图构造器**    | `GraphState` → 标准 `DGLGraph`（可被 GNN 消费） | 低                |
| **B** | **编码器+分类器** | 部分子图 → graph embedding → 类别预测          | 中（选型 + 训练） |
| **C** | **训练-评估管线** | 训练尺子 → 用尺子量各种 Reader                  | 中（循环编排）    |

三块都不涉及 Reader 内部逻辑修改——纯组合接入。

---

## 三、工程设计

### 3.1 子图构造器：`SubgraphBuilder`

**职责**：将 `GraphState`（揭示状态）转换为标准的 `DGLGraph`。

```
GraphState                     →  DGLGraph sub_g
──────────────────────────────────────────────────
V_revealed = {0, 1, 3, 5}     →  4 个节点
revealed_edge_feats:            →  2 条边，边特征 = 原始 7 维
  {(0,1): feat, (1,3): feat}
frontier_edges:                 →  不加入子图
  {(3,9), (5,7)}                  （未揭示端点不在子图中）
```

**关键决策**：


| 决策点   | 选择                                | 理由                                |
| -------- | ----------------------------------- | ----------------------------------- |
| 节点范围 | 仅 V_revealed                       | 未揭示节点对分类器不可见            |
| 边范围   | 仅已揭示边（双方均揭示）            | 这些边有完整特征；frontier 边无特征 |
| 节点特征 | `nn.Embedding(n_total, d)` 可学习表 | PPA 无原始节点特征；常见做法        |
| 边特征   | 直接取原始 7 维向量                 | 不做变换，交给 GNN 的 edge_encoder  |

```python
# 伪代码
def build(state: GraphState) -> dgl.DGLGraph:
    nodes = sorted(state.V_revealed)
    # 建立旧 ID → 新 ID 映射（子图节点从 0 开始编号）
    old_to_new = {old: i for i, old in enumerate(nodes)}

    src, dst, edge_feats = [], [], []
    for (u, v), feat in state.revealed_edge_feats.items():
        if u in old_to_new and v in old_to_new:
            src.extend([old_to_new[u], old_to_new[v]])  # 双向边
            dst.extend([old_to_new[v], old_to_new[u]])
            edge_feats.extend([feat, feat])

    sub_g = dgl.graph((src, dst), num_nodes=len(nodes))
    sub_g.edata['feat'] = torch.stack(edge_feats)
    # ndata['feat'] 由编码器的 node_emb 在 forward 时注入
    return sub_g, nodes  # nodes 是原始 ID，用于查 embedding
```

### 3.2 编码器：`GINEEncoder`

```
GINEEncoder:
  node_emb:     nn.Embedding(n_nodes_total, hidden_dim)   ← 可学习节点初始特征
  edge_encoder: nn.Linear(7, hidden_dim)                  ← 边特征映射
  convs:        ModuleList[GINEConv × L]                  ← 带边特征的 GIN
  bns:          ModuleList[BatchNorm1d × L]
  pool:         mean readout                              ← 图级汇聚
```

**为什么是 GINEConv（不是 GINConv）**：PAA 有 7 维边特征，GINEConv 支持边特征注入消息函数，OGBG-PPA 排行榜验证有效。

**参考超参**（OGB-PPA 排行榜常用配置）：

```
hidden_dim = 300
n_layers   = 5
dropout    = 0.5
pool       = mean
```

### 3.3 分类器：`GraphClassifier`

```
GraphClassifier:
  encoder: GINEEncoder
  head:    Sequential(
             Linear(hidden_dim, hidden_dim),
             ReLU(),
             Dropout(0.5),
             Linear(hidden_dim, num_classes)
           )
```

### 3.4 训练策略

**核心原则：尺子是中立的。它不能偏袒任何一种 Reader 策略。**


| 阶段     | 数据     | 策略                       | 预算 | 目的                         |
| -------- | -------- | -------------------------- | ---- | ---------------------------- |
| **训练** | trainset | **random**（唯一无偏策略） | B    | 让尺子见过各种"部分观测"模式 |
| **评估** | testset  | random / bfs / max-degree  | B    | 用尺子度量每种 Reader 的质量 |

训练细节：

```
for each epoch:
  for each graph in trainset:
    B ~ Uniform{1%, 2%, 5%, 10%, 20%, 50%} × |V|
    用 random 策略执行 reveal_loop(budget=B)   ← 不依赖特定 Reader
    构造子图 → 前向 → 计算 CE loss → 反向
```

**为什么训练用 random 策略**：random 无偏——不偏向连通区域（BFS）也不偏向枢纽节点（max-degree）。在 random 生成的各种部分图上训练，尺子(Encoder和Classifier)对任何 Reader 读出的子图都公平。


|                   | 训练（校准尺子）                           | 评估（用尺子量）               |
| ----------------- | ------------------------------------------ | ------------------------------ |
| **用什么 Reader** | random（唯一无偏）                         | 三种各自跑                     |
| **尺子见过谁**    | 只见过 random 的"随机拼图"                 | 没见过 BFS/maxdeg 的子图模式   |
| **目的**          | 让尺子学会「从拼图碎片猜全貌」这个通用能力 | 公平度量三种 Reader 的产出质量 |

### 3.5 全图上界

按 M3 协议要求，单独训练一个 GIN 在**完整图**上：

```
训练: full graph → GINEEncoder → head → CE loss
测试: full graph → accuracy
```

这是所有预算-性能曲线的天花板。任何部分观测下的准确率都应低于此线。

---

## 四、新增模块架构

```
active_graph_reader/
├── ... (现有: engine / strategies / cost / dataset / config — 零修改)
│
├── encoder/                          ← 新增
│   ├── __init__.py
│   ├── subgraph_builder.py           # GraphState → DGLGraph
│   ├── gin_encoder.py                # GINEConv × L
│   └── classifier.py                 # encoder + MLP head
│
├── experiment/                       ← 新增
│   ├── __init__.py
│   ├── runner.py                     # 单次: reader → encoder → predict
│   ├── trainer.py                    # 训练循环
│   └── evaluator.py                  # 多策略 × 多预算评估
│
└── viz/                              ← 新增
    └── curves.py                     # 预算-准确率曲线
```

**与现有模块的关系**：纯组合，不修改任何现有代码。`experiment/runner.py` 调用 `engine/RevealLoop` + `encoder/SubgraphBuilder` + `encoder/GraphClassifier`。

---

## 五、研究问题与 M3 要回答什么

### 5.1 M3 直接能回答的


| 问题                             | 测量方式                                          |
| -------------------------------- | ------------------------------------------------- |
| 三种 Reader 在同等预算下谁更好？ | 同一尺子，三条预算-准确率曲线对比                 |
| 预算多大时性能趋于饱和？         | 看曲线何时接近全图上界                            |
| 方差有多大？（种子敏感性）       | 同图多种子 → mean ± std                         |
| 是否存在"小预算高回报"空间？     | kill criterion：BFS <10% 已达 ~90% 上界 → 没空间 |

### 5.2 M3 无法回答（留给 M4/M5）


| 问题                                    | 为什么 M3 答不了                                      |
| --------------------------------------- | ----------------------------------------------------- |
| 如何让 Reader 学习探索？                | M3 的策略是固定的（random/bfs/maxdeg），没有学习组件  |
| 哪里是"关键子图"？                      | M3 不引入任务信号指导阅读，Reader 只看拓扑            |
| 学习到的 Reader 能比 heuristic 好多少？ | 需要 M4 的 trajectory scoring + M5 的 policy learning |

### 5.3 通向 M4/M5 的路径

```
M3: 固定策略 + 尺子度量          →  回答"哪个方向有空间？"
M4: 轨迹收集 + 质量评分          →  回答"好的阅读轨迹长什么样？"
M5: 策略模仿学习                  →  回答"能不能学会像好轨迹那样读？"
                      
  Reader 策略
  ┌──────────────────────────────────────────────┐
  │ M3: random / bfs / max-degree (固定规则)      │
  │ M4: oracle / heuristic (轨迹评分)             │
  │ M5: learned policy (GNN 决策网络)             │  ← 目标
  └──────────────────────────────────────────────┘
```

**M3 的核心价值**：用最简单的 Reader 跑通全管线，拿到第一条预算-准确率曲线。如果 kill criterion 不触发（即有 headroom），M4/M5 才有推进的必要。

---

## 六、实现计划

### Phase 1: 尺子搭建（1 天）

- [X]  `subgraph_builder.py`：GraphState → DGLGraph
- [X]  `gin_encoder.py`：GINEConv × 5 + node_emb + edge_encoder
- [X]  `classifier.py`：encoder + MLP head
- [X]  冒烟测试：随机部分图可前向通过，无 shape 错误

### Phase 2: 训练尺子（1 天）

- [ ]  `trainer.py`：标准 OGB 训练循环
- [ ]  **全图上界**：完整图训练 → test accuracy（参考 ~68%）
- [ ]  **混合预算训练**：random 策略 × 混合预算 → 保存模型
- [ ]  验证：训练 loss 正常下降

### Phase 3: 用尺子量 Reader（1 天）

- [ ]  `runner.py`：(strategy, budget, graph, seed) → prediction
- [ ]  `evaluator.py`：3 策略 × 6 预算 × 多种子
- [ ]  种子平均（每图 5~10 个种子）
- [ ]  汇总：mean accuracy ± std

### Phase 4: 可视化与报告（半天）

- [ ]  `curves.py`：三条预算-准确率曲线 + 全图上界虚线
- [ ]  Kill criterion 检查
- [ ]  产出分析文字

---

## 七、风险与预案


| 风险                                               | 预案                                                      |
| -------------------------------------------------- | --------------------------------------------------------- |
| **Kill criterion 触发**（BFS <10% 已达 ~90% 上界） | 说明 PPA 太简单 → 换更硬的 G_0（方案 A）或换更难的数据集 |
| **方差过大**（同种子同策略准确率 ±20%）           | 增加种子数，报告分布而非均值；考虑分层（按图规模分组）    |
| **random 训练的尺子对 BFS 子图不公平**             | 验证：用 BFS 子图 fine-tune 一轮，看是否显著提升          |
| **训练时间长**（PPA ~158k 张图）                   | Phase 1/2 用 5000 张子集快速验证，Phase 3 全量            |

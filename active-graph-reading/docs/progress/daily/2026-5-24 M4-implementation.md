# M4 阶段实现方案：Teacher / Trajectory 基线

> 日期：2026-05-24
> 状态：方案设计
> 上游：M3 基线尺子（Encoder-Classifier）就绪
> 下游：M5 第一个可学习的 Reader 策略

---

## 一、M4 在整体路线中的定位

```
M3: 固定策略 + 尺子度量          →  回答"哪个方向有空间？"
M4: 轨迹收集 + 质量评分          →  回答"好的阅读轨迹长什么样？"   ← 本文档
M5: 策略模仿学习                  →  回答"能不能学会像好轨迹那样读？"
```

M4 是一个**分析+准备阶段**，核心产出不是训练好的模型，而是：

1. 一批高质量的 teacher 阅读轨迹（用于 M5 行为克隆训练）
2. 对"什么是好轨迹"的量化理解（用于 M5 损失函数设计）
3. oracle 性能上界（用于判断 M5 学习到的策略还有多少提升空间）
4. 对轨迹蒸馏方案的可行性判断（决定 M5 走 BC 还是端到端）

---

## 二、M4 四大任务与核心问题

### 任务 1：生成多策略阅读轨迹

**问题**：需要多少种 teacher？哪些 teacher 值得实现？

**核心矛盾**：好的 teacher 需要利用全图信息（甚至标签信息）来确定"哪些节点重要"，但 teacher 产生的轨迹将用于训练一个只能看到部分观测的 student Reader。teacher 的"全知全能"能力不能泄露到 M5 的测试阶段。

**解决方案**：区分两种 teacher：

- **训练 teacher**：可以访问全图 + 标签，仅用于离线生成训练轨迹
- **推理 teacher**：仅使用 student 推理时可获得的信息，用于公平对比

### 任务 2：轨迹质量评分

**问题**：给定一批轨迹，如何区分好的和坏的？

**核心挑战**：一条轨迹的好坏是**多维的**——既要准确（最终分类正确），又要高效（预算利用率高），还要稳健（不一定唯一最优路径）。

**解决方案**：多维度评分 + 过滤 + 排序，每图保留 top-K 条轨迹。

### 任务 3：Oracle 性能上界估计

**问题**：理论上，在给定预算 B 下，Reader 能做到的最好性能是多少？

**核心难点**：不能把标签泄露到 test-time 策略中，但可以用标签构建 oracle teacher 来估计上界。

**解决方案**：对训练集/验证集的每张图，用 oracle teacher 在预算 B 下生成轨迹 → 记录最终准确率 → 以此作为"给定预算下可达的性能上限"。注意：**oracle 仅用于上界估计，不用于 M5 训练**（M5 的训练轨迹来自无标签泄露的训练 teacher）。

### 任务 4：轨迹蒸馏可行性判断

**问题**：是否值得用 BC 训练 student Reader？还是直接端到端训练更好？

**判断标准**（来自 reader-training-plan.md §4.5）：

- 如果 teacher 轨迹的 Final Accuracy 显著高于 BFS baseline → BC 可行
- 如果 teacher 轨迹间的一致性高（多条 teacher 选择相似的节点）→ 学习目标明确
- 如果 trajectory 的质量方差大 → 需要严格的过滤策略

---

## 三、Teacher 设计：生成高质量阅读轨迹

### 3.1 设计原则

一个好的 teacher 应该满足：

1. **信息优势**：能利用 student Reader 在推理时无法获得的信息（全图拓扑、完整边特征、标签）
2. **可复现性**：在相同输入下产生确定性的阅读顺序（种子固定时）
3. **计算可行性**：在 PPA 数据集（158K 张图，每图 ~300 节点，2200边）上可行
4. **多样性**：不同类型的 teacher 可能识别不同类型的"重要节点"

### 3.2 Teacher 方案详细设计

#### Teacher T1：梯度归因 Teacher（首选）

**原理**：在全图上训练好的 GIN 分类器上做反向传播，计算每个节点嵌入对正确类别的梯度范数作为"重要性"分数。

**来源**：基于 Grad-CAM (Selvaraju et al., ICCV 2017) 在 GNN 上的推广（Pope et al., 2019 "Explainability Methods for GCNs"）；与 Integrated Gradients (Sundararajan et al., ICML 2017) 思想一致。

**具体步骤**：

```
1. 全图 G_orig → 冻结的φ（M3 尺子）→ 前向 → 预测 ŷ
2. 取正确类别 logit y_c → 反向传播梯度
3. 对每一层 l，计算每个节点的 embedding 梯度范数：
     s_v^{(l)} = ||∂ŷ_c / ∂h_v^{(l)}||₂
4. 汇总（多层平均或仅末层）：s_v = mean_l(s_v^{(l)})
5. 按 s_v 降序排列，作为"揭示优先级"
6. 模拟 reveal 过程：从 seed node 出发，每次揭示 frontier 中 s_v 最高的节点
   直到预算耗尽或 frontier 为空
```

**关键实现细节**：

- 梯度计算需要对正确类别（而非预测类别）求导，以避免确认偏差
- 如果 φ 预测错误，该轨迹直接标记为错误 → 过滤掉
- PPA 无节点特征，重要性来源是节点在消息传递中的信息汇聚角色
- 多任务平均：如果有多个 GNN 模型（如不同随机种子训练的），可以聚合它们的梯度信号

**文献支撑**：

- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- Pope et al., "Explainability Methods for Graph Convolutional Neural Networks", CVPR 2019
- Baldassarre & Azizpour, "Explainability Techniques for Graph Convolutional Networks", ICML 2019 Workshop

#### Teacher T2：信息增益 Teacher（备选）

**原理**：对每个 frontier 节点，模拟揭示它，用冻结尺子前向计算当前置信度，选置信度提升最大的节点。

**来源**：基于决策树中经典的信息增益（Quinlan, 1986）推广到图序列决策；与 Bayesian Active Learning by Disagreement (BALD, Houlsby et al., NeurIPS 2011) 理念相通；Fuchsgruber et al. (ICML 2024) 验证了信息论方法在图上的有效性。

**具体步骤**：

```
状态 s_t：已揭示子图 G_t
动作 a ：揭示 frontier 节点 v
收益   ：IG(v) = H(p_t) - H(p_{t|v})
         其中 H(p) = -∑_c p(c) log p(c) 是分类器输出的熵
         p_t = φ(G_t) 的 softmax 输出
         p_{t|v} = φ(G_t 揭示 v 后) 的 softmax 输出

for each v in frontier_t:
    1. 模拟揭示：state_t + reveal_node(v) → G_{t|v}
    2. 前向计算：p_{t|v} = φ(G_{t|v})
    3. 计算 IG(v) = H(p_t) - H(p_{t|v})

选择 v* = argmax_v IG(v)
```

**关键实现细节**：

- 每步需要对 |frontier| 个候选各做 1 次前向（无需反向）
- frontier 大小通常 5-50，在 GPU 上可批量处理
- 信息增益可以为负（揭示后熵增大）——排除此类候选
- 温度参数：对 softmax 输出应用温度 τ < 1 可以放大信息增益信号（降低低概率类别的平滑效应）

**文献支撑**：

- Houlsby et al., "Bayesian Active Learning for Classification and Preference Learning", NeurIPS 2011
- Fuchsgruber et al., "Uncertainty for Active Learning on Graphs", ICML 2024
- Gal et al., "Dropout as a Bayesian Approximation", ICML 2016

#### Teacher T3：结构 Teacher（快速基线）

**原理**：使用图拓扑度量（度中心性、PageRank、betweenness）作为节点重要性。

**来源**：经典图论度量；PageRank (Page et al., 1999)；多项工作验证了度中心性在 GNN 解释任务中作为简单基线的有效性 (Ying et al., NeurIPS 2019)。

**具体步骤**：

```
1. 计算每个节点的结构重要性分数（可选择）：
   - 度中心性：s_v = deg(v) / (|V|-1)
   - PageRank：s_v = PR(v), damping factor α = 0.85

2. 按 s_v 降序排列
3. 模拟 reveal 过程：选择 frontier 中 s_v 最高的节点
```

**关键实现细节**：

- 结构 teacher 不依赖任何 GNN 模型，可以离线预计算
- 度中心性计算最快（O(|E|)），PageRank 适中（迭代到收敛），Betweenness 最慢（O(|V|×|E|)）
- PageRank 适合捕捉"枢纽"节点（即使度不高，但连接了重要节点）
- 度中心性在无标度网络中效果好（PPA 图符合此特性）

**计算开销**：极低。预计算整数据集的节点分数可在几分钟内完成。

**文献支撑**：

- Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks", NeurIPS 2019
- Page et al., "The PageRank Citation Ranking: Bringing Order to the Web", 1999

#### Teacher T4：混合 Teacher（增强多样性）

**原理**：混合 T1 + T3（梯度 + 结构），加权组合节点分数。

**具体步骤**：

```
s_v^{hybrid} = α · s_v^{grad} + (1-α) · s_v^{struct}
其中 α 在 [0.3, 0.7] 中随机采样（每张图），增加多样性
也可：α = 0.5 固定
```


### 3.3 Teacher 对比总结


| Teacher      | 信号来源                        | 计算复杂度                   | 轨迹质量 | 多样性 | 推荐角色     |
| ------------ | ------------------------------- | ---------------------------- | -------- | ------ | ------------ |
| T1: 梯度归因 | 冻结φ的反向梯度                | 低（1 fwd+1 bwd）            | 高       | 中     | 核心信号源   |
| T2: 信息增益 | 冻结φ的多次前向                | 高（K×fwd，K=frontier大小） | 最高     | 低     | 质量验证基准 |
| T3: 结构     | 图拓扑（度/PageRank）           | 极低（预计算）               | 中       | 高     | 多样性补充   |
| T4: 混合     | T1 梯度分数 + T3 结构分数加权低 | 等同 T1                      | 高       | 高     | 首选         |

**推荐策略**：T1 提供核心梯度信号，T3 补充结构视角，T2 仅在验证集上作为质量参考基准（不对训练集全量运行）。

### 3.4 Teacher 多样性策略

**为什么需要多样性**：

- 单一 teacher 产生的轨迹可能过于"窄"——总是选择同一类节点
- BC 训练需要足够多的 (state, action) 样本覆盖不同的阅读场景
- 多种 teacher 可以教导 student 识别不同类型的"重要节点"

**实现方式**：

```
每张图生成 5-8 条轨迹，来源于：
  1. T1 梯度归因（α=1.0）  → 2 条（不同 GNN 随机种子）
  2. T3 度中心性           → 1 条
  3. T3 PageRank           → 1 条
  4. T4 混合 α=0.3         → 1 条
  5. T4 混合 α=0.7         → 1 条
```

---

## 四、轨迹质量评分与过滤

### 4.1 为什么要评分和过滤

- teacher 也会犯错：梯度归因可能被噪声误导，信息增益可能贪心过头
- BC 训练的质量上限取决于训练轨迹的质量：**垃圾进，垃圾出**
- 需要在"质量"和"数量"之间平衡——过滤太严格会减少训练数据

### 4.2 评分维度

#### 维度 1：Final Accuracy（必要条件）

```
Acc_traj = 1(φ(G_B) 的预测 == 真实标签)

判定：Acc_traj == 0 → 直接丢弃（错误轨迹不用于 BC 训练）
```

#### 维度 2：Budget Efficiency（预算效率）

```
BE = Acc_traj / B_used

含义：单位预算获得的"正确性"。预算是稀缺资源，高效的轨迹应该用更少节点达到正确分类。
```

**归一化处理**：由于不同图的节点数不同，使用相对预算 `B_rel = B / |V|` 更公平：

```
BE_norm = Acc_traj / (B_used / |V| + ε)
```

其中 ε 是小常数（如 1e-6），防止除零。

#### 维度 3：Information Gain（信息增益轨迹）

```
IG_sum = Σ_{t=1}^{B} [H(p_{t-1}) - H(p_t)]

含义：轨迹中每一步的信息增益之和。高 IG 表示每一步都在"减少不确定性"。
```

**细节**：

- 早期步的 IG 通常更大（从无到有）
- 后期步的 IG 趋于递减（边际收益下降）
- 如果 IG 在早期步就接近饱和，且后续步的 IG → 0，说明早期节点质量极高
- 可以通过 `IG_sum / B` 衡量平均每步增益

#### 维度 4：轨迹稳定性（同图不同种子的一致性）

```
Stability = |∩_i N_i| / |∪_i N_i|

其中 N_i 是第 i 条 teacher 轨迹选择的节点集合。Jaccard 相似度。
高值 → teacher 意见一致，学习目标明确；低值 → teacher 分歧大。
```

### 4.3 综合评分公式

```
Score(traj) = Acc_traj · (
    w1 · BE_norm +
    w2 · IG_sum_norm +
    w3 · Stability
)

默认权重：w1=0.5, w2=0.3, w3=0.2
```

### 4.4 过滤策略

```
Step 1: 丢弃 Acc_traj == 0 的轨迹
Step 2: 每张图按 Score 降序排列
Step 3: 每张图保留 top-K 条轨迹（K=3~5）
Step 4: 保留的轨迹中，确保至少有 1 条来自不同 teacher 类型（多样性保证）
```

### 4.5 轨迹评分与过滤的代码结构

```python
# m4/trajectory_generator.py
class TrajectoryGenerator:
    def __init__(self, teachers, classifier, subgraph_builder):
        ...
    def generate(self, graph, label, seed, budget) -> Trajectory:
        ...

# m4/trajectory_scorer.py
class TrajectoryScorer:
    def score(self, trajectory, graph, label) -> dict:
        return {
            'final_acc': ...,
            'budget_efficiency': ...,
            'ig_sum': ...,
            'total_score': ...,
        }

# m4/trajectory_filter.py
class TrajectoryFilter:
    def filter(self, trajectories, scores, top_k=3) -> list:
        # 1. discard Acc==0
        # 2. sort by Score
        # 3. top-K per graph
        # 4. diversity guarantee
        ...
```

---

## 五、Oracle 性能上界估计

### 5.1 核心问题

在给定预算 B 下，Reader 策略的理论最优性能是多少？这个值本身无法精确计算（需要穷举所有 frontier 序列），但可以通过 oracle teacher 给出一个**经验上界**。

### 5.2 方法论：全图上界、下界、经验上界

**完整图上界（M3 已给出）**：φ 在全图上的准确率（~68% for GIN on PPA）

**部分图下界（M3 已给出）**：random Reader 在预算 B 下的准确率

**Oracle 经验上界（M4 产出）**：oracle teacher 在预算 B 下的准确率

```
准确率
  ↑
  │  ┌───────────────────────────────── 全图上界（~68%）
  │  │     ┌────────────────────────── oracle 经验上界
  │  │     │    ┌───────────────────── BFS baseline
  │  │     │    │    ┌──────────────── random baseline
  │  │     │    │    │
  │  │·····│····│····│····  ← 这些虚线是 M3 产出的曲线
  │  │     │    │    │
  └──┴─────┴────┴────┴──→ 预算 B
```

### 5.3 Oracle 的两种定义

**Oracle Type I（标签感知型）—— 用于上界估计**：

在全图上，已知标签 y，已知冻结尺子φ。对每个 possible 的选择，选择能最大化 `φ(G_t + reveal(v))_y`（正确类别概率）的那个 v。

这种 oracle 在概念上是最优的（因为它知道标签），但会高估可达上限，因为它利用了标签信息。它给出的是**绝对上界**。

**Oracle Type II（梯度感知型）—— 用于训练**：

使用 T1（梯度归因 teacher），它不知道标签本身（梯度通过正确类别 logit 计算，但正确类别是从标签反推的），而是基于模型内部表征。这种 oracle 给出的上界更接近"可学习的上界"。

**建议**：报告 Type II 作为**可学习经验上界**，Type I 作为**绝对上界**（仅用于分析，非训练信号）。

### 5.4 头空间（Headroom）分析

```
Headroom_oracle = Acc(Oracle Type II, B) - Acc(BFS, B)   # BFS vs oracle 的差距
Headroom_random = Acc(Oracle Type II, B) - Acc(random, B) # random vs oracle 的差距

解释：
  Headroom_oracle > 5%  → 有显著提升空间 → 推进 M5
  Headroom_oracle < 2%  → 固定策略已接近最优 → 重新评估方案
```

### 5.5 无标签泄露的严格上界

如果担心 Oracle Type II 仍然间接泄露了标签信息（因为梯度计算时指定了正确类别），可以构建一个**完全无标签的 oracle**：

**PageRank + 梯度混合（T4 的 α=0.7 版本）**：梯度信号来自 GNN 模型的任意预测（而非正确类别），加上强结构先验。这保证了 test-time 的 oracle 不接触标签。

两套 oracle 分别用于：

- **训练 oracle**：T1/T2/T4，可使用标签 → 生成 BC 训练轨迹
- **上界估计 oracle**：T4 α=0.7（无标签梯度版）→ 公平估计
- **报告**：同时报告两个值，展示标签泄露的实际影响

---

## 六、轨迹蒸馏可行性判断

### 6.1 决策矩阵


| 条件                                 | 判断                   | M5 路线                   |
| ------------------------------------ | ---------------------- | ------------------------- |
| Oracle Type II 显著 > BFS (gap > 5%) | 轨迹蒸馏值得做         | BC 预训练 → 端到端微调   |
| Oracle Type II 略 > BFS (gap 2-5%)   | 可以做，但预期收益有限 | 仅 BC（M5a），不进入 M5b  |
| Oracle Type II ≈ BFS (gap < 2%)     | BC 可能无法超越 BFS    | 直接端到端训练（跳过 BC） |
| 同图多条 teacher 轨迹高一致性        | BC 学习信号强          | BC + label smoothing      |
| 同图多条 teacher 轨迹低一致性        | 目标不明确             | 端到端训练为主，BC 为辅   |

### 6.2 定量指标

```
蒸馏可行指数 (Distillation Feasibility Index, DFI) =

  0.4 × (Oracle_Acc - BFS_Acc) / max(BFS_Acc, ε)     # 相对提升
+ 0.3 × Stability_score                               # teacher 一致性
+ 0.3 × (1 - Acc_variance_across_trajectories)        # 轨迹质量方差低

DFI > 0.5 → 轨迹蒸馏可行且推荐
DFI 0.3-0.5 → 轨迹蒸馏可行，需要额外过滤
DFI < 0.3 → 推荐跳过 BC，直接端到端训练
```

---

## 七、M4 阶段软件架构

### 7.1 目录结构

```
mycode/
├── m4/                               # M4 新增模块
│   ├── __init__.py
│   ├── teachers/                     # Teacher 实现
│   │   ├── __init__.py
│   │   ├── base.py                   #   Teacher 抽象接口
│   │   ├── gradient_teacher.py       #   T1: 梯度归因 teacher
│   │   ├── info_gain_teacher.py      #   T2: 信息增益 teacher
│   │   ├── structural_teacher.py     #   T3: 结构 teacher
│   │   └── hybrid_teacher.py         #   T4: 混合 teacher
│   ├── trajectory_generator.py       # 轨迹生成器（teacher → 轨迹）
│   ├── trajectory_scorer.py          # 轨迹评分器
│   ├── trajectory_filter.py          # 轨迹过滤器
│   ├── oracle_estimator.py           # Oracle 上界估计
│   ├── feasibility_analyzer.py       # 蒸馏可行性分析
│   └── generate_trajectories.py      # 主脚本：生成全部 M4
│                                      #   产出
├── active_graph_reader/              # M3 现有代码（只读）
│   └── ...
└── data/                             # M4 数据产出
    └── trajectories/
        ├── raw/                      #   原始轨迹
        ├── scored/                   #   评分后轨迹
        └── filtered/                 #   过滤后轨迹（供 M5 使用）
```

### 7.2 Teacher 接口

```python
# m4/teachers/base.py

class Teacher(ABC):
    """生成图阅读轨迹的 teacher 抽象接口。

    每个 teacher 实现两个核心方法：
    - score_nodes: 对图中所有节点给出重要性分数（利用全图信息）
    - generate_trajectory: 模拟 reveal 过程，生成阅读轨迹
    """

    @abstractmethod
    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: nn.Module,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """返回每个节点的"重要性"分数 (shape: |V|, )."""
        ...

    def generate_trajectory(
        self,
        graph: dgl.DGLGraph,
        score: torch.Tensor,
        seed_node: int,
        budget: int,
    ) -> Trajectory:
        """从 seed 出发，每次选择 frontier 中分数最高的节点揭示。"""
        ...
```

### 7.3 Trajectory 数据结构

```python
@dataclass
class Trajectory:
    graph_idx: int           # 图索引
    seed_node: int           # 种子节点
    budget_used: int         # 使用的预算
    reveal_order: list[int]  # 揭示节点顺序 [v_1, v_2, ..., v_B]
    scores_per_step: list[float]      # 每步的 teacher 分数
    confidence_per_step: list[float]  # 每步的φ预测置信度
    entropy_per_step: list[float]     # 每步的φ输出熵
    final_pred: int                   # 最终预测类别
    is_correct: bool                  # 预测是否正确
    teacher_type: str                 # 哪一个 teacher 生成的
    teacher_params: dict              # teacher 的具体参数
```

### 7.4 主脚本流程

```python
# m4/generate_trajectories.py

def main():
    # 1. 加载冻结的尺子（M3 产出）
    classifier = load_frozen_classifier("best_model.pt")

    # 2. 初始化所有 teacher
    teachers = [
        GradientTeacher(classifier),              # T1
        StructuralTeacher(method='degree'),       # T3-degree
        StructuralTeacher(method='pagerank'),     # T3-PageRank
        HybridTeacher(classifier, alpha=0.3),     # T4
        HybridTeacher(classifier, alpha=0.7),     # T4
    ]

    # 3. 对训练集每张图生成轨迹
    generator = TrajectoryGenerator(teachers, classifier, subgraph_builder)
    raw_trajs = generator.generate_all(train_indices, budgets=[5,10,15,20])

    # 4. 评分
    scorer = TrajectoryScorer(classifier)
    scored_trajs = scorer.score_all(raw_trajs)

    # 5. 过滤
    filter = TrajectoryFilter(top_k=3)
    filtered_trajs = filter.filter(scored_trajs)

    # 6. Oracle 上界估计
    oracle = OracleEstimator(classifier)
    upper_bounds = oracle.estimate(filtered_trajs)

    # 7. 可行性分析
    feasibility = FeasibilityAnalyzer()
    dfi_report = feasibility.analyze(upper_bounds, filtered_trajs)

    # 8. 保存产出
    save_trajectories(filtered_trajs, "data/trajectories/filtered/")
    save_report(dfi_report, "m4_report.json")
```

---

## 八、关键风险与缓解

### 风险 1：梯度归因 Teacher 对 PPA 无节点特征图的适用性

**风险**：PPA 无节点特征，所有节点的初始化嵌入相同（共享 `Embedding(1, 300)`）。梯度归因是否仍能区分不同节点的重要性？

**分析**：可以。虽然初始嵌入相同，但不同节点在图中的位置不同（度、邻居结构、参与的边特征），导致它们在 GIN 消息传递后获得不同的中间表示。梯度归因正是利用这些因拓扑位置不同而产生的表示差异。**本质上是衡量每个节点在图结构中扮演的信息汇聚角色**。

**验证方法**：检查不同节点的 `||∂ŷ_c/∂h_v||` 是否确实有差异（标准差 > 0）。

**缓解**：如果梯度方差过小（所有节点分数几乎相同），切换到 T2（信息增益 teacher）作为主要 teacher。

### 风险 2：PPA 图较大，信息增益 Teacher 计算过慢

**风险**：PPA 图平均约 300 节点，每步 frontier 可能达 20-50 候选。B=10 时，每张图需 ~300 次前向传播。训练集 158K 张图，总前向次数过大。

**缓解**：

- 仅在验证集（~20K 张图）上使用 T2，用于验证 T1 的质量
- 对训练集使用 T1 + T3 + T4（计算轻量）
- 如果 T2 轨迹显著优于 T1，可考虑对训练集随机抽样 20-30K 张图使用 T2

### 风险 3：Teacher 轨迹分歧大

**风险**：不同 teacher 对不同节点给出高度分歧的重要性排序，导致目标不一致。

**缓解**：

- 先用 Stability 指标量化分歧程度
- 如果 Stability 低，考虑只使用效果最好的 1-2 个 teacher（而非所有 5 个）
- BC 训练时增加 label smoothing (ε=0.2) 以容忍 teacher 的不确定性

### 风险 4：Oracle 上界估计过高，产生虚假期望

**风险**：Oracle Type I（标签感知型）给出过于乐观的上界，偏离 M5 实际可达的性能。

**缓解**：

- 始终以 Oracle Type II 作为"可学习上界"的主要指标
- Oracle Type I 仅作为参考（标注"理论最大，不可达"）
- M5 的目标设定为 Oracle Type II 的 80-90%（而非 100%）

---

## 九、文献支撑

### 核心参考


| 论文                                                                          | 与 M4 的关系                                                                                                                                                   |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Policy-Learn** (Bevilacqua et al., ICLR 2024)                               | 最接近的工作：Gumbel-Softmax 端到端训练子图选择策略。M4 的 teacher 轨迹生成可以看作 Policy-Learn 的"教师版"，M5 将使用 Policy-Learn 的 Gumbel-Softmax 训练方法 |
| **GNNExplainer** (Ying et al., NeurIPS 2019)                                  | 节点/边重要性评估方法：互信息最大化找到关键子图。T1 梯度 teacher 的梯度归因是 GNNExplainer 体系的一个子方法                                                    |
| **PGExplainer** (Luo et al., NeurIPS 2021 / TPAMI 2024)                       | 参数化可归纳解释器：用共享神经网络生成解释。M4 的 teacher 可以看作"手工设计的 explainer"，M5 的 student 则是"学习到的 explainer-like reader"                   |
| **Uncertainty for Active Learning on Graphs** (Fuchsgruber et al., ICML 2024) | 验证了贝叶斯不确定性在图节点选择中的有效性。T2 信息增益 teacher 与其节点选择策略理念一致                                                                       |
| **Grad-CAM** (Selvaraju et al., ICCV 2017) → GNN 版 (Pope et al., CVPR 2019) | 梯度归因方法在 GNN 上的适用性验证                                                                                                                              |
| **Recognizing Predictive Substructures with SIB** (Yu et al., TPAMI 2024)     | 信息瓶颈子图选择：识别"预测子结构"。T2 teacher 的内在逻辑与 SIB 一致——选择对分类最具预测性的节点集合                                                         |

### 背景与对比


| 论文                                                                             | 与 M4 的关系                                                               |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **GALMI** (Huang et al., Expert Systems with Applications, 2024)                 | 用 RL + 互信息学习节点选择策略。M5 的端到端训练路径的参考                  |
| **DiffusAL** (2023, ECML PKDD)                                                   | 多准则节点选择（不确定性+多样性+重要性）。M4 的多 teacher 多样性策略的参考 |
| **Efficient Subgraph GNNs by Learning Effective Selection Policies** (ICLR 2024) | Policy-Learn 原文——M5 端到端训练的架构参考                               |

---

## 十、M4 产出清单与验收标准

### 代码产出


| 产出          | 文件                          | 验收标准                                                       |
| ------------- | ----------------------------- | -------------------------------------------------------------- |
| Teacher 模块  | `m4/teachers/`                | 每种 teacher 至少输出 100 条轨迹，且平均 Acc > random baseline |
| 轨迹生成器    | `m4/trajectory_generator.py`  | 支持多 teacher × 多预算 × 多种子，断点续传                   |
| 轨迹评分器    | `m4/trajectory_scorer.py`     | 4 维度评分，批量处理                                           |
| 轨迹过滤器    | `m4/trajectory_filter.py`     | top-K 保留 + 多样性保证                                        |
| Oracle 估计器 | `m4/oracle_estimator.py`      | 输出 Oracle Type I / Type II 上界                              |
| 可行性分析    | `m4/feasibility_analyzer.py`  | 输出 DFI 指数 + 可视化 + M5 推荐路线                           |
| 主脚本        | `m4/generate_trajectories.py` | 一键运行，产出所有 M4 数据和报告                               |

### 数据产出


| 产出            | 位置                          | 格式                            |
| --------------- | ----------------------------- | ------------------------------- |
| 原始轨迹        | `data/trajectories/raw/`      | `.pt` 文件                      |
| 评分后轨迹      | `data/trajectories/scored/`   | `.pt` 文件（带 score 字段）     |
| 过滤后轨迹      | `data/trajectories/filtered/` | `.pt` 文件（供 M5 BC 训练使用） |
| Oracle 上界报告 | `m4_report.json`              | JSON                            |

### 分析产出


| 产出             | 内容                                                       |
| ---------------- | ---------------------------------------------------------- |
| 预算-准确率曲线  | random / BFS / maxdeg / T1 / T2 / T3 / T4 / Oracle Type II |
| 头空间分析       | 各预算下的 Headroom_oracle 和 Headroom_random              |
| DFI 蒸馏可行指数 | 数值 + 决策建议                                            |
| M5 路线推荐      | "BC 预训练 → Gumbel-Softmax 微调" 或 "直接端到端训练"     |

---

## 十一、执行时间线


| 步骤     | 内容                                                | 工期         | 依赖            |
| -------- | --------------------------------------------------- | ------------ | --------------- |
| 1        | 实现 Teacher 接口 + T1（梯度归因）+ T3（结构）      | 2-3 天       | M3 冻结尺子就绪 |
| 2        | 验证 T1 质量：在验证集上生成轨迹、评分、与 BFS 对比 | 1 天         | 步骤 1          |
| 3        | 实现 T4（混合）+ 轨迹多样性生成                     | 1 天         | 步骤 1          |
| 4        | 实现 T2（信息增益）+ 在验证子集上验证               | 2 天         | 步骤 1          |
| 5        | 轨迹评分 + 过滤                                     | 1 天         | 步骤 2-4        |
| 6        | Oracle 上界估计                                     | 1 天         | 步骤 5          |
| 7        | 可行性分析 + M5 路线决策                            | 1 天         | 步骤 6          |
| 8        | 全量训练集轨迹生成（可后台运行）                    | 1-2 天       | 步骤 5          |
| **总计** |                                                     | **10-12 天** |                 |

---

## 十二、与 reader-training-plan.md 的关系

本文档聚焦于 M4（轨迹收集 + 质量评分 + oracle 上界估计），是 reader-training-plan.md 定义的完整培训方案中的 **Stage 2**。reader-training-plan.md 进一步覆盖了：

- M5a：BC 预训练（使用本文档产出的过滤后轨迹）
- M5b：端到端 Gumbel-Softmax 训练 / REINFORCE 训练
- M5c：交替微调

当 M4 完成产出数据和 DFI 报告后，reader-training-plan.md 的 M5 部分将根据 DFI 分析结果进一步细化。

# Policy-Learn 论文精读报告

> **论文**: Efficient Subgraph GNNs by Learning Effective Selection Policies
> **作者**: Beatrice Bevilacqua, Moshe Eliasof, Eli Meirom, Bruno Ribeiro, Haggai Maron
> **发表**: ICLR 2024（CCF-A）
> **arXiv**: [2310.20082](https://arxiv.org/abs/2310.20082)
> **代码**: [github.com/beabevi/policy-learn](https://github.com/beabevi/policy-learn)
> **精读日期**: 2026-05-16（更新：2026-05-23）
> **与本项目关系**: 最接近的已有工作（预算约束下的迭代子图选择）

---

## 一、论文概要

### 1.1 核心问题

子图 GNN（Subgraph GNN）已被证明具有超越 WL 检验的表达能力，但需要处理大量子图（通常 O(n) 个），计算代价极高（时间复杂度 O(n²·d) vs 标准 MPNN 的 O(n·d)，其中 d 为最大节点度）。本文提出：**能否从大量候选子图中，学习选择一小部分最具信息量的子图？**

### 1.2 核心贡献

| 维度         | 贡献                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| **理论**     | 证明存在高效子图选择策略：对于 WL 无法区分的图族，只需少量子图即可完全区分 |
| **方法**     | 提出 Policy-Learn——一种迭代式学习子图选择策略的架构                      |
| **表达能力** | 证明 Policy-Learn 可以学到理论最优策略，而随机策略和先前的 OSAN 方法做不到 |
| **实验**     | 在 ZINC、Reddit-Binary、OGB 等多个基准上超越随机和 OSAN，逼近全子图袋性能  |

---

## 二、关键定义与前置知识

### 2.1 子图 GNN（Subgraph GNN）形式化定义

**定义（Subgraph GNN）**：给定输入图 $G = (A, X)$，子图 GNN 首先根据预定义的生成策略将 $G$ 变换为一个子图袋（bag / multiset）$\mathcal{B}_G = \{\{S_1, S_2, \dots, S_m\}\}$，其中 $S_i = (A_i, X_i)$ 表示第 $i$ 个子图。然后，通过堆叠（节点-和子图-）置换等变层处理 $\mathcal{B}_G$，最后通过池化函数获得图级或节点级表示用于最终预测。

子图 GNN 的两个关键设计维度：
1. **子图生成策略（Generation Policy）**：决定如何从原图生成子图
2. **等变架构（Equivariant Architecture）**：决定子图间如何共享信息

### 2.2 节点标记策略（Node Marking, NM）

**定义（Node Marking）**：每个子图通过标记原图中一个节点生成——保持图结构不变，仅在节点特征中拼接该节点的 one-hot 指示向量。

具体地，对于每个子图 $S_i = (A_i, X_i)$，存在节点 $v_i$ 使得：
- $A_i = A$（邻接矩阵不变）
- $X_i = X \oplus \chi_{v_i}$（特征通道维拼接 one-hot 向量 $\chi_{v_i}$）

称 $v_i$ 为子图 $S_i$ 的**根节点（root）**，等价记作 $S_{v_i}$。

**关键性质**：由于每个子图恰好标记一个节点，节点与子图之间存在**双射（bijection）**，因此 NM 策略下子图总数 $m = n$。NM 被选择是因为它泛化了多种其他策略（Zhang et al., 2023），任何保持节点-子图双射的其他节点基准策略均可等价使用。

**注**：严格来说 NM 不生成传统意义上的"子图"，而是生成原始图的"标记版本"（marked versions），但文献中通常将其视为子图（Papp & Wattenhofer, 2022）。

### 2.3 DS-GNN（Deep Sets GNN）定义

**定义（DS-GNN）**：DS-GNN（Bevilacqua et al., 2022）是子图 GNN 的一种具体架构，其核心特征是：
- **每个子图独立处理**：对袋子中的每个子图分别进行消息传递（使用共享的 MPNN 编码器）
- **跨子图池化**：通过一个置换不变集合函数（如 Deep Sets 的 sum/mean 池化）聚合所有子图的表示
- **对称群**：对 $S_m \times S_n$ 等变，即独立置换子图（$S_m$）和每个子图内的节点（$S_n$）

DS-GNN 是最轻量的子图 GNN 变体，在 Policy-Learn 中同时用作选择网络和预测网络的基础架构。

### 2.4 DSS-GNN 与子图 GNN 架构层级

为完整理解 DS-GNN 的定位，以下是主要子图 GNN 架构的关系：

| 架构 | 子图间信息共享 | 对称群 | 表达能力上界 |
|------|----------------|--------|-------------|
| DS-GNN | 无（独立处理） | $S_m \times S_n$ | 3-WL |
| DSS-GNN | 有（同节点跨子图聚合） | $S_n$（单个） | 3-WL |
| SUN (Frasca et al., 2022) | 统一框架，泛化以上 | $S_n$ | 3-WL |
| GNN-SSWL (Zhang et al., 2023) | 最强节点基准变体 | $S_n$ | 3-WL |

**SUN 的关键洞察**：节点基准策略诱导了节点与子图间的双射，因此正确的对称群是单个 $S_n$（而非 $S_m \times S_n$）——这是一个更小的群，允许更有表达力的操作。

### 2.5 完整子图袋与选择策略

**定义（完整子图袋）**：在 NM 策略下，$\mathcal{B}_G = \{\{S_1, ..., S_n\}\}$，共 $n$ 个子图。

**定义（子图选择策略 $\pi$）**：$\pi(G)$ 以输入图 $G$ 为输入，返回 $\mathcal{B}_G$ 的一个子集 $\mathcal{B}_G^T \subset \mathcal{B}_G$，其中 $\mathcal{B}_G^T = \{\{G, S_{v_1}, S_{v_2}, \dots, S_{v_T}\}\}$，共包含原始图 $G$ 和 $T$ 个被选中的子图。

**目标**：学习策略 $\pi(G)$，使下游预测网络 $g$ 基于 $\mathcal{B}_G^T$ 的输出 $\hat{y} = g(\mathcal{B}_G^T)$ 尽可能准确。

---

## 三、方法详解

### 3.1 问题形式化

**输入**：无向图 $G = (A, X)$，$n$ 个节点，$A \in \mathbb{R}^{n \times n}$ 是邻接矩阵，$X \in \mathbb{R}^{n \times c}$ 是节点特征矩阵。

**子图生成**：采用节点标记（Node Marking）策略。每个子图 $S_i$ 由标记一个节点 $v_i$ 生成——保持图结构不变，仅在特征中追加该节点的 one-hot 指示向量：

- $A_i = A$（邻接矩阵不变）
- $X_i = X \oplus \chi_{v_i}$（特征拼接 one-hot）

**完整子图袋**：$\mathcal{B}_G = \{\{ S_1, ..., S_n \}\}$，共 $n$ 个子图。

**目标**：学习策略 $\pi(G)$，从 $n$ 个子图中选出 $T+1$ 个（含原始图 $G$），用选出的 $\mathcal{B}_G^T$ 做下游预测。

### 3.2 Policy-Learn 总体架构

Policy-Learn 由两个子图 GNN 端到端训练：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Policy-Learn                              │
│                                                                  │
│  ┌──────────────────────┐         ┌──────────────────────────┐  │
│  │  选择网络 f (DS-GNN)  │ ──迭代──→│  预测网络 g (DS-GNN)       │  │
│  │                       │   T 步   │                          │  │
│  │  ℬ_G⁰ = {G}          │          │  输入: ℬ_Gᵀ              │  │
│  │     ↓                 │          │  输出: ŷ                  │  │
│  │  p₁ = f(ℬ_G⁰)        │          │                          │  │
│  │  v₁ ~ GumbelSoftmax   │          └──────────────────────────┘  │
│  │  ℬ_G¹ = ℬ_G⁰ ∪ {Sᵥ₁} │                                       │
│  │     ↓                 │                                       │
│  │  ... 重复 T 次         │                                       │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 迭代选择过程（Algorithm 1）

```
Algorithm 1: Policy-Learn 前向传播

1:  初始化子图袋: ℬ_G⁰ := {{ G }}                    ▷ 只有原图
2:  for t = 0, ..., T-1 do
3:      p_{t+1} = f(ℬ_Gᵗ)                             ▷ 选择网络输出 n 个节点的 logits
4:      v_{t+1} ~ GumbelSoftmax(p_{t+1})               ▷ 可微采样选出一个节点
5:      ℬ_G^{t+1} = ℬ_Gᵗ ∪ {{ S_{v_{t+1}} }}          ▷ 将标记了 v_{t+1} 的子图加入袋中
6:  end for
7:  π(G) = ℬ_Gᵀ                                        ▷ 最终选出 T+1 个子图
8:  ŷ = g(ℬ_Gᵀ)                                        ▷ 预测网络消费最终子图袋
```

**关键设计——顺序采样**：每一步的选择依赖之前已选的子图。这使策略能够：
- 避免重复选择相同类型的节点
- 动态调整概率分布（如将已标记组件中的节点概率设为零）
- 在 $(n, \ell)$-CSL 图族中，能从不同连通分量选根节点

### 3.3 选择网络 f 的详细架构

选择网络是 DS-GNN，每个子图独立做消息传递后跨子图池化。

**子图编码器**（对每个子图 $s = 0, \dots, t$）：

两路独立的消息传递（各 $L_1$ 层 GIN）：

1. **标记更新（$f_p$）**：跟踪标记信息在子图上的传播
   $$p_s^{(\ell)} = f_p^{(\ell)}(p_s^{(\ell-1)}, A_s)$$

2. **特征更新（$f_h$）**：标记拼接到特征后做消息传递
   $$h_s^{(\ell)} = f_h^{(\ell)}(h_s^{(\ell-1)} \oplus p_s^{(\ell-1)}, A_s)$$

3. **合并**：
   $$h_s = p_s^{(L_1)} + h_s^{(L_1)}$$

4. **跨子图池化 + MLP 得到节点选择概率**：
   $$p_{t+1} = \text{MLP}\left(\text{pool}\left(\{\{ h_0, \dots, h_t \}\}\right)\right) \in \mathbb{R}^n$$
   
   其中 pool 为 mean 聚合，MLP 将通道降为 1，得到每个节点被选中的 logits。

**初始化**：
- $p_s^{(0)} = \chi_{v_s}$（该子图根节点的 one-hot 向量）
- $h_s^{(0)} = X$（原始节点特征）

**架构细节**：
- 消息传递使用 **GIN 层**（Xu et al., 2019）
- MLP 由 **2 层线性层 + ReLU** 组成
- 使用 **Batch Normalization**，但 `track_running_stats=False`（因选择网络中每步子图数变化）
- 通常使用**残差连接**
- 跨子图池化使用 **mean 聚合**

### 3.4 预测网络 g

与选择网络结构相同（DS-GNN + GIN 层），但**权重独立**。以最终子图袋 $\mathcal{B}_G^T$ 为输入，经 $L_2$ 层消息传递后：
- 先 mean 池化各子图内节点得子图表示
- 再 mean 池化所有子图表示得图表示
- 最后 MLP 输出预测

### 3.5 可微采样：Straight-Through Gumbel-Softmax

使用 Straight-through Gumbel-Softmax 估计器（Jang et al., 2016; Maddison et al., 2017），使离散的节点选择可导。

**Gumbel-Max 技巧**（精确离散采样）：
对于类别概率分布 $\pi_1, \dots, \pi_k$，离散样本为：
$$z = \text{one-hot}\left(\arg\max_i [g_i + \log \pi_i]\right)$$
其中 $g_i \sim \text{Gumbel}(0, 1)$，即 $g_i = -\log(-\log(u))$，$u \sim \text{Uniform}(0, 1)$。

**Gumbel-Softmax 松弛**（连续可导近似）：
$$y_i = \frac{\exp((\log \pi_i + g_i) / \tau)}{\sum_j \exp((\log \pi_j + g_j) / \tau)}$$

其中温度 $\tau$ 控制离散程度：
- $\tau \to 0$：趋近 one-hot（离散）
- $\tau = 1$：标准 softmax
- $\tau \to \infty$：趋近均匀分布

**Straight-Through (ST) 估计器**：
- **前向传播**：$z = \arg\max y$（硬 one-hot）
- **反向传播**：$\nabla_\theta z \approx \nabla_\theta y$（假装离散化未发生）

PyTorch 实现模式：
```python
y_soft = softmax((logits + gumbel_noise) / tau, dim=-1)
y_hard = one_hot(argmax(y_soft), num_classes)
y = (y_hard - y_soft).detach() + y_soft  # ST estimator
```

$\chi_{v_{t+1}} = \text{GumbelSoftmax}(p_{t+1}) \in \{0,1\}^n$

**温度调优**：$\tau \in \{0.33, 0.66, 1, 2\}$

**训练时的探索机制**：
- 在节点概率分布上应用 **dropout**（概率 $\in \{0, 0.3, 0.5\}$），防止对特定节点过度自信
- $T \geq 5$ 时额外**屏蔽已选子图对应的节点**，避免重复选择
- **推理时**用 $\arg\max$ 替代采样（确定性选择）

### 3.6 训练配置

整个系统端到端训练，使用下游任务损失（MAE/交叉熵）。梯度通过 Gumbel-Softmax 回传到选择网络 $f$。

**通用配置**：
- 框架：PyTorch + PyTorch Geometric
- 选择网络和预测网络使用**独立的 Adam 优化器**
- 预测网络：学习率 0.001，每 300 epoch 衰减 0.5
- 选择网络：学习率 0.001，**无衰减**（保持探索能力）
- 最大 epoch：多数数据集 1000（molhiv/moltox21 为 500）

**各数据集超参数**：

| 数据集 | 预测网络层数 | 嵌入维度 | 残差连接 | 其他 |
|--------|-------------|---------|---------|------|
| ZINC-12k | 6 | 64 | 是 | T=2→1000ep, T=5→600ep |
| Reddit-Binary | 4 | 32 | 否 | 10-fold CV, lr tuned in {0.01, 0.001} |
| OGB-molhiv | 2 | 64 | — | 恒定学习率，层间 dropout 0.5 |
| OGB-moltox21/molbace | 5 | 300 | — | 层间 dropout 0.5，batch size Full=32 |
| OGB-molesol | {2,3} | 64 | — | — |
| Alchemy | 6 | 64 | — | 类似 ZINC 设置 |
| CSL/EXP | 6 | 64 | EXP 有 | CSL lr={0.01,0.001}, EXP lr=0.0001 |

---

## 四、理论结果

### 4.1 CSL 图与 $(n, \ell)$-CSL 图族

**定义 1（CSL 图，Murphy et al., 2019）**：设 $n, k$ 满足 $n$ 与 $k$ 互质且 $k < n-1$。CSL($n$, $k$) 是无向图，节点标记为 $\{0, \dots, n-1\}$，边集由环边 $(j, j+1)$ 和跳跃边（skip links）组成。跳跃边序列递归定义：$s_1 = 0$, $s_{i+1} = (s_i + k) \text{ mod } n$，边为 $(s_i, s_{i+1}) \in E$。

CSL 图是 4-正则图（环边 2 度 + 跳跃边 2 度），已知是 1-WL 不可区分的，但通过节点标记子图袋可区分。

**定义 2（$(n, \ell)$-CSL 图）**：$(n, \ell)$-CSL$(n, (k_1, \dots, k_\ell))$ 是由 $\ell$ 个不相连的非同构 CSL 子图组成的图，每个 CSL 子图有 $n$ 个节点。总节点数为 $\ell \cdot n$。$\ell$ 的最大值取决于 $n$（因 CSL 图需不同 $k$ 值，且 $k$ 须与 $n$ 互质）。

### 4.2 核心定理

**定理 1（$(n, \ell)$-CSL 图族 WL 不可区分）**：设 $\mathcal{G}_{n,\ell}$ 为所有非同构 $(n, \ell)$-CSL 图构成的族。则 $\mathcal{G}_{n,\ell}$ 中所有图两两 1-WL 不可区分。

*证明思路*：任意两个 $(n, \ell)$-CSL 图中所有节点度均为 4（因每个节点属于一个 CSL 子图），一轮 WL 后所有节点颜色相同，算法终止，无法区分。

**命题 2（存在高效 $\pi$ 完全识别 $(n, \ell)$-CSL 图）**：节点标记子图选择策略 $\pi$ 能完全识别 $\mathcal{G}_{n,\ell}$ 中任意图 $G$ 的同构类型，**当且仅当**其子图袋在每个 CSL 子图中至少有一个标记。

*充分性证明*：若每个 CSL 子图至少有一个标记节点，则对应的标记子图足以识别该 CSL 子图的同构类型（Cotta et al., 2021, Theorem 2：标记任意节点足以区分任意 CSL 图对）。$\ell$ 个直方图集合即可识别所有 $\ell$ 个 CSL 子图的同构类型。

*必要性证明*（逆否命题）：若存在某个 CSL 子图未被任何标记节点覆盖，则可构造仅在未覆盖子图不同的非同一图 $G'$，策略无法区分 $G$ 和 $G'$。

**命题 3（随机策略无法高效识别 $(n, \ell)$-CSL 图）**：对 $\mathcal{G}_{n,\ell}$ 中任意图，随机均匀采样 $\ell$ 个子图能成功识别同构类型的概率为 $\ell! / \ell^\ell$，当 $\ell \to \infty$ 时趋近于 0。

**引理 6（随机策略的期望子图数——优惠券收集问题）**：随机策略完全识别所需的期望子图数 $T = \Theta(\ell \ln \ell)$。

*证明思路*：问题等价于优惠券收集问题（Coupon Collector Problem）。设 $t_i$ 为已收集 $i-1$ 个不同 CSL 子图后收集第 $i$ 个所需次数，$\mathbb{E}[t_i] = \frac{\ell}{\ell-i+1}$。由期望线性：
$$\mathbb{E}[T] = \sum_{i=1}^{\ell} \frac{\ell}{\ell-i+1} = \ell \cdot H_\ell = \ell \cdot \Theta(\ln \ell) = \Theta(\ell \ln \ell)$$

### 4.3 Policy-Learn 的表达能力定理

**定理 4（Policy-Learn 可识别 $\mathcal{G}_{n,\ell}$）**：存在选择网络 $f$ 和预测网络 $g$ 的权重，使得 Policy-Learn 以 $T = \ell$ 可识别 $\mathcal{G}_{n,\ell}$ 中任意图 $G$ 的同构类型。

*证明思路*：选择网络可实现策略 $\pi$：返回 $\ell$ 个子图，每个子图在不同 CSL 子图中标记恰好一个节点。
具体地：
1. DS-GNN 选择网络可区分"已标记 CSL 子图中的节点"与"其余节点"（因标记信息通过 MP 在子图内传播）
2. 将已标记组件的节点概率设为零，其余节点保持均匀概率
3. 因选择网络比 MPNN 更有表达力（DS-GNN > MPNN），它能做出这种区分
4. 顺序采样确保每步可基于当前袋状态动态调整

由命题 2 的充分性条件，满足该条件的策略即可完全识别。

**定理 5（OSAN 无法高效识别 $\mathcal{G}_{n,\ell}$）**：对于 $\mathcal{G}_{n,\ell}$ 中任意图，OSAN（Qian et al., 2022）以 $T = \ell$ 成功识别同构类型的概率为 $\ell! / \ell^\ell$，随 $\ell$ 增大趋近于 0——等同于随机策略。

*证明思路*（两个架构差异导致）：
1. OSAN 的选择网络是 **MPNN**（表达能力弱于 DS-GNN），无法区分 WL 不可区分 CSL 子图中的所有节点
2. OSAN 的 $\ell$ 个子图**同时（one-shot）而非顺序（sequential）**采样——无法基于先前选择动态调整
3. 因此 OSAN 在 $\mathcal{G}_{n,\ell}$ 上的表现等价于均匀随机采样

**命题 7（Policy-Learn 可学到依赖局部结构的策略）**：Policy-Learn 还可以学到依赖局部图结构的策略，例如返回所有根节点度数大于指定阈值的子图（证明见附录 C）。

策略多样性的来源：
1. **DS-GNN 选择网络** 的增强表达能力（相较于 MPNN）
2. **顺序采样** 允许每步基于历史信息动态调整概率分布
3. 从 WL 不可区分的组件中"挑出"特定组件的能力

---

## 五、实验

### 5.1 数据集

| 数据集                     | 任务类型       | 图规模        | 说明           |
| -------------------------- | -------------- | ------------- | -------------- |
| ZINC-12k                   | 回归（MAE）    | 小分子        | 500k 参数预算  |
| Reddit-Binary              | 分类（准确率） | avg. 430 节点 | 全子图袋会 OOM |
| OGB-molesol/bace/tox21/hiv | 回归/分类      | 分子          | 标准 OGB 基准  |
| Alchemy                    | 回归           | 量子化学      | 消融实验       |
| CSL / EXP                  | 分类（准确率） | 合成数据      | 表达力验证     |
| Counting (3数据集)         | 回归（MAE）    | 合成数据      | 子结构计数     |

### 5.2 基线方法

对每个任务包含两个核心基线：
- **Random**：每 epoch 均匀随机采样 $T$ 个子图（使用与 Policy-Learn 相同的预测网络）
- **Full**：使用完整子图袋（所有 $n$ 个子图）

另包括 MPNN 基线（GCN, GIN, PNA）、全袋子图 GNN（NGNN, DS-GNN, DSS-GNN, GNN-AK+, SUN, GNN-SSWL+）、以及 OSAN（Qian et al., 2022）。

### 5.3 ZINC-12k 结果

| 方法                      | T   | ZINC (MAE ↓)      |
| ------------------------- | --- | ------------------ |
| GIN                       | —   | 0.163 ± 0.004     |
| PNA                       | —   | 0.133 ± 0.011     |
| CIN                       | —   | 0.079 ± 0.006     |
| GNN-SSWL+                 | —   | 0.070 ± 0.005     |
| Full                      | —   | 0.087 ± 0.003     |
| Random-OSAN               | 2   | 0.214 ± 0.007     |
| OSAN                      | 2   | 0.177 ± 0.016     |
| Random                    | 2   | 0.136 ± 0.005     |
| **Policy-Learn**          | 2   | **0.120** ± 0.003 |
| Random                    | 3   | 0.128 ± 0.004     |
| **Policy-Learn**          | 3   | **0.116** ± 0.008 |
| Random                    | 5   | 0.113 ± 0.006     |
| **Policy-Learn**          | 5   | **0.109** ± 0.005 |
| Random                    | 8   | 0.102 ± 0.003     |
| **Policy-Learn**          | 8   | **0.097** ± 0.005 |

**关键发现**：
- Policy-Learn 在所有 $T$ 值下稳定优于 Random
- $T$ 越小差距越大（$T=2$ 时 MAE 差距 0.016，$T=8$ 时差距 0.005）
- $T=8$ 时 0.097 已接近 Full 的 0.087
- OSAN 甚至弱于 Random（因其预测网络实现不同）

### 5.4 Reddit-Binary 结果

| 方法                 | T   | RDT-B (ACC ↑)   |
| -------------------- | --- | --------------- |
| GIN                  | —   | 92.4 ± 2.5     |
| CIN                  | —   | 92.4 ± 2.1     |
| Full                 | —   | OOM             |
| Random (DS-GNN)      | 2   | 91.3 ± 1.6     |
| Random (DSS-GNN)     | 2   | 92.7 ± 0.8     |
| Random               | 20  | 92.6 ± 1.5     |
| Random               | 2   | 92.4 ± 1.0     |
| **Policy-Learn**     | 2   | **93.0** ± 0.9 |

**关键发现**：
- Full 因 OOM 无法运行（avg. 430 节点），Policy-Learn 可以
- Policy-Learn $T=2$ 超越 Random $T=20$（93.0 vs 92.6）
- Policy-Learn $T=2$ 超越 Random DSS-GNN $T=2$（93.0 vs 92.7）

### 5.5 OGB 结果

| 方法              | T | molesol (RMSE↓) | moltox21 (AUC↑) | molbace (AUC↑) | molhiv (AUC↑) |
| ----------------- | - | --------------- | --------------- | -------------- | ------------- |
| GCN               | — | 1.114 ± 0.036  | 75.29 ± 0.69   | 79.15 ± 1.44  | 76.06 ± 0.97 |
| GIN               | — | 1.173 ± 0.057  | 74.91 ± 0.51   | 72.97 ± 4.00  | 75.58 ± 1.40 |
| OSAN              | 10| 0.984 ± 0.086  | N/A             | 72.30 ± 6.60  | N/A           |
| Full              | — | 0.847 ± 0.015  | 76.25 ± 1.12   | 78.41 ± 1.94  | 76.54 ± 1.37 |
| Random            | 2 | 0.951 ± 0.039  | 76.65 ± 0.89   | 75.36 ± 4.28  | 77.55 ± 1.24 |
| **Policy-Learn**  | 2 | **0.877** ± 0.029 | **77.47** ± 0.82 | **78.40** ± 2.85 | **79.13** ± 0.60 |
| Random            | 5 | 0.900 ± 0.032  | 76.62 ± 0.63   | 78.14 ± 2.36  | 77.30 ± 2.56 |
| **Policy-Learn**  | 5 | **0.883** ± 0.032 | **77.36** ± 0.60 | **78.39** ± 2.28 | **78.49** ± 1.01 |

**关键发现**：
- Policy-Learn 在所有 4 个 OGB 数据集上均优于 Random
- molesol 上从 0.951→0.877（$T=2$），显著逼近 Full 的 0.847
- molhiv 上 **79.13**（$T=2$）甚至超越 Full（76.54）——采样起到正则化效果
- OSAN 使用 $T=10$ 仍不及 Policy-Learn 的 $T=2$（除 molesol）
- molbace 上 Policy-Learn $T=2$ 几乎等于 Full（78.40 vs 78.41）

### 5.6 表达力验证（CSL / EXP）

在合成表达力基准上验证理论：
- **CSL**：Policy-Learn 以 $T = \ell$ 达到 100% 准确率，验证定理 4
- **EXP**：同样达到完美区分（100%），证明策略在非 CSL 图族上也有效

### 5.7 Alchemy 与 Counting 数据集

在 Alchemy 量子化学数据集上，Policy-Learn 同样优于随机基线和 OSAN。在子结构计数任务（Counting datasets）上，Policy-Learn 在 3 个数据集上与 Full 持平或接近，且显著优于 MPNN。

### 5.8 时间复杂度对比

| 数据集 | 图规模 (n) | Full 时间 | Policy-Learn (T=2) | 加速比 |
|--------|-----------|----------|-------------------|--------|
| Reddit-Binary | ~430 | OOM | 可行 | ∞ |
| ZINC-12k | ~23 | 基准 | 相近 | ~2-3x |

当 $T \ll n$ 时加速比可达到 ~$n$ 倍。

---

## 六、复杂度分析

### 6.1 高效实现说明

论文描述了等价的高效实现：选择网络每步仅处理**最近添加的子图**，复用之前已存储的子图表示，而非每步重新处理整个袋子。

### 6.2 时间复杂度

记 $\Delta_{\max}$ 为最大节点度，$n$ 为节点数，$T$ 为选中子图数：

- **选择网络 $f$**：$\mathcal{O}(T \cdot n \cdot \Delta_{\max})$
  - $T$ 次迭代，每次对一个子图进行一次 MPNN（$\mathcal{O}(n \cdot \Delta_{\max})$）
- **预测网络 $g$**：$\mathcal{O}((T+1) \cdot n \cdot \Delta_{\max})$
  - $T+1$ 个子图各做一次 MPNN
- **总计**：$\mathcal{O}(T \cdot n \cdot \Delta_{\max})$

**对比全袋子图 GNN**：$\mathcal{O}(n^2 \cdot \Delta_{\max})$

当 $T \ll n$ 时，Policy-Learn 约快 $n$ 倍。

### 6.3 空间复杂度

- **选择网络 $f$**：$\mathcal{O}(T \cdot n + (n + n \cdot \Delta_{\max}))$ — 存储 $T$ 个子图的节点特征 + 单次 MPNN 所需空间
- **预测网络 $g$**：$\mathcal{O}((T+1) \cdot (n + n \cdot \Delta_{\max}))$
- **总计**：$\mathcal{O}(T \cdot (n + n \cdot \Delta_{\max}))$

**对比全袋子图 GNN**：$\mathcal{O}(n \cdot (n + n \cdot \Delta_{\max}))$

**实例**：Reddit-Binary（$n \approx 430$, $T = 2$），$T \ll n$，时间和空间均大幅缩减。

---

## 七、消融实验与设计选择的启示

### 7.1 顺序 vs 一次性选择

理论分析（定理 4 vs 定理 5）证明了顺序选择的必要性：
- OSAN 的一次性选择等价于随机（在不可区分图上）
- Policy-Learn 的顺序选择使每步选择依赖历史信息

### 7.2 DS-GNN vs MPNN 做选择器

DS-GNN > MPNN 的表达力优势是 Policy-Learn > OSAN 的另一根源：
- MPNN 选择器在 WL 不可区分组件间"失明"
- DS-GNN 选择器可追踪标记传播，区分已标记/未标记组件

### 7.3 Gumbel-Softmax 温度与 Dropout

- **温度 $\tau$**：调优于 {0.33, 0.66, 1, 2}，控制离散化程度
- **Dropout on 概率分布**：防止模型过早对某节点产生 100% 置信度，保留探索能力
- **Mask 已选节点**（$T \geq 5$）：训练时屏蔽已选子图，避免重复，鼓励覆盖不同区域

### 7.4 选择网络与预测网络的解耦训练

- 选择网络用**独立优化器 + 无学习率衰减**，保持策略灵活性
- 预测网络用**学习率衰减**，稳定收敛

---

## 八、与本项目的对比分析

### 8.1 相似之处

| 维度         | Policy-Learn                          | 本项目（Budgeted Graph Reading） |
| ------------ | ------------------------------------- | -------------------------------- |
| **核心思想** | 从 n 个子图中迭代选择 T 个            | 从图 G 的节点/边中迭代选择 B 个  |
| **选择方式** | 学习型策略（DS-GNN + Gumbel-Softmax） | 启发式/学习型策略                |
| **预算约束** | 固定预算 T                            | 固定预算 B                       |
| **下游任务** | 图分类/回归                           | 图分类                           |
| **采样方式** | 顺序采样（每步依赖前步）              | 顺序揭示（每步依赖前步）         |
| **训练**     | 端到端                                | 教师-学生/模仿学习               |

### 8.2 关键差异

| 维度         | Policy-Learn                           | 本项目                                          |
| ------------ | -------------------------------------- | ----------------------------------------------- |
| **选择粒度** | 子图（从 n 个预生成的候选中选）        | 原子证据（节点/边/邻域）                        |
| **候选池**   | 固定（每个节点生成一个子图）           | 无预生成——直接从原始图揭示                    |
| **图可见性** | 选择网络看到完整图（只是"选哪个子图"） | 模型只看到部分观测 $G_{<b}$（更像真实"阅读"场景） |
| **初始状态** | 已有完整图 + n 个子图候选              | $G_0$ 仅知节点 ID，无边无特征                     |
| **研究焦点** | 子图 GNN 的表达能力与效率              | 证据获取——"在预算内学会往哪里看"              |
| **隐喻**     | 子图选择                               | 图阅读（类比 LLM agent 检索上下文）             |
| **方法阶段** | 教师-学生：先用启发式生成轨迹，再蒸馏  | 初始阶段：先建立启发式基线                      |

### 8.3 对我们的启示

1. **顺序选择优于一次性选择**：Policy-Learn 的理论分析证明，顺序采样使策略能适应已揭示的信息——这正是我们项目"图阅读"的核心直觉。
2. **DS-GNN 做选择器可行**：子图 GNN 作为策略网络在实验中表现良好，可以作为我们 M4/M5 阶段学习型策略的参考架构。
3. **Gumbel-Softmax 是有效的可微采样方案**：从离散节点/边选择中回传梯度的标准做法。
4. **T 值的选取**：Policy-Learn 中 T=2 就能获得大部分收益，T=5 接近全袋性能。对我们来说，B 的选取需要类似的实验分析。
5. **随机基线很强**：Policy-Learn 的随机基线（均匀采样 T 个子图）并不弱——提醒我们简单的启发式基线需要认真对待。
6. **过自信问题**：Policy-Learn 训练中策略容易对特定节点产生过高置信度——通过概率分布上的 dropout 缓解。我们在训练学习型策略时也可能遇到类似问题。
7. **策略多样性**：除了"从不同组件各选一个"，Policy-Learn 还能学到依赖局部结构的策略（如按度选择，命题 7），这启示我们策略空间设计的丰富性。

### 8.4 我们的增量贡献空间

| 维度           | Policy-Learn 没做的          | 我们可以做的                              |
| -------------- | ---------------------------- | ----------------------------------------- |
| **问题设定**   | 完整图可见，只决定选哪个子图 | $G_0$ 几乎为空，逐步揭示——更像真实"阅读"  |
| **动机类比**   | 子图 GNN 效率                | LLM agent 上下文检索——论文叙事空间更大  |
| **选择对象**   | 预生成子图                   | 原始图证据（节点/边/特征）——更细粒度    |
| **初始方法**   | 端到端强化学习               | 教师-学生蒸馏——更可控、可解释           |
| **基线丰富度** | 只有随机 + OSAN              | 随机、BFS、度中心性、任务启发式——更多样 |

---

## 九、相关工作的更广阔背景

### 9.1 子图 GNN 发展脉络

```
Reconstruction GNN (Cotta et al., 2021)
    ↓
ESAN / DS-GNN & DSS-GNN (Bevilacqua et al., 2022, ICLR 2022 Spotlight)
    ↓
SUN 统一框架 (Frasca et al., 2022, NeurIPS 2022)
    ↓
GNN-SSWL 表达力层级 (Zhang et al., 2023, ICML 2023)
    ↓
OSAN 可学子图选择 (Qian et al., 2022)
    ↓
Policy-Learn (本文, ICLR 2024)
    ↓
k,l-WL Universal Framework (Zhou, Wang, Zhang, 2023, ICML 2023)
```

### 9.2 子图 GNN 的表达力层级

```
1-WL (= 2-WL)
  ≺ 1,1-WL (= SWL, ≡ DS-WL with NM policy)
    ≺ 1,2-WL
      ≺ 3-WL (= 2-FWL)
        ≺ ...
```

- DS-GNN（NM 策略）：介于 1-WL 和 3-WL 之间
- DSS-GNN/SUN：同样上界为 3-WL（但实际强于 DS-GNN）
- $k$-OSAN 上界为 $(k+1)$-WL
- 节点基准子图 GNN 的上界为 3-WL（Frasca et al., 2022; Qian et al., 2022）

---

## 十、结论

Policy-Learn 是当前与 Budgeted Graph Reading 最接近的已发表工作。它在"预算约束下的迭代子图选择"这一核心设定上与本项目高度一致。

**本质上**：Policy-Learn 做的是"从预生成的子图菜单中点菜"，而我们做的是"在原图上一步步探险"。前者的图是先验已知的，后者的图是逐步揭开的——这个差异既是增量的创新点，也是论文定位的锚点。

建议在 M2（相关工作图谱）中将 Policy-Learn 列为**最核心的对比对象**，重点阐述上述差异，作为本项目独特贡献的论证基础。

---

## 附录 A：论文关键公式

### 选择网络的核心公式

**标记传播**（每层 $\ell = 1...L_1$）：
$$p_s^{(\ell)} = f_p^{(\ell)}(p_s^{(\ell-1)}, A_s)$$

**特征传播**：
$$h_s^{(\ell)} = f_h^{(\ell)}(h_s^{(\ell-1)} \oplus p_s^{(\ell-1)}, A_s)$$

**合并**：
$$h_s = p_s^{(L_1)} + h_s^{(L_1)}$$

**节点选择概率**：
$$p_{t+1} = \text{MLP}(\text{pool}(\{\{ h_0, ..., h_t \}\}))$$

**可微采样**：
$$\chi_{v_{t+1}} = \text{GumbelSoftmax}(p_{t+1})$$

**Gumbel-Softmax 连续松弛**：
$$y_i = \frac{\exp((\log \pi_i + g_i) / \tau)}{\sum_j \exp((\log \pi_j + g_j) / \tau)}, \quad g_i \sim \text{Gumbel}(0,1)$$

---

## 附录 B：完整定理清单

| 编号 | 名称 | 结论 |
|------|------|------|
| 定义 1 | CSL 图 | 4-正则循环跳跃图，1-WL 不可区分 |
| 定义 2 | $(n, \ell)$-CSL 图 | $\ell$ 个不相连的非同构 CSL 子图组成 |
| 定理 1 | $(n, \ell)$-CSL WL 不可区分 | 族内所有图两两 1-WL 不可区分 |
| 命题 2 | 存在高效策略 | 完全识别 $\iff$ 每个 CSL 子图至少一个标记 |
| 命题 3 | 随机策略低效 | 成功概率 $\ell!/\ell^\ell \to 0$ |
| 引理 6 | 随机期望子图数 | $T = \Theta(\ell \ln \ell)$（优惠券收集） |
| 定理 4 | Policy-Learn 可识别 | 以 $T=\ell$ 即可识别（存在性构造） |
| 定理 5 | OSAN 等价随机 | 在 $\mathcal{G}_{n,\ell}$ 上成功概率同随机 |
| 命题 7 | 策略多样性 | Policy-Learn 还能学到依赖局部结构的策略 |

---

## 附录 C：参考资料

- Bevilacqua et al. (2022). Equivariant Subgraph Aggregation Networks. ICLR 2022.
- Cotta et al. (2021). Reconstruction for Powerful Graph Representations. NeurIPS 2021.
- Frasca et al. (2022). Understanding and Extending Subgraph GNNs by Rethinking Their Symmetries. NeurIPS 2022.
- Zhang et al. (2023). A Complete Expressiveness Hierarchy for Subgraph GNNs. ICML 2023.
- Qian et al. (2022). Ordered Subgraph Aggregation Networks. NeurIPS 2022.
- Jang et al. (2016) / Maddison et al. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR 2017.
- Murphy et al. (2019). Relational Pooling for Graph Representations. ICML 2019.
- Papp & Wattenhofer (2022). A Theoretical Comparison of Subgraph GNNs. ICML 2022.
- Zhou, Wang, Zhang (2023). A Universal Framework for More Expressive GNNs via the k,l-WL Hierarchy. ICML 2023.

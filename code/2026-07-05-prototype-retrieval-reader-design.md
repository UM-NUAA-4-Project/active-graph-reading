# 原型检索式 Reader 设计（v6）

> 日期：2026-07-05
> 状态：设计定稿，待实施
> 前置：POMDP v5 设计（`code/2026-6-12 reader分析和改进灵感.md`）、Phase 4 归因报告
> 核心转变：**移除 LSTM 循环信念累积，改为每步闭式高斯查询 + 可学习原型跟随**

---

## 一、设计背景

POMDP v5 采用 DRBPN 式 Dirichlet 证据 + LSTM 累积信念。该框架的信念更新依赖 Kalman 增益调制，存在冷启动陷阱与累积不稳定问题。v6 的核心思路是**绕过累积机制本身**：

- 信念不再由 LSTM 逐步累积，而是每步独立查询原型库得到
- 探索的收益隐含在"子图逐步变完整 → 每步嵌入质量提升 → 查询逐步变准"中
- 训练目标大幅简化：只剩策略网络 + 原型跟随，无 LSTM、无证据头、无 EDL 损失

这是一个从"参数化信念推断"到"检索式判别"的范式转换。

---

## 二、核心决策汇总


| #   | 决策           | 选择                                                                   |
| --- | -------------- | ---------------------------------------------------------------------- |
| D1  | 聚类数据来源   | 全图编码（不做子图聚类纠偏）                                           |
| D2  | 降维           | PCA，先跑定目标维度 d（95% 累计方差）                                  |
| D3  | 聚类算法       | 按 37 类分桶，每桶内独立 HDBSCAN（自主定簇数）                         |
| D4  | 信念机制       | 每步独立高斯查询，替代 LSTM；无 Submit-Early                           |
| D5  | 先验数学形式   | 高斯似然，对角协方差 + shrinkage                                       |
| D6  | 分类器能力问题 | 延后（终端分类直接用 b_T 的 argmax，不过 f_class）                     |
| D7  | 子图编码器     | 另训 2 层 GNN，从 f_class GNN 权重初始化，微调                         |
| D8  | 空间对齐       | 原型跟随：μ 可学习，随 g_φ 训练更新                                  |
| D9  | 终端分类       | ŷ = argmax_k b_T(k)（最后一步查询结果）                               |
| D10 | 策略输入       | b_t(37) + frontier_embed(128) + budget_ratio(1) + struct_v(3) = 169 维 |

---

## 三、整体架构

```
┌─────────────────────────────────────────────────────────┐
│  离线阶段（一次性）                                       │
│                                                         │
│  训练集全图 ──→ f_class GNN(冻结) ──→ 300维嵌入          │
│       └──→ PCA 拟合 ──→ 降维 d 维                        │
│            └──→ 按37类分桶 ──→ 每桶 HDBSCAN              │
│                 ──→ 原型库 {(μ⁰_{k,m}, Σ_{k,m}, π_k)}    │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓ 原型库作为可学习参数初始化
┌─────────────────────────────────────────────────────────┐
│  在线阶段（每步 t）                                      │
│                                                         │
│  子图_t ──→ g_φ(2层GNN, f_class初始化, 微调) ──→ 300维   │
│       └──→ 固定PCA ──→ ṽ_t (d维)                        │
│            └──→ 高斯查询原型库                            │
│                 b_t(k) ∝ π_k Σ_m w_{k,m} N(ṽ_t;μ_{k,m},Σ)│
│                 ──→ 策略网络(b_t, frontier, ...) ──→ 动作 │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓ 跑到预算 T
┌─────────────────────────────────────────────────────────┐
│  终端：ŷ = argmax_k b_T(k)                               │
└─────────────────────────────────────────────────────────┘
```

---

## 四、离线阶段：原型库构建

### 4.1 全图编码

使用已训练的 f_class GNN 编码器（val_acc 64-70%），对训练集全部 N≈8 万张图提取 300 维全图嵌入。

- 编码器冻结，不参与后续训练
- 输出：$E \in \mathbb{R}^{N \times 300}$，标签 $y \in \{1,...,37\}^N$

### 4.2 PCA 降维

对 $E$ 拟合 PCA，目标保留 95% 累计方差。

- 先跑 PCA，输出 explained variance ratio 曲线
- 取累计方差 ≥ 95% 的最小主成分数作为 d（预期 d ≈ 30~80）
- 保存 PCA 变换矩阵 $W_{\text{PCA}} \in \mathbb{R}^{300 \times d}$，**在线阶段固定不变**
- 降维后：$\tilde{E} = E \cdot W_{\text{PCA}} \in \mathbb{R}^{N \times d}$

**PCA 固定的理由**：在线阶段 $g_\phi$ 微调后输出分布会漂移，但 PCA 变换不变。粗对齐由 PCA 提供，精对齐由可学习原型 $\mu$ 补偿。这是一个分层设计：PCA 锁定降维方向，$\mu$ 锁定类内位置。

### 4.3 分桶 HDBSCAN 聚类

**按 37 个类别标签分桶**，每个类别桶内独立运行 HDBSCAN：

```
for k = 1 to 37:
    E_k = {ṽ_i : y_i = k}          # 该类所有降维嵌入
    clusters_k = HDBSCAN(E_k)       # 自主决定簇数
    μ⁰_{k,m} = centroid(cluster_m)  # m = 1,...,M_k
```

- HDBSCAN 自主决定簇数 $M_k$，无需预设 K
- 每个簇的质心作为原型初始值 $\mu^0_{k,m}$
- 少数类样本不足时，HDBSCAN 可能输出 $M_k = 1$（单簇）或 $M_k = 0$（全噪声）——此时该类退化为单原型或需特殊处理

**少数类处理**：

- 若 $M_k = 0$（全部标为噪声）：取该类全部样本均值作为单原型
- 若某类样本数 $n_k < 2d$：强制单原型 $M_k = 1$，$\mu^0_{k,1} = \bar{v}_k$

### 4.4 协方差估计与 shrinkage

每个原型 $(k,m)$ 的协方差采用**对角形式 + Ledoit-Wolf shrinkage**：

$$
\hat{\sigma}^2_{k,m,i} = \frac{1}{|C_{k,m}|-1}\sum_{\tilde{v} \in C_{k,m}} (\tilde{v}_i - \mu^0_{k,m,i})^2
$$

Ledoit-Wolf shrinkage 向各维度方差均值收缩：

$$
\sigma^2_{k,m,i} \leftarrow (1-\rho)\,\hat{\sigma}^2_{k,m,i} + \rho\,\bar{\sigma}^2_{k,m}
$$

其中 $\bar{\sigma}^2_{k,m} = \frac{1}{d}\sum_i \hat{\sigma}^2_{k,m,i}$，$\rho$ 由 Ledoit-Wolf 公式自动确定。

- **对角形式选择理由**：PCA 后主成分正交，维度间相关性已被消除，对角假设损失极小；参数量每类 $M_k \times d$，远低于全协方差 $M_k \times d^2/2$
- **shrinkage 理由**：少数类/小簇样本不足时，某维方差估计可能为 0 或极小，导致高斯似然数值爆炸；shrinkage 保证下界
- **协方差固定**：$\Sigma_{k,m}$ 在离线估计后**不参与原型跟随**（仅 $\mu$ 跟随），理由见 §6.3

### 4.5 类先验

$$
\pi_k = \frac{n_k}{N}
$$

即训练集中该类样本占比。少数类 $\pi_k$ 小，在查询时自然被压制。

### 4.6 原型库数据结构

```python
prototype_bank = {
    'mu':       Tensor(K_total, d),      # 所有原型均值，可学习
    'sigma2':   Tensor(K_total, d),      # 对角方差，固定
    'class_id': Tensor(K_total),         # 每个原型的类别标签
    'pi':       Tensor(37),              # 类先验，固定
    'pca_W':    Tensor(300, d),          # PCA 变换矩阵，固定
    'M_k':      Tensor(37),              # 每类原型数
}
# K_total = Σ_k M_k
```

---

## 五、在线阶段：每步高斯查询

### 5.1 子图编码器 $g_\phi$

- 结构：2 层 GNN（与 f_class GNN 同构），隐藏维 300，输出 300 维图嵌入
- 初始化：从 f_class GNN 权重拷贝
- 训练：微调（参与 BC/RL 训练）

**注意**：$g_\phi$ 的输入是**子图**（5-15% 节点的诱导子图），而 f_class 训练时输入是全图。从 f_class 初始化提供良好起点，微调让 $g_\phi$ 适应子图分布。

### 5.2 降维与查询

每步 t：

$$
v_t = g_\phi(\text{subgraph}_t) \in \mathbb{R}^{300}
$$

$$
\tilde{v}_t = v_t \cdot W_{\text{PCA}} \in \mathbb{R}^{d}
$$

$W_{\text{PCA}}$ 固定（离线拟合），不随训练更新。

### 5.3 混合高斯似然与信念计算

对类别 $k$，其似然为该类所有原型的混合高斯：

$$
p(\tilde{v}_t \mid c=k) = \sum_{m=1}^{M_k} w_{k,m}\,\mathcal{N}(\tilde{v}_t;\,\mu_{k,m},\,\Sigma_{k,m})
$$

其中混合权重 $w_{k,m} = 1/M_k$（均匀混合，或按簇大小比例）。

高斯密度（对角协方差展开）：

$$
\mathcal{N}(\tilde{v}_t;\mu_{k,m},\Sigma_{k,m}) = \frac{1}{(2\pi)^{d/2}\prod_i \sigma_{k,m,i}} \exp\!\left(-\frac{1}{2}\sum_{i=1}^{d}\frac{(\tilde{v}_{t,i}-\mu_{k,m,i})^2}{\sigma^2_{k,m,i}}\right)
$$

**数值稳定**：实际计算用 log 域：

$$
\log p(\tilde{v}_t \mid c=k) = \text{logsumexp}_m\!\left[\log w_{k,m} + \log\mathcal{N}(\tilde{v}_t;\mu_{k,m},\Sigma_{k,m})\right]
$$

$$
\log\mathcal{N} = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_i \log\sigma^2_{k,m,i} - \frac{1}{2}\sum_i \frac{(\tilde{v}_{t,i}-\mu_{k,m,i})^2}{\sigma^2_{k,m,i}}
$$

**后验信念**（Bayes）：

$$
b_t(k) = \frac{\pi_k \cdot p(\tilde{v}_t \mid c=k)}{\sum_{k'=1}^{37} \pi_{k'} \cdot p(\tilde{v}_t \mid c=k')}
$$

log 域计算后 softmax：

$$
b_t = \text{softmax}\!\left(\log\pi_k + \log p(\tilde{v}_t \mid c=k)\right)_{k=1}^{37}
$$

**关键性质**：

- 每步独立计算，无累积、无状态
- $b_t$ 仅依赖当前子图嵌入 $\tilde{v}_t$ 和当前原型库 $\{\mu,\Sigma,\pi\}$
- 探索的收益：子图扩张 → $\tilde{v}_t$ 质量提升 → $b_t$ 逐步变准

### 5.4 终端分类

$$
\hat{y} = \arg\max_k b_T(k)
$$

最后一步 $b_T$ 的 argmax，**不过 f_class classifier**。f_class 仅用于离线编码建库和 $g_\phi$ 初始化。

---

## 六、原型跟随机制

### 6.1 动机

$g_\phi$ 从 f_class 初始化后微调，其输出空间会漂移。原型库 $\mu$ 若固定在 f_class 空间，则 $\tilde{v}_t$（来自 $g_\phi$）与 $\mu$（来自 f_class）坐标系不一致，高斯查询失效。

原型跟随让 $\mu$ 作为可学习参数，随 $g_\phi$ 一起训练，始终对齐 $g_\phi$ 当前输出空间。

### 6.2 实现：可学习原型

```python
class PrototypeBank(nn.Module):
    def __init__(self, mu_init, sigma2, class_id, pi, pca_W):
        super().__init__()
        self.mu = nn.Parameter(mu_init.clone())     # (K_total, d) 可学习
        self.register_buffer('sigma2', sigma2)       # (K_total, d) 固定
        self.register_buffer('class_id', class_id)   # (K_total,) 固定
        self.register_buffer('pi', pi)               # (37,) 固定
        self.register_buffer('pca_W', pca_W)         # (300, d) 固定
        self.register_buffer('M_k', M_k)             # (37,) 固定

    def query(self, v_t):
        """v_t: (batch, 300) → b_t: (batch, 37)"""
        vtilde = v_t @ self.pca_W                    # (batch, d)
        # 对每个原型计算 log N
        diff = vtilde.unsqueeze(1) - self.mu.unsqueeze(0)  # (batch, K, d)
        mahal = (diff**2 / self.sigma2).sum(-1)             # (batch, K)
        log_det = self.sigma2.log().sum(-1)                 # (K,)
        log_N = -0.5 * (d * log(2*pi) + log_det + mahal)    # (batch, K)
        # 按类别聚合 (logsumexp over prototypes of same class)
        log_p = scatter_logsumexp(log_N, self.class_id, dim=1, n_classes=37)  # (batch, 37)
        log_post = log_p + self.pi.log()                    # (batch, 37)
        b_t = F.softmax(log_post, dim=-1)                   # (batch, 37)
        return b_t
```

$\mu$ 是 `nn.Parameter`，梯度通过 BC/RL 损失反传自动更新。

### 6.3 协方差为何不跟随

$\Sigma_{k,m}$ 固定，不跟随。理由：

1. **稳定性**：协方差估计需要足够样本，在线每步更新易震荡
2. **充分性**：$\mu$ 跟随已解决空间对齐的主要矛盾；$\Sigma$ 的微小偏差对后验影响远小于 $\mu$ 的偏差
3. **简单**：减少可学习参数量和训练复杂度

若实验发现 $\Sigma$ 失配严重，可后续引入 EMA 更新：$\Sigma \leftarrow (1-\beta)\Sigma + \beta\hat\Sigma_{\text{batch}}$。

### 6.4 坍缩风险与监控

原型跟随的风险：所有类原型坍缩到同一点，查询退化为均匀分布。

**监控指标**：

- 原型间距离矩阵 $\|\mu_{k,m}-\mu_{k',m'}\|$ 的均值，训练中不应单调下降
- 类间原型中心距离 $\|\bar{\mu}_k - \bar{\mu}_{k'}\|$，应保持非零
- $b_t$ 的平均熵 $H(b_t)$，不应趋近 $\log 37$

**缓解措施**（若坍缩发生）：

- 原型间排斥正则：$\mathcal{L}_{\text{repel}} = -\lambda_r \sum_{(k,m)\neq(k',m')}\|\mu_{k,m}-\mu_{k',m'}\|$
- 或限制 $\mu$ 的学习率小于 $g_\phi$

---

## 七、策略网络

### 7.1 输入

$$
s_t = [\,b_t\;(37) \;\|\; \text{frontier\_embed}\;(128) \;\|\; \text{budget\_ratio}\;(1) \;\|\; \text{struct\_v}\;(3)\,] \in \mathbb{R}^{169}
$$

- $b_t$：当前步高斯查询后验（37 维）
- frontier_embed：候选 frontier 节点嵌入（128 维，与 base reader 一致）
- budget_ratio：剩余预算比例 $= 1 - t/T$（1 维）
- struct_v：结构特征（度、聚类系数等，3 维，与 base reader 一致）

### 7.2 结构

沿用 base reader 的 ReaderMLP：

$$
\text{MLP}: 169 \to 128 \to 128 \to 1
$$

输出每个 frontier 节点的得分，选 top-1 扩张。

### 7.3 探索信号

无 LSTM 累积后，探索方向性来自 $b_t$ 的不确定性：

- $H(b_t)$ 高 → 当前子图判别力弱 → 需继续探索
- $H(b_t)$ 低 → 已有强判别 → 可停止（但无 Submit-Early，仍跑到 T）

策略网络通过 $b_t$ 隐式利用这个信号——MLP 学会在高熵 $b_t$ 时选信息增益大的 frontier 节点。

---

## 八、训练流程

### 8.1 阶段划分

```
Phase -1（离线，一次性）:
  f_class GNN → 全图嵌入 → PCA → 分桶HDBSCAN → 原型库初始化

Phase 0: 子图编码器 g_φ 预热
  - g_φ 从 f_class 初始化
  - 在 teacher 轨迹子图上用 MSE 对齐 f_class 输出（可选）
  - 或直接进入 Phase 1

Phase 1: BC 训练（策略 + 原型跟随）
  - 用 teacher 轨迹做行为克隆
  - g_φ 和 μ 联合训练
  - 损失：L_BC（策略模仿）+ [可选] L_repel

Phase 2: RL 训练（REINFORCE）
  - 以 b_T 的 argmax 准确率为奖励
  - g_φ 和 μ 继续联合训练
  - 损失：L_RL（策略梯度）
```

### 8.2 损失函数

**Phase 1（BC）**：

$$
\mathcal{L}_{\text{BC}} = -\sum_t \log \pi_\theta(a_t^* \mid s_t)
$$

其中 $a_t^*$ 是 teacher 轨迹的动作。$s_t$ 中的 $b_t$ 由当前 $g_\phi,\mu$ 计算（含梯度）。

梯度路径：$\mathcal{L}_{\text{BC}} \to \pi_\theta \to s_t \to b_t \to \mu$ 和 $\to g_\phi$。即 **BC 损失同时训练策略网络、原型 $\mu$、子图编码器 $g_\phi$**。

**Phase 2（RL）**：

$$
\mathcal{L}_{\text{RL}} = -\sum_t \log \pi_\theta(a_t \mid s_t) \cdot A_t
$$

奖励 $R = \mathbb{1}[\hat{y}=y]$（终端分类正确为 1）。

### 8.3 训练监控


| 指标                         | 含义           | 期望趋势       |
| ---------------------------- | -------------- | -------------- |
| $H(b_t)$ 均值                | 查询信念熵     | 随子图扩张下降 |
| $b_T$ 的 $\max_k$            | 终端置信度     | 上升           |
| 终端准确率                   | $\arg\max b_T$ | 上升           |
| 原型间距离均值               | 坍缩监控       | 稳定不降       |
| $g_\phi$ vs f_class 输出距离 | 编码器漂移     | 缓慢上升       |

---

## 九、伪代码

### 9.1 离线建库

```python
def build_prototype_bank(f_class, train_graphs, train_labels, pca_var=0.95):
    # 1. 全图编码
    embeddings = []
    for g in train_graphs:
        e = f_class.encode(g)               # (300,)
        embeddings.append(e)
    E = torch.stack(embeddings)              # (N, 300)

    # 2. PCA
    pca = PCA(n_components=pca_var)
    E_tilde = pca.fit_transform(E)           # (N, d)
    W_pca = pca.components_.T                # (300, d)

    # 3. 分桶 HDBSCAN
    mu_list, sigma2_list, class_id_list = [], [], []
    for k in range(37):
        idx = (train_labels == k)
        E_k = E_tilde[idx]
        if len(E_k) < 2 * E_tilde.shape[1]:
            # 少数类：单原型
            mu_list.append(E_k.mean(0))
            sigma2_list.append(ledoit_wolf_diag(E_k))
            class_id_list.append(k)
            continue
        clusterer = HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(E_k)
        for m in set(labels) - {-1}:         # 排除噪声
            members = E_k[labels == m]
            mu_list.append(members.mean(0))
            sigma2_list.append(ledoit_wolf_diag(members))
            class_id_list.append(k)

    # 4. 组装
    bank = PrototypeBank(
        mu_init=torch.stack(mu_list),
        sigma2=torch.stack(sigma2_list),
        class_id=torch.tensor(class_id_list),
        pi=(bincount(train_labels, 37) / len(train_labels)),
        pca_W=torch.tensor(W_pca),
    )
    return bank
```

### 9.2 在线推理

```python
def run_reader(g_phi, bank, policy, graph, budget_T):
    subgraph = init_seed(graph)
    b_t = None
    for t in range(budget_T):
        # 编码 + 降维 + 查询
        v_t = g_phi.encode(subgraph)         # (300,)
        b_t = bank.query(v_t.unsqueeze(0)).squeeze(0)  # (37,)

        # 策略选节点
        frontiers = get_frontiers(subgraph)
        s_t = build_state(b_t, frontiers, t, budget_T)
        scores = policy(s_t)
        a_t = scores.argmax()

        # 扩张
        subgraph = expand(subgraph, frontiers[a_t])

    # 终端分类
    y_hat = b_t.argmax()
    return y_hat
```

### 9.3 BC 训练步

```python
def bc_train_step(g_phi, bank, policy, batch):
    loss = 0
    for graph, teacher_actions, label in batch:
        subgraph = init_seed(graph)
        for t, a_star in enumerate(teacher_actions):
            v_t = g_phi.encode(subgraph)
            b_t = bank.query(v_t.unsqueeze(0)).squeeze(0)
            frontiers = get_frontiers(subgraph)
            s_t = build_state(b_t, frontiers, t, budget_T)
            log_prob = policy.log_prob(s_t, a_star)
            loss -= log_prob
            subgraph = expand(subgraph, frontiers[a_star])
    loss.backward()    # 梯度流经 policy, bank.mu, g_phi
    optimizer.step()
```

---

## 十、公式速查

### 10.1 高斯查询

$$
b_t = \text{softmax}_k\!\left(\log\pi_k + \text{logsumexp}_m\!\left[\log w_{k,m} - \frac{1}{2}\sum_i\left(\log\sigma^2_{k,m,i} + \frac{(\tilde{v}_{t,i}-\mu_{k,m,i})^2}{\sigma^2_{k,m,i}}\right)\right]\right)
$$

### 10.2 可学习参数


| 参数                    | 形状                    | 可学习 | 初始化                |
| ----------------------- | ----------------------- | ------ | --------------------- |
| $g_\phi$（2层GNN）      | -                       | ✓     | f_class GNN 权重      |
| $\mu$（原型均值）       | $(K_{\text{total}}, d)$ | ✓     | HDBSCAN 簇质心        |
| $\Sigma$（对角方差）    | $(K_{\text{total}}, d)$ | ✗     | Ledoit-Wolf shrinkage |
| $\pi$（类先验）         | $(37,)$                 | ✗     | 类频率                |
| $W_{\text{PCA}}$        | $(300, d)$              | ✗     | PCA 拟合              |
| $\pi_\theta$（策略MLP） | $169\to128\to128\to1$   | ✓     | 随机/预训练           |

### 10.3 数据流

```
subgraph → g_φ → v_t(300) → ×W_PCA → ṽ_t(d)
                                        ↓
    μ(K,d) ──→ 高斯查询 ──→ b_t(37) ──→ 策略 ──→ 动作
    Σ(K,d) ──→
    π(37)  ──→
```

---

## 十一、风险与待验证项

### 11.1 设计风险


| 风险                              | 严重度 | 缓解                                               |
| --------------------------------- | ------ | -------------------------------------------------- |
| 原型坍缩（$\mu$ 趋同）            | 中     | 监控原型间距 + 排斥正则                            |
| $g_\phi$ 漂移过大，PCA 粗对齐失效 | 中     | 监控$g_\phi$ vs f_class 输出距离；必要时重拟合 PCA |
| 少数类原型质量差（样本不足）      | 中     | 分桶 HDBSCAN 退化为单原型；$\pi_k$ 小自然压制      |
| 无累积导致每步噪声大              | 低-中  | $b_T$ 用最后步（子图最完整）；策略 MLP 隐式平滑    |
| 全图聚类 vs 子图分布偏移          | 中     | $g_\phi$ 微调适应子图分布；$\mu$ 跟随补偿          |

### 11.2 待实验验证


| 实验                                     | 目的                           |
| ---------------------------------------- | ------------------------------ |
| 跑 PCA 定 d                              | 确定 95% 方差对应维度          |
| HDBSCAN 分桶聚类                         | 确认每类原型数分布、少数类处理 |
| Phase 1 BC 训练                          | 策略收敛 + 原型跟随稳定性      |
| Phase 2 RL 训练                          | 终端准确率提升                 |
| 消融：原型跟随 vs 固定原型               | 验证跟随必要性                 |
| 消融：全图聚类 vs 子图聚类（若后续需要） | 验证分布偏移影响               |

### 11.3 与 v5 的关系

v6 **不是** v5 的增量改进，而是信念机制的范式替换：


| 维度         | v5                       | v6                     |
| ------------ | ------------------------ | ---------------------- |
| 信念机制     | LSTM 累积 Dirichlet 证据 | 每步闭式高斯查询       |
| 状态         | 有（LSTM 隐状态）        | 无（每步独立）         |
| 证据头       | 有（Linear(128,37)）     | 无（高斯查询替代）     |
| EDL 损失     | 有                       | 无                     |
| Submit-Early | 设计中有                 | 移除                   |
| 训练复杂度   | 高（多组件联合）         | 低（策略+原型+编码器） |
| 可学习参数   | GSAE+证据头+LSTM+策略    | $g_\phi$+$\mu$+策略    |

v5 的 GSAE、证据头、LSTM 信念更新、Dempster 融合等组件在 v6 中**全部移除**。

---

*设计基于 2026-07-05 讨论确认的 10 项核心决策。实施前需先跑 PCA 确定 d。*

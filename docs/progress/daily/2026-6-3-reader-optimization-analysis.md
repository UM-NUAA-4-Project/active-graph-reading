# 有限视野 MDP 下 Reader 策略优化分析

> 日期：2026-06-03
> 状态：分析文档，用于 M5 实施参考，暂不部署代码

---

## 一、问题回顾

Reader 在一个**部分可观测 MDP** 中做序列决策：

```
State:  G_t = (V_revealed, E_known, X_E, X_V)  — 仅见已揭示+前沿
Action: a_t = v ∈ frontier_nodes                 — 每步从 ~10-50 个候选里选
Reward: R = Accuracy(classifier(sub_G))          — episode 结束才有
Horizon: ~15 步（PPA |V|≈300, budget=5%）
```

核心困难：**argmax 不可微** → 梯度从 reward 到 Reader 参数 θ 的链路断开。

三种策略优化方法（BC → Gumbel-Softmax → REINFORCE）各有短板，需要额外的优化措施提升学习效果。

---

## 二、问题诊断

| 问题 | 影响 | 影响范围 |
|------|------|----------|
| **部分可观测** | Reader 看不到完整图结构，无法评估全局重要性 | 所有方法 |
| **稀疏奖励** | 唯一 reward 在 episode 结束，~15 步 credit assignment 困难 | Gumbel-Softmax, REINFORCE |
| **分布偏移** | BC 中 student 访问的 s_t 偏离 teacher 的 s_t，复合误差 | BC |
| **高方差** | 单一标量 reward × 多步 log_prob，梯度噪声大 | REINFORCE |
| **动态 action space** | frontier 每步变化，候选节点数不固定 | 所有方法 |
| **STE 梯度有偏** | softmax 梯度 ≠ argmax 真实梯度，长 trajectory 弥散 | Gumbel-Softmax |

---

## 三、推荐优化措施（按优先级）

### 高收益 / 低成本

#### 1. DAgger 替代纯 BC

**解决的问题**：BC 的分布偏移 — student 访问的状态 teacher 没见过，一步偏差逐层放大。

**方案**：
1. Round 1：在 M4 teacher 轨迹上训练 BC
2. Round 2：用 student policy rollout 收集轨迹，teacher 在 student 的状态上重打标签（relabel），与原始数据 50/50 混合训练
3. Round 3（可选）：重复一次

**实现要点**：
- Student rollout 时不用 argmax，用 categorical sampling 增加状态多样性
- Relabel 用同一种 teacher（如 hybrid-0.5 或 gradient）保持一致性
- 每轮后评估 validation 是否提升

**预期收益**：显著缩小 student-teacher gap，BC 性能逼近 teacher 上限。

#### 2. Step-level reward shaping（Classifier confidence）

**解决的问题**：稀疏 episode reward → ~15 步后才有信号。

**方案**：
- 定义中间奖励：`r_t = Confidence(classifier(sub_G_{t+1})) - Confidence(classifier(sub_G_t))`
- Confidence = max(softmax(logits)) 或 -entropy(logits)
- 总 loss = `Σ_t baseline_loss - λ * r_t * log_prob`
- λ 控制 dense reward 权重（建议从 0.1 开始）

**适用范围**：Gumbel-Softmax（将 r_t 视为 per-step advantage）、REINFORCE（替换或补充 episode reward）

**注意事项**：
- 每步多一次 classifier 前向传播，训练速度下降约 30%（可用 batch 分摊）
- 需要 normalization（同图内 r_t 减均值除标准差）

#### 3. 图级上下文特征（Graph-level context）

**解决的问题**：ReaderMLP 逐节点独立打分，不知道"我们已经看了多少"、"还剩多少预算"。

**方案**：
```python
# 当前：frontier_embeds[i] → ReaderMLP → score
# 改进：concat graph summary
graph_summary = frontier_embeds.mean(dim=0)  # 或 max pool
augmented = torch.cat([frontier_embeds, graph_summary.expand(N, -1)], dim=-1)
scores = reader_mlp(augmented)
```

**成本**：ReaderMLP 第一层 `Linear(2*hidden_dim, hidden_dim)` 增加 ~16K 参数。

**预期收益**：Reader 隐式感知 frontier 规模、已揭示比例。

---

### 中收益 / 中成本

#### 4. Frontier self-attention

**解决的问题**：Frontier 节点独立打分，无法建模节点间的竞争/互补关系。

**方案**：在 ReaderMLP 前加一层多头自注意力（1 head, no FFN）：
```python
frontier_embeds = self.attn(frontier_embeds.unsqueeze(0)).squeeze(0)
scores = self.reader_mlp(frontier_embeds)
```

**成本**：~33K 参数，每步多一次 attention 计算（可接受，frontier ≤ 50）。

**预期收益**：Reader 学会避免选择"冗余"节点（两个 frontier 看到同一个区域）。

#### 5. 辅助任务：Frontier degree 预测

**解决的问题**：Encoder embedding 可能未充分捕获 frontier node 的拓扑重要性。

**方案**：
- 在 FrontierNodesEncoder 上加 aux head：`Linear(hidden_dim, 1)` 预测每个 frontier node 的 `frontier_degree`（已揭示邻居数）
- 联合 loss：`L_total = L_main + λ_aux * MSE(pred_degree, true_degree)`
- `frontier_degree` 已经在 `GraphState._frontier_degree` 中维护，零额外计算

**成本**：一个 Linear 层 ~128 参数。

**预期收益**：Encoder 学会识别"hub" frontier 节点（揭示它能打开很多新区域）。

#### 6. 多 seed 共识训练

**解决的问题**：单 seed 的初始状态可能不典型，训练数据偏差。

**方案**：
- 每张图用 k=3 个不同 seed 生成轨迹
- 选出最优轨迹（最低 classifier CE）作为正例
- 可选：最差轨迹作 hard negative（只对 BC：不给这些 action 正 label）

**成本**：训练数据量 ×3，但只在 BC 阶段使用，可提前生成。

**预期收益**：Reader 学到对 seed 位置鲁棒的策略。

---

### 低优先级（兜底方案）

#### 7. Entropy regularization
- `L_total = L_policy - λ * H(probs)`
- 鼓励探索，防过早收敛到次优策略
- 适用：REINFORCE、Gumbel-Softmax

#### 8. Budget-aware conditioning
- 将 `budget_remaining / total_budget` 作为额外 feature 输入 ReaderMLP
- 让 Reader 在不同预算区间调整行为

#### 9. Distance-from-seed positional encoding
- 在 FrontierNodesEncoder 中给每个节点加距离种子的步数编码
- 成本低，但 PPA 无节点特征，收益不确定

---

## 四、推荐的 M5 训练流水线

```
Phase 1: BC + DAgger
  ├── Round 1: 纯 BC（M4 teacher trajectories）
  ├── Round 2: Student rollout → teacher relabel → 混合训练
  └── Round 3: 重复（如验证集仍提升）
  措施：多 seed 共识、frontier degree aux loss
  目标：稳定逼近 teacher 性能上限

Phase 2: Gumbel-Softmax 端到端微调
  ├── 加载 Phase 1 最优 checkpoint
  ├── Step-level confidence reward shaping
  ├── 图级上下文特征
  ├── Frontier self-attention
  └── Temperature annealing: 2.0 → 0.1
  目标：超越 teacher，端到端优化 Reader + Classifier

Phase 3 (backup): REINFORCE
  ├── 仅当 Phase 2 不收敛时
  ├── Value function baseline（小 MLP on graph summary）
  ├── Multi-trajectory rollout × 多 seed
  └── Entropy regularization
  目标：无偏梯度，兜底收敛
```

## 五、进入条件重申

M5 启动的前提（与 policy-optimization-report 一致）：
1. M3 gate PASS：BFS 在 ≤10% budget 未达 full-graph accuracy 90%
2. M3 三种策略差异显著（需大样本确认）
3. M4 teacher trajectory 质量够（accuracy 上限显著高于 random）

# M5 训练经验记录

## 现象：BC 准确率高，RL 掉到 1/5

**原因**：BC 只用了 ~20% 训练图的 teacher trajectories，reader 内部分类头没见过 80% 的图。RL 依赖这个弱分类头算 reward，reward 信号噪声大 → REINFORCE 更新方向错误。

**风险**：RL 完全不更新分类头（`compute_reward` 中 `.item()` 断图），分类头停留在 BC 结束时的状态。

## Loss 设计

### 当前公式（正确）

```
L = α · (-R_i · Σ_{t} log π_t(a_t)) + (1-α) · CE(y_i, f(h_i))
```

- RL 项和 CE 项**独立相加**，CE 不受 reward 控制
- 如果 CE 被 reward 乘（`R_i · (... + CE)`），R_i 为负时 CE 梯度被反转，优化器会去增大 CE——**错误设计**

### 量级问题

- RL_loss ≈ -R · Σlogπ：logπ ≈ -2~-3，budget 30 步 → 约 60-90
- CE_loss：训练中通常 0.5-6.0
- α=0.5 时 RL 项比 CE 项大 30-100 倍

### 稀疏奖励问题

- 初始 acc ≈ 0.15（随机水平），85% 轨迹 R=0 → RL_loss 为 0，仅 CE 提供梯度
- 15% 正确轨迹的 RL 梯度量级大 + 方差高，会把参数往其方向拽
- 用 `tanh` reward 让每条轨迹都有非零 R：
  - 正确时 CE≈0.5 → R≈tanh(0.86)≈0.70
  - 错误时 CE≈3.5 → R≈tanh(-0.03)≈-0.03

### Reward 对比

|            | binary               | tanh               | asymmetric         |
| ---------- | -------------------- | ------------------ | ------------------ |
| 正确 R     | 1.0                  | ~0.7               | 1.0                |
| 错误 R     | 0                    | ~-0.3              | -0.1               |
| 梯度方向   | 正确推近，错误无信号 | 正确推近，错误推远 | 正确推近，错误推远 |
| 稀疏性问题 | 85% 轨迹无梯度       | 所有轨迹有梯度     | 所有轨迹有梯度     |

## 文献参考

### 固定加权求和（最常用）

`L = L_RL + λ · L_SL`，λ 固定。

- **UNREAL** (Jaderberg et al., 2017, ICML) — A3C + 辅助任务，辅助 loss 与 reward 独立
- **Self-Supervised RL for RecSys** (Xin et al., 2020, SIGIR) — 两个输出头，CE 早期主导，RL 逐渐接管
- **Hybrid-AIRL** (Silue et al., 2026) — IRL + BC loss，λ 固定

### 自适应加权

- **Gradient Similarity** (Du et al., 2018) — α = max(0, cos(grad_RL, grad_SL))，梯度冲突时关掉辅助 loss
- **QCS** (Kim et al., 2024, NeurIPS) — β = f(return)，低回报轨迹多 RL，高回报多 SL

### Gradient 修改

- **Delightful Policy Gradient** (Osband, 2026) — REINFORCE gradient 乘以 sigmoid(advantage × action_surprisal)，使梯度更接近 CE 方向

## 锁步循环关键规则

### Budget 检查（必须）

每步 `reveal_node(state, chosen)` 后，检查：

```python
if state.get_budget_used() >= state._budget_limit:
    active[j] = False
```

最后一步达到 budget 的节点**包含在 trajectory 中**。不检查会导致小图在锁步循环中超出预算。

### 自环规则（曾出 bug）

- `SubgraphBuilder.build()` 内部已调用 `dgl.add_self_loop()`
- 禁止外层再包一层 `dgl.add_self_loop(builder.build(state)[0])` → 双重自环
- 从磁盘加载的全图需 `dgl.add_self_loop(g)` 后再创建 GraphState
- `FrontierGraphBuilder` 内部也加自环

## eval_heldout.py 变量覆盖 bug（2026-08-04）

lockstep 缓存变量 `all_subgraphs` 与输出积累列表同名，在 chunk 循环内被重新赋值导致前序 chunk 数据丢失。修复：缓存改名为 `cache_subgraphs`。

# Feedback: 2026-05-26 - M3/M4 Code Review and Next Steps

**日期**: 2026-05-26
**主题**: 对 M3/M4 代码提交后的反馈，以及在本地 laptop / 云机器条件下的下一步安排。
**背景**: 当前 repo 已经包含 M3/M4 代码：Reader engine、OGB-style GINE classifier、M3 benchmark、M4 teacher / trajectory / oracle / DFI 模块。下一步需要先把 M3 变成可复现的 empirical gate，再决定是否推进 M4/M5 learned Reader。

---

## 一、核心反馈

现在先不要继续扩展新的 M4/M5 方法。当前最重要的目标是回答一个更基础的问题：

> 在 OGBG-PPA 这种较大图分类任务上，partial graph reading 是否真的非平凡？在固定 budget 下，不同 reading policy 是否会产生明显不同的分类效果？

也就是说，下一阶段的重点不是“再设计一个更复杂的 Reader”，而是先把 M3 跑成一个可复现的 baseline gate。

如果 M3 结果显示 random / BFS / max-frontier-degree 在小 budget 下差异明显，并且距离 full-graph upper bound 还有空间，那么 M4/M5 的 learned Reader 才有意义。反过来，如果 BFS 在很小 reveal ratio 下已经接近 full graph accuracy，那么这个 benchmark 对当前问题可能太简单，需要调整数据集、初始观测或 protocol。

## 二、Frontier Edge Feature 的协议决定

我们已经决定：

> Reader 在选择下一个 frontier node 之前，可以看到连接已揭示节点和 frontier node 的 frontier edge feature。

这个决定需要在代码和文档里明确实现，而不是只停留在讨论里。

目前代码里 `GraphState` 已经维护了 `frontier_edges`，但需要检查：

- `G_0` 时，seed node 到 frontier nodes 的 edge feature 是否已经可查询；
- 在每一步 reveal 后，新加入 frontier 的边特征是否在“选择 frontier node 之前”可见；
- strategy / 未来 learned Reader 是否有清晰 API 读取这些 frontier edge features。

建议加一个显式接口，例如：

```python
state.get_frontier_edge_feat(u, v)
```

或者维护类似：

```python
state.frontier_edge_feats[(u, v)]
```

同时加一个最小测试：初始化 `G_0` 后，不消耗 budget，也能读到 seed-to-frontier edge feature。

这不是一个可选细节。因为如果 Reader 只知道 `b / c / d` 这些 frontier nodes 存在，但看不到任何 edge feature，那么这些候选节点在信息上几乎等价，只能随机选，后续 learned Reader 很难有真实决策信号。

## 三、Laptop 上应该先完成的任务

智辉目前主要用自己的 laptop，这没有问题。以下任务都应该是 laptop 上可做的，不需要等云机器。

| 任务 | Laptop 是否可做 | 说明 |
|---|---:|---|
| 修正 / 明确 frontier edge feature 可见性 | 可以 | 小规模代码改动 + unit test，不依赖完整训练。 |
| 写最小测试证明 `G_0` 已暴露 seed-frontier edge feature | 可以 | 可以用 synthetic graph，不一定需要 OGBG-PPA。 |
| 让 `smoke_test_phase1.py` 可配置数据路径 | 可以 | 当前硬编码了本地路径，应该改成 CLI 参数或配置。 |
| 加一个 synthetic graph smoke test | 可以 | 避免每次 smoke test 都依赖完整 OGBG-PPA 下载和环境。 |
| 写 `code/README.md` 说明运行方式 | 可以 | 纯文档任务：环境、数据路径、smoke test、quick benchmark。 |
| 在 1-10 张小图上跑 Reader + classifier shape smoke test | 可以 | CPU 足够，目标是发现代码错误，不是追求性能。 |
| 在 10-50 张 validation graphs 上跑 quick benchmark | 大概率可以 | 如果数据已在本地，CPU 会慢但可接受；只做 sanity check。 |

Laptop 阶段的目标是：

1. 证明 protocol 实现正确；
2. 证明代码能端到端跑通；
3. 产生一个很小规模的 sanity benchmark；
4. 把云机器到位后要跑的命令准备好。

Laptop 阶段不要求得到正式实验结果。

## 四、应该等云机器 / GPU 后再做的任务

以下任务不建议在 laptop 上硬跑，除非只是非常小规模 sanity check：

| 任务 | 原因 |
|---|---|
| OGBG-PPA full-graph GINE 正式训练 | CPU laptop 会很慢，且容易因为环境 / 内存 / 时间浪费精力。 |
| 500+ graphs、multi-seed、multi-budget 的正式 M3 curves | 计算量较大，应该在云机器上跑。 |
| M4 info-gain teacher 全量轨迹生成 | 每步要对多个 frontier candidates 做模拟前向，成本高。 |
| M4 oracle / teacher 大规模 trajectory collection | 可以先写代码和小样本测试，但正式数据生成等机器。 |
| M5 learned Reader training | 必须等 M3 gate 清楚之后再做，否则容易在错误方向上投入太多。 |

因此，近期不要把“没有云机器”当作停滞理由。云机器影响的是正式训练和正式曲线，不影响 protocol 修正、测试、文档和 quick sanity run。

## 五、Policy Optimization Presentation

另一个本周可以在 laptop 上完成的任务，是准备一次关于 policy optimization 的小报告。这个报告的目的不是马上实现 M5，而是确认你真的理解后面 learned Reader 可能会用到的优化方法。

报告建议控制在 `10-15` 分钟，重点讲清楚以下内容：

1. **问题形式**
   - Reader 的 state / action / reward 或 loss 分别是什么；
   - 为什么选择 frontier node 是一个离散决策问题；
   - 为什么普通 backprop 不能直接穿过 `argmax` / discrete sampling。

2. **Behavior Cloning**
   - teacher trajectory 如何生成；
   - student Reader 学什么；
   - loss 怎么写；
   - 优点：稳定、简单；
   - 缺点：受 teacher 质量上限限制。

3. **Gumbel-Softmax / Straight-Through Estimator**
   - 它解决什么问题：让离散选择近似可微；
   - forward 和 backward 分别发生什么；
   - temperature 的作用；
   - 优点：低方差、适合端到端微调；
   - 风险：梯度有偏、temperature schedule 敏感。

4. **REINFORCE / Policy Gradient**
   - 基本公式：`log prob(action) * reward`;
   - 为什么它可以处理不可微的离散采样；
   - baseline / advantage 为什么能降方差；
   - 优点：理论上无偏，不需要可微 reveal；
   - 风险：方差大、训练慢、需要多次 rollout。

5. **在本项目里的使用顺序**
   - 首选路线：M3 benchmark gate → M4 teacher trajectories → M5 behavior cloning pretrain → Gumbel-Softmax fine-tune；
   - REINFORCE 作为 backup，不作为第一选择；
   - 如果 M3 没有显示 policy choice 有明显空间，就不进入这些复杂训练。

报告不需要证明很复杂的数学定理，但需要能画出训练流程图，并能回答：“梯度/学习信号到底从哪里来？”

## 六、M3 Gate：下一份结果应该长什么样

下一份 deliverable 建议叫：

```text
M3 benchmark sanity report
```

至少包含：

1. **代码运行状态**
   - 环境是否能安装；
   - 数据路径如何配置；
   - smoke test 是否通过；
   - quick benchmark 是否能运行。

2. **协议检查**
   - `G_0` 是什么；
   - seed 是否不消耗 budget；
   - frontier node / frontier edge feature 在什么时候可见；
   - `reveal_node(v)` 消耗什么 budget，并暴露什么新信息。

3. **小规模 quick benchmark**
   - 可以先用 `10-50` 张 validation graphs；
   - `1-3` 个 seeds；
   - 少量 budget ratios，例如 `5% / 10% / 20%`；
   - 比较 `random / bfs / max-frontier-degree`；
   - 如果没有 pretrained full-graph model，就先说明 blocker，不要伪造 full-graph number。

4. **正式实验计划**
   - 云机器到位后要跑哪些命令；
   - 预计需要哪些输入文件；
   - 输出保存在哪里；
   - 需要多长时间 / 多大显存的粗略估计。

## 七、具体行动项

| Owner | Action | 机器要求 |
|---|---|---|
| 智辉 | 明确实现 frontier edge feature 在 reveal 前可见，并补最小测试 | Laptop |
| 智辉 | 把 `smoke_test_phase1.py` 的数据路径改成参数，不要硬编码本机路径 | Laptop |
| 智辉 | 增加 synthetic graph smoke test，确保无 OGBG-PPA 时也能测 Reader 协议 | Laptop |
| 智辉 | 写 `code/README.md`：环境、数据路径、smoke test、quick benchmark、正式 benchmark 命令 | Laptop |
| 智辉 | 跑一个小规模 quick benchmark，目标是证明脚本可运行 | Laptop |
| 智辉 | 准备 `10-15` 分钟 policy optimization 小报告：Behavior Cloning / Gumbel-Softmax / REINFORCE | Laptop |
| Meng / 王老师 | 准备云机器或 GPU 资源，用于 full-graph training 和正式 multi-seed curves | Cloud/GPU |
| 智辉 | 云机器就绪后跑正式 M3 benchmark curves + kill criterion | Cloud/GPU |

## 八、暂时不要做的事

暂时不要把主要精力放在：

- 继续扩展 M4 teacher 种类；
- 直接开始 M5 learned Reader training；
- 调复杂的 Gumbel-Softmax / REINFORCE；
- 写很长的方法综述但没有 runnable benchmark；
- 在 laptop 上硬跑完整 OGBG-PPA 训练。

这些事情不是不重要，而是顺序应该靠后。现在的 priority 是把 M3 做成一个可信的 gate。

## 九、下次 meeting 重点

下次 meeting 不需要讲太多新设计，重点讲清楚三件事：

1. **协议是否和 5/25 讨论一致**：frontier edge feature 是否在选择 frontier node 前可见？
2. **代码是否能复现运行**：smoke test / quick benchmark 是否跑通？
3. **M3 是否有初步信号**：在小样本上 random / BFS / max-frontier-degree 是否有差异？
4. **policy optimization 是否讲清楚**：Behavior Cloning / Gumbel-Softmax / REINFORCE 的学习信号、优缺点、以及本项目中使用顺序是什么？

如果这三件事清楚了，再进入 full benchmark 和 learned Reader；如果这三件事还不清楚，就先继续修 M3。

# AgentNet：工作流与 AGR 接入方案

记录 AgentNet（ICLR 2023, *Agent-based Graph Neural Networks*，
arXiv 2206.11010，用户本地 `AgentNet-main`，PyG 官方参考实现）的原版
端到端工作流，以及移植为 AGR 协议基线的接入方案。

**状态：已实现（2026-09-15，cifar10 冒烟通过）。** 实现落点：
`baseline/agentnet.py`（AgentNetModel 单 agent DGL 移植 + AgentnetStrategy）+
`baseline/run_agentnet.py`；决策表见 `protocol-adaptation.md` §7。
AgentNet 是二者中唯一的**原生序贯模型**（多智能体逐显式移动选点），与
AGR 的揭示循环同构度远高于 SAGPool/Soft-Mask 这类打分层；但它的移动假设
完整图可见、可往返、无预算语义，接入是"策略直译"而非"打分器移植"，
改造面比 SAGPool 大。

---

## 1. 原版工作流（AgentNet-main）

### 1.1 端到端训练管线（`graph_classification.py`）

```
TUDataset（MUTAG/.../REDDIT-BINARY）
  → 单一 train/val/test 划分，DataLoader（bs=32）
  → AgentNet 前向 → log_softmax → NLLLoss
  → Adam(lr=1e-4, wd=0.01)，cosine LR（warmup）
  → gumbel 温度 τ 余弦退火（参考命令取常数 2/3）
  → 梯度裁剪 clip_grad=1.0
  → 汇报 test acc
```

参考超参（README）：num_agents=18 / num_steps=16 / hidden=128 /
dropout=0 / lr=1e-4 / bs=32 / epochs=350 / `--self_loops --input_mlp --global_agent_pool --bias_attention`。

### 1.2 模型前向（`model.py` AgentNet）——多智能体游走

```
输入 x, edge_index, batch（可选 edge_feat）
节点投影 input_proj → node_emb；agent_emb = Embedding(K 个 agent)
初始位置：随机均匀撒（或 importance_init：对全节点注意力选址）
逐 step i = 0..num_steps：
  1) 移动：对每个 agent，在【当前位置的全部邻居】上算注意力
       score = Q(agent_emb)·K([邻居 emb ‖ 当前节点 emb ‖ (边 emb)])
       + BSEU 偏置（back/stay/explored/unexplored 可学习标量，
         visited 集合按 visited_decay=0.9 衰减）
     gumbel-softmax(hard, straight-through) 选一条边 → agent 移动过去
  2) 节点更新：仅更新【被 agent 占据的节点】：agent→node 消息 +
     邻域卷积（active 节点），残差 + LayerNorm
  3) agent 更新：MLP([agent_emb ‖ 所在节点 emb ‖ 所走边 emb])，残差
  4) 逐步读出：agent 维 max+mean → fc → logits，跨步累加求平均
→ log_softmax → 分类
```

全流程端到端可微（gumbel straight-through），**训练信号只有分类 NLL**——
"该往哪走"完全由分类损失间接塑形。推理用 `--test_argmax`（τ→1e-6）
近似确定性。变体：`--random_agent`（随机游走）、`--basic_agent`
（agent 专属 BSEU 参数）。

关键性质（接入 AGR 前必须认清的根源）：

- 移动候选 = 当前位置的**全部邻居**（完整图结构，评测态不可见）；
- agent 可**往返/停留**（self-loop stay、back 偏置），访问集≠揭示集；
- 无预算概念：K 个 agent × 16 步 ≫ AGR 的 5%~15% 节点预算；
- 初始选址用全图信息（随机选址除外）。

---

## 2. AGR 接入方案（候选基线：agentnet）

与 SAGPool/Soft-Mask 的"打分器移植"不同，AgentNet 的接入是**策略直译**：
把 agent 游走一步 ≙ 揭示一个 frontier 节点。它天然回答的对比问题是：
"端到端分类监督训出的游走策略，对比 random-walk（深度型启发式）强多少"——
即**无序贯奖励的学习型深度策略**基线，与 random-walk（同深度型、无学习）
形成成对对照。

### 2.1 游走 → 揭示的映射（单 agent 严格版）

```
eval.py --strategy agentnet → AgentnetStrategy(BaseReader)
  状态：agent 当前位置 v_t ∈ V_revealed（揭示区始终连通，P3 天然满足）
  每步 select()：
    1) 候选 = N(v_t) ∩ frontier            ← 只能沿当前节点的边走（P3）
    2) 可见子图重算：邻居节点特征对 frontier 置零（P4）；
       one-ended 边特征喂入 K 的 edge_emb 通道（模型 edge_feat 路径现成）
    3) 注意力打分（含 BSEU 偏置：visited=V_revealed ∪ 历史位置，
       decay 状态为策略内部状态，合法）；评测态 τ→0 取 argmax（确定性）
    4) 返回得分最高的候选 frontier 节点 → reveal_node，agent 移动过去
  预算记账：reveal_node 首次揭示才消耗（原则 3）；agent 空步/停留不消耗
  死角处理：N(v_t) ∩ frontier 为空（走到局部死角）→ 在已揭示区内用
  BSEU 打分导航（区域内移动不消耗预算，步数上限 4n；rollout 契约不瞬移，
  瞬移实际会揭示节点并消耗预算），导航耗尽返回 None 结束 rollout
  终判：读满预算 → 冻结分类器（P5）→ 统一 eval_fixed.json
```

**训练**（两档，按工程量排序）：

- 档 1（对齐 SAGPool 适配，推荐起步）：完整图上端到端分类监督训练
  AgentNet（原则 2：训练全知合法），产 checkpoint；评测按上表在可见
  子图上重打分。改造量：DGL 复刻移动/更新模块 + registry 数据接线
  （无节点特征 → 度数特征或共享 token；边特征走现成 edge_feat 通道）。

### 2.2 多 agent 的取舍

原版 K=16~26 个 agent 并行游走。AGR 预算按"首次揭示节点数"计：

- **单 agent**（推荐）：预算语义逐字节对齐（每步揭示 1 个节点）；

### 2.3 设计要点与风险

| 点                    | 决策                                 | 说明                                                                                                                                                                                                                                                                                                 |
| --------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 与 random-walk 的关系 | **候选结构等价，选点分布不同** | 两者候选集同为"当前位置邻居 ∩ 未揭示"（深度型；都不是严格 DFS，无栈回溯纪律）。random-walk 均匀随机、无状态；AgentNet 学习注意力分布 + BSEU 偏置（back/stay 可学出回溯/停留，区域内移动不计预算）+ visited 衰减记忆。二者构成控制变量成对对照（唯一变量：选点分布是否学习得到），非第三种独立启发式 |
| 移动范围              | 当前位置的 frontier 邻居             | 原版全邻居→协议违规（P1/P3）；限 frontier 后与揭示语义严格对齐                                                                                                                                                                                                                                      |
| BSEU/visited          | 保留，作为策略内部状态               | visited ⊆ 已揭示区 + 历史位置，不引入未观测信息（原则 1）                                                                                                                                                                                                                                           |
| gumbel 噪声           | 评测关闭（argmax）                   | 原则 4 确定性；训练档 1 保留原版 τ=2/3                                                                                                                                                                                                                                                              |
| 初始位置              | seed 节点                            | 对齐 AGR 种子协议（P2），弃用原版随机/注意力选址                                                                                                                                                                                                                                                     |
| 死角                  | 已揭示区内 BSEU 导航                 | 局部死角在区域内移动（不计预算，上限 4n）到有 frontier 候选处；瞬移会揭示节点并消耗预算，rollout_strategy 亦不调用瞬移，导航耗尽返回 None 结束（与固定策略契约一致）                                                                                                                                                          |
| 已知风险              | 无文献先例                           | 原版每步更新的是"被占据节点"的表征，限 frontier 后节点更新范围缩小，分类器输入分布变化更大；属基线设定，结果须如实呈现                                                                                                                                                                               |

### 2.4 复现命令（实现后）

```bash
# 档 1 训练（新建 baseline/run_agentnet.py，产物 baseline/artifacts/{ds}/agentnet.pt）
python baseline/run_agentnet.py --dataset ppa --root D:\ogb-master\dataset --max-graphs 30000

# 评测（eval.py choices / strategies 注册 / artifacts 默认路径，三处接线同 sagpool）
python code/train_reader/eval.py --dataset ppa --root D:\ogb-master\dataset \
    --strategy agentnet --classifier <上界分类器>.pt --max-graphs 2000
```

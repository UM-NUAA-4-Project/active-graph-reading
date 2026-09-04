# Evidence Reader 设计文档（v2：带证据累积）

状态：v2 已实施（2026-09-03）；2026-09-04 完成决策-分类一体终判 + 奖励源决议，
test_evidence_reader.py（7 组）+ smoke 全绿 ｜ 日期：2026-09-03 / 修订 2026-09-04 ｜
作者：与 ZCode 协作

修订史：
- v1（纯 base + 证据头、无递归）：经评审认为"少了点意思"，未单独实施；
- v2（2026-09-03）：加入 LSTM + 注意力表达证据累积，逐步预测携带历史信息。
  v1 自动降级为 v2 的消融档（去 LSTM + 去 attention-pool 即 v1），不另立设计；
- 2026-09-04：评估终判改为自身证据头（决策-分类一体）；奖励源决议维持冻结
  外部分类器；终局判定统一为"读满后补充前向"（修复最后一跳证据缺口）。

上一版 reader（attention）因臃肿缝合不收敛而废弃；**本文档全程贯彻简洁原则**，
v2 的每个新模块都有且只有一个职责。

---

## 1. 诊断：attention reader 为什么更大却更差

参数实测（h128, L2, edge7, m4）：

| 模块 | base | attention |
|---|---|---|
| frontier_encoder（GIN，共享） | 135,682 | 135,682 |
| 决策头（ReaderMLP / FrontierPointer） | 33,665 | 153,793 |
| BeliefState | — | 265,088 |
| NewInfoEncoder（第二个 GIN！） | — | 217,218 |
| **合计** | **169,347** | **771,781（4.6x）** |

三条病因：
1. **双编码器冗余**：NewInfoEncoder 整图重编码"新信息"，与主编码器完全重复；
2. **递归被接进了决策**：pointer 用 belief 打分，选点路径背负跨步链式梯度；
3. **学习信号密度没变**：4.5 倍参数，每步仍只有 1 bit 动作监督。

**注意：病因是"递归进了决策路径 + 编码器翻倍"，不是"存在递归"本身。**
v2 的全部设计就是精确避开这两点，同时保留递归的表达力。

## 2. 历史信息到底加不加：v1 论证的修正

v1 论证：已揭示子图是历史的充分统计量，递归冗余。**这个论证在信息层面
成立，在计算层面不成立**：

- 已揭示子图在**信息**上无损——读过的都在；
- 但编码它的 GIN 只有固定 2 层：每步从零编码，**跨步揭示的远距离证据
  （图中相距 >2 跳的两个已读节点）永远无法在单次前向中交互**。已揭示
  子图连通但直径可以很长（collab 稀疏长直径图正是如此）；
- 沿阅读轨迹的递归 = **时间维度上的无限深积分**：第 1 步读到的证据与
  第 20 步读到的证据在 h_t 中充分混合，不受图直径限制。

结论：递归不是"记忆被遗忘的东西"（信息论角度冗余），而是"把散落的证据
**增量式深度积分**"（计算角度必要）。这正是"表达证据累积的过程"的需求，
也是 GAM 的 LSTM 在它自己设定下的真实作用——**GAM 每步观测过贫（单节点），
历史必须住在状态里；我们每步观测是整个子图，历史应该住在积分器里，
职责不同。**

## 3. 协议定位：T0/T1/T2 与 evidence 的边界

观测分层（协议决策，2026-09 定稿）：

| 层 | 内容 | 状态 |
|---|---|---|
| T0 | 前沿节点的**存在性**与挂接关系 | 所有 reader 必备 |
| T1 | **前沿边特征**（半揭示边，reveal 前可查询、零预算） | **保留**（v1 协议区分点） |
| T2 | 候选节点属性 / 未读区域的任何内容 | **禁止** |

evidence reader 对三层的使用：

- **选点路径**：T1 全额使用——前沿边特征经 GINE 边编码器进入消息传递，
  `get_frontier_edge_feat` 免费可查询（Visible ≠ revealed），与 base 一致；
- **预测路径（证据流）**：只读已揭示节点的嵌入 `h_all[:n_rev]`。
  前沿节点的嵌入（`h_all[n_rev:]`）虽在同一次前向中算出，但**只供选点，
  绝不进 z_t**——单测以前沿嵌入投毒（×100 随机）断言 logits 不变；
- **永不使用**：T2（未读节点属性、度/聚类等前沿侧结构描述子进预测头）。
  防守线：**分类只应是已揭示内容的函数**——同一 checkpoint 在不同预算档
  揭示了相同节点集合时，预测必须相同；喂预算比例/结构特征会破坏这一点。

决策相关参数（剩余预算比例 1 维 + 结构特征 3 维）**只拼进选点打分**
（`_policy_scores`，base 同款），不进预测头——它们是"下一步去哪"的信息，
不是"读到了什么"的信息。

## 4. 架构 v2：一条增量流、一个积分器、一次历史引导的读出

```
每步 t（锁步批内逐 state）：
  扩展子图 → FrontierNodesEncoder（共享，唯一 GNN，不变）
      ├─ 前沿嵌入 → [budget+结构特征] → ReaderMLP（不变，无状态）→ 选点打分
      ├─ 已揭示嵌入 R_t ──────────────────────────────┐
      └─ 新揭示节点嵌入 Δ_t = emb(v_{t-1})（同一次前向免费取出）│
                                                        ↓
  h_t = LSTM(Δ_t, h_{t-1})                ← 证据积分器（历史）
  z_t = AttnPool(query=h_t, K/V=R_t)      ← 历史引导的注意力读出
  logits_t = MLP([z_t ; h_t])             ← 逐步图类别预测（带历史）
```

三个模块，各司一职：

1. **增量流 Δ_t**：刚揭示节点的嵌入。**取自共享编码器同一次前向的输出**
   （编码器本来就算了它）——不再有第二个 GIN。attention reader 的 217K
   NewInfoEncoder 就此消失。
2. **积分器 h_t**：一层 LSTM(d→d)。h_{t-1} 携带此前全部阅读历史，Δ_t 是
   本步新增证据；逐步预测的 query 由它给出。每图独立，rollout 开始时
   h_0 = LSTM(emb(seed), 0)。
3. **历史引导注意力读出 z_t**：单头缩放点积注意力，query = h_t，K/V =
   当前全部已揭示节点嵌入。语义精确：**累积状态决定"此刻预测该重读已读
   证据的哪一部分"**——这是注意力在本设计中唯一的职责（读出接口），
   不做候选打分、不做信念更新。

**选点路径与 base reader 完全一致（无状态、无历史）**——递归只住在预测
头里。这是与 attention reader 的根本结构差异，也是收敛性的防火墙：
LSTM/注意力的梯度只来自监督 CE（良态的序列建模），永不污染策略梯度。

### 4.1 张量形状流转（d = hidden_dim，N = 批内 state 数，n = 子图节点数）

| 阶段 | 张量 | 形状 | 来源 |
|---|---|---|---|
| 子图批 | subgraphs | N 个 FrontierSubgraph | `frontier_builder.build(s)` |
| 共享编码 | h_all[k] | (n_k, d)，**前 n_rev 行=已揭示（升序），其余=前沿** | `forward_all_batch`（与 forward_batch 同特征同 GNN） |
| Δ_t 收集 | deltas | (N, d)；last_revealed 不在已揭示集时取零向量 | h_all[k][pos]，pos = index of last_revealed in revealed_sorted |
| LSTM 单步 | (h_new, c_new) | 各 (N, d) | `evidence_lstm(stack(deltas).unsqueeze(0), (stack hs, stack cs))`，seq=1, batch_first=False |
| 选点打分 | scores[k] | (n_frontier_k,) | `_policy_scores(h_all[k][n_rev:], sg, state)`——与 base 逐字节一致 |
| 历史读出 | z_t | (d,) | `evidence_pool(h_new[k], h_all[k][:n_rev])`；n_rev=0 时返回零向量 |
| 逐步预测 | ev_logits[k] | (n_classes,) | `evidence_head(cat([z_t, h_new[k]]))`，头为 Linear(2d→d)+ReLU+Linear(d→C) |

### 4.2 三个新增模块的精确定义

```python
class HistoryPool(nn.Module):            # evidence_pool
    q/k/v_proj = nn.Linear(d, d)         # 单头缩放点积
    forward(query (d,), kv (n, d)) -> (d,):
        attn = softmax(K q / sqrt(d), dim=0); return attn @ V
        n=0 → 零向量（无可读证据）

self.evidence_lstm = nn.LSTM(d, d)       # seq=1, batch_first=False；每次调用恰一步
self.evidence_head = Sequential(Linear(2d, d), ReLU(), Linear(d, n_classes))
```

`init_evidence(n, device)` → [(zeros(1,d), zeros(1,d))] × n——(h, c) 由
调用方（BC/RL/eval 循环）逐 state 维护，reader 本体无状态。

## 5. 状态管理语义（训练/评估循环侧）

| 量 | 初始化 | 逐步更新 | 边界 reset |
|---|---|---|---|
| (h, c) | `init_evidence(n_total, device)`，每 batch/chunk 一次 | 每次 `forward_states_evidence` 返回 new_hc 写回 | BC：每 batch；RL：每 batch；eval：每 chunk |
| last_revealed[j] | `seed_node`（Δ_0 = seed 嵌入） | 选点 reveal 后 = chosen；**teleport 后 = 跳转节点**（teleport 也是一次 reveal） | 同上 |

**终局判定（2026-09-04 定稿）**：逐步前向发生在 reveal 之前，最后一次
前向只见过 B−1 个揭示节点——第 B 个节点的证据从未进过任何判定。修复：
**预算耗尽后补充一次 `forward_states_evidence` 前向**（Δ = 最后一次 reveal
的节点，KV = 全部 B 个已揭示节点），其 ev_logits 即终局预测。BC/RL/eval
三处语义一致（BC/RL 的终局 CE 计入 L_ev；eval 的终局 argmax 计入 acc_*）。
任何 state（至少有 seed）都能做该前向，无 None 分支。

## 6. 训练：损失与奖励公式

### 6.1 BC（`core/bc.py`，教师轨迹，lockstep）

$$L_{BC} = \underbrace{\frac{1}{|S|}\sum_{s}\sum_{t}-\log \pi(a_t^{teacher}\mid s,t)}_{L_{act}\text{（教师动作 NLL，逐步）}} \;+\; \lambda\cdot L_{ev}$$

$$L_{ev}=\frac{1}{|E|}\Big[\sum_{s}\sum_{t=0}^{B_s-1}\mathrm{CE}(z_{s,t},y_s)+\sum_{s}\mathrm{CE}(z^{final}_s,y_s)\Big],\qquad |E|=\sum_s (B_s+1)$$

- $z_{s,t}$：第 t 次前向的预测（见过 seed + 前 t−1 次揭示 + T1 前沿边信息）。
  t=0 只看 seed——逼模型在每个阅读阶段都给出校准的猜测；
- $z^{final}_s$：读满 B 后的补充前向（§5），教"读完整怎么判"；
- **|E| 是 batch 内 CE 项总数**（逐步 + 终局），即 `ev_losses` 列表长度，
  `.mean()` 的隐含分母。注意三个计数的分工：`ev_acc` 分母 = 逐步项数，
  `ev_acc_final` 分母 = 轨迹数，|E| = 两者之和；
- λ 默认 1.0（`--evidence-lambda`）；**λ=0 时证据流整体关闭**（不前向、
  不训练、不记 ev_acc）——此类消融只用于测"L_ev 对编码器的整形作用"，
  其 eval 终判（随机头）无意义；
- 日志：`train_loss`（总量）、`act_acc`（教师动作 top-1 一致率，2026-09-04 新增，
  三种 reader 都有）、`ev_acc`（逐步）、`ev_acc_final`（终局）。

### 6.2 RL（`core/rl.py`，on-policy REINFORCE）

奖励（来源：**冻结外部分类器**判最终揭示子图，2026-09-04 决议维持）：

$$R_s=\begin{cases}\mathbb{1}[\argmax g(G_B)=y_s] & \text{binary（默认）}\\ \tanh(1-\mathrm{CE}(g(G_B),y_s)/\ln K) & \text{tanh}\\ +1\,/\,-0.1 & \text{asymmetric}\end{cases}$$

g = 冻结上界分类器，G_B = 预算耗尽时的揭示子图，K = n_classes，ln K 预计算。

优势（batch 级）：

$$A_s=\frac{R_s-\bar R}{\sigma_R+10^{-6}}\cdot\begin{cases}\text{neg\_scale} & A_s<0\\ 1 & A_s\ge 0\end{cases}\qquad(\texttt{--no-adv-normalize}\Rightarrow A_s=R_s)$$

策略损失与总损失：

$$L_{RL}=\frac{1}{N}\sum_s -A_s\sum_t\log\pi(a_t^s),\qquad L^{total}=L_{RL}+\lambda\cdot L_{ev}$$

- L_ev 公式与 BC 相同，但轨迹是 on-policy 采样；逐步 CE 在 rollout 当下
  收集，终局 CE 在 rollout 后补前向收集，batch 末合并；
- **梯度路由（护栏③）**：$L_{RL}$ 只经 $\sum_t\log\pi$ 回传，递归三件套
  不在其计算图上；L_ev 的监督梯度到达 evidence_head/pool/lstm 与（经
  Δ_t、KV）共享编码器。共享编码器因此拿两份梯度——策略梯度 + 监督梯度，
  这正是"L_ev 整形编码器"的机制；
- 奖励通路保持单一：A_s 与证据头无关。**自举风险明确否决**（决策迁就
  自身弱头的偏见、早期奖励噪声大），见 §15。

### 6.3 实施清单（已落地）

- `gnn/frontier/encoder.py`：`forward_all_batch`（全节点嵌入，与
  forward_batch 同特征同 GNN；device 从参数读取）；
- `reader/evidence.py`：`EvidenceReader(LearnableReader)` + HistoryPool +
  evidence_lstm + evidence_head；`_policy_scores` 自 base 提取共享；
- `active_graph_reader/config.py`：kind='evidence'（必填 n_classes；validate；
  to_dict/from_dict/from_legacy；`infer_reader_config` 靠 `evidence_lstm*`
  键判 kind、n_classes 从 evidence_head 权重 shape[0] 推断）；`build_reader` 分派；
- `train_reader/core/bc.py`/`rl.py`：λ·L_ev（含终局项）+ ev/act 计数器；
- `train_reader/core/eval_loop.py`：evidence 终判（§7）；
- `tests/test_evidence_reader.py`：7 组测试（§13.3）。

## 7. 评估（`eval.py` → `core/eval_loop.py`）

**终判 = 自身证据头**（2026-09-04 定稿，决策-分类一体落地）：

- kind=evidence 的 checkpoint：rollout 走 `forward_states_evidence`
  （**选点仍与 base 逐字节一致**，温度 1.0、argmax），chunk 结束后对全部
  state 做一次终局前向，`argmax(ev_logits)` 为预测计入主指标 `acc_*`；
- 冻结上界分类器对同一批最终揭示子图的判定降为参照：控制台 `[ref]` 行 +
  JSON 顶层 `acc_clf_reference`（逐预算 mean/std/n）+ results 内
  `acc_clf_*`/`clf_outcomes`——这是**选点质量**的诊断口径；
- base/attention 行为不变（上界分类器即终判，无 acc_clf_* 键）；
- 评估预算自动取 `spec.budget_options`（collab 已改 0.10/0.15/0.20 与
  GAM 对齐）；每图 `--n-seeds 4` 个确定性 seed（`deterministic_seed`）取平均。

**跨 reader 对比口径（重要）**：同一张 acc 图里，base/attention 的 acc_*
含义是"上界分类器判所选子图"，evidence 的含义是"自己判"。选点质量的
公平对比要用 evidence 的 `acc_clf_*` 对 base 的 `acc_*`；读图能力对比
才是 evidence 的 `acc_*` 对 base 的 `acc_*`（此时 evidence 多了"自判"
这一能力维度，是决策-分类一体的本意）。

训练期 ev_acc 与评估期 ev_acc 的口径差异：训练 ev_acc 在 on-policy（RL）
或 teacher 轨迹（BC）上按步统计；评估是 held-out 图 + 确定性 argmax
rollout + 终局判定。训练 ev_acc_final 可作评估终判的先导指标。

## 8. 参数与复杂度

| 模块 | 参数 | 备注 |
|---|---|---|
| frontier_encoder | 135,682 | 共享唯一 GNN，与 base 相同 |
| ReaderMLP（选点） | 33,665 | 与 base 相同 |
| LSTM 积分器 | ≈131,840 | d→d 单层 |
| AttnPool（Q/K/V 单头） | ≈49,152 | |
| 证据头 d→d→n_classes | ≈17,664 | 随 n_classes 线性增长 |
| **合计** | **≈368,000** | **2.2x base，0.48x attention** |

实测参考（cifar10，h128 两层、n_classes=10）：总参数 384,397
（base 168,579 / attention 770,245 同口径实测）。

运行时：每 state 每步多一次 LSTM 单步 + 一次对已揭示集的单查询注意力，
相对每步必付的子图 GIN 前向可忽略；无第二编码器，总前向开销 ≈ base。
终局补充前向每轨迹多一次共享编码前向（≈ 1/(B+1) 开销）。

## 9. 与 attention reader 的三点结构差异（为什么这次不会重蹈覆辙）

1. **单编码器**：Δ_t 来自共享编码器的既有输出，砍掉 217K 重复 GIN——
   病因 1 根除；
2. **递归不进决策**：pointer 打分退回 base 的无状态 MLP，h_t 只服务预测
   头，L_ev 是监督损失——病因 2 根除，策略优化形态与 base 完全一致；
3. **参数换信号**：132K 的 LSTM 换来每步 K 类稠密监督（病因 3 缓解），
   每步监督信号从 1 bit（动作）升到 1 bit + log(n_classes)。

## 10. GAM 对照表

| GAM 机制 | 判断 | 落点 |
|---|---|---|
| LSTM 历史累积 | **拿** | 积分器 h_t（但住在预测侧，不进选点） |
| 每步分类 | **拿** | 证据头逐步 CE（+ 终局项，GAM 没有的"读满终判"） |
| 混合损失（似然+奖励调制） | **拿** | L_act + λ·L_ev 两阶段各持一项（但 L_ev 永不进策略梯度，比 GAM 的 γ^(T−t) 调制干净） |
| 注意力 | **拿** | 历史引导读出 z_t（GAM 没有的角色，比它的类型注意力更守协议） |
| 节点类型注意力（看未读邻居） | 不拿 | 违反 T1/T2 |
| 随机起点多次读取投票 | 不拿 | 违反预算 |
| intent-vs-realized 状态 | 不拿 | h_t 的历史引导读出已覆盖其意图 |
| 奖励 = 自身预测对错 | **不拿** | 自举风险；奖励源 = 冻结外部分类器（§15） |

## 11. 代码地图

```
code/
├── active_graph_reader/
│   ├── config.py                  # kind='evidence'：validate（n_classes>0）、
│   │                              #   to/from_dict、from_legacy、checkpoint 推断
│   ├── gnn/frontier/encoder.py    # forward_all_batch：全节点嵌入（已揭示行在前）
│   └── reader/
│       ├── learnable_reader.py    # _policy_scores（budget+结构特征+MLP，base/evidence 共享）
│       └── evidence.py            # HistoryPool + EvidenceReader（forward_states_evidence、
│                                  #   init_evidence）
├── train_reader/
│   ├── core/bc.py                 # L_act + λ·L_ev（逐步+终局）、act_acc、ev_acc(_final)
│   ├── core/rl.py                 # REINFORCE + λ·L_ev、teleport 维护 last_revealed
│   ├── core/eval_loop.py          # evidence 终判 + acc_clf 参照 + acc_clf_reference 落盘
│   ├── train_bc.py                # --evidence-lambda（默认 1.0；kind=evidence 时自动
│   │                              #   注入 spec.n_classes）
│   ├── train_rl.py                # --evidence-lambda、--no-adv-normalize
│   └── eval.py                    # 无需新参数；kind 从 checkpoint config 重建
├── comparison/plot_training_curves.py  # 多面板曲线 + {name}_ev_bar.png（终局为主指标）
└── tests/test_evidence_reader.py  # 7 组测试（见 §13.3）
```

## 12. 使用命令

```bash
# BC（evidence；n_classes 从数据集 spec 自动注入）
python code/train_reader/train_bc.py --dataset collab --root D:\data\collab \
    --data-dir data/collab/trajectories_filtered --reader-kind evidence \
    --evidence-lambda 1.0 --epochs 30

# RL（bootstrap 自 BC；奖励源 = 冻结上界分类器）
python code/train_reader/train_rl.py --dataset collab --root D:\data\collab \
    --reader-kind evidence --bootstrap <bc.pt> \
    --classifier result/collab/classifier/<cls>.pt --reward binary

# 评估（终判 = 证据头；[ref] = 冻结分类器参照；无需新参数）
python code/train_reader/eval.py --dataset collab --root D:\data\collab \
    --checkpoint <rl.pt> --classifier <cls>.pt --split test

# 可视化（多面板曲线 + 终局证据 bar）
python code/comparison/plot_training_curves.py --json \
    result/collab/experiments/exp1/evidence/bc/training_log.json \
    result/collab/experiments/exp1/evidence/rl/training_log.json \
    --labels evidence/bc evidence/rl
python code/comparison/plot_eval_results.py --json <eval_heldout_test_reader.json>
```

## 13. 成功判据、消融与测试

### 13.1 判据（同预算同数据集）

1. 精度 ≥ base reader（重点观察 collab 稀疏长直径子集）；
2. 参数 ≤ base × 2.5、前向开销 ≈ base；
3. 逐步证据准确率随预算上升，v2 > v1 的差距集中在长直径图；
4. 终判（acc_*）与选点质量参照（acc_clf_*）的差距 = 自判能力的净值，
   预期随训练扩大后收敛。

### 13.2 消融（同一份代码的开关）

- 去 LSTM + 去 AttnPool（mean-pool + 直接头）= **v1**；
- 去 LSTM 留 AttnPool（query=可学习向量）；
- 去 AttnPool 留 LSTM（logits=MLP([pool_mean; h_t])）；
- λ ∈ {0, 0.5, 1.0}（λ=0 档见 §6.1 使用禁忌）。

v1 即 λ 消融之外的天然基线：**若 v2 ≈ v1，说明长直径积分假设不成立，
论文里就是一个诚实的负结果**。

### 13.3 测试覆盖（`tests/test_evidence_reader.py`，7 组全绿）

1. 选点路径与 base 逐字节一致（strict=False 载入 base 权重，scores allclose）；
2. 证据流形状/手工复算/前沿嵌入投毒不变性（×100 随机 → logits 不变）；
3. (h,c) 按 state 独立、reset 语义；
4. config 往返 / 权重 shape 推断 / checkpoint 重建；
5. 评估冒烟：acc_* + acc_clf_* 双套键、样本数、base 路径无 acc_clf_*；
6. BC 循环冒烟（λ=1 有 ev_acc/ev_acc_final/act_acc，λ=0 全无）；
7. RL 循环冒烟（teleport/last_revealed 维护 + ev_acc_final）。

## 14. 已知限制与使用注意

1. **λ 语义（2026-09-04 起）**：L_ev 含终局项后每轨迹 B+1 项，λ=1 时终局
   CE 与每步 CE 等权；与旧实验的 loss 数值不可直接比；
2. **终局项权重随预算稀释**：λ·L_ev 中终局占比 = 轨迹数/(逐步+轨迹数)，
   B=3 时 1/4、B=15 时 1/16。若终判欠拟合可考虑终局单独加权（暂不加，
   遵循简洁原则）；
3. **ev_acc 的 on-policy/teacher-forced 之别**：RL 与 BC 的 ev_acc 纵向
   对比时注意轨迹来源不同；
4. **λ=0 + evidence kind 的 eval 无意义**（随机头终判），消融对照用 base；
5. **跨 reader acc 语义**（§7）：base/attention = 分类器判子图；evidence =
   自判。选点质量对比走 acc_clf_*；
6. **冒烟陷阱**：用会小子图塌缩的分类器（如 finetune3）做奖励源/参照时，
   参照值恒为类 0 基础率（实测 0.1000），不代表管线错误——正式实验换
   不塌缩的分类器。

## 15. 决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-03 | v2 设计定稿并实施：单编码器 + 递归只住预测头 | attention reader 病因（§1）；简洁原则 |
| 2026-09-03 | T1 保留（前沿边特征通道） | v1 协议区分点；分类仍只读已揭示内容 |
| 2026-09-04 | 评估终判 = 自身证据头（决策-分类一体） | reader 定位即"自己判类别"；分类只应是已揭示内容的函数 |
| 2026-09-04 | 终局判定 = 读满后补充前向 | 逐步前向在 reveal 之前，最后一跳证据必须进终判；BC/RL/eval 语义统一 |
| 2026-09-04 | **RL 奖励源维持冻结外部分类器** | 外部分类器稳定、直接反映"是否提取到关键子图特征"，适合帮决策；证据头利用决策带来的历史积累判标签，是 reader 视角；own-head reward 有自举风险（决策迁就弱头偏见、早期奖励噪声） |
| 2026-09-04 | 预测头不加预算比例/结构特征 | 它们是策略侧信息；预测应是已揭示内容的函数（§3） |
| 2026-09-04 | 训练 bar/汇总图用 ev_acc_final（终局） | 逐步平均混合了"证据稀早期"的低准确率，终局才是判读能力 |
| 2026-09-04 | collab 预算档改 0.10/0.15/0.20 | 与 GAM 单智能体对齐实验一致 |

## 16. 明确否决的方案（防复活记录）

- ❌ 第二编码器 / NewInfoEncoder 复活（Δ_t 一律取自共享编码器）；
- ❌ 递归/信念进入选点打分（h_t 不给 ReaderMLP）；
- ❌ 多头大注意力块 / FFN Transformer 层堆叠（单头读出足矣）；
- ❌ 证据头参与 RL 奖励或选点（单一奖励通路）——**2026-09-04 复议维持**：
  奖励源 = 冻结外部分类器（稳定、反映关键子图提取质量）；自举风险
  （决策迁就自身弱头、早期奖励噪声）明确否决 own-head reward；
- ❌ λ 调度 / 课程学习（先证简单版有效）；
- ❌ 预测头拼接预算比例 / 前沿结构描述子（预测只应是已揭示内容的函数，
  且防"臃肿缝合"复发）。

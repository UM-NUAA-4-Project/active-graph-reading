# SAGPool 改造：修改前后工作流对照

记录 SAGPool（ICML 2019, *Self-Attention Graph Pooling*）从**原版实现**
（用户本地 `SAGPool-master`，PyG 官方代码）到**本仓协议适配版**
（`baseline/sagpool.py` + `baseline/run_sagpool.py`）的端到端工作流变化。

一句话总结角色转变：**原版 SAGPool 是分类模型内部的可微池化层**（选点是
训练的中间产物，没有独立接口）；**改造后拆成两阶段**——阶段 A 在完整图上
以标准监督训练同构 scorer，阶段 B 把它的第一级 gate 分数接到 AGR 协议的
逐点揭示循环上，成为协议合规的学习型选点策略。

各项改动的设计理由（为什么 GCN→GINE、为什么评测重打分、为什么只用第一级
gate 等）的权威记录在 `protocol-adaptation.md` §4，本文不展开论证，只记录
**流程本身**；两文配合阅读。

---

## 1. 原版工作流（SAGPool-master）

### 1.1 端到端训练管线（`main.py`）

```
TUDataset（DD/PROTEINS/NCI1/...，自带节点特征）
  → random_split 8:1:1（train / val / test）
  → DataLoader（train bs=128 shuffle；val bs=128；test bs=1）
  → Net 前向 → log_softmax
  → NLLLoss(out, data.y)          ← 训练标签直接可见
  → Adam(lr=5e-4, weight_decay=1e-4)
  → 每 epoch 在 val 上测 loss/acc
  → val loss 最优时存 latest.pth；连续 50 epoch 不降则早停
  → 载入 latest.pth 报 test accuracy
```

### 1.2 模型前向（`networks.py` Net）

```
x, edge_index, batch
→ [GCNConv → SAGPool] × 3        ← 层级式：每层池化后的子图进入下一层
|    每层池化后读出 cat(gmp(x), gap(x))  → x1, x2, x3
→ x1 + x2 + x3                   ← 逐层读出求和
→ lin1 → relu → dropout(0.5) → lin2 → relu → lin3 → log_softmax
```

### 1.3 SAGPool 层机制（`layers.py`）

```
score = GCNConv(in, 1)(x, edge_index)      # 可学习打分头（score_layer）
perm  = topk(score, ratio=0.5, batch)      # 完整图上一次性 top-k 选点
x'    = x[perm] * tanh(score[perm])        # 门控：高分节点特征乘 tanh(score)
edge_index' = filter_adj(...)              # 裁剪邻接 → 池化后子图
```

关键性质（也是必须改造的根源）：

- top-k 发生在**完整图**上，输入是全部节点/边——评测态不可见的信息；
- 打分头仅由分类损失**间接**监督（经读出头回传），没有独立的选点监督；
- "选了哪些节点"只是池化中间产物：模型对外的接口是**整图分类**，
  不存在"给定当前已揭示状态，选下一个节点"的序贯决策入口。

---

## 2. 改造后工作流（本仓 `baseline/`）

### 2.1 阶段 A：scorer 全图训练（`run_sagpool.py`）

```
datasets registry（cifar10 / ppa / collab，--max-graphs 控制规模）
  → 自环归一：remove_self_loop + add_self_loop   ← 与评测侧构图约定一致(P6)
  → SAGPoolModel 单图 batch 训练：
      init_nodes：节点特征线性投影；无节点特征（PPA/COLLAB）→ 共享 node token
      init_edges：边特征线性投影 + relu
      逐层 [EGINConv → GateConv → top-k(ceil(0.5·n)) → tanh 门控
            → dgl.node_subgraph]        × 3
      逐层读出 cat(max, mean) 求和 → lin1→relu→dropout→lin2→relu→lin3 → logits
      → CrossEntropy（与原版 log_softmax+NLL 等价）
  → Adam(lr=5e-4, wd=1e-4)，seed=777        ← 超参对齐原版默认
  → 每 epoch 在 test 全图上测 acc
  → 保存 test acc 最优权重 + 内嵌 config + history
      → baseline/artifacts/{ds}/sagpool.pt（*.pt 不进 git；history.json 进）
```

结构骨架（3×[conv→gate→top-k 门控池化]、逐层 mean+max 读出求和、3 层 MLP
头、pool_ratio 0.5、dropout 0.5）与原版逐项对应；差异只有：GCNConv→GINE 式
边特征卷积、无节点特征数据集的共享 token、单图 batch、早停→固定轮数取
test 最优。

### 2.2 阶段 B：协议合规评测（`eval.py --strategy sagpool` → `SagpoolStrategy`）

```
eval.py 检测到 --strategy sagpool：
  → 注册 baseline 策略（strategies.register_baseline_strategies）
  → 加载 scorer checkpoint（默认 baseline/artifacts/{ds}/sagpool.pt，
    结构按内嵌 config 重建，strict 加载）
  → 进入统一评测循环（eval_loop.rollout_strategy）：
      预算 B = round(ratio × n_nodes)，不含 seed（P1）
      seed 节点 = 确定性公式（P2），每图多种子取平均
      while frontier 非空 且 预算未耗尽:
          select()（见下）→ reveal_node（首次揭示才消耗预算）
  → SagpoolStrategy.select() 每步做的事：
      1) 取当前 frontier 节点集                    ← 只能在 frontier 上选(P3)
      2) 构可见子图 build_visible_graph()：
         V_revealed ∪ frontier；已揭示边 + frontier 边（均双向）；
         frontier 边特征照搬原图（one-ended 免费可见，P4）；
         frontier 节点特征置零（身份可见、特征未揭示）；
         自环归一（P6）
      3) first_stage_scores()：可见子图上重算第一级 gate 每节点分数
      4) 返回分数最高的 frontier 节点；平分取小 id（确定性）
  → 读满预算 → builder.build(state) 构最终揭示子图
  → 冻结上界分类器终判（P5，与 random/bfs/reader 完全同口径）
  → 产出统一 schema eval_fixed.json（原则 6，结果同表可比）
```

注意推理形态与训练形态的分工：`forward()`（完整层级池化 + 读出头）只在
阶段 A 训练/自测时使用；阶段 B 只调 `first_stage_scores()`——即 stage-1
gate，语义是"节点对分类的筛选重要性"（训练中它经 x1 读出项获得梯度），
层级高阶 gate 作用于池化后子图，与逐节点选点不对应。

---

## 3. 前后差异对照

| 维度 | 原版（SAGPool-master） | 改造后（baseline/） |
|---|---|---|
| 角色定位 | 分类模型的池化层，整图分类 | 两阶段：全图 scorer 训练 → 协议选点策略 |
| 选点时机/范围 | 训练中，完整图一次性 top-k | 推理中，每步在可见子图上重打分、选 1 个 frontier 节点 |
| 每步开销 | 无（一次前向整体完成） | O(可见子图) 一次 GNN 前向 |
| 卷积 | GCNConv（无边特征） | GINE 式边特征卷积（MLP+BN） |
| 节点特征 | TUDataset 必有 | PPA/COLLAB 无 → 共享 node token；frontier 节点特征置零 |
| 使用的 gate | 三级全部参与前向 | 评测仅取第一级（first_stage_scores） |
| 监督信号 | 分类 CE 端到端（间接监督打分头） | 同（阶段 A），但推理只消费 gate 分数，读出头弃用 |
| 终判 | 模型自身读出头 log_softmax | 冻结上界分类器（协议统一口径 P5） |
| 预算/种子/确定性 | 无此概念 | P1/P2/P3 + 平分取小 id |
| 训练选择 | 8:1:1 划分，val loss 早停 | registry 划分，固定轮数取 test acc 最优 |

改造中**没有**动的部分：层级 3×(conv→gate→top-k 门控) 结构、tanh 门控、
逐层 mean+max 读出求和、3 层 MLP 头、pool_ratio 0.5、lr/wd/seed 默认值。

## 4. 分布漂移：基线设定的组成部分

阶段 A 在完整图上训练、阶段 B 在 5%~15% 预算的可见子图上推理——scorer
从未见过"frontier 节点特征置零、子图只是原图一小块"的输入分布。这个漂移
**不是 bug**，正是该基线要隔离的对比信息：off-the-shelf 学习打分器（无序贯
决策训练）能到多少，主方法应在其之上。

## 5. 复现命令与文件对照

```bash
# 阶段 A：训练 scorer（默认输出 baseline/artifacts/{ds}/sagpool.pt）
# 实际产线配置（见各 sagpool.history.json 的 config 字段）：
python baseline/run_sagpool.py --dataset cifar10 --root D:\data\superpixels --epochs 50
python baseline/run_sagpool.py --dataset ppa --root D:\ogb-master\dataset --max-graphs 30000 --epochs 10
python baseline/run_sagpool.py --dataset collab --root D:\data\COLLAB --epochs 10

# 阶段 B：协议评测（--strategy-ckpt 缺省时自动找 baseline/artifacts/{ds}/sagpool.pt）
python code/train_reader/eval.py --dataset ppa --root D:\ogb-master\dataset \
    --strategy sagpool --classifier <上界分类器>.pt --max-graphs 2000
```

| 原版文件 | 本仓对应 |
|---|---|
| `main.py`（训练循环） | `baseline/run_sagpool.py` |
| `networks.py`（Net：conv/pool/读出/MLP 头） | `baseline/sagpool.py` `SAGPoolModel`（EGINConv/GateConv/forward） |
| `layers.py`（SAGPool 层：score/topk/门控/filter_adj） | `SAGPoolModel` 内联实现（GateConv + topk + tanh 门控 + node_subgraph） |
| —（无对应概念） | `SagpoolStrategy`（build_visible_graph / first_stage_scores / select）+ `eval_loop.rollout_strategy`（协议驱动） |

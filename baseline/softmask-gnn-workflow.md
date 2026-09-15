# Soft-Mask GNN：工作流与 AGR 接入方案

记录 Soft-Mask（WWW 2021, *Soft-Mask: Adaptive Substructure Extractions for
GNNs*，用户本地 `soft-mask-gnn-main`，PyG 官方代码）的原版端到端工作流，
以及移植为 AGR 协议基线的接入方案。

**状态：方案设计，未实现。** 接入后应在 `protocol-adaptation.md` 增补
决策表（理由论证放那边，本文只记录流程与方案）。

一句话总结：Soft-Mask 与 SAGPool 同类——**分类模型内部的软掩码打分层**。
每层由掩码生成器给每个节点输出 $m_v\in(0,1)$，消息传递被掩码调制
（$x\odot m$ 参与聚合），端到端分类监督下"逐层软提取子结构"。原版没有
序贯/预算/揭示概念，掩码打分发生在完整图上，接入方案与 SAGPool 适配
同构：全图训 scorer → 评测在可见子图上重算第一层掩码分数 → 贪心选
frontier 最高分。

---

## 1. 原版工作流（soft-mask-gnn-main）

### 1.1 端到端训练管线（`classification_main.py`）

```
TUDataset（MUTAG/PROTEINS/COLLAB/...；edge_attr 被丢弃）
  → 无节点特征的数据集自动补特征：
      max_degree < 1000 → OneHotDegree；否则归一化度数（NormalDegree）
  → StratifiedKFold 10 折（seed 777），每折 9:1 train/val
  → DataLoader（bs=128，train shuffle）
  → SMG 前向 → log_softmax
  → NLLLoss
  → Adam(lr=5e-4) + StepLR(step=50, γ=0.75)，151 epochs
  → 跨折按 epoch 平均准确率，取最优 epoch 的均值±std 汇报
```

超参考例（`configs/COLLAB.json`）：layers=3 / hidden=256 / bs=128 /
lr=5e-4 / step=50 / decay=0.75 / seed=777。**COLLAB 在原版配置里现成**，
与我们第三个数据集同源。

### 1.2 模型前向（`models/classification.py` SMG）

```
x → lin0 → hidden
→ × num_layers 层：
    mask_val = WeightConv(x, edge_index, 上一层 mask_val)   # 掩码生成器
    x        = relu( SparseConv(x, edge_index, mask_val) )  # 掩码调制卷积
→ global_add_pool → lin1 → relu → dropout(0.5) → lin2 → log_softmax
```

变体：`SMG_JK`（逐层 add_pool 后 JumpingKnowledge 拼接）、`SMG_2h`（掩码
生成走两跳：一阶聚合特征 ma1h → 二阶生成掩码 ma2h）、`SMG_2h_JK`。

### 1.3 两个核心算子（`conv/`）

```
WeightConv1（掩码生成器）：
    x* = x ⊙ mask_prev（首层 mask=None 跳过）
    aggr = Σ_{u∈N(v)} lin1(x*)_u                # 邻域聚合
    m_v  = σ( lin4( relu( lin3( [aggr ‖ lin2(x*)_v] ) ) ) )   ∈ (0,1)

SparseConv（掩码调制卷积）：
    h    = (x ⊙ m) W
    out_v= Σ_{u∈N(v)} h_u + lin(x_v)            # 输出整体再 ⊙ m
    （内部 remove_self_loops；edge_weight 可作消息权重，分类管线未用）
```

另有 `SparseConvR`（NNConv 式，`nn(edge_attr)` 生成边权重矩阵）与
`WeightConv2`（MLP 版掩码生成器）——**边特征感知的积木原码已存在**，
只是分类管线把 `edge_attr` 丢了（TU 无边特征）。

关键性质（接入 AGR 前必须认清的根源）：

- 掩码每层动态重算、软连续（sigmoid 概率），**不是排序/选点**——全节点
  保留、按权重缩放，"子结构"体现在表征层，没有硬 top-k；
- 掩码生成的聚合用**完整图邻域**（评测态不可见）；
- 掩码仅由分类损失间接监督，无独立选点监督；无预算/时序概念。

---

## 2. AGR 接入方案（候选基线：soft-mask）

角色对标 SAGPool 适配（两阶段），实现上可大量复用 `baseline/sagpool.py`
的骨架（`build_visible_graph` / Strategy 包装 / artifacts 约定）。

### 2.1 阶段 A：scorer 全图训练（新建 `baseline/run_softmask.py`）

```
datasets registry（cifar10/ppa/collab）
  → 节点特征：有则线性投影；无（PPA/COLLAB）→ 对齐原版方案用度数特征
    （OneHotDegree 或归一化度数；比 SAGPool 适配的共享 token 更忠实原版），
    或共享 token（与 sagpool.py 约定统一，二选一须在 config 记录）
  → 边特征：复用 SparseConv 的 edge_weight 通道或 SparseConvR 思路，
    把 7 维边特征投影为标量/向量消息权重（PPA 必需；cifar10 边特征同理）
  → SMG 前向（DGL 复刻 WeightConv1 + SparseConv）→ CE，完整图标准监督
  → 单图 batch、自环 remove+add 归一（与 sagpool.py 约定一致）
  → 产物 baseline/artifacts/{ds}/softmask.pt + 内嵌 config
```

### 2.2 阶段 B：协议合规评测（新建 `SoftmaskStrategy`）

```
eval.py --strategy softmask（choices 与注册三处接线同 sagpool：
  eval.py choices / strategies.register / artifacts 默认路径）
  → 每步 select()：
      1) 取当前 frontier（P3）
      2) build_visible_graph：V_revealed ∪ frontier，one-ended 边特征，
         frontier 节点特征置零，自环归一（P4/P6；直接复用 sagpool.py）
      3) 在可见子图上重算第一层掩码分数 m_v（WeightConv1 单层前向；
         首层无 mask_prev，输入不依赖未揭示信息）
      4) 贪心选 m_v 最高的 frontier 节点；平分取小 id（原则 4）
  → 读满预算 → builder.build → 冻结分类器终判（P5）→ 统一 eval_fixed.json
```

### 2.3 设计要点与风险

| 点 | 决策 | 说明 |
|---|---|---|
| 取哪层掩码 | **第一层** | 与 SAGPool 取 stage-1 gate 同理：首层输入是原始特征，无依赖；深层掩码作用于被掩码调制的表征，与逐节点选点不对应 |
| 分数语义 | m_v∈(0,1) 直接比大小 | 无需 top-k；sigmoid 单调，也可用 pre-sigmoid logit 避免饱和平分 |
| 与 SAGPool 的对照价值 | 软加权 vs 硬 top-k | 回答"打分器的监督形态（软掩码调制 vs 硬选择池化）对序贯揭示迁移的影响" |
| 已知风险 | 分布漂移同 SAGPool | 全图训练、5%~15% 可见子图推理；属基线设定一部分（原则 2） |

### 2.4 复现命令（实现后）

```bash
python baseline/run_softmask.py --dataset collab --root D:\data\COLLAB
python code/train_reader/eval.py --dataset collab --root D:\data\COLLAB \
    --strategy softmask --classifier <上界分类器>.pt
```

# 机制例图：CIFAR10 选低前沿度节点 = 前沿扩张（graph 78 / seed 1 / 第 6 步）

**一句话**：在真实 CIFAR10 超像素图上，reader（epoch6）以 p = 0.144 选中一个
前沿度只有 1 的节点，一步换来 **+18 条新可见前沿边特征**；而前沿度最大的
候选（fd = 3）同样揭示只会带来 +6 条——低 fd 选择最大化了每步新获取的
边界信息。

## 例图参数

| 项 | 值 |
|---|---|
| checkpoint | `code\result\cifar10\experiments\ev-R-read\evidence\rl\checkpoint_epoch6.pt` |
| graph_idx（test 本地）/ seed | 78 / seed_i = 1 → 种子节点 27（deterministic_seed(78, 1, 115)） |
| 图规模 / 预算 | n = 115，B = 11（10%·n） |
| 类别 / 终判 | label = 3，证据头 ev_pred = 3（正确） |
| 展示步 | dump 第 5 步（0 基；已揭示 6 节点，即第 6 次揭示选择点）→ 标题记作 Step 6/7 |
| 协议 | 与 eval 严格同款：argmax 确定性 rollout、无 teleport |
| 数据来源 | 表格数字 = `candidate_rollouts_b10_g100s4.json`；拓扑 = 同款协议重放 |

## 标注含义

### 面板 a（Step 6：揭示前）

| 元素 | 含义 |
|---|---|
| 钢蓝圆形（中心） | 已揭示节点；深藏青菱形 = 种子节点 |
| 外环圆圈（奶油→深红渐变） | 27 个前沿候选，按 reader 选择概率着色（底部色条 0–0.14） |
| 灰虚线 / 实线 | 前沿边（揭示区↔候选，协议中免费可见）/ 已揭示区内部边 |
| 红色虚线圆圈 | reader 本步实际选中（argmax）的节点 n.78，旁注 p = 0.144 |
| 深灰虚线圆圈 | 对照候选 = 当步前沿度最大的节点 n.50（fd = 3） |

### 面板 b（Step 7：揭示后）

| 元素 | 含义 |
|---|---|
| n.78 变为蓝色 | 被选节点进入已揭示区（红虚线圈保留追踪） |
| **橙色实线段（核心标注）** | 揭示 n.78 后**新变为可见的前沿边**（共 18 条：n.78 的 19 个邻居中除 1 条连已揭示区外的全部） |
| 白色圆圈 | 揭示后的新前沿（|F| 27 → 43） |

### 面板 c（表格）

| 列 | 含义 |
|---|---|
| fd（frontier degree） | 该候选与已揭示区的连接边数 |
| deg | 开邻域大小（不含自环） |
| ΔE | 揭示该节点后新增可见前沿边数 = deg − fd |
| p | reader 给该候选的选择概率 |

## 为什么 fd 低 → ΔE 大

协议规定前沿边的特征在揭示前**免费可见**（一端已揭示即可查询）。节点 v 的
边分成两类：连向已揭示区的 fd 条（揭示前已可见，不带来新信息）和连向未
揭示区的 deg − fd 条（揭示 v 后才变为可见的前沿边）。所以
**ΔE(v) = deg(v) − fd(v)**：fd 越低、度数越高，一步揭示换来的新边特征越多。
fd 和 deg 都是揭示前可观测的信号——reader 学会了利用它们做信息最大化选择。

## 本例读数

| 候选 | fd | deg | ΔE | p |
|---|---|---|---|---|
| **Reader 选中 n.78** | 1 | 19 | **+18** | 0.144 |
| 最大前沿度 n.50 | 3 | 9 | +6 | 0.001 |

选中节点的 ΔE 是最大 fd 候选的 3 倍；|F| 由 27 增至 43。

## 聚合证据（防止"挑例子"质疑）

epoch6 reader，test 前 100 图 × 4 seed，共 4,516 个策略步
（`candidate_rollouts_b10_g100s4.json`，子集口径 ev_acc = 0.375）：

| 统计 | 值 |
|---|---|
| P(选中 fd ≤ 2) | **100%**（4516/4516） |
| 有 fd≥3 候选的步 | 3,439 步（76%）——其中选 fd≥3 的次数：**0** |
| ΔE（被选） | 9.22 [9.16, 9.29]（bootstrap 95% CI） |
| ΔE（最大 fd 反事实） | 8.47 [8.38, 8.56] |
| ΔE 相对前沿平均 | +0.29/步（严格更优 55% / 并列 13% / 更差 32%） |
| 揭示子图内边密度 | 0.81 edges/node（对照 COLLAB 2.87，见另一张机制图） |

## 结论与措辞注意

1. **结论**：CIFAR10 上 epoch6 reader 的"轮廓行走"（99% 选 fd=1）在功能上
   等价于前沿扩张策略——每步最大化新可见前沿边特征。选点打分利用了协议
   免费提供的可观测结构信号（fd、deg）。
2. **定位**：这是事后行为分析（机制解释），不是精度来源证明；ΔE = deg − fd
   本身是算术恒等式，主张落在"reader 真的在按这个可观测信号选择并兑现"
   （100% 不碰 fd≥3 + ΔE 显著高于反事实），例图只是让机制可见。
3. 引用行为规律时注意 checkpoint 代际：fd=2"凹角填充"的旧结论只对 epoch3
   成立（见 `selection_process_b10.md` 的行为对照表）。

## 产物与复现

- 图：`mechanism_expansion_cifar10_g78s1.{svg,pdf,tiff,png}` +
  对齐/碰撞审计文件（QA：面板对齐 1.5pt strict、碰撞 0 fail、文本 ≥5pt、
  visual-judge PASS）
- 逐候选数据：`candidate_rollouts_b10_g100s4.json`（graph_idx=78, seed_i=1, step=5）
- 重生成：

```bash
python code/diagnosis/plot_mechanism_example.py --claim expansion \
    --dataset cifar10 --root D:\\data\\superpixels \
    --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt \
    --dump code/result/cifar10/experiments/ev-R-read/evidence/rl/behavior-analysis/candidate_rollouts_b10_g100s4.json \
    --graph-idx 78 --seed-i 1 --step 5
```

- 姊妹篇：`code/result/collab/experiments/evidence1/evidence/rl/behavior-analysis/mechanism_densification_collab_g7s0.md`（COLLAB 致密化机制，两图配对使用）

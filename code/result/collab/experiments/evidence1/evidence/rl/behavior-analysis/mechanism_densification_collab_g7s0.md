# 机制例图：COLLAB 选高前沿度节点 = 闭合社区中心（graph 7 / seed 0 / 第 6 步）

**一句话**：在真实 COLLAB 共现图上，reader（epoch6）以 p = 0.930 的最高
置信选中当步前沿度最大的候选 n.35——一个连接全图 61/62 个节点的枢纽
（社区中心）；揭示它把已揭示区一口气闭合成致密核（新闭合 15 个三角形，
内边密度 2.5 → 3.0），而不是像低 fd 选择那样只延伸出一条稀疏路径。

## 例图参数

| 项 | 值 |
|---|---|
| checkpoint | `code\result\collab\experiments\evidence1\evidence\rl\checkpoint_epoch6.pt` |
| graph_idx（test 本地）/ seed | 7 / seed_i = 0 → 种子节点 38（deterministic_seed(7, 0, 62)） |
| 图规模 / 预算 | n = 62，B = 6（10%·n） |
| 类别 / 终判 | label = 0，证据头 ev_pred = 0（正确） |
| 展示步 | dump 第 5 步（0 基；已揭示 6 节点，即第 6 次揭示选择点）→ 标题记作 Step 6 |
| 协议 | 与 eval 严格同款：argmax 确定性 rollout、无 teleport |
| 数据来源 | 表格数字 = `candidate_rollouts_b10_g100s1.json`；拓扑 = 同款协议重放 |

## 标注含义

### 面板 a（Step 6：揭示前）

| 元素 | 含义 |
|---|---|
| 钢蓝圆形（中心） | 已揭示的 6 个节点；深藏青菱形 = 种子节点 |
| 外环圆圈（奶油→深红渐变） | 14 个前沿候选，按 reader 选择概率着色（色条 0–0.93） |
| 灰虚线 / 实线 | 前沿边（免费可见）/ 已揭示区内部边 |
| 红色虚线圆圈 | reader 本步选中 n.35，旁注 p = 0.930（全前沿最高） |
| 深灰虚线圆圈 | 对照候选 = 当步前沿度最小的节点 n.39（fd = 2） |

### 面板 b（Step 7：揭示后；布局与 a 完全相同，便于对照）

| 元素 | 含义 |
|---|---|
| n.35 变为蓝色 | 被选枢纽进入已揭示区（红虚线圈保留） |
| **深红实线辐条（核心标注）** | n.35 指向 6 个已揭示邻居的 fd 条边——fd = 6 即它把已揭示区所有可连的边都收了回来 |
| **琥珀色边（核心标注）** | 这 6 个已揭示邻居**彼此之间已有的 15 条边** = 揭示 n.35 新闭合的三角形；邻居互连致密正是"社区"的结构证据 |
| 白色圆圈 | 其余前沿节点（未变） |

脚注注明 `|F| = 14 → 55`：枢纽的 55 个新邻居形成的新前沿**未画入**
（画了会糊成一团；本图重点是已揭示区内的致密化，不是前沿扩张）。

### 面板 c（表格）

| 列 | 含义 |
|---|---|
| fd / deg / ΔE | 与 cifar10 图同义（连已揭示区边数 / 开邻域 / 新增可见前沿边 = deg − fd） |
| tri | 揭示该节点新闭合的三角形数 = 其已揭示邻居之间已有的边数 |
| p | reader 选择概率 |

## 为什么高 fd → 社区中心

共现图的社区 = 致密子图，社区中心 = 高度枢纽。一个候选若与已揭示区有
很多连接（fd 大）**且**这些已揭示邻居彼此也互连致密（闭合三角形多），
它就坐在社区核的中心：揭示它不是把路径向外延伸一格，而是把整个致密核
收进观察范围。子图计数统计（度、密度）正是 COLLAB 分类信号的来源。

## 本例读数

| 候选 | fd | deg | ΔE | tri | p |
|---|---|---|---|---|---|
| **Reader 选中 n.35** | 6（当步最大） | 61/62 | +55 | **15** | 0.930 |
| 最小前沿度 n.39 | 2 | 3 | +1 | 1 | 0.001 |

揭示后：内边 15 → 21（密度 2.50 → 3.00 edges/node），|F| 14 → 55。
对照候选（fd=2, deg=3）是"沿稀疏路径再走一格"的选择，既拿不到新信息
（ΔE=1）也闭合不了社区（tri=1）。

## 聚合证据（防止"挑例子"质疑）

epoch6 reader，test 前 100 图 × 1 seed，共 663 个策略步
（`candidate_rollouts_b10_g100s1.json`，ev_acc = 0.54，多数类水平）：

| 统计 | 值 |
|---|---|
| P(选中 fd = 当步最大 fd) | **100%**（663/663） |
| 闭合三角形（被选） | 29.5（比前沿平均高 +2.5） |
| 揭示子图内边密度 | 2.87 edges/node（cifar10 对照 0.81——扩张 vs 致密化） |

## 结论与措辞注意

1. **结论**：COLLAB 上 reader 的选点功能上等价于"锁进社区核"——100% 选
   当步最大 fd，被选节点系统性地闭合更多三角形；与 CIFAR10 的扩张策略
   合起来构成一个对称故事：同一 reader 在两类图上把免费可观测的 fd 信号
   用向相反但同样服务信息获取的方向。
2. **不要写置信跳变**：实测子图分类器与 reader 证据头的逐步熵都基本不动
   （分类器 ΔH = +0.02、翻对率 ≈ 0）——类决策只在终局形成，机制证据是
   结构性的（闭合、密度），不是置信度的。
3. **精度口径**：COLLAB 上 reader 是多数类水平（ev_acc ≈ 0.52–0.54），
   本图解释的是"选点策略在做什么"，不是"精度从哪来"；与导师定的类水平
   表述不冲突。
4. "读对 vs 读错 episode 的平均 fd 不同（3.4 vs 7.4）"是步数/图规模混杂，
   不要引用。

## 产物与复现

- 图：`mechanism_densification_collab_g7s0.{svg,pdf,tiff,png}` +
  对齐/碰撞审计文件（QA：面板对齐 1.5pt strict、碰撞 0 fail、文本 ≥5pt、
  visual-judge PASS）
- 逐候选数据：`candidate_rollouts_b10_g100s1.json`（graph_idx=7, seed_i=0, step=5）
- 重生成：

```bash
python code/diagnosis/plot_mechanism_example.py --claim densification \
    --dataset collab --root D:\\data\\COLLAB \
    --checkpoint code/result/collab/experiments/evidence1/evidence/rl/checkpoint_epoch6.pt \
    --dump code/result/collab/experiments/evidence1/evidence/rl/behavior-analysis/candidate_rollouts_b10_g100s1.json \
    --graph-idx 7 --seed-i 0 --step 5 --device cuda:0
```

- 姊妹篇：`code/result/cifar10/experiments/ev-R-read/evidence/rl/behavior-analysis/mechanism_expansion_cifar10_g78s1.md`（CIFAR10 扩张机制，两图配对使用）

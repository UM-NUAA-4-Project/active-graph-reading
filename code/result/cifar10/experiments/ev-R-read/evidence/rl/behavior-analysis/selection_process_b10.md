# 选择过程图：真实 CIFAR10 图上的"打分 → 揭示 → 新前沿"（预算 10%）

**一句话**：冻结 reader（`checkpoint_epoch6.pt`）在一张真实 CIFAR10 超像素图上，
第 5 步对 24 个前沿候选逐一给出选择概率（a 面板 viridis
着色），选中概率最高的节点（红星，p = 0.147）揭示后，前沿推进为
30 个新候选（b 面板空心圈）。

## 例图参数

| 项 | 值 |
|---|---|
| checkpoint | `code\result\cifar10\experiments\ev-R-read\evidence\rl\checkpoint_epoch6.pt` |
| graph_idx（test 本地序号）/ 种子 | 6 / deterministic_seed(6, 0, n) → 节点 84 |
| 图规模 / 预算 | n = 109，B = 10（10%·n） |
| 类别 / 终判 | label = 1，证据头 ev_pred = 1（正确） |
| 协议 | 与 eval 严格同款：argmax 确定性 rollout、无 teleport |

## 第 5 步快照

- 揭示前：已揭示 5 节点（含种子），前沿 |F| = 24；
- 被选节点：节点 89，前沿度数 = 1（与已揭示区恰有 1 条连接：沿前沿延伸路径（contour walking）。这是补训后 checkpoint（epoch6）的主行为——见下方"行为注意"），
  选择概率 p = 0.1468（次高 0.1467）；
- 揭示后：已揭示 6 节点，新前沿 |F| = 30。

## 读图

- **a**：蓝 = 已揭示，菱形 = 种子；前沿候选按 reader 选择概率 viridis 着色
  （底部色条），红星 = argmax 被选节点；灰虚线 = 前沿边（协议中免费可见）。
  只画读者可观察的已揭示区 + 前沿，未揭示节点不显示。
- **b**：揭示被选节点（红描边蓝点）后的新前沿（空心圈）：紧邻新邻居入列。

## 行为注意：epoch3 → epoch6 策略变化

用完全相同的 eval 协议 rollout 在 test 前 400 图上逐步统计"被选节点前沿度数"：

| checkpoint | 第 2 步 | 第 3–9 步 | 行为 |
|---|---|---|---|
| epoch3（旧主结果，行为分析 README 所用） | 98.3% 选 fd=2 | 84–91% 选 fd=2 | 凹角向内补，紧致圆盘生长 |
| epoch6（补训后，本图所用） | 99.8% 选 fd=1 | ≈99% 选 fd=1 | 沿前沿延伸路径，轮廓行走 |

补训不只提升精度（B10% ev_acc 29.9 → 33.8），还**定性地改变了读图策略**。
`behavior-analysis/README.md`（基于 epoch3）的"度数=2 凹角填充"结论对 epoch3
成立，但不再适用于 epoch6 主结果；引用行为规律时注意区分。

## 产物

`selection_process_b10_step5.json`（逐步快照 +
每个候选的概率表）、`.svg/.pdf/.tiff/.png`、对齐/碰撞审计文件。
源码：`code/diagnosis/visualize_selection_process.py`。

## graph18 拓扑图 + reader 头 IG 表（2026-09-25）

`selection_process_b10_graph18_topo.*`（源码
`code/diagnosis/plot_selection_topology.py`）：三面板 220×72 mm——
a = 第 2 步揭示前（前沿按分数渐变成环 + 红虚线选中圈），b = 揭示后
（|F| 12→15），c = 三种选点口径对 **reader 证据分类头**的一步前瞻
信息增益表。IG 定义：IG(v) = H(reader | R) − H(reader | R ∪ {v})
（nats）——从 checkpoint 重放 eval rollout 到第 2 步取证据流现状
（(h, c) 与逐步预测分布），对每个候选以同款证据流前进一步算熵降；
重放与 step2to3 JSON 逐项校验一致（seed/chosen/前沿集/逐点分数/
前沿度数）。逐候选数据在 `selection_process_b10_graph18_topo_ig.json`。

| 选点口径 | 节点 | p | fd | IG (nats) |
|---|---|---|---|---|
| Reader 选中 | n.50 | 0.498 | 1 | +0.0238 |
| 最大前沿度（5 个 fd=2 并列，取组内 IG 最大） | n.94 | 0.001 | 2 | +0.0329 |
| 前沿最大 IG | n.57 | 0.002 | 1 | +0.0824 |

现状 reader 头熵 H = 1.885 nats。**读法**：reader 以 p = 0.498 高概率
选中的 n.50 并不是对自己分类头增益最大的节点——n.57 的 IG（+0.082）
是它的 3.5 倍，但选择概率只有 0.002。即 epoch6 的选点行为（轮廓行走
fd=1）不能解释为"为自己证据头挑最大熵降"：选点打分与证据增益解耦，
与 evidence reader 设计（证据流不参与选点）自洽。

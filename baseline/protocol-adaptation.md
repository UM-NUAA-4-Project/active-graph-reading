# baseline 对 AGR 协议的适配说明

本文档专门记录 `baseline/` 下各复现基线对 **AGR(Active Graph Reading)协议**
的适配改动与理由。协议本体定义见 `AGENTS.md`(Protocol / Self-loop rules /
Budget check),架构决策见 `docs/decisions.md`;本文不重复协议定义,只回答
一个问题:**每个基线从"原文设定"到"协议可行 agent"改了什么、为什么。**

## 1. 协议要点(评测侧约束)

| # | 约束 | 内容 |
|---|---|---|
| P1 | 预算语义 | Budget = 揭示节点数(**不含 seed**),按图取比例 `B = round(ratio × n_nodes)`;预算档 cifar10/ppa = 0.05/0.10/0.15,collab = 0.10/0.15/0.20 |
| P2 | 种子 | 确定性公式 `(graph_idx*31 + seed_i*17 + 7) % n_nodes`,每图多种子取平均 |
| P3 | 选点约束 | 每步只能从**当前 frontier**(与已揭示区相邻的未揭示节点)中选;`reveal_action` 只接受 frontier 节点 → 揭示区域与 seed 始终连通 |
| P4 | 可见信息 | 已揭示节点的特征 + 已揭示边(含特征)+ **frontier 边的 one-ended 特征(免费可见)**;frontier 节点的**身份**可见,其**节点特征不可见** |
| P5 | 终判 | 读满预算后,冻结上界分类器对最终揭示子图分类(learner 侧评测走统一 eval.py) |
| P6 | 自环 | 揭示子图构图时恰好 1 条自环/节点(remove+add 归一) |

## 2. 通用适配原则

所有 baseline 复现一律遵循,每份基线的具体改动都是这些原则的实例化:

1. **评测态协议可行**:策略在推理时只能消费 P4 列出的可见信息。任何在
   完整图上预计算的全局量(全局中心性、全图 top-k 分数、全图注意力)
   都等于偷看未观测数据,禁止出现在选点逻辑中。
2. **训练全知、评测可见**:训练阶段允许使用完整图监督(与主方法的
   BC/RL 训练同权);推理时切换到可见子图。由此产生的分布漂移是基线
   设定的一部分,不是 bug。
3. **预算记账**:只有"首次揭示"消耗预算;游走重访、搜索回溯、瞬移不消耗。
4. **确定性**:同一 (graph, seed_i) 必须产出同一轨迹。策略内部随机源从
   `(seed_node, n_nodes)` 算术播种(禁止 `hash(tuple)`,其进程间随机化);
   平分 tie-break 取小 id。
5. **无标签推理**:评测时图标签不可用。凡原方法依赖"已知预测类别"的
   机制(如解释器的 Shapley 目标),必须替换为类别无关奖励。
6. **统一产出**:基线评估一律走主仓 `code/train_reader/eval.py`
   (`--strategy <name>`),产出统一 schema 的 `eval_fixed.json`,
   与 random/bfs 及 reader 结果同表可比;不为 baseline 另造评估格式。

## 3. random-walk(GraphSAGE 一族)——协议适配

对应 GraphSAGE 式邻域随机采样读取;无训练,纯启发式。

| 维度 | 原文/直觉设定 | 本实现 | 理由 |
|---|---|---|---|
| 游走 | 随机游走连续访问节点 | 沿随机边游走,**首次到达**未揭示节点才揭示 | P1:重访不消耗预算,预算=首次揭示数 |
| 选点范围 | 游走天然沿边走 | 每步返回值必须 ∈ 当前 frontier(P3) | 游走到达的未揭示节点必然与已揭示区相邻,二者天然一致 |
| 可见信息 | — | 仅当前节点的邻接(`state.g` 的 succ∪pred,去自环) | 爬虫站在当前节点上可见其邻居;不触碰全局结构 |
| 死循环防护 | — | 游走步数上限 `4·n_nodes+16`,超限退化 frontier 均匀随机(瞬移) | 连通分量内可能反复重访;瞬移不改变预算语义 |
| 确定性 | — | RNG = `random.Random(seed_node*1_000_003 + n_nodes)` | 原则 4;`random` 全局 RNG 不可复现 |

与现有 `random`(frontier 均匀采样,广度型)互补:random-walk 是深度型。

## 4. SAGPool(ICML 2019)——协议适配

复刻官方实现骨架(用户本地 SAGPool-master,PyG 版):3×(conv → gate →
top-k 门控池化) 层级结构、逐层 mean+max 读出求和、3 层 MLP 头;超参对齐
原默认(hidden 128 / pool_ratio 0.5 / dropout 0.5 / lr 5e-4 / wd 1e-4)。

| 维度 | 原文设定 | 本实现 | 理由 |
|---|---|---|---|
| 卷积 | GCNConv(无边特征) | GINE 式边特征卷积(MLP+BN) | PPA/COLLAB 无节点特征,消息传递必须吃边特征;边特征是协议可见信息 |
| 训练 | 完整图 top-k 池化 + 分类损失 | 完整图训练(单图 batch,与原实现单图处理一致),自环 remove+add 归一 | 原则 2:训练全知合法;构图约定与评测侧一致(P6) |
| **评测选点** | **完整图上一次性 top-k** | **每步在 `V_revealed ∪ frontier` 可见子图上重算第一级 gate 分数,取最高分 frontier 节点** | 原文选点用未观测的全局结构,且不保证与 seed 连通(P3 违规);重打分后每步仍是 O(可见子图) 一次前向 |
| frontier 节点输入 | 不存在该概念 | 身份与边特征可见(One-ended);**节点特征置零** | P4:节点特征未揭示;置零而非插值,避免虚构信息 |
| 评分函数 | 各级 gate(Conv(in,1)) | 评测取**第一级 gate**(`first_stage_scores`) | 训练中 stage-1 gate 经 x1 读出项获得梯度,语义即"节点对分类的筛选重要性";层级高阶 gate 作用于池化后子图,与逐节点选点不对应 |
| tie-break | topk 内部序 | 平分取小 id(`max(key=(score, -id))`) | 原则 4 |
| 读出头 | log_softmax | logits(CE 内置 softmax) | 与主仓 classifier 约定一致 |

评测形态要点:SAGPool scorer 在完整图上训练后,放到 5%~15% 预算的可见
子图上存在分布漂移——这是"off-the-shelf 学习打分器"基线设定的组成部分,
主方法应在其之上;这正是该基线要隔离出的对比信息(有无序贯决策训练的差)。

## 5. 计划中基线的适配预决

**SubgraphX(ICML 2021)**——实现前先固定三条适配,避免临场发挥:

1. **连通性约束搜索**:原版在完整图上搜索任意子图;改为以 seed 为根的
   树搜索,MCTS 子节点 = 当前可加的 frontier 节点(P3),搜索状态 = 揭示集合。
2. **无标签奖励**:原版 Shapley 目标假设已知预测类别;改为类别无关,
   如 `max_c logit_c(classifier(S))` 或 top-1/top-2 margin(原则 5)。
3. **算力上限**:Shapley 采样 × 每候选一次分类器前向,单图成本高;设
   每 graph 候选评估次数上限并作为超参报告(与 P1 预算分列,两本账)。

## 6. 已否决选项(决策记录,防止重复讨论)

| 选项 | 否决理由 |
|---|---|
| **GAM**(KDD 2018) | ① step 网络按"节点标签注意力"选步,cifar10 连续特征不可用、collab/ppa 无节点标签只能度数分桶合成(信息口径已失真);② 迁移到 frontier 边特征需改动其源码,既非忠实复现也非干净新方法。代码已清理(git 历史可查 `baseline/gam_src` 等)。 |
| **全局中心性 top-k**(degree/PageRank/k-core) | ① 信息违规:需要完整图拓扑,评测时不可见(原则 1);② 机制不可行:top-k 序列中大部分节点不在 frontier 上,P3 下无法揭示。合规变体 = 局部 frontier-degree greedy,已在主仓实现为固定策略 `max-frontier-degree`,无需在 baseline/ 重复。 |

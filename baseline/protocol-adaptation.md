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
   (唯一例外:planner 型基线 SubgraphX,见 §5 的口径说明,结果行须显式标注。)
2. **训练全知、评测可见**:训练阶段允许使用完整图监督(与主方法的
   BC/RL 训练同权);推理时切换到可见子图。由此产生的分布漂移是基线
   设定的一部分,不是 bug。
3. **预算记账**:只有"首次揭示"消耗预算;游走重访、搜索回溯不消耗。
   瞬移(teleport_if_stuck,reader lockstep 评估在 frontier 空且预算未满时
   调用)= 一次伪随机 reveal,**消耗 1 预算**(reveal_action.py docstring
   明确);固定策略 rollout 契约无瞬移,frontier 空(预算未满)即终止
   rollout,剩余预算作废——个别策略的内部兜底见 §3/§5/§6/§7。
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
**端到端工作流(原版训练管线 → 两阶段改造:全图 scorer 训练 + 协议合规
评测)的逐步对照见 `sagpool-workflow.md`**;本节只记录决策与理由。

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

## 5. SubgraphX(ICML 2021)——协议适配(已实现)

复现自官方实现(用户本地 SubgraphX-master):MCTS 搜索连通子图(Q+UCB
选择、state_map 去重、逐 expansion 批量评估子节点奖励)+ 冻结分类器打分;
默认超参对齐 Configures.py(n_rollout=20 / c_puct=5 / expand_atoms=12)。

| 维度 | 原文设定 | 本实现 | 理由 |
|---|---|---|---|
| 搜索方向 | 完整图自顶向下**删点**(explainer 设定) | 以 seed 为根自底向上**生长**,子节点 = coalition 的 frontier 节点,终止 = 大小到达 budget+1(含 seed) | 生长方向天然连通(无需 connected_components 剪枝);目标大小即预算语义(P1) |
| 奖励 | mc_l_shapley(zero_filling 构图保留**完整图边结构**作上下文)+ **目标类**概率 | **弃用 Shapley**,采用官方 reward 家族中的 gnn_score 成员:候选 coalition 的导出子图(双向边 + 自环归一,与终判构图同约定)整体打分;类别无关聚合 `max_c prob`(默认)/`logit`/`margin` | ① zero_filling 上下文需要完整图边结构,评测态不可见(P4 违规);② 目标类来自完整图预测,隐含标签信息(原则 5) |
| **可见性口径** | 解释器全知 | **planner 型(oracle)**:MCTS 规划期使用完整图结构,未揭示节点的身份/特征参与搜索 | 见下方口径说明 |
| 算力 | 无显式上限 | `max_evals`(单次规划的唯一 coalition 评估数)硬上限,触顶节点视作叶子;预算未满时退化 max-frontier-degree 补齐 | 算力上限作为可报告超参(原则 3 的算力面) |
| 揭示序 | 无(子图即答案) | 最优 coalition 内自 seed 的 BFS 序(每步必然在当前 frontier 上,P3) | 产出必须是揭示动作序列 |
| 确定性 | Shapley 随机采样 | 无随机源(gnn_score 无采样),tie-break 用元组序 | 原则 4 |

**Planner 口径说明**(原则 1 的显式例外):SubgraphX 作为解释器天生全知,
移植为揭示策略时保留规划期的完整图访问——与 teacher(梯度/信息增益)同属
oracle 口径。它回答的问题是:"学习到的序贯策略,相对'直接对同一冻结分类器
做测试时搜索'有多少增益?"因此其结果行必须标注 planner;若需严格 P4 合规
变体,可退化为每步 greedy lookahead(仅对当前 frontier 逐一评估,无树搜索),
留作 ablation,暂未实现。

## 6. Soft-Mask(WWW 2021)——协议适配(已实现)

复刻官方实现骨架(用户本地 soft-mask-gnn-main,PyG 版,`baseline/softmask.py`
DGL 移植):SMG 变体,3×(WeightConv1 掩码生成 → SparseConv 掩码调制卷积)、
add-pool 读出、2 层 MLP 头;超参对齐原默认(hidden 128 / dropout 0.5 /
lr 5e-4 / wd 1e-4 / seed 777)。工作流详见 `softmask-gnn-workflow.md`。

| 维度 | 原文设定 | 本实现 | 理由 |
|---|---|---|---|
| 卷积 | GCN 式消息传递 | WeightConv1 掩码生成 + SparseConv 掩码调制(DGL 移植) | 原版骨架;DGL 复刻,聚合排除自环(原版 remove_self_loops,以 not_self 边权实现) |
| 边特征 | 分类管线弃用(TU 无边特征) | 小 MLP 编码为**每边标量权重**,走 SparseConv 原版 edge_weight 消息通道 | PPA/cifar10 边特征是协议可见信息,必须消费;**末层零初始化**使初始行为与无边权原版逐位一致,调制由训练习得 |
| 无节点特征 | 不存在该概念(TU 必有) | 共享 node token | 对齐 sagpool.py 约定 |
| 训练 | 10 折 CV + NLL + 早停 | 完整图标准监督(单图 batch CE),自环 remove+add 归一 | 原则 2:训练全知合法;与 run_sagpool 同约定 |
| **评测选点** | **无选点(软掩码加权,全节点保留)** | **每步在可见子图上重算第一层掩码分数,取最高分 frontier 节点** | 软掩码 m∈(0,1) 无序贯入口;重打分口径与 SAGPool 适配一致(P3/P4);平分取小 id(原则 4) |
| 评分函数 | 各层动态重算 | 评测取**第一层**(pre-sigmoid logit) | 首层输入为原始特征,与逐节点选点对应;sigmoid 单调,排序不变 |
| 终判 | 模型自身读出头 | 冻结上界分类器(P5) | 与全部基线同口径 |

## 7. AgentNet(ICLR 2023)——协议适配(已实现)

复现官方实现核心(用户本地 AgentNet-main,PyG 版,`baseline/agentnet.py`
DGL 单 agent 移植):移动头 Q·K 注意力([邻居‖当前‖边])+ BSEU 探索偏置
(back/stay/explored/unexplored,visited 衰减 0.9),gumbel-softmax(hard,
straight-through) 选边,节点/agent 残差更新 + 逐步读出累加。超参对齐
README(hidden 128 / steps 16 / lr 1e-4 / wd 0.01 / clip 1.0 / τ=2/3 /
seed 777 / mlp_width_mult 2 / reduce log / batch 32)。2026-09-19 源码
逐行审查:恢复此前未声明丢失的 mlp_width_mult/reduce,训练坍缩根因见
`docs/decisions.md`。工作流详见 `agentnet-workflow.md`。

| 维度 | 原文设定 | 本实现 | 理由 |
|---|---|---|---|
| agent 数 | K=18~26 并行游走 | **单 agent** | 预算语义逐字节对齐(每步揭示 1 节点);训练/评测同构 |
| time conditioning | TimeEmbedding(固定 num_steps+1) | **移除** | 评测预算步数可变,固定表会越界;单 agent 下收益有限 |
| 初始位置 | 随机均匀撒/注意力选址 | **seed 节点**(P2) | 对齐 AGR 种子协议 |
| **移动候选** | **当前位置全部邻居(完整图)** | **当前位置的 frontier 邻居** | 原版用未观测结构(P1/P3 违规);限缩后为深度型揭示(与 random-walk 候选结构等价,构成"均匀随机 vs 学习型选点分布"控制变量对照) |
| 往返/停留 | self-loop stay + back 偏置,可任意往返 | 已揭示区内**BSEU 打分导航**(不消耗预算,步数上限 4n);揭示步=frontier 候选 argmax | 移动语义保留;区域内移动不消耗预算(原则 3);rollout_strategy 不瞬移,导航耗尽返回 None 结束 |
| 训练 | TU 数据端到端分类 NLL(游走 gumbel 可微) | 完整图同款端到端监督(自环归一) | 原则 2;评测切换可见子图(分布漂移属基线设定) |
| MLP 宽度 | mlp_width_mult=2(README) | **已恢复**(2026-09-19 审查;此前未声明丢失,分类头坍缩主因之一) | 忠实性 |
| 聚合归约 | reduce='log'(sum/log2(deg+1), README) | **已恢复**(此前用 sum) | 忠实性;尺度差部分被 MLP 前置 LN 吸收 |
| global_agent_pool | README 开启(agent 间全局通信,附录 C.2) | 不实现 | K=1 下退化为自环广播,无信息增益 |
| batch/优化器 | batch 32 / AdamW+cosine / 350 epochs | batch 32(梯度累积,前向单图)/ Adam / 100 epochs | 原版配方;前向含序列 gumbel 采样不可 dgl.batch |
| 单 agent 表达力 | Theorem 12:存在 2 agent 可解、1 agent 任意倍步数不可解的结构 | **接受为已知局限** | 协议每步 1 节点约束下的必然取舍;acc_own 参照口径如实记录 |
| 游走步数 | 固定 num_steps(16) | 训练/own_predict 读出游走 **steps=max(1,int(ratio×n))** 逐样本按预算档采样 | 与评测预算驱动的策略游走长度对齐;模型无 time conditioning、读出按 1/(T+1) 归一,天然支持变步数 |
| 图对称化 | TU 对称图 | **无向化重建并保留边特征**(`symmetrize_with_feat`;dgl.to_bidirected 丢 edata) | cifar10 DGL 图为单向边,移动/卷积按无向语义查边,与协议无向 frontier 语义对齐 |
| 确定性 | test_argmax(τ→1e-6) | argmax + 平分取小 id(原则 4) | 注意:原版 test_argmax 的选择仍含 gumbel 噪声(τ 只影响直通值),本实现比原版更严格确定 |
| 终判 | 模型自身读出头 | 冻结上界分类器(P5) | 与全部基线同口径 |

## 8. 已否决选项(决策记录,防止重复讨论)

| 选项 | 否决理由 |
|---|---|
| **GAM**(KDD 2018) | ① step 网络按"节点标签注意力"选步,cifar10 连续特征不可用、collab/ppa 无节点标签只能度数分桶合成(信息口径已失真);② 迁移到 frontier 边特征需改动其源码,既非忠实复现也非干净新方法。代码已清理(git 历史可查 `baseline/gam_src` 等)。 |
| **全局中心性 top-k**(degree/PageRank/k-core) | ① 信息违规:需要完整图拓扑,评测时不可见(原则 1);② 机制不可行:top-k 序列中大部分节点不在 frontier 上,P3 下无法揭示。合规变体 = 局部 frontier-degree greedy,已在主仓实现为固定策略 `max-frontier-degree`,无需在 baseline/ 重复。 |

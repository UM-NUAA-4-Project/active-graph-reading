# 相关工作 v2.3（中文同步版）

> 与 `related-work-v2.tex` 同步，用于替换 `main_zh.md` §2。共四段：池化 / 采样 / 解释 / 学习式序贯读取器。
> v2.3：① 按作者删改同步，移除"预算化获取"与"复杂度"两段（其引用键在文末标为备用，材料若挪去引言或理论节可复用）；
> ② "学习式序贯读取器"段扩写为四个最近邻工作（GAM、RWE-SGNN、AgentNet、RAW），逐一覆盖
> **行为特征 / 任务差异 / 如何违反协议**；③ 维持无前向指涉——T0/T1/T2 在引言已定义，此处属回指。引用占位沿用【REF：…】。

## 2 相关工作

**全图观测下的图分类与池化。** GIN 等消息传递分类器消费整张图【REF：gin】，池化方法则以任务感知的注意力在完整图上为节点打分、据此稀疏化计算，从可微簇分配到自注意力 top-$k$ 选择【REF：diffpool、sagpool、pooling-survey】。这些方法回答的都是"整图已读之后哪些节点重要"；而在节点预算下，同一问题必须在图可见**之前**、仅凭前沿信息回答。SAGPool 是我们对照的代表——改造为匹配预算下的静态选择器。

**图解释。** 解释方法为训练好的模型在**已观测**图上的预测识别理由。我们的设定把这一步的方向倒过来：证据必须在"有预测可解释"**之前**获取，解释式打分在部分观测下按构造不可得。

**学习式序贯读取器。** 与本文设定最亲近的是"预算内选择性读图"的智能体。GAM 用强化学习训练循环游走器做预算化整图分类：rank 网络检视当前节点的邻居并采样下一步，LSTM 积分阅读历史，多个游走器可把各自历史池化为共享记忆【REF：gam】。RWE-SGNN 保留游走但改用 DQN、以下游损失的下降为奖励，以**采样器与预测器分离**的双模型为自监督图分类生成类相关子结构【REF：rwesgnn】。AgentNet 用直通 Gumbel 转移替代强化学习，让一群携带身份编码的智能体可微且次线性地游走，与所访节点每步交换状态【REF：agentnet】。RAW 把游走建成 POMDP 做半监督**节点**分类，把边内容纳入观测，并聚合相关度过阈值的邻居特征【REF：raw】。四者的任务恰好包夹本文：RAW 保护单个节点的预测、图其余部分默认可观测；另外三者与本文同为图级目标。但它们都违反了本文协议强化的纪律。其一，未揭示内容进入决策乃至预测表示：GAM 移动前检视邻居内容，RAW 聚合从未访问邻居的特征，AgentNet 每步吸收邻居状态、可微转移还把梯度回传进未访问候选的嵌入。其二，没人维持"单种子、单调增长的已揭示集"：预算按游走步数或智能体数记账，GAM 可重访节点，RWE-SGNN 从全图 top 打分的入口发起多条游走，AgentNet 把智能体撒满全图。其三，GAM、RAW 与 AgentNet 都从驱动选点的同一循环状态读出预测；RWE-SGNN 的双模型拆分与本文解耦同路，却没有任何揭示纪律。本文的读取器则从构造上保证：预测路径只是已揭示集合的函数。

---

## 引用键 → 完整文献（供后续建 bibliography）

**本节使用（12 键）：**

| 键                                  | 文献                                                                                                     |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| gin                                 | Xu et al., ICLR 2019                                                                                     |
| diffpool / sagpool / pooling-survey | Ying et al., NeurIPS 2018 / Lee et al., ICML 2019 / Grattarola et al., IEEE TNNLS 2022                   |
| graphsage                           | Hamilton et al., NeurIPS 2017                                                                            |
| crawl                               | Tönshoff et al., AISTATS 2021                                                                           |
| gnnexplainer / subgraphx            | Ying et al., NeurIPS 2019 / Yuan et al., ICML 2021                                                       |
| gam                                 | Lee et al., KDD 2018                                                                                     |
| rwesgnn                             | Self-supervised Subgraph Neural Network with Deep Reinforcement Walk Exploration, arXiv 2502.01809, 2025 |
| agentnet                            | Martinkus et al., Agent-based Graph Neural Networks, ICLR 2023                                           |
| raw                                 | Akujuobi et al., WSDM 2020                                                                               |

**暂未使用（原"预算化获取"与"复杂度"两段所用，材料挪往他处时复用）：**

| 键                                             | 文献                                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| gpa / igp / gflowgnn / afa-shim                | Hu et al., NeurIPS 2020 / Zhang et al., ICLR 2022 / Li et al., arXiv 2023 / Shim et al., NeurIPS 2018 |
| ram / grl-survey                               | Mnih et al., NeurIPS 2014 / Nie et al., IEEE TETCI 2022                                               |
| games-against-nature / pomdp-complexity / ssat | Papadimitriou, JCSS 1985 / Papadimitriou & Tsitsiklis, Math. OR 1987 / Majercik & Littman, AAAI 1998  |
| adaptive-submodularity                         | Golovin & Krause, JAIR 2011                                                                           |

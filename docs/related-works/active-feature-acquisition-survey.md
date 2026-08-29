# 主动特征获取（AFA）与信息价值（VoI）相关论文：结构化梳理与对比分析

> **检索日期**：2026-08-28
> **检索来源**：arXiv + ICML/NeurIPS Proceedings + arXiv + IEEE CDC（本地 PDF 精读，2026-08 收集至 `papers/PDFs/active-feature-acquisition/`）
> **收录标准**：高质量期刊/会议（ICML、Statistical Science、IEEE CDC）或近期权威综述预印本
> **检索方向**：Active Feature Acquisition (AFA) 的 POMDP 统一框架、myopic vs non-myopic 获取策略、Conditional Mutual Information (CMI)、Expected Information Gain / Value of Information (EIG/VoI)、Adaptive Informative Path Planning (AIPP)
> **与本项目关系**：本项目的核心问题是**预算受限的图节点序贯揭示**（图版本 AFA）——每一步从 frontier 中选一个节点揭示，换取最终分类精度。这 8 篇构成 AFA/VoI 方法家族：AFA 综述给出统一 **AFA-POMDP 框架**与四类方法分类（本项目的 RL reader 属于 model-free 类，gradient/structural 教师属于 myopic 近似类）；SEFA/BRiG-AFA/TAFA 给出三条**非近视（non-myopic）**路线；CMI 串起贪心基线的信息论形式与缺陷；ModernBED 给出 EIG=VoI 的正规定义；AIPP 与 VoI-Deceptive 提供机器人/控制域的跨域对偶

---

## 目录

1. [综述：AFA 问题框架与本项目的交叉](#综述afa-问题框架与本项目的交叉)
2. [论文 1：A Survey on Active Feature Acquisition Strategies（arXiv 2026, 综述）](#论文-1a-survey-on-active-feature-acquisition-strategiesarxiv-2026-综述)
3. [论文 2：Stochastic Encodings for Active Feature Acquisition, SEFA（ICML 2025, CCF-A）](#论文-2stochastic-encodings-for-active-feature-acquisition-sefaicml-2025-ccf-a)
4. [论文 3：BRiG-AFA: Bellman Risk-to-Go Learning（arXiv 2026）](#论文-3brig-afa-bellman-risk-to-go-learningarxiv-2026)
5. [论文 4：Information Templates: A New Paradigm for Intelligent AFA, TAFA（arXiv 2026）](#论文-4information-templates-a-new-paradigm-for-intelligent-afa-tafaarxiv-2026)
6. [论文 5：Towards Dynamic Feature Acquisition on Medical Time Series by Maximizing CMI（ICML 2024 Workshop）](#论文-5towards-dynamic-feature-acquisition-on-medical-time-series-by-maximizing-cmiicml-2024-workshop)
7. [论文 6：Modern Bayesian Experimental Design（Statistical Science 2024）](#论文-6modern-bayesian-experimental-designstatistical-science-2024)
8. [论文 7：Learning-based Methods for Adaptive Informative Path Planning（arXiv 2024, 综述）](#论文-7learning-based-methods-for-adaptive-informative-path-planningarxiv-2024-综述)
9. [论文 8：Value of Information-based Deceptive Path Planning Under Adversarial Interventions（IEEE CDC 2025）](#论文-8value-of-information-based-deceptive-path-planning-under-adversarial-interventionsieee-cdc-2025)
10. [总结对比矩阵](#总结对比矩阵)
11. [阅读优先级建议](#阅读优先级建议)
12. [附：方法论映射——从已有方法到本项目](#附方法论映射从已有方法到本项目)

---

## 综述：AFA 问题框架与本项目的交叉

### AFA 的标准定义（Aronsson 综述）

对每个实例 x，序贯地选择要获取的特征：

```
x (特征全未知) → π 选择 a ∈ U (未获取特征) → 支付成本 c_a → 观测 x_a
                                    ↓
                          直到 π 选择 STOP → f(x_S) → ŷ

目标:  min_{f,π} E[ ℓ(f(x_{π[x]}), y) + α·c(π[x]) ]  s.t. c(π[x]) ≤ B
```

- 硬预算 B：每个实例最多花 B；软预算 α：成本-精度权衡系数（本文本项目只有硬预算）
- 与静态特征选择的本质区别：**自适应**——先获取的观测决定下一步获取哪个特征（实例相关）

### 方法论交叉图

```
预算受限图阅读（本项目）：从 frontier 选节点揭示 → 子图分类
    │
    ├── 线索 A: 统一问题框架
    │   ├── 论文 1: AFA 综述 — AFA-POMDP（信息状态为充分统计量、四类方法分类）
    │   └── 论文 6: Modern BED — EIG = VoI 的贝叶斯形式化（增量 EIG ↔ 自适应序列获取）
    │
    ├── 线索 B: myopic 贪心基线
    │   ├── 论文 5: CMI-MedicalTS — CMI 的端到端隐式逼近（可微获取器）
    │   └── 论文 2 对 CMI 的批评: 指示器问题 + 熵最小化 ≠ 识别最可能类
    │
    ├── 线索 C: non-myopic 三条路线
    │   ├── 论文 2: SEFA — 随机潜空间 + 信息瓶颈（监督训练，绕开 RL）
    │   ├── 论文 3: BRiG-AFA — 预算特定 risk-to-go 直接回归（Bellman 后向拟合）
    │   └── 论文 4: TAFA — 特征模板库（联合信息量 + 子模性下界 + 可解释策略）
    │
    └── 线索 D: 跨域对偶
        ├── 论文 7: AIPP 综述 — 机器人领域同一信息获取 POMDP 的连续动作版本
        └── 论文 8: VoI-Deceptive — 控制领域显式 VoI（极小化对手 VoI = 欺骗）
```

### 关键映射：从 AFA 范式到本项目

| AFA 范式 | 标准 AFA | → | 本项目对应 |
|---------|---------|----|-------------|
| 特征 | 表格特征 x_i（独立） | → | 图节点 v（结构依赖、有拓扑相关性）— 根本差异 |
| 动作 | 选未观测特征 | → | 选 frontier 节点（可选集合随历史变化） |
| "观测" | 特征值 x_i | → | 节点边特征（揭示前即可见——frontier 特征通道） |
| 停止 | STOP 动作（软/硬预算） | → | 固定的揭示数 B（暂无 STOP——见 contingent 综述） |
| 终局奖励 | ℓ(ŷ, y) + α·成本 | → | Accuracy(classifier(sub_G))（episode-level） |
| 策略类 | 概率策略 / Q 网络 | → | LearnableReader（RL）+ teacher 轨迹（BC） |

---

## 论文 1：A Survey on Active Feature Acquisition Strategies（arXiv 2026, 综述）

| 维度 | 内容 |
|------|------|
| **论文名称** | A Survey on Active Feature Acquisition Strategies |
| **作者** | Linus Aronsson, Arman Rahbar, Morteza Haghir Chehreghani（Chalmers / Univ. of Gothenburg） |
| **发表** | **arXiv 预印本**（cs.LG, v2, 2026-02） |
| **期刊等级** | 综述（无 CCF 评级） |
| **arXiv** | [2502.11067](https://arxiv.org/abs/2502.11067) |

### 摘要

主动特征获取（AFA）研究如何针对每个数据实例序贯获取特征，在预测性能与获取成本之间权衡。本篇为**首个以显式部分可观测马尔可夫决策过程（POMDP）统一刻画 AFA** 的综述。作者将该框架置于最优信息获取文献（尤其是信息获取型与感知型结构化 POMDP 家族）中，基于此提出贴近求解 POMDP 标准方法的 AFA 方法分类：(i) 嵌入式成本感知预测器（成本敏感决策树/集成）、(ii) 基于模型（学习概率组件做规划）、(iii) 免模型（从模拟 episode 学习获取策略）、(iv) 混合方法。作者认为 POMDP 中心视角厘清了方法间联系、催生更原则化的算法设计，并指出由于大量既有工作属于启发式、缺乏形式保证，可将 AFA 与自适应随机优化（adaptive stochastic optimization）衔接以获取近似保证。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | AFA 文献零散在特征选择/医学诊断/机器人各域，缺乏统一语言——如何系统比较问题设定与方法？ |
| **核心贡献** | ① 完整 AFA-POMDP 形式化（MOMDP 结构、信念依赖奖励、STOP 动作）② 四类方法分类法 ③ 把 myopic 贪心（VoI 规则）的泛滥归因于自适应优化的计算难处理性 |

### 核心思路

```
AFA-POMDP 关键要素：
  状态 S = (S, x, y) — 注意: 状态包含真实 x, y（对 agent 部分可观测）
  动作 A_B(S) = {a ∈ U : c(S)+c_a ≤ B} ∪ {STOP}
  转移: 获取 a → 观测 x_a → S = S ∪ {a}（确定性，信息状态扩充）
  终局: 由 STOP 触发，stochastic shortest path / 有限时域 episodic
  观测建模: 两种常见做法 — ① mask (m, x̃) 拼接 + 占位符或 imputation
            ② 深度集合编码器（permutation-invariant）
            ③ f(x_S) = E_{x_U|x_S}[f(x_S, x_U)]（隐含边缘化，估计困难）
```

1. **固定预算的双形式**：硬预算 B（严格限额）vs 软预算 α（成本项，允许难样本花更多）——α 使成本随实例变化，对应"难图多揭示"
2. **训练配方**：预测器 f 通常两种——(i) 离线在随机 mask 上预训练（本项目 classifier 的用法）；(ii) 与策略联合在 on-policy 经验上训练；在线变体（训练时也获取特征）仅少数工作（Kachuee 2019 / Rahbar 2023, 2025）
3. **可解性诊断**：自适应最优信息获取 ⟺ 自适应随机优化，普遍不可处理（Krause & Guestrin 2009）→ myopic VoI 贪心规则是主流近似；需通过自适应子模性获得渐进保证
4. **历史前身**：贝叶斯医学专家系统 Pathfinder（Heckerman 1992）与成本敏感决策树（Núñez 1991 / Norton 1989）——任何决策树都可视为合法的 AFA 策略

### 与本项目的对比分析

##### 相似之处

| 维度 | AFA 综述 | 本项目 |
|------|---------|--------|
| **形式化** | AFA-POMDP：信念/信息状态为充分统计量 | 同一框架：信息状态 = 已揭示子图 + frontier |
| **硬预算** | B = 每个实例的总获取成本上限 | B = 揭示节点数（unit cost，硬预算） |
| **策略类映射** | (iii) model-free：从模拟 episode 学策略 | RL reader（REINFORCE）落在此类，(ii) 以上界 GIN 为概率组件 |
| **任务结构** | 每实例独立、预测器 f 需处理任意特征子集 | 每图独立、分类器需处理任意已揭示子图 |

##### 关键差异

| 维度 | AFA 综述 | 本项目 |
|------|---------|--------|
| **获取单元** | 独立特征 | 结构耦合的节点（揭示暴露邻居、边特征）——动作空间内生变化 |
| **奖励语义** | ℓ(ŷ,y) + α·成本（含停止） | 仅终点 accuracy（无停止、无软预算） |
| **观测顺序与成本** | 特征值获取即付成本 | frontier 边特征可见但不付成本（v1 核心协议差异） |
| **保证路线** | 子模性/自适应随机优化（特征独立假设下成立） | 图结构下的子模性不显然（需重新论证，见论文 4 的借鉴） |

> **启示**：这篇综述是本项目最合适的"定位锚"——用 AFA-POMDP 的术语重写形式化，把"确定性转移"措辞改为"观测动力学确定"，把 gradient/structural 教师明确定位为 **myopic VoI 规则**，把 RL reader 定位为 **model-free POMDP 策略**。综述第 5 节（决策树即 AFA 策略）值得深挖：对于图版，BFS/固定度中心策略即"决策树式"的非自适应基线。

---

## 论文 2：Stochastic Encodings for Active Feature Acquisition, SEFA（ICML 2025, CCF-A）

| 维度 | 内容 |
|------|------|
| **论文名称** | Stochastic Encodings for Active Feature Acquisition |
| **作者** | Alexander Norcliffe, Changhee Lee, Fergus Imrie, Mihaela van der Schaar, Pietro Liò（Cambridge / Korea Univ. / Oxford） |
| **发表** | **ICML 2025**（PMLR 267） |
| **期刊等级** | **CCF-A**（机器学习顶级会议） |
| **arXiv** | [2508.01957](https://arxiv.org/abs/2508.01957) |
| **精读笔记** | `papers/paper-notes/SEFA/`（中英对照） |

### 摘要

AFA 是实例相关的序贯决策问题：基于当前观测，对每个测试实例独立地决定测哪一维特征。常见做法要么用强化学习（训练困难），要么贪心最大化标签与未观测特征的条件互信息（导致近视获取）。SEFA 引入潜变量模型，以**监督方式**训练：在随机潜空间中对许多可能的未观测实现进行推理后再决定获取。在大量合成与真实数据集（含癌症分类）上的评估表明 SEFA 稳定优于各类基线。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | RL 训练困难（稀疏奖励、探索/利用、致命三要素）与 CMI 贪心近视——无监督训练下的 AFA 怎么做？ |
| **核心贡献** | ① 理论+实例证明 CMI 的两个缺陷（见下）② SEFA：特征级随机编码器 + 信息瓶颈 + 潜空间获取目标（天然非近视）③ 消融显示每个设计选择都必要 |

### 核心思路

```
SEFA 架构（encoder-predictor + 隐变量 Z）:
  p(y|x_S) = E_{p_θ(z|x_S)} [p_φ(y|z)]     ← 预测（对大量潜实现取期望）
  特征级随机编码器: z_i = e_i(x_i) + ε   → 每个特征独立编码、观测特征插值
  信息瓶颈正则化:   min I(Z;X) - β·I(Z;Y) ← 仅保留标签相关信息，滤噪
  获取目标: 候选 f 的分数 = 潜空间中对"预测似然对 z_f 分量的梯度范数"
            在多个随机潜实现上的期望  ← 未观测特征的效果被"想象"出来
```

1. **非近视的设计原理**：每个候选特征的效果跨多种可能的未观测实现进行评估——不把未观测特征边缘化掉（这正是 CMI 失败的根源），而是"采样它们"
2. **监督训练**：只优化预测 loss（不做策略/RL 训练），获取在推理时用定义好的目标
3. **CMI 缺陷一（指示器问题）**：Proposition 4.1——任何用边缘分布 p(x_i, y) 打分的目标都不会先选"指示器特征"（其与标签边缘独立，但指示后续哪个特征有效）；d+1 位合成例证中贪心 CMI 期望 3−1/d 次获取 vs 最优 2 次。Proposition 4.2——考虑未观测特征值（E_{p(x_U|x_O)}[I(X_i;Y|x_O,x_U)]）的贪心即可最优，但不可计算
4. **CMI 缺陷二（0-1 loss 视角）**：CMI 最大化 ⟺ 熵最小化，而**熵最小不等于找出最可能类别**：H([0.5,0.5,0])=0.693 < H([0.7,0.15,0.15])=0.819，后者却已确信最可能的类。从 Rényi 熵角度：识别单类应最小化 min-entropy H_∞ = −log max p，但 min-H_∞ 不再等价于最大化 KL/D_∞ 散度，理论基础丢失
5. **对敏感度类方法的批评**（与本文项目直接相关）：sensitivity-based（Kachuee 2017/2018）用标签对特征的梯度做选择，但缺失值填零 + 测量不连续 → **梯度不可靠地表征真实敏感度**

### 与本项目的对比分析

##### 相似之处

| 维度 | SEFA | 本项目 |
|------|------|--------|
| **问题设定** | 实例相关序贯选择 + 硬预算 B | 图相关序贯揭示 + 硬预算 B |
| **批评对象一致** | 贪心（CMI/敏感度）近视且次优 | gradient teacher（敏感度式）与贪心结构教师均属此类 |
| **监督训练** | 预测 loss 训练，推理时用获取目标 | BC（教师轨迹蒸馏 = 监督）+ RL 微调 |
| **框架位置** | 绕开 RL 的非近视 AFA | 本项目的 RL 属于"必须 RL"那一侧（定位差异，见下） |

##### 关键差异

| 维度 | SEFA | 本项目 |
|------|------|--------|
| **信息单元** | 特征级（条件独立假设下可分解） | 图节点级（拓扑相关，无法分解成独立编码器） |
| **其潜空间** | 特征值潜编码，插值缺失值 | 无此构造——子图结构本身是观测（GNN 编码） |
| **获取目标** | 潜空间梯度范数（需清晰的可微潜模型） | 策略网络直接输出 softmax 分数 |
| **训练路线** | 纯监督（刻意回避 RL） | BC→RL（RL 允许奖励直接对精度，但方差大） |

> **启示**：SEFA 与本项目是两个对照面——它证明"非近视不一定要 RL"，我们则选"teacher 蒸馏 + RL"。论文最宝贵的两处：(1) **CMI 两个缺陷的证明可写进论文的动机分析**（我们的贪心教师同样会犯"忽视 context 节点"与"降熵≠辨类"两类错）；(2) 其 related work 对 sensitivity-based 方法的批评（填零+不连续→梯度失真）**几乎原样适用于我们的 GradientTeacher**——这是论证"教师只是弱监督、RL 才是主线"的现成文献弹药。

---

## 论文 3：BRiG-AFA: Bellman Risk-to-Go Learning（arXiv 2026）

| 维度 | 内容 |
|------|------|
| **论文名称** | BRiG-AFA: Bellman Risk-to-Go Learning for Non-Myopic Active Feature Acquisition |
| **作者** | Jiaorong Feng, Qian Li, Ying Li（Curtin University） |
| **发表** | **arXiv 预印本**（cs.LG, 2026-08） |
| **期刊等级** | 预印本（无 CCF 评级） |
| **arXiv** | [2608.02305](https://arxiv.org/abs/2608.02305) |

### 摘要

AFA 回答"预算下下一步测哪个特征"。贪心规则易训练但会忽略"其价值要通过后续获取才兑现"的上下文特征；RL 与生成式方法则引入艰难的优化或条件密度估计。BRiG-AFA 是一种可部署的监督替代方案：为**每个剩余预算学习一个候选条件化的 risk-to-go 函数**。从单步终局分类风险出发，用 Bellman 目标**后向拟合**；推理时仅用观测值、mask、候选身份、剩余预算贪心最小化所学终局风险。受控非近视基准（CUBE-NM）显示：预算 2、3 时相对单步消融提高 4.84±2.17 / 4.39±1.10 个百分点；Fashion-MNIST-20 在 4 次获取时提高 10.20±0.74 点，跨预算平均配对增益 3.50±0.37；MiniBooNE 在小预算混合、8/16 次获取为正（作者明确这是"当前边界"而非普适主张）。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 不牺牲非近视、不引入 RL/生成模型的 AFA——能否直接回归终局风险？ |
| **核心贡献** | 预算特定 Bellman 形式化：把非近视 AFA 变成**监督风险回归**（无在线 RL 循环、无生成模型），并配 "one-step 消融"做受控对比 |

### 核心思路

```
离线（后向拟合, fitted DP）:
  Q_r(x_m, m, a) := 剩余 r 次、先选 a 后的条件期望终局损失
  T_1 = ℓ_ϕ(x, y, m+)                     ← 一步: 终局分类 CE
  T_r = min_{a'} Q_{r-1}(x⊙m+, m+, a')    ← r>1: 递归自上一预算网络
  θ_r = argmin E[ (Q_θr(x_m,m,a,r/d) − T_r)² ]
  （Q_{r-1} 冻结后再训 Q_r → 拟合动态规划，非时序差分）

推理（部署, 无标签/无隐藏值/无特权信息）:
  m=0; for t=0..B−1: a_t = argmin_a Q_{B−t}(x⊙m, m, a, (B−t)/d)
  （一次性用到 0 budget 的 myopic Q_1 即为消融基线，隔离长视界递归的价值）
```

1. **训练目标 = 终局风险**：直接回归预测器的交叉熵损失（mask-aware 预训练后冻结），逐预算后向拟合——把"价值"降格为"风险"，绕开互信息估计/策略 RL
2. **通用状态采样**：空 mask + 均匀变化长度的随机特征前缀混合——不读标签、不用合成数据、不手工分组
3. **受控基准实验设计**：CUBE-NM 的"门控特征 0（噪声门识别两种 regime）→ 全局特征 1-2 → regime 特定特征 3-5/6-8 → 干扰项 9-11"精确构造出"必须先知门控、再选专业集"的结构——等价于 AFA 文献里的指示器问题构造，专门测非近视能力
4. **结果措辞谨慎**：明确区分"机制级证据"（受控基准+五种子）与"SOTA 比较仍缺"的边界——这对实验报告是好的示范

### 与本项目的对比分析

##### 相似之处

| 维度 | BRiG-AFA | 本项目 |
|------|----------|--------|
| **决策时信息约束** | 只用观测值 + mask + 候选 + 剩余预算（无特权） | 只用已揭示子图 + frontier 特征（无标签/无全图） |
| **非近视动机** | context 特征（门控）价值只有多步后才兑现 | 教师轨迹中的"先选 hub 再选叶子"同样多步兑现 |
| **受控比较方法** | 与一步消融配对对比（隔离"非近视"增益） | 可用同样结构：myopic 教师 vs RL reader 在同一预算下的差距 |
| **预算特定函数** | 每个剩余预算一个 Q 网络 | 预算档位在 m7 中是离散比率（0.05..0.20） |

##### 关键差异

| 维度 | BRiG-AFA | 本项目 |
|------|----------|--------|
| **目标空间** | 标量终局风险回归（MSE） | 策略分布（CE + RL 奖励） |
| **学习范式** | 无策略学习——拟合"值函数" | 策略学习（BC 监督 + REINFORCE 奖励） |
| **推理开销** | 每步评估所有候选的 Q | 每步一次前向 |
| **图结构** | 无（特征均独立、动作空间静态） | 拓扑相关、动作空间内生 |

> **启示**：(1) BRiG 的"预算特定 + Bellman 后向拟合"可直接移植为图版：`Q_r(revealed_subgraph, frontier_candidate, r)` → 训练目标用上界分类器在**真实揭示结果**上的终局 CE，训练数据来自轨迹生成器（教师给出的多步结果）。这可能是比纯 REINFORCE 方差更小、训练更稳的"价值面"监督信号——与本项目的 RL 形成互补（值回归 vs 策略梯度的双翼）。(2) 其"myopic 消融"设计建议直接复用到本项目的实验表：**同一 reader 架构在 B=每步一步lookahead vs 全预算策略**——把"非近视有效"这个卖点做出机制级证据。

---

## 论文 4：Information Templates: A New Paradigm for Intelligent AFA, TAFA（arXiv 2026）

| 维度 | 内容 |
|------|------|
| **论文名称** | Information Templates: A New Paradigm for Intelligent Active Feature Acquisition |
| **作者** | Hung-Tien Huang, Dzung Dinh, Junier B. Oliva（UNC Chapel Hill） |
| **发表** | **arXiv 预印本**（cs.AI, v2, 2026-02） |
| **期刊等级** | 预印本（无 CCF 评级） |
| **arXiv** | [2508.18380](https://arxiv.org/abs/2508.18380) |

### 摘要

AFA 在推理时由策略序贯选取特征并支付成本。现有方法要么训练 RL 策略（应对困难 MDP），要么用无法兼顾特征联合信息量、或需要底层数据分布的贪心策略。TAFA 提出**基于模板的 AFA**：学习一个小型特征模板库（联合信息量高的特征集合），用模板库指导后续获取。模板识别不仅显著压缩策略考量的动作空间，也免去对数据分布的估计。大量实验显示 TAFA 优于现有 SOTA 基线，同时实现更低的总体获取成本与计算量。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | AFA 的决策空间随维度指数膨胀——什么是驱动成本/收益的"典型特征子集"（canonical subsets）？ |
| **核心贡献** | ① 模板库 = 特征子集字典，动作空间压缩 ② 遗传算法+连续松弛求模板（目标函数子模、含近似保证）③ cost/benefit 准则构成最优 AFA MDP 值的下界 ④ 蒸馏出带 ACQUIRE-IF 规则的可解释决策树 |

### 核心思路

```
模板搜索（oracle 视角, 离线）:
  g(B) = E_{x,y}[ min_{b∈B} ℓ(ŷ_b(x_b), y)   + λ Σ_{j∈b} c(j) ]
                     └── 从库 B 选最优模板 ──┘  └─ 成本项 ─┘
  B⋆ = argmin g(B)   （g 子模: 定理 3.1, facility-location 型证明; 贪心 O(B²D) 不可行）
  求解: 遗传算法（变异-贪心）+ 离散 Gumbel-Softmax 连续细调 + parametric agent

部署（两阶段策略）:
  阶段1 模板选择: b⋆ = argmin_{b∈B} { ℓ_b(x_o)   + λ Σ_{j∈b\o} c(j) }
                                   └ 当前部分观测的损失 ┘  └ 未获取部分成本 ┘
  阶段2 特征获取: 依次获取 b⋆ 中尚未观测的特征（动作被模板约束）

可解释学生的蒸馏: 模板库作为"教师" → 决策树学生, 节点 = ACQUIRE-IF 规则
```

1. **子模性纳入 AFA**：首个（据作者称）在 AFA 中利用子模性——模板集合目标子模（增加模板的边际收益递减），贪心优化有近似界
2. **代价-收益准则 = MDP 值下界**：模板的每次选择对应的 (ε = loss + λ·cost) 是"最优 AFA 值"的可计算下界——提供理论底座
3. **与 SEFA/ACO 的区分**：SEFA 需要额外潜模型且不与 AFA MDP 奖励直接挂钩；ACO 虽免 RL/生成模型，但推理需枚举指数级未来序列，极其昂贵
4. **可解释性**：策略=树 + 明确的 ACQUIRE-IF 规则，"哪个信息对哪个类型的实例重要"一目了然

### 与本项目的对比分析

##### 相似之处

| 维度 | TAFA | 本项目 |
|------|------|--------|
| **动作空间灾难** | 特征子集指数级 | frontier 组合指数级（平均 ~24.6 个候选/步） |
| **强调联合信息** | 模板 = 联合信息量高的集合 | 图的 motif/高效应子结构（hub-叶子组合） |
| **非近视诉求** | 模板先选"该拿的集合" | 教师轨迹教会"先 hub 后叶子" |
| **可解释性** | 决策树 + ACQUIRE-IF | 若做结构教师规则（度中心/pagerank）可写出一张规则表 |

##### 关键差异

| 维度 | TAFA | 本项目 |
|------|------|--------|
| **对象** | 特征子集（维度间可枚举组合） | 图子结构（节点/边模式，组合由拓扑定义） |
| **值函数来源** | oracle 评分（≥MDP 值下界） | 无 analytic oracle——分类器精度是关键黑箱 |
| **动作约束** | 硬约束到模板 β⊆B | 无约束——每步全 frontier 打分 |
| **训练成本** | 免 RL（但需模板搜索的进化优化） | BC + RL |

> **启示**：TAFA 是"把动作空间压低"的最直接借鉴——图版即为**子图模板**：从训练数据中挖掘高频"高信息量子图模式"（例如度-中心性-边特征聚簇的组合），将每步的 frontier 候选投影到模板节点候选上，既可压缩动作空间、又提供"结构先验"。此外其"oracle 下界"思想可借给本项目：**教师（上界分类器）在预算 B 下能达到的精度可以作为 AFA-MDP 值的参照下界**，使实验报告里的百分比有理论坐标。

---

## 论文 5：Towards Dynamic Feature Acquisition on Medical Time Series by Maximizing CMI（ICML 2024 Workshop）

| 维度 | 内容 |
|------|------|
| **论文名称** | Towards Dynamic Feature Acquisition on Medical Time Series by Maximizing Conditional Mutual Information |
| **作者** | Fedor Sergeev, Paola Malsot, Gunnar Rätsch, Vincent Fortuin（ETH Zürich / Helmholtz AI / TU Munich） |
| **发表** | **ICML 2024** Next Generation of Sequence Modeling Architectures Workshop |
| **期刊等级** | 工作坊论文（无 CCF 评级） |
| **arXiv** | [2407.13429](https://arxiv.org/abs/2407.13429) |

### 摘要

知道多变量时间序列的哪些特征该测、何时测，是医学、可穿戴设备与机器人领域的关键任务。更好的获取策略能在保持甚至提升下游预测器性能的同时降低开销。受条件互信息最大化启发，作者提出**仅用下游损失端到端训练获取器**的方法。实验显示：优于随机获取策略、与不限预算的完整获取持平，但**尚未超过静态获取策略**。作者坦率指出假设前提并给出未来方向。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 医学时间序列上逐时步推荐"下一步测哪些特征"——获取器怎么训？ |
| **核心贡献** | CMI 的**隐式可微实现**：获取器输出近似 CMI 的 logits，Gumbel-Softmax 采样 + 基于下游分类 CE 端到端更新（不显式估计 CMI、不用 RL） |

### 核心思路

```
每时间步 τ 的获取循环（full-observation 模拟测量）:
  logits = acquirer({t, b, mask_{τ−1}, model state})
  while i < budget:
      m = GumbelSoftmax(logits)
      logits −= penalty(m)        ← 防止重复选同一特征
      m_τ += m; i += 1
  preds = classifier(x' = m_τ ⊙ x_τ)  ← 仅下游 CE 反向传播整链
```

1. **免估计 CMI**：CMI 只做"设计动机"，不进入目标函数——直接用分类 loss 训练获取器，规避 CMI 估计（pVAE 类生成模型）困难
2. **Gumbel-Softmax 做离散选择的可微中继**：配合重复选择惩罚，一个前向得到整批特征选择
3. **诚实的结果呈现（对本项目最有价值）**：random < ours ≈ complete，但 **static 固定子集 > ours**——即"实例自适应选择"在这个医学时间序列设定下**没有赢过固定子集**；作者标注假设（每步预算常数、数据全观测、等成本）与未来方向
4. 实验添加 30 个 fake 特征（zeros/Gaussian/GP）验证获取器是否学到区分真实特征（扫描图显示真实特征被持续选到）

### 与本项目的对比分析

##### 相似之处

| 维度 | CMI-MedicalTS | 本项目 |
|------|---------------|--------|
| **信息论动机** | CMI 指导设计，目标函数用下游损失 | gradient/structural 教师 = VoI 规则的启发式实现（同样"动机上正确"） |
| **端到端** | 获取器+分类器联合训练 | BC+RL（与分类器解耦但有上界 GIN） |
| **基准组** | random / complete / static | random / bfs（固定类）/ learnable |
| **诚实对照** | static 比本方法强（意外发现） | 必须与 static/random 对照才有说服力 |

##### 关键差异

| 维度 | CMI-MedicalTS | 本项目 |
|------|---------------|--------|
| **时序结构** | 同一特征可跨时步重复（时间序列） | 每个节点只能揭示一次（图） |
| **观测机制** | 全观测模拟（meet 隐含 0 填充） | 真部分观测 + 可见 frontier 边特征 |
| **预算** | 每步常数 | 全局固定 B |
| **其核心难点** | 时间维度的状态演化 | 拓扑维度的候选探索 |

> **启示**：这篇的价值是"反例与警钟"——**实例自适应 ≠ 自动更好**：它的 end-to-end CMI 获取器打不过静态固定子集。对本项目的具体含义：m7 eval 里必须把 random 与 bfs（固定策略）都作为汇报基线，且若 learnable 未显著超 bfs，说明"自适应"的故事失败。CMI 直觉 + 下游 loss 训练的组合（免估计、免 RL）也值得抄：如果 RL 方差难压，可给 reader 加一个可微获取器头做"CMI 对齐"预热。

---

## 论文 6：Modern Bayesian Experimental Design（Statistical Science 2024）

| 维度 | 内容 |
|------|------|
| **论文名称** | Modern Bayesian Experimental Design |
| **作者** | Tom Rainforth, Adam Foster, Desi R Ivanova, Freddie Bickford Smith（Oxford / MSR AI4Science） |
| **发表** | **Statistical Science**（综述文章，2024） |
| **期刊等级** | 统计类**顶级期刊**（SCI 一区） |
| **arXiv** | [2302.14545](https://arxiv.org/abs/2302.14545) |

### 摘要

贝叶斯实验设计（BED）为优化实验设计提供强有力的一般框架，但其部署面临重大计算挑战。本综述梳理近年研究如何系统解决这些瓶颈——先介绍 BED 的信息论形式化与自适应序列变体，再展示嵌套蒙特卡洛（nested MC）、变分 EIG 估计、摊销化（amortized）等现代方法如何令 BED 在实际中可用，最后讨论未来方向。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | "选哪个实验/测什么"的贝叶斯最优决策如何可计算？ |
| **核心贡献** | ① EIG=VoI 的严格定义 ② 自适应设计的"增量 EIG"及其自洽性论证 ③ 计算瓶颈的现代解法（嵌套 MCMC/变分界/摊销） |

### 核心思路

```
信息增益（单步）:  InfoGain(ξ,y) = H[p(θ)] − H[p(θ|y,ξ)]
期望信息增益 EIG:  EIG(ξ) = E_{p(θ)p(y|θ,ξ)}[ log p(y|θ,ξ) − log p(y|ξ) ]
                = I(θ; y|ξ)  （= 后验→先验 KL 的期望 = 预测不确定性下降的期望）

自适应设计 BAD:  ξ_t 的选择基于历史 h_{t−1}:
  EIG(ξ_t | h_{t−1}) = E_{p(θ|h)p(y|θ,ξ_t,h)}[ log p(y|θ,ξ_t,h) − log p(y|ξ_t,h) ]
  循环: 推断 → 优化 EIG → 跑实验 → 更新信念（所有模型参数都要更新,
        保证 Σ 增量 EIG = 总 EIG）
```

1. **EIG 的等价视角**：互信息、期望熵减、期望 KL（后验-先验）——三视角互证；θ 不必是参数，可以是任意"想获取信息的目标量"（函数最优点、下游预测等）
2. **对频率派 FIM 的批判**（连带为 EIG 辩护）：Fisher 信息矩阵需选迹/行列式等汇总（对参数化敏感）、依赖未知真实 θ、序列设定下不可加——都不满足自洽决策需求
3. **计算难点与解法**：后验/边缘似然不可解析 → 双层嵌套估计（nested MC）、变分代理（VNMC、ACE/prior-contrastive 估计）、摊销化（离线训练网络减少在线成本）
4. **与 AFA 的关系**：AFA 的"获取哪个特征"= 设计选择 ξ=特征候选；CMI/VoI 贪心 = 每步最大化 EIG 的 myopic 特例

### 与本项目的对比分析

##### 相似之处

| 维度 | Modern BED | 本项目 |
|------|-----------|--------|
| **决策对象** | 选择实验设计 ξ（测什么） | 选择揭示节点 v（测什么） |
| **自适应** | Bayesian adaptive design（逐步选、逐步更新信念） | 每步选 frontier 节点、更新子图 |
| **效用基准** | EIG = 期望信息增益 | VoI 结构等价（上界分类器信息增量） |
| **贝叶斯理由** | 自洽整合所有已有信息 | 信念（POMDP）要求贝叶斯式平衡 |

##### 关键差异

| 维度 | Modern BED | 本项目 |
|------|-----------|--------|
| **设计空间** | 连续/组合设计参数 | 离散节点（frontier） |
| **模型** | 显式概率生成模型 p(θ), p(y\|θ,ξ) | 黑箱分类器（图 GNN），无 p(y\|x) 解析式 |
| **求解范式** | 显式 EIG 优化（难算） | 隐式策略学习（learner）+ 启发式（教师） |
| **视图** | 实验设计一般理论 | 工程化 RL/BC 实体的交付 |

> **启示**：BED 是 **VoI 的"出处"**——用 EIG 语言给本项目套一层理论外衣：`揭示节点 v 的 EIG_t(v) = I(θ; y_v | h_t)`（θ=该图的类别标签潜在变量），gradient/structural 教师是 EIG 的不同启发式近似；RL reader 是"摊销化策略"——BED 的 amortized 范式与 model-free AFA 策略正好同构（离线训练网络、在线只前推）。论文里"参数化敏感性/序贯不可加"对 FIM 的批评也可以用来**论证梯度教师（FIM/敏感度式）不是可靠尺度**——又是一条文献弹药。

---

## 论文 7：Learning-based Methods for Adaptive Informative Path Planning（arXiv 2024, 综述）

| 维度 | 内容 |
|------|------|
| **论文名称** | Learning-based Methods for Adaptive Informative Path Planning |
| **作者** | Marija Popović, Joshua Ott, Julius Rückin, Mykel J. Kochenderfer（TU Delft / Univ. Bonn / Stanford） |
| **发表** | **arXiv 预印本**（cs.RO, v3, 2024-07；投稿 Elsevier） |
| **期刊等级** | 综述（无 CCF 评级） |
| **arXiv** | [2404.06940](https://arxiv.org/abs/2404.06940) |

### 摘要

自适应信息路径规划（AIPP）是众多机器人应用的关键——让移动机器人高效收集关于初始未知环境的有用数据。本综述探索**机器人学习应用于 AIPP** 的研究，弥合两个领域之间的鸿沟。先给出 AIPP 的统一数学问题定义，再从 (i) 学习算法与 (ii) 机器人应用两个互补视角建立分类法；探讨协同效应与近期趋势，并讨论挑战与未来方向，附论文目录（含开源实现仓库）。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 在未知环境中规划一条最大化所获信息的轨迹，同时尊重资源约束——学习型方法如何帮助？ |
| **核心贡献** | 统一数学定义 + 学习算法/应用双分类法 + 开放目录（dmar-bonn.github.io/aipp-survey） |

### 核心思路

```
AIPP 统一定义:
  ψ* = argmax_ψ I(ψ)   s.t.  C(ψ) ≤ B
  （ψ = 机器人动作序列: 位姿/传感器/充电站折返；I = 信息准则; B = 时间/能量预算）

  "环境映射" F(x,t) 未知 → 维护信念 p(F | z_1..z_t, a_1..a_t)（在线贝叶斯更新）
  → 每次测量后重规划（在线 replanning），使 I 准则一直基于最新信念

映射方法分类: 占据栅格 | 高斯过程 | 图方法（active SLAM）| 隐式神经表示（NeRF）
学习算法分类: 监督学习 | 主动学习 | 强化学习 | 模仿学习
应用分类:    环境监测 | 探索与搜索 | 语义场景理解 | 主动 SLAM
```

1. **信息准则的信念基础**：AIPP 的 I(·) 必须基于对特征映射的信念而非真值（真值不可得）——与本项目"基于已揭示子图而非全图"同构
2. **四类学习算法正对应 AFA 综述的四类**：SL（代理/预测）、AL（采样选择）、RL（策略）、IL（教学）：这与 AFA 分类法（嵌入/基于模型/免模型/混合）形成跨领域同构，间接支持"信息获取问题的统一框架"叙事
3. **不确定性来源**：传感噪声、定位/执行误差、动态环境——AIPP 比 AFA 多了"动作执行不确定性"（本项目图版没有这一层，是纯确定性转移的"信息获取"子类）
4. **评价维度**：信息增益 vs 成本、覆盖/收敛、仿真-真机差距

### 与本项目的对比分析

##### 相似之处

| 维度 | AIPP | 本项目 |
|------|------|--------|
| **核心结构** | 未知环境信息获取 POMDP + 硬预算 | 未知图信息获取 POMDP + 硬预算 |
| **信念更新** | 测量后更新 F 的信念 | 揭示后更新信息状态（子图+frontier） |
| **自适应** | 在线重规划 | 每步重新打分 |
| **价值函数** | MI/熵/预测不确定性类准则 | 分类器精度/梯度（代理） |

##### 关键差异

| 维度 | AIPP | 本项目 |
|------|------|--------|
| **动作** | 连续路径（长时程、运动约束） | 离散节点揭示（unit cost） |
| **不确定性** | 传感+执行+环境 | 仅被观测信息的内容（无执行噪声） |
| **价值计算** | 需环境模型（GP/栅格） | 黑箱图分类器 |
| **终止** | 预算结束或任务达成 | 预算结束（episode 固定） |

> **启示**：AIPP 是"同构问题、异构领域"的最好例证——引用它可以把本项目定位到**广义信息获取 POMDP 家族**（AFA 综述已点出该家族的共性与工具互通：Krause & Guestrin 2009、Lauri et al. 2023）。写作时可用一句话连通：*"本工作是与机器人 AIPP 平行、以图分类为终局的离散版信息获取 POMDP"*——审稿人对机器人侧的工作会有直觉。其"四类学习算法"分类也可作为我们 Teacher 定性（识别/启发式类推）的旁证。

---

## 论文 8：Value of Information-based Deceptive Path Planning Under Adversarial Interventions（IEEE CDC 2025）

| 维度 | 内容 |
|------|------|
| **论文名称** | Value of Information-based Deceptive Path Planning Under Adversarial Interventions |
| **作者** | Wesley A. Suttle, Jesse Milzman, Mustafa O. Karabag, Brian M. Sadler, Ufuk Topcu（US Army Research Lab / UT Austin） |
| **发表** | **IEEE CDC 2025** |
| **期刊等级** | 控制领域顶级会议（IEEE Conference on Decision and Control） |
| **arXiv** | [2503.24284](https://arxiv.org/abs/2503.24284) |

### 摘要

现有欺骗性路径规划（DPP）方法假设观测者是**被动的**，不适用于观测者能对规划者实施对抗性干预的场景。本文提出一个包含对抗性干预的 DPP-MDP 模型，并设计全新的**基于信息价值（VoI）的欺骗目标**：规划者选择对观测者**信息价值低**的轨迹，诱导观测者做出次优干预。借助 MDP 线性规划理论导出高效策略合成方法。网格世界实验表明：在干预实际发生时，相对被动观测者 DPP 方法与保守路径规划（CPP）方法，VoI-DPP 能以更低成本达成目标。

### 核心问题与贡献

| 维度 | 内容 |
|------|------|
| **核心问题** | 观测者能主动干预（删边/权值）时，如何通过误导其信息来超越最坏保底？ |
| **核心贡献** | ① 敌方干预 DPP-MDP 模型 ② 两个 VoI 欺骗度量（基于观测者诱导信念的信息值）③ LP（占用测度）高效合成 |

### 核心思路

```
观测者信念模型（最大熵路线）:
  P(G|ζ_0:T) ≈ P̃(G|s_T) = e^{V_G(s_T)−V_G(s_0)}·P(G) / Σ_{G'} e^{V_G'(s_T)−V_G'(s_0)}·P(G')
  V_G: 目标 G 的 softmax（最大熵）值函数 — 观测者认为"理性但略带低效 α"的规划者行为

欺骗-信息关系:
  agent 选择轨迹 ζ 以最小化观测者的 VoI(ζ)（对目标的信念区分度）
  └─ 夸张式欺骗: 靠近诱饵目标（decoy）→ 让观测者信错
  └─ 模糊式欺骗: 长时间保持对多个目标的等后验 → 观测者无法干预到位

求解: 欺骗成本函数 g(s,a) + 可达性约束 → 线性规划（状态-动作占用测度 λ）
  π*(a|s) = λ*(s,a) / Σ λ*(s',a)    ← 经典 MDP 对偶 LP + R_max(G*) 可达约束
```

1. **VoI 的主体相对性**：论文的 VoI 是"对**观测者**提供多少信息"——欺骗 = **极小化对方的 VoI**；与本项目（极大化自己的 VoI/分类精度）构成同一数学对象的两极
2. **最大熵观测者模型**：用 softmax 值迭代求观测者信念，使"对方怎么推理"有可计算模型（线性规划中的概率参数由 π 诱导）
3. **保守对照**：robust MDP / 零和博弈只保证最坏情况；VoI-DPP 通过操纵信念争取**好于最坏情况**
4. **LP 的工程细节**：辅助第二个 LP 在最优值 v* 下找最快到达真目标的最优策略（避免最优占用过多零成本态）

### 与本项目的对比分析

##### 相似之处

| 维度 | VoI-DPP | 本项目 |
|------|---------|--------|
| **VoI 显式使用** | VoI 作为规划目标（虽取极小化面） | VoI/VOO 作为教师/评估的信号（取极大化面） |
| **信念更新** | 观测者信念随轨迹更新 | 读者信息状态随揭示更新 |
| **MDP 形式** | 图上的 MDP（节点=状态、边=动作） | 图上的 POMDP（揭示节点=动作） |
| **计算工具** | LP/占用测度（离线合成） | RL/BC（离线+在线） |

##### 关键差异

| 维度 | VoI-DPP | 本项目 |
|------|---------|--------|
| **信息受益方** | 反方（观测者）——欺骗方要*隐瞒* | 己方——阅读方要*获取* |
| **对手** | 主动对抗性观测者 | 无对手（被动分类器） |
| **动作语义** | 空间移动（影响观测到的东西） | 信息揭示（直接改变观测集） |
| **解决方式** | 离线 LP 最优（小状态空间） | 学习策略（大规模、无解析模型） |

> **启示**：(1) VoI-DPP 证明 VoI 是**主体相对**的（同一条轨迹，给观测者 0 信息、给自己 1 信息）——综述写作中可用两篇对照（ActiveVOO 极大化自己的 VOO vs 本文极小化对手的 VoI）形成"信息价值谱系"，把本项目放中间（给自己极大化、无对手）。(2) 其最大熵观测者信念（softmax 值迭代）是图版分类器信念的可迁移技术——若未来做"分类器期望 + 读者决策"的联合推理，可用类似思路建模"分类器在子图上会怎么想"。

---

## 总结对比矩阵

| # | 论文 | 会议/期刊 | 级别 | 年份 | 核心方法 | 近视性 | 获取单元 | 与本项目相似度 | 核心借鉴点 |
|---|------|-----------|------|------|----------|:--:|----------|:--:|-----------|
| 1 | AFA Survey | arXiv | 综述 | 2026 | AFA-POMDP 统一形式化+四类分类 | 分析性 | 特征 | ⭐⭐⭐⭐⭐ | 定位框架、术语、保证路线 |
| 2 | SEFA | ICML | CCF-A | 2025 | 随机潜编码+信息瓶颈+梯范数目标 | **非近视** | 特征 | ⭐⭐⭐⭐ | CMI 两缺陷的证明→批评 myopic 教师 |
| 3 | BRiG-AFA | arXiv | 预印本 | 2026 | 预算特定 risk-to-go 后向回归 | **非近视** | 特征 | ⭐⭐⭐⭐⭐ | 值回归训练 + one-step 消融实验设计 |
| 4 | TAFA | arXiv | 预印本 | 2026 | 特征模板库（子模+遗传算法） | **非近视** | 特征子集 | ⭐⭐⭐⭐ | 动作空间压缩 + oracle 下界 |
| 5 | CMI-MedicalTS | ICML WS | 工作坊 | 2024 | Gumbel-Softmax 隐式 CMI 端到端 | myopic | 时间序列特征 | ⭐⭐⭐ | static 更强的反面教训+可微获取器 |
| 6 | Modern BED | Statistical Science | SCI 顶刊 | 2024 | EIG/VoI 定义+变分/摊销估计 | 分析性 | 实验设计 | ⭐⭐⭐⭐ | EIG=VoI 定义、FIM 批判、摊销=策略学习 |
| 7 | AIPP Survey | arXiv | 综述 | 2024 | 信息获取 POMDP 机器人版双分类 | 分析性 | 路径/测量 | ⭐⭐⭐ | 跨域同构叙事 |
| 8 | VoI-Deceptive | IEEE CDC | 控制顶会 | 2025 | 极小化对手 VoI 的 LP 欺骗规划 | 非近视 | 路径/节点 | ⭐⭐ | VoI 主体性、最大熵观测者模型 |

---

## 阅读优先级建议

| 优先级 | 论文 | 理由 |
|--------|------|------|
| **P0 — 必读** | AFA Survey (#1) | 统一术语与分类——论文定位与形式化章节的骨架 |
| **P0 — 必读** | SEFA (#2) | "为什么贪心教师会失败"的直接理论弹药 + 非近视监督学习的标杆 |
| **P0 — 必读** | BRiG-AFA (#3) | 与其最接近谱系（决定论教师→值回归），且给出机制级实验设计 |
| **P1 — 重要** | Modern BED (#6) | EIG=VoI 定义出处——写"信息价值"相关章节必须引用 |
| **P1 — 重要** | TAFA (#4) | 动作空间压缩与子模下界——若做图模板方向则升为 P0 |
| **P2 — 参考** | CMI-MedicalTS (#5) | static 基线的反例警示 + 可微获取器备选 |
| **P2 — 参考** | AIPP Survey (#7) | 跨域综述引用（一句话定位） |
| **P2 — 参考** | VoI-Deceptive (#8) | VoI 主体相对性论述需两篇对照（ActiveVOO 在其侧） |

---

## 附：方法论映射——从已有方法到本项目

| 已有方法 | 概念 | → | 本项目对应组件 |
|---------|------|----|-------------|
| AFA-POMDP（综述 #1） | 信息状态=充分统计量、STOP、MOMDP | → | 本项目的状态=已揭示子图+frontier；STOP 见 contingent 综述扩展 |
| AFA 四分类法（#1） | model-free / model-based / 嵌入成本 | → | LearnableReader=model-free；上界 GIN=概率组件；BFS/random=嵌入式（近）静态策略 |
| CMI 贪心（#1,#5） | myopic 最大化 I(X_i;Y\|x_S) | → | gradient/structural 教师的理论定位（近似 EIG 的一步启发式） |
| CMI 缺陷（#2） | 指示器问题+熵≠辨类 | → | 论证策略学习（而非贪心教师）是本项目卖点的必要依据 |
| SEFA 潜空间非贪心（#2） | 对未观测实现取期望打分 | → | 可扩展方向：frontier 候选打分的"多实现"化 |
| BRiG 预算特定 Q（#3） | 终局风险回归 + 后向 Bellman | → | 值回归辅助训练（与 REINFORCE 互补）、One-step 消融实验设计 |
| TAFA 模板（#4） | 联合信息特征子集库 | → | 子图模板：动作空间压缩 + 可解释结构策略（可解释图阅读） |
| TAFA oracle 下界（#4） | MDP 值下界 | → | 教师（上界分类器）预算 B 精度的下界参照 |
| BED EIG（#6） | 增量 EIG = MI 的序贯化 | → | "揭示节点 v 的期望信息增益"的形式化定义（报告/评估用） |
| BED 摊销化（#6） | 离线训练→在线前推 | → | 与 RL reader 同构：离线 BC+RL、在线一步前推 |
| AIPP 信念更新（#7） | 测后更新 p(F\|z) 再重规划 | → | 每步重打分 reader 的同构论据（跨域） |
| VoI 主体性（#8） | 信息价值取决于"谁的信念" | → | 对比 ActiveVOO 的 VOO：我方获取、对方隐瞒的两极定位 |
| Gumbel-Softmax 获取器（#5） | 可微离散选择 | → | 若 RL 方差无法收敛：可微获取头做 BC 热启动的备选 |

---

> **最后更新**：2026-08-28
> **关联文档**：`contingent-graph-reading-survey.md`（submit-or-continue 扩展——AFA 的 STOP 动作即其核心）、`evidential-pomdp-reader-survey.md`（信念 POMDP 读者架构）、`active-learning-graphs-survey.md`（图主动学习/图探索）、`papers/paper-notes/SEFA/` 与 `papers/paper-notes/AFA-Survey/`（中英对照精读笔记）

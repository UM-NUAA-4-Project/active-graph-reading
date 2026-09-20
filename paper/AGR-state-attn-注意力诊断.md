# state-attention 读出头注意力诊断

数据：CIFAR10 与 COLLAB 全量评测，2026-09-20。
来源：`code/result/{cifar10,collab}/experiments/state-attn/` 下的 `report.json` 与 `eval_state_attention.json`；
脚本 `code/train_reader/train_state_attention.py` 与 `eval_state_attention.py`。

## 0. 一行结论

state-attention 读出的净增益来自对逐步状态序列的重加权，其主导机制是**下权早期无信息的步**；
该收益随轨迹长度与步间信息不均度增长。CIFAR10 上是真实但温和的架构增益，对 uniform 对照
+1.9 个百分点；COLLAB 上对现行终判 +3.4 个百分点，其中注意力形态退化为近似硬选终局位，
说明收益主要来自排除早期稀释而非回看历史。两处形态不同，机制同源。

## 1. 对象与评估口径

**评估对象。** 冻结已训好的 evidence reader（RL 后 checkpoint），在其 rollout 上采集逐步证据
状态 $x_t=[z_t\,;\,h_t]\in\mathbb R^{2d}$，其中 $z_t$ 为历史引导池化读出、$h_t$ 为 LSTM 积分器隐状态；
序列由 rollout 各步揭示前状态加末位的终局补充前向状态构成。聚合头对序列做 masked attention
池化：$x_t$ 拼接归一化时间标量 $t/T$，经 LayerNorm 后与零初始化的可学习查询打分，softmax 加权
求和后接 MLP 输出类别。查询零初始化使该头在训练起点严格等价于均匀池化，特化完全来自训练。

**五个口径。** 全部口径消费同一批 rollout，仅在读出侧不同：

| 口径 | 含义 |
| --- | --- |
| state-attn | 学习注意力权重，主对象 |
| uniform | 同一 MLP、权重恒均匀，架构净增益的直接对照 |
| terminal-only | 冻结 reader 自身终判，参照下限 |
| ev-last | 证据头逐步输出的最后一步，与 terminal-only 恒等，作一致性检查 |
| ev-ens | 逐步输出平均集成，负对照 |

**规模与确定性。** CIFAR10 test 为 10000 图 × 4 种子 = 40000 episodes，CI95 ±0.45；
COLLAB test 为 501 图 × 4 种子 = 2004 episodes，CI95 ±2.2。两次评测 `terminal_mismatch` 均为 0，
缓存与在线 rollout 逐状态一致。head 训练配置：$d_{attn}=128$、MLP 隐宽 128、40 epoch、
学习率 $10^{-3}$、种子 777；模型选择用 train 内部切出的 15% 验证集，eval split 只做最终报告。

## 2. 主结果

test split，单位 %。

**CIFAR10（10 类）**

| 口径 | 总体 | 5% | 10% | 15% |
| --- | --- | --- | --- | --- |
| state-attn | **31.10** | 26.69 | 31.61 | 34.95 |
| uniform | 29.20 | 24.39 | 29.96 | 33.21 |
| terminal-only | 29.25 | 25.40 | 30.00 | 32.33 |
| ev-ens | 27.40 | 24.32 | 27.78 | 30.08 |

**COLLAB（3 类）**

| 口径 | 总体 | 10% | 15% | 20% |
| --- | --- | --- | --- | --- |
| state-attn | **56.99** | 55.59 | 58.22 | 57.21 |
| uniform | 53.14 | 47.79 | 59.14 | 52.75 |
| terminal-only | 53.59 | 52.35 | 56.37 | 52.15 |
| ev-ens | 52.45 | 52.35 | 54.53 | 50.52 |

两点直接读数。其一，ev-ens 在两个数据集上全面最差，逐步平均把早期弱证据计入终判，
是稀释危害的负对照。其二，ev-last 与 terminal-only 数值逐格相同，与证据头终判取第 $B$ 步
输出的定义一致，口径链无矛盾。

## 3. 净增益的分离：uniform 对照为什么是关键

state-attn 对 terminal-only 的差距混合了两种收益：重新训练读出头的优化收益，以及注意力
机制的架构收益。uniform 对照与 state-attn 用同样的数据、同样的 40 epoch、同一个 MLP，
唯一差别是权重恒均匀，因此 **state-attn 减 uniform 才是架构净增益**。

| 差值（百分点） | CIFAR10 5/10/15% | 总体 | COLLAB 10/15/20% | 总体 |
| --- | --- | --- | --- | --- |
| state-attn − terminal | +1.29 / +1.61 / +2.62 | +1.85 | +3.24 / +1.85 / +5.06 | +3.40 |
| state-attn − uniform | +2.30 / +1.65 / +1.74 | +1.90 | +7.80 / −0.92 / +4.46 | +3.85 |
| uniform − terminal | −1.01 / −0.04 / +0.88 | −0.05 | −4.56 / +2.77 / +0.60 | −0.45 |

CIFAR10 上第三行几乎为零，即均匀池化全部步与只看终局位等价：**终局状态已是充分摘要**，
证据沿轨迹近似均匀到达，没有可剔除的稀释段。学习注意力在此之上仍买到 +1.9，靠的是温和
重加权而非丢弃某段。

COLLAB 上第三行整体为负且 10% 档达 −4.6：早期步是实质性稀释源，均匀池化会明显伤害终判。
这正是学习注意力在 COLLAB 上收益更大的原因，也解释了它为何在 10% 档对 uniform 拉开 +7.8。

## 4. 注意力形态诊断

两个诊断量定义如下。**末位质量** alpha_last_mass 为注意力 softmax 落在序列末位即终局补充
前向状态上的质量占比，对 episode 取平均。**归一化熵** alpha_norm_entropy 为
$H=-\sum_t p_t\log p_t\,/\,\log T$，对有效步数取对数归一，0 表示全部质量集中在单步，
1 表示完全均匀。

| 数据集 | 末位质量 | 归一化熵 | 形态解读 |
| --- | --- | --- | --- |
| CIFAR10 | 65.26% | 0.348 | 温和重加权，约三分之一质量留在早中段 |
| COLLAB | 94.49% | 0.024 | 近似硬选终局位，回看历史几乎不发生 |

两个数据集的形态相反，但可以统一成一条机制陈述：**学习注意力学到的是"该忽略哪些步"，
其收益由步间信息分布的不均程度与轨迹长度决定**。COLLAB 前几步揭示的节点几乎不带类信息，
终局状态信息密度最高，注意力于是收敛到近似 one-hot，其价值在于避免稀释，这在短预算下
最明显，10% 档 +7.8 的 uniform 差距是直接证据。CIFAR10 超像素的证据沿轨迹均匀积累，
没有可剔除的段落，注意力只能做温和的重加权，收益封顶在 2 个百分点附近。

由此可得一个对论文有用的推论：该组件的收益随预算增长而有差异，报告时应逐预算给出，
而不是只报总体。CIFAR10 对 terminal 的增益 +1.29/+1.61/+2.62 随预算单调上升；
COLLAB 的增益集中在 10% 与 20% 两档。

另一处值得记录的结构性事实：COLLAB 的近 one-hot 形态可能部分由输入中的时间标量 $t/T$
驱动。实现上 $t/T=(t+1)/T$ 按各自序列长度归一，终局位的通道值恒等于 1.0，即模型可以仅凭
时间通道识别终局位，形成"挑最后一步"的捷径而不必评估任何步的内容。区分"内容驱动的
重加权"与"时间捷径"需要在训练时打乱或屏蔽 $t/T$ 做对照，属建议补做的诊断；它不影响
当前精度结论的成立，但直接决定归因措辞的强度。

## 5. 噪声边界与异常项

COLLAB 的评测规模只有 2004 episodes，CI95 约 ±2.2，以下异常不作过度解读。其一，15% 档
uniform 59.14 是全场最高，超过 state-attn 的 58.22，差距在置信区间内；选定单头后逐档如实
报告即可，不允许逐档挑选口径。其二，state-attn 在 20% 档回落到 57.21，低于 15% 档，预算
响应非单调。其三，COLLAB 总体 +3.4 的增益若按独立样本 CI 误差传播约为 ±3.1，显著性边缘；
同批 rollout 配对比较会收紧这一区间，正式采用前应以配对检验复核。

CIFAR10 侧有一个数据源差异需要说明：`report.json` 中 uniform 为 30.49，`eval_state_attention.json`
中为 29.20。前者是训练期日志口径，后者是独立 test 评测口径，本文一律引用后者。

## 6. 泄漏与协议核查

无选择泄漏：模型选择只用 train 内部切出的验证集，eval split 只做最终报告。协议合法性成立：
头的全部输入来自已揭示区的表征与可观测标量 $t/T$，不含前沿节点信息。缓存一致性成立：
两次评测 `terminal_mismatch` 为 0。

两项遗留。其一，COLLAB 训练 config 中 cache 默认路径印的是 CIFAR10 的路径，episode 数与
mismatch 表明实际使用了 COLLAB 缓存，未构成误用，但默认值应清理以免复现时误导。其二，
PPA 未跑同一评估；三数据集口径一致前需补跑，预期增益有限，因为 PPA 的证据轨迹本身已塌缩，
读出侧没有可重加权的对象，这一负结果本身也是机制陈述的一部分。

## 7. 对论文的落点

1. §4.3 终判读出的表述从"取第 $B$ 步输出"改为对逐步状态序列的学习注意力读出，口径校准为
   learned readout over the step-state sequence，不把 COLLAB 的增益归因于"回看历史"。
2. 结果表换数前必须统一评测管线：eval_state_attention 与 eval_learnable 两条管线对同一
   checkpoint 的终判相差约 0.3 个百分点，表内数字必须同源。
3. Table 4 消融增加 uniform 行，作为读出侧架构增益的直接对照。
4. 若换用该头，CIFAR10 主表变为 26.69/31.61/34.95，15% 档对最优基线领先由 11.3 扩至约
   13.6 个百分点；COLLAB 恢复全图测试精度的比例变为约 80.5/84.3/82.9%。
5. PPA 补跑同一评估后再定第三数据集的表述。
6. 注意力分布素材的产出方式：eval_state_attention.py 已扩展，逐预算档输出
   `attn_dist` 字段，步位按位序/n_nodes 归一化到 [0, ratio] 比例轴后 20 等宽分箱、对
   episode 取平均，附 bin_edges/bin_centers 与实际计数。热力图 = 预算档 × 箱的 dist
   矩阵，分布曲线 = 各预算档 dist，逐预算熵与末位质量已按档给出。字段与"消融和分析"
   的图直接对接；服务器重跑 eval 后即可绘图。
7. 纯重加权归因口径：train_state_attention.py 已加 `--freeze-mlp`，把 evidence head
   MLP 权重右补零列（时间通道置零，t/T=0 处与原判据严格等价）加载进聚合头并冻结，
   只训 attention 打分器。读数规则：frozen learned ≈ fresh learned ⇒ 增益归因于注意力
   分布本身；frozen ≪ fresh ⇒ 差值即重训占比。与屏蔽 t/T 的对照互补，共同支撑
   "哪一阶段信息最该重视"的归因强度。

## 8. 数据与复现出处

| 项 | CIFAR10 | COLLAB |
| --- | --- | --- |
| 冻结 checkpoint | `result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch3.pt` | `result/collab/experiments/evidence1/evidence/rl/checkpoint_epoch6.pt` |
| 训练日志 | `state-attn/report.json`，trained_at 2026-09-20 17:11 | `state-attn/report.json`，trained_at 2026-09-20 12:13 |
| 独立评测 | `state-attn/eval_state_attention.json` | `state-attn/eval_state_attention.json` |
| 训练批量 | 64 | 16 |

收敛性：CIFAR10 验证精度在第 12 epoch 达 31.39 后平台波动，40 epoch 内无过拟合迹象；
COLLAB 验证精度 58.42，同期平台。特征缓存落盘后变体复评无需重采集。

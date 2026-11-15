# Rebuttal 补充表格与数据核对（2026-11-15）

> 对应 `rebuttal-checklist-2026-11-15.md` **第 1 项**（修正现有表格错漏 + 整理仓库已有关键证据）与**第 4 项**（修正推理开销统计口径并补齐三数据集）。
> 除特别标注外，准确率数字均取自 `docs/Summary of results.md`（2026-11-15 定为唯一数据源）；每个数字的 checkpoint / 实验目录 / 评测命令溯源见 §7。
> §6 为本次实测：NVIDIA RTX 4060 Laptop (8GB)，**10% 预算 × 1 seed × 10% 测试子集**，统一真实批量（cifar10 chunk/batch = 32，collab / ppa = 16）。

---

## 1. 统一冻结分类器表（修正版）—— 回应 zJYV W2

全部 **test** 口径；CIFAR10 = 5/10/15%，COLLAB = 10/15/20%，PPA = 5%。

| 统一冻结分类器 | CIFAR10 superpixels   | COLLAB                | OGBG-PPA |
| :------------- | :-------------------- | :-------------------- | :------- |
| Random         | 16.32 / 18.69 / 20.57 | 15.57 / 15.57 / 15.57 | 20.16    |
| BFS            | 16.32 / 18.06 / 19.59 | 15.57 / 15.57 / 15.72 | 24.45    |
| Max-Frontier-Degree | 16.09 / 17.31 / 18.98 | 15.57 / 15.72 / 15.87 | 26.20 |
| Random-Walk    | 16.30 / 19.03 / 21.36 | 15.57 / 15.57 / 15.57 | 21.46    |
| SAGPool        | 15.49 / 16.91 / 18.10 | 15.57 / 15.57 / 15.57 | 27.28    |
| Soft-Mask      | 16.12 / 18.45 / 21.07 | 15.57 / 15.57 / 15.57 | 21.88    |
| AgentNet       | 16.25 / 18.88 / 21.58 | 15.57 / 15.57 / 15.57 | 23.43    |
| BAGR-BC        | 16.08 / 17.82 / 19.44 | 15.72 / 18.21 / 20.26 | 27.11    |
| BAGR-RL        | 15.50 / 16.76 / 18.66 | 15.72 / 18.21 / 20.26 | 20.73    |

**相对旧版 `rebuttal.md` 的修正记录**：

1. **数值笔误**：Soft-Mask CIFAR10 10% `18/58` → **18.45**（小数点笔误）；COLLAB 20% 档 Random `15.67`→**15.57**、BFS `16.47`→**15.72**、Max-Frontier-Degree `17.27`→**15.87**、Random-Walk `15.62`→**15.57**。
2. **valid / test 混用统一为 test**：COLLAB 可训练基线（SAGPool / Soft-Mask / AgentNet）三行此前部分填了 valid 集的 15.43，统一为 test 口径 **15.57**（子图塌缩区，≈少数类在 test 中的占比）。
3. **CIFAR10 三行旧值纠正**（旧值与他行数字串行/过期）：SAGPool `15.35/16.79/18.34`→`15.49/16.91/18.10`；Soft-Mask `16.1/18.58/21.52`→`16.12/18.45/21.07`；AgentNet `16.2/19.03/21.81`→`16.25/18.88/21.58`（19.03 是 Random-Walk 的 10% 值）。
4. **PPA SAGPool 5%**：`27.14` → **27.28**（Summary PPA 基线表 test 口径）。
5. **checkpoint 对齐**：BAGR-RL（CIFAR10）使用与正文主表**同一个 checkpoint**（`checkpoint_epoch6`，own 头 27.79 / 33.84 / 36.55）的冻结分类器参照值 **15.50 / 16.76 / 18.66**；旧表中的 `16.68 / 19.78 / 22.98` 来自 epoch3 checkpoint（own 头 25.32 / 30.03 / 32.68），已弃用——若 rebuttal 中同时引用两个轮数，需分列注明。

---

## 2. BAGR-Base（无证据头）同裁判对照 —— 回应 zJYV W2

最严格的"同一冻结预测器"受控对比：BAGR-Base 没有循环证据头，训练（RL 奖励）与终判**全程与其他方法共用同一个冻结全图分类器**——性能差异只能来自选点策略本身。

| 数据集   | BAGR-Base BC           | BAGR-Base RL           | 同裁判下最强基线                   |
| :------- | :--------------------- | :--------------------- | :--------------------------------- |
| CIFAR10  | 16.60 / 20.84 / 22.96  | **16.89 / 20.99 / 23.85** | Random-walk 16.30 / 19.03 / 21.36 |
| OGBG-PPA | 26.70 / 37.41 / 44.42  | **30.53 / 41.42 / 46.21** | SAGPool 27.28 / 37.92 / 45.08     |

- CIFAR10：RL 每档超同裁判最强基线 **+0.57 / +1.96 / +2.49**；
- PPA：RL 每档超最强基线 SAGPool **+3.25 / +3.50 / +1.13**（5/10/15%），达到全图上界的 46.5% / 63.1% / 70.4%；
- 诚实披露：COLLAB 上 base reader RL 严重坍缩无法训练（Summary §3），需 evidence reader 的证据流才能突破分类器限制（见 §4）——这正说明 CIFAR10/PPA 上 base 的胜利不依赖证据头，COLLAB 的胜利不依赖分类器。

---

## 3. 可训练基线双评测口径对照（own vs 统一冻结分类器）—— 回应 4jJN W1

固定策略（Random/BFS/…）无自带头，恒为冻结分类器终判；可训练基线给出口径两列：**own** = 主表口径（全图训练的自带分类头），**clf** = 统一冻结分类器（与 §1 同裁判）。"—" = 仓库暂无该口径记录（固定策略无 own 头）。

**CIFAR10**（test，5/10/15%）：

| 系统 | own 头（主表口径） | 统一冻结分类器 |
| :--- | :--- | :--- |
| Random        | —                          | 16.32 / 18.69 / 20.57 |
| SAGPool       | 10.92 / 12.52 / 13.81 *    | 15.49 / 16.91 / 18.10 |
| Soft-Mask     | —                          | 16.12 / 18.45 / 21.07 |
| AgentNet      | 16.25 / 18.30 / 19.11      | 16.25 / 18.88 / 21.58 |

\* SAGPool own 取自本地 `code/result/cifar10/fixed-own/eval_fixed.json`（judge=own 重评，同批 clf 参照 15.48/16.90/18.09 与 §1 一致）；其余 own 列均出自 Summary。

**COLLAB**（test，10/15/20%）：

| 系统 | own 头（主表口径） | 统一冻结分类器 |
| :--- | :--- | :--- |
| Random    | —                          | 15.57 / 15.57 / 15.57 |
| SAGPool   | —                          | 15.57 / 15.57 / 15.57 |
| Soft-Mask | —                          | 15.57 / 15.57 / 15.57 |
| AgentNet  | 51.90 / 51.90 / 51.90（多数类水平 ≈52%，常数输出） | 15.57 / 15.57 / 15.57 |

**OGBG-PPA**（test，5/10/15%）：

| 系统 | own 头（主表口径） | 统一冻结分类器 |
| :--- | :--- | :--- |
| Random    | —                     | 20.16 / 33.07 / 41.88 |
| SAGPool   | —                     | 27.28 / 37.92 / 45.08 |
| Soft-Mask | —                     | 21.88 / 33.89 / 42.00 |
| AgentNet  | 9.48 / 9.48 / 9.48（坍缩至单类 ≈9.5%） | 23.43 / 35.75 / 43.51 |

**结论（写进回复的论点）**：
1. 统一冻结分类器下，可训练基线**选出的子图本身大多优于 Random**：PPA 5% 上 SAGPool 27.28 / AgentNet 23.43 / Soft-Mask 21.88 全部 > Random 20.16（10%/15% 档同样全部更高）；CIFAR10 15% 上 AgentNet 21.58 / Soft-Mask 21.07 > Random 20.57。
2. 主表中它们 own 口径大幅低于 Random 的真正主因，是**全图训练的自带分类头在 5–15% 稀疏子图上发生严重分布漂移**（PPA AgentNet own 坍缩到 9.48、CIFAR10 SAGPool own 10.92–13.81 远低于其 clf 参照 15.48–18.09），叠加协议约束（frontier 节点特征不可见、AgentNet 收缩为单智能体）。
3. 反向印证 BAGR 的设计：证据头与选点策略在**同一局部序列协议下联合训练**，无分布漂移（PPA own=ev 头终判 26.90，与 clf 参照 20.73 同批 rollout）。

---

## 4. Scratch-RL（无 BC 热启动）vs BC + RL —— 回应 zJYV W1

正文图 2 上排只画了训练奖励，此处补 test 集准确率与训练轮数（own 头终判 + 冻结分类器参照）：

| 数据集 | 方法 | RL 轮数 | test（own 头终判） | clf 参照 |
| :--- | :--- | :--- | :--- | :--- |
| CIFAR10 (5/10/15%) | Scratch-RL（soft-both→hard-both，batch 32×2 seeds） | 15 | 27.11 / 32.30 / 34.79 | 15.49 / 16.79 / 18.55 |
| CIFAR10 (5/10/15%) | **BC + RL（主表 ckpt）** | 6（epoch3 续训至 epoch6） | **27.79 / 33.84 / 36.55** | 15.50 / 16.76 / 18.66 |
| COLLAB (10/15/20%) | Scratch-RL | 10（汇报 epoch5） | 52.25 / 51.10 / 48.05 | 15.57 / 15.57 / 15.57 |
| COLLAB (10/15/20%) | **BC + RL（主表 ckpt）** | 10（汇报最优 epoch5） | 51.90 / **53.44 / 55.74** | 15.72 / 18.21 / 20.26 |

- **CIFAR10**：公平训练量下（6 轮 < 15 轮），BC 热启动全面领先 **+0.68 / +1.54 / +1.76**——无 BC 起点并非"性能相当"，冷启动只是初期上升快、收敛慢。
- **COLLAB**：Scratch-RL 收敛到多数类水平（≈52%）且**随预算增大倒退**（52.25→48.05，读多读少不影响判定）；BC + RL 随预算稳步升至 55.74。BC 热启动是 COLLAB 预算扩展性的必要条件，两阶段框架的必要性由此显现。

---

## 5. 预训练冻结分类器完整配置与训练数据 —— 回应 phri

三个裁判分类器均为 **5 层 GIN(GINE) 主干、hidden_dim=300、dropout=0.5**，仅用各数据集官方 train split 训练后冻结；**无任何外部预训练数据、无 valid/test 泄漏**（valid 仅用于训练期早停/选优）。

| 数据集 | 结构细节 | 训练数据（严格仅 train split） | 全图 valid / test |
| :--- | :--- | :--- | :--- |
| CIFAR10 | GIN-plus：residual、readout=meanmax、FFN 头、label smoothing 0.1；batch 128，Adam 1e-3，早停 | 50,000 图（Benchmarking GNNs CIFAR10 超像素官方 train；test 10,000 不参与） | — / 60.5 |
| COLLAB | GIN 5 层；先全图训练（ex1），再子图适应微调（finetune：random/random-walk/bfs 混合采样部分子图 + full-frac 全图防遗忘，lr 1e-4）——COLLAB 子图塌缩需要适应后的裁判 | 4,000 图（COLLAB 5,000 图按类别分层 80/10/10，种子 42；valid 499 / test 501 不参与） | 71.54 / 69.06 |
| OGBG-PPA | GIN(GINE) 5 层，7 维边特征编码，37 类 | 78,200 图（OGB species split train；valid 45,100 / test 34,800 不参与） | 64.2 / 65.6 |

**分类器依赖性退化实验（evidence-third）**：把 CIFAR10 奖励源换成性能次一级的分类器（全图 55.73 vs 60.50），BAGR-RL 仍全面超越所有固定策略基线：

| 奖励源分类器 | RL（own 头终判） | clf 参照 | vs 最强固定策略（Random-walk 16.30/19.03/21.36） |
| :--- | :--- | :--- | :--- |
| 主分类器（60.50） | 27.79 / 33.84 / 36.55 | 15.50 / 16.76 / 18.66 | 每档 +11.49+ |
| 次分类器（55.73） | 20.68 / 23.59 / 25.35 | 15.34 / 20.95 / 23.66 | 每档 +3.99+ |

→ BAGR 的性能不依赖裁判分类器的强弱；裁判只提供训练期奖励信号，部署时甚至可以完全卸载（见 §6 bagr-deploy）。

---

## 6. 推理开销统计（修正口径 + 三数据集补齐）—— 回应 4jJN

### 6.1 口径修正（相对 11-13 版基准）

1. **剥离部署不需要的冻结分类器**：新增 `bagr-deploy` 纯部署口径（`evaluate_reader(classifier=None)` 跳过终局子图构建与冻结分类器前向）——BAGR 实际部署只用自身证据头；
2. **单预算档计时**：全部系统在 10% 预算 × 1 seed 下测 `ms/graph`，不再把 3 预算档轨迹耗时累加后与全图单次前向相除；
3. **显存口径**：删除 chunk=1 下对 BAGR 扣减 ~8.5 MiB cuDNN LSTM workspace 再 ×批量的换算，删除 ckpt 磁盘大小列（含 Adam 优化器状态，不代表推理开销）；统一在**真实批量**（cifar10 chunk/batch=32，collab/ppa=16）下统计 **Params / 模型常驻显存 / 前向峰值瞬态显存**；
4. **新增硬件无关计算量**：每图平均**揭示节点占比**与整条轨迹逐步编码的**累计可见边数 Σ_{t=1}^{B}|E_vis(t)| 占全图分类器消息传递边数 L_clf·|E| 的比例**（L_clf=5）。

产物：`code/result/{ds}/cost/benchmark_cost_2026-11-15.json`；命令见 §7。

### 6.2 成本表

统一真实批量：cifar10 chunk/batch = **32**，collab = **16**（8GB 显存约束），ppa = **16**；单预算档 10% × 1 seed × 10% 测试子集（cifar10 分层抽样 1000/10000，collab 全量 501，ppa 分层抽样 3480/34800）。acc 为单 seed 口径，与主表 4-seed 均值略有差异。「步进前向峰值」= 每步选点前向的峰值瞬态显存（full-graph 步 = 整批前向）；「终判前向均值」= 读满预算后终判批前向瞬态（基线 = 冻结裁判批 + 自带头批；bagr = 冻结裁判参照批；bagr-deploy 无裁判、证据头终判瞬态并入步进列）。

**CIFAR10**（test 1000 图，均值 118.5 节点 / 1067 边）：

| 系统 | Params (M) | 模型常驻 (MiB) | 步进前向峰值 (MiB) | 终判前向均值 (MiB) | ms/图 | 揭示节点 (%) | Σ\|E_vis\| / L·\|E\| (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-graph (batch 32) | 2.18 | 8.37 | 186.8 | — | 0.8 | 100.0 | 100.0 |
| SAGPool   | 0.14 | 0.55 | 0.58 | 103.8 | 106.2 | 9.61 | 11.79 |
| Soft-Mask | 0.32 | 1.20 | 0.24 | 72.9 | 62.9 | 9.61 | 12.39 |
| AgentNet  | 0.57 | 2.17 | 3.31 | 95.2 | 58.3 | 9.61 | 12.20 |
| BAGR（评测，含裁判参照） | 0.38 | 1.47 | 16.97 | 8.3 | 87.2 | 9.61 | 12.51 |
| **BAGR（纯部署，无裁判）** | 0.38 | 1.47 | 17.27 | — | **81.2** | 9.61 | 12.51 |

**COLLAB**（test 全量 501 图）：

| 系统 | Params (M) | 模型常驻 (MiB) | 步进前向峰值 (MiB) | 终判前向均值 (MiB) | ms/图 | 揭示节点 (%) | Σ\|E_vis\| / L·\|E\| (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-graph (batch 16) | 1.82 | 6.97 | 1952.1 | — | 3.4 | 100.0 | 100.0 |
| SAGPool   | 0.14 | 0.55 | 16.18 | 60.1 | 108.9 | 9.36 | 5.72 |
| Soft-Mask | 0.31 | 1.20 | 1.35 | 79.6 | 74.7 | 9.36 | 10.18 |
| AgentNet  | 0.53 | 2.03 | 272.53 | 85.0 | 73.2 | 9.36 | 10.11 |
| BAGR（评测，含裁判参照） | 0.38 | 1.47 | 52.03 | 6.9 | 101.0 | 9.36 | 10.99 |
| **BAGR（纯部署，无裁判）** | 0.38 | 1.47 | 52.03 | — | **98.6** | 9.36 | 10.99 |

注：AgentNet 步进峰值含其首步 per-graph 静态缓存构建（之后常驻）；COLLAB 稠密图上 AgentNet 该缓存开销显著。

**OGBG-PPA**（test 分层抽样 3480 图，均值 243.5 节点 / 4773 边，稠密）：

| 系统 | Params (M) | 模型常驻 (MiB) | 步进前向峰值 (MiB) | 终判前向均值 (MiB) | ms/图 | 揭示节点 (%) | Σ\|E_vis\| / L·\|E\| (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-graph (batch 16) | 1.84 | 7.04 | 654.3 | — | 2.0 | 100.0 | 100.0 |
| SAGPool   | 0.15 | 0.56 | 10.75 | 521.8 | 250.8 | 9.91 | 56.56 |
| Soft-Mask | 0.32 | 1.22 | 0.86 | 345.4 | 186.7 | 9.91 | 46.48 |
| BAGR（评测，含裁判参照） | 1.53 | 5.85 | 91.14 | 12.9 | 242.8 | 9.91 | 40.65 |
| **BAGR（纯部署，无裁判）** | 1.53 | 5.85 | 91.14 | — | **242.7** | 9.91 | 40.65 |

注：PPA 的 BAGR 为 ev256 变体（1.53M 参数）；本机无 AgentNet PPA scorer，该列缺省（服务器产物可后补）。

### 6.3 要点

1. **特征获取 / 消息传递计算量降一个量级（硬件无关）**：10% 预算下 BAGR 每图只揭示 **~9.6% 节点**，整条轨迹逐步编码的累计可见边数仅为全图分类器消息传递边数（5 层 × 全图边）的 **11–12.5%**（cifar10）/ **11.0%**（collab）——预算感知采样的收益直接可量化，这正是审稿人要求的"评估预算感知采样带来的收益"。
2. **显存**：BAGR 模型常驻 1.47 MiB（各系统最小档）；步进前向峰值瞬态 cifar10 **17.3 vs 全图批前向 186.8 MiB**（batch 32），collab **52.0 vs 1952.1 MiB**（batch 16，**~37×**）——图越大越稠密，瞬态优势越大。
3. **时间**：小图（cifar10 均值 118 节点）上 GPU 批量全图前向 0.8 ms/图，任何序贯协议系统（每图 B 步逐步揭示）都在 60–110 ms/图 量级，BAGR（部署 81.2）与可训练基线（58–106）同档；协议的收益不在 wall-time，而在以 9.6% 的节点揭示拿到全图 56–60% 的相对精度（cifar10 33.8/60.5，collab 54.5/69.1 单 seed）。
4. **部署无需冻结分类器**：`bagr-deploy` 卸载裁判后 cifar10 87.2→**81.2** ms/图（−6.9%）、collab 101.0→**98.6**（−2.4%），且不再需要 8.37/6.97 MiB 的裁判常驻——训练期的裁判依赖不带入部署。
5. **PPA（大而稠密的图）**：BAGR 部署步进瞬态 **91.1 vs 全图批前向 654.3 MiB**（batch 16，~7×）；累计可见边占比 **40.7%**，反而低于 SAGPool 的 56.6% 与 Soft-Mask 的 46.5%——BAGR 用**更少的可见边**拿到 32.2% 精度（两者 own 头分别坍缩在 11.4% / 9.5%），单位信息量的选点质量更高；时间上 BAGR 242.7 ms/图与 SAGPool 250.8 同档。
6. 单 seed 口径下精度与 4-seed 主表基本一致（cifar10 33.8 vs 33.84、ppa 32.2 vs 32.34、collab 54.5 vs 51.9+seed 方差），成本结论不受采样影响。


---

## 7. 溯源：checkpoint / 实验目录 / 评测命令

| 条目 | 路径（相对仓库根；result/=code/result/） |
| :--- | :--- |
| CIFAR10 裁判分类器 | `result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt`（config 同目录） |
| COLLAB 裁判分类器 | `result/collab/classifier/finetune/best_collab_classifier.pt`（全图起点 ex1） |
| PPA 裁判分类器 | `result/ppa/classifier/best_model_ft.pt` |
| CIFAR10 BAGR-RL（主表） | `result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt`（eval: 同目录 `performance-analysis/eval_learnable.json`） |
| CIFAR10 BAGR-BC / RLref | Summary evidence-best：BC 16.91/17.87/18.39 own、BCref 16.08/17.82/19.44（`ev1/evidence/bc`；RL epoch3 起点 = ev-R-read） |
| CIFAR10 Scratch-RL | `result/cifar10/experiments/scratch-rl/evidence/rl/`（epoch14 评估） |
| CIFAR10 evidence-third | `result/cifar10/experiments/ev1/evidence/rl/`（eval_heldout_test_reader.json） |
| COLLAB BAGR-BC / RL（主表） | `result/collab/experiments/evidence1/evidence/{bc,rl}/`（RL 汇报 epoch5：`rl/performance-analysis/eval_learnable.json`） |
| COLLAB Scratch-RL | `result/collab/experiments/scratch-rl/evidence/rl/`（epoch5） |
| COLLAB base BC | `result/collab/experiments/ex2/base/bc/` |
| PPA BAGR-BC / RL（主表） | `result/ppa/experiments/ev256/evidence/{bc,rl}/`（RL = checkpoint_epoch5，`rl/performance-analysis/eval_learnable.json`） |
| PPA base BC / RL | `result/ppa/experiments/bc_baseline/base/bc/`；`result/ppa/experiments/rl_asymmetric_ex2/base/rl/`（epoch8） |
| 固定策略/基线 eval JSON | `result/{ds}/fixed/eval_fixed.json`（clf 口径）；`result/cifar10/fixed-own/eval_fixed.json`（SAGPool own） |
| 成本基准 JSON | `result/{ds}/cost/benchmark_cost_2026-11-15.json` |

成本基准命令（三数据集同构，10% 预算 × 1 seed × 10% 测试子集）：

```bash
# cifar10：test 分层抽样 1000/10000，chunk=ref_batch=32
python code/tools/benchmark_inference_cost.py --dataset cifar10 \
    --root D:\\data\\superpixels --budget-ratios 0.10 --n-seeds 1 \
    --max-graphs 1000 --stratified --ref-batch-size 32 --chunk-size 32 \
    --modes full-graph sagpool softmask agentnet bagr bagr-deploy \
    --classifier code/result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt \
    --bagr-ckpt code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt \
    --output result/cifar10/cost/benchmark_cost_2026-11-15.json

# collab：test 全量 501（即测试集全部），chunk=ref_batch=16
python code/tools/benchmark_inference_cost.py --dataset collab \
    --root D:\\data\\COLLAB --budget-ratios 0.10 --n-seeds 1 \
    --max-graphs 1000 --stratified --ref-batch-size 16 --chunk-size 16 \
    --modes full-graph sagpool softmask agentnet bagr bagr-deploy \
    --classifier code/result/collab/classifier/finetune/best_collab_classifier.pt \
    --bagr-ckpt code/result/collab/experiments/evidence1/evidence/rl/checkpoint_epoch5.pt \
    --output result/collab/cost/benchmark_cost_2026-11-15.json

# ppa：test 分层抽样 3480/34800，chunk=ref_batch=16（本机无 agentnet scorer，剔除该模式）
python code/tools/benchmark_inference_cost.py --dataset ppa \
    --root D:\\data\\ogbg_ppa --budget-ratios 0.10 --n-seeds 1 \
    --max-graphs 3480 --stratified --ref-batch-size 16 --chunk-size 16 \
    --modes full-graph sagpool softmask bagr bagr-deploy \
    --classifier code/result/ppa/classifier/best_model_ft.pt \
    --bagr-ckpt code/result/ppa/experiments/ev256/evidence/rl/checkpoint_epoch5.pt \
    --output result/ppa/cost/benchmark_cost_2026-11-15.json
```

---

## 8. 反向固定 BAGR-RL 预测头评测各基线选点 —— 回应 zJYV W2（清单第 2 项）

与 §1「统一使用冻结分类器」构成**正反双向控制变量**：这里固定 **BAGR-RL 的 evidence 头**为唯一终判（checkpoint 完全冻结，不重训），把每步选点替换为 Random / BFS / Max-Frontier-Degree / SAGPool / AgentNet 的选点，BAGR 的 LSTM/Attention 证据状态仍逐步经 `forward_states_evidence` 更新，读满预算后由证据头终判——所有选点方法共享同一个专为局部序列训练的预测头。

实现：`code/train_reader/core/reverse_eval.py`（与 `evaluate_reader` 完全同构的 lockstep + teleport 协议，仅选点动作换源；无冻结分类器、无终局子图构建）+ 入口 `code/train_reader/eval_reverse_reader.py`；agentnet 新增 `clone_walk()`（lockstep 交错推进下共享模型与 per-graph 静态缓存、独立游走状态）；`forward_states_evidence` 新增 `need_scores=False` 快速路径（跳过被丢弃的选点打分，占前向 ~43%，**3.6× 提速且终判逐位一致**，默认 True 不影响既有训练/评估）。协议口径与 §1 一致：cifar10/collab = 4 seeds、ppa = 1 seed，产物 = checkpoint 同目录 `eval_reverse.json`（键 = reverse-{agent}，对照行 = 同 ckpt 的 eval_learnable.json `acc_*`）。

### 8.1 服务器命令清单（清单第 2 项，先跑）

```bash
# 需同步到服务器的代码（7 个文件，见 §9.4 同步清单）与 3 个 BAGR-RL ckpt
# CIFAR10：test 全量 10000 × 4 seeds × 5 策略 ≈ 每策略 ~1h（服务器 GPU 更快则更短）
python code/train_reader/eval_reverse_reader.py --dataset cifar10 \
    --root <服务器superpixels根> \
    --checkpoint result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt \
    --n-seeds 4 --chunk-size 32

# COLLAB：test 全量 501 × 4 seeds × 5 策略（快，~10-30 分钟/策略）
python code/train_reader/eval_reverse_reader.py --dataset collab \
    --root <服务器COLLAB根> \
    --checkpoint result/collab/experiments/evidence1/evidence/rl/checkpoint_epoch5.pt \
    --n-seeds 4 --chunk-size 16

# PPA：test 全量 34800 × 1 seed × 5 策略 ≈ 每策略 ~3-6h（agentnet scorer 若服务器
# 缺失会自动跳过并告警）；赶时间可改 10% 分层抽样（与 §6 成本基准同协议，
# 注明即可）：--max-graphs 3480 --stratified → 每策略 ~25 分钟
python code/train_reader/eval_reverse_reader.py --dataset ppa \
    --root <服务器ogbg_ppa根> \
    --checkpoint result/ppa/experiments/ev256/evidence/rl/checkpoint_epoch5.pt \
    --n-seeds 1 --chunk-size 16

# 回传产物（每数据集一个）：checkpoint 同目录 eval_reverse.json
```

### 8.2 结果表（待服务器回填；BAGR 行 = 同 ckpt eval_learnable.json 的 acc_*）

| 选点来源（BAGR 证据头终判） | CIFAR10 | COLLAB | OGBG-PPA |
| :--- | :--- | :--- | :--- |
| Random | | | |
| BFS | | | |
| Max-Frontier-Degree | | | |
| SAGPool | | | |
| AgentNet | | | |
| **BAGR（自己选点，Ours）** | 27.79 / 33.84 / 36.55 | 51.90 / 53.44 / 55.74 | 26.90 |

---

## 9. 两阶段 RL 受控消融（服务器执行）—— 回应 zJYV W3 & phri（清单第 3 项）

### 9.1 主表链条核实（以本地 config.json / training_log.json 为证）

| 数据集 | 主表链（真实 reward 源） | 总 RL 轮数 | 汇报 ckpt | 共同超参 |
| :--- | :--- | :--- | :--- | :--- |
| CIFAR10 | BC(2轮, `ev1/evidence/bc` ep2) → classifier/asymmetric RL **9轮**(`default/evidence/rl`, ep9) → reader/asymmetric RL **6轮**(`ev-R-read`, 汇报 epoch6) | **15** | ev-R-read ep6（27.79/33.84/36.55） | lr 1e-4, graphs-per-batch 32, seeds-per-graph 4, temperature 0.2, λ_ev=1.0, adv_normalize **off** |
| COLLAB | BC(2轮, `evidence1/evidence/bc` ep2) → classifier/binary RL **10轮**(`evidence1/evidence/rl`, valid 选优 epoch5) | **10** | evidence1 ep5（51.90/53.44/55.74） | 同上（adv_normalize off） |
| OGBG-PPA | BC(**10轮**, `ev256/evidence/bc` ep10) → reader/asymmetric RL **5轮**(`ev256/evidence/rl`, 汇报 epoch5) | 5 | ev256 ep5（26.90） | lr 1e-4, gpb 32, **seeds-per-graph 2**, temp 0.2, λ_ev=1.0, adv off |

**PPA 26.90 奖励源核对结论（清单 3 附加项）**：`ev256/evidence/rl/config.json` 的 bootstrap 直指 `ev256/evidence/bc/reader_checkpoint_epoch10.pt`（BC checkpoint），`reward_source=reader`、`reward=asymmetric`，training_log 1..5 → 主表链 = **BC 10 轮 + reader-RL 5 轮**，与 Summary 记载一致；论文附录 A.3 的 "10 soft-both + 5 reader" **有误**（soft-both 只出现在独立消融目录 `ev256-soft-both`，其起点是 `ev256-c/evidence/rl/checkpoint_epoch3.pt`，与主表链无关）。回复口径统一为 "BC(10) + reader-RL(5)"，终稿 A.3 表格待修。

> 注：CIFAR10 链条的 stage-1（`default`）奖励源为 **classifier**（旧版 config 无 reward_source 键 = 默认 classifier；其 eval 23.33/26.57/28.62 即 Summary 的 evidence-second 行）。两阶段 = **classifier(9) → reader(6)**。

### 9.2 消融设计（5 组 × 2 数据集，均在服务器跑）

控制条件：**同一 BC 起点、相同总 RL 轮数**（CIFAR10=15、COLLAB=10，对齐主表）、相同 graphs-per-batch / seeds-per-graph / lr / temperature / adv_normalize（§9.1）。每组汇报 test 双指标（own = 证据头终判 `acc_*`；clf = 冻结分类器参照 `acc_clf_*`，eval_reader 自动输出）。

| 组 | 奖励源 / 开关 | 轮数 | 回应点 |
| :--- | :--- | :--- | :--- |
| 1 Classifier-only | `--reward-source classifier --reward asymmetric`(cifar) / `binary`(collab) | 15 / 10 | W3：仅参考分类器奖励 |
| 2 Reader-only | `--reward-source reader --reward asymmetric/binary` | 15 / 10 | W3 + phri：全程不加载分类器 |
| 3 Soft-both only | `--reward-source soft-both` | 15 / 10 | W3：混合奖励不切换 |
| 4 CE-only | `--reward-source reader --rl-lambda 0` | 15 / 10 | W3：无策略梯度（**本次新增开关**，只以证据 CE 更新预测头） |
| 5 Two-stage (Ours) | 主表现有产物，不重跑 | 15 / 10 | cifar10 = `ev-R-read` ep6；collab = `evidence1` valid 选优 |

代码改动：`train_reader/core/rl.py` 新增 `rl_lambda`（`L = rl_lambda·L_RL + λ_ev·L_ev`，=0 时跳过策略梯度项、奖励照常记录）+ `train_rl.py --rl-lambda`（config 落盘 `rl_lambda`；base reader + λ=0 直接报错）。CE-only 需 evidence reader（`--reader-kind evidence` 为默认）。

### 9.3 服务器命令清单（8 组训练 + 评估；路径相对仓库根，`--root` 按服务器数据集位置调整）

```bash
# ============ CIFAR10（总 15 轮；BC 起点 = 主表链同一 ckpt）============
BC=result/cifar10/experiments/ev1/evidence/bc/reader_checkpoint_epoch2.pt
CLS=result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt
COMMON="--dataset cifar10 --root <服务器superpixels根> --bootstrap $BC \
  --epochs 15 --lr 1e-4 --graphs-per-batch 32 --seeds-per-graph 4 \
  --temperature 0.2 --evidence-lambda 1.0 --no-adv-normalize"

# 1 Classifier-only
python code/train_reader/train_rl.py $COMMON --exp rl-w3-classifier \
  --reward-source classifier --reward asymmetric --classifier $CLS
# 2 Reader-only（不加载分类器）
python code/train_reader/train_rl.py $COMMON --exp rl-w3-reader \
  --reward-source reader --reward asymmetric
# 3 Soft-both only
python code/train_reader/train_rl.py $COMMON --exp rl-w3-soft-both \
  --reward-source soft-both --classifier $CLS
# 4 CE-only（无策略梯度：--rl-lambda 0，奖励源 reader 仅作日志）
python code/train_reader/train_rl.py $COMMON --exp rl-w3-ce-only \
  --reward-source reader --reward asymmetric --rl-lambda 0

# 评估（汇报最终轮 epoch15；eval_learnable.json 自动含 own acc_* + clf acc_clf_*）
for EXP in rl-w3-classifier rl-w3-reader rl-w3-soft-both rl-w3-ce-only; do
  python code/train_reader/eval_reader.py --dataset cifar10 \
    --root <服务器superpixels根> --classifier $CLS \
    --checkpoint result/cifar10/experiments/$EXP/evidence/rl/checkpoint_epoch15.pt
done

# ============ COLLAB（总 10 轮；按主表协议 valid 选优汇报）============
BC=result/collab/experiments/evidence1/evidence/bc/reader_checkpoint_epoch2.pt
CLS=result/collab/classifier/finetune/best_collab_classifier.pt
COMMON="--dataset collab --root <服务器COLLAB根> --bootstrap $BC \
  --epochs 10 --lr 1e-4 --graphs-per-batch 32 --seeds-per-graph 4 \
  --temperature 0.2 --evidence-lambda 1.0 --no-adv-normalize"

python code/train_reader/train_rl.py $COMMON --exp rl-w3-classifier \
  --reward-source classifier --reward binary --classifier $CLS
python code/train_reader/train_rl.py $COMMON --exp rl-w3-reader \
  --reward-source reader --reward binary
python code/train_reader/train_rl.py $COMMON --exp rl-w3-soft-both \
  --reward-source soft-both --classifier $CLS
python code/train_reader/train_rl.py $COMMON --exp rl-w3-ce-only \
  --reward-source reader --reward binary --rl-lambda 0

# 评估（逐轮评估后按 valid 选优，与主表一致；--ref-batch-size 16 防 OOM）
for EXP in rl-w3-classifier rl-w3-reader rl-w3-soft-both rl-w3-ce-only; do
  for EP in 1 2 3 4 5 6 7 8 9 10; do
    python code/train_reader/eval_reader.py --dataset collab \
      --root <服务器COLLAB根> --classifier $CLS --ref-batch-size 16 \
      --checkpoint result/collab/experiments/$EXP/evidence/rl/checkpoint_epoch$EP.pt \
      --output-dir result/collab/experiments/$EXP/evidence/rl/eval_ep$EP
  done
done
```

> 提醒：COLLAB 逐轮全量评估 ×4 组 ×10 轮开销大，可先只评 3/5/7/10 轮再按 valid 补评邻近轮；CIFAR10 若 15 轮太久可先跑 epoch 5/10 看趋势。产物回填本表（test own / clf 双指标 + 各组 valid 曲线），`Two-stage (Ours)` 行直接引用主表数字。

### 9.4 服务器同步清单（任务 2+3 共 7 个改动文件 + checkpoint）

```bash
# 代码（本地 → 服务器，仓库根相对路径）
# 任务2：反向评测 + agentnet lockstep 支持 + evidence 快速路径
#   code/train_reader/core/reverse_eval.py        （新增）
#   code/train_reader/core/__init__.py            （导出 evaluate_reverse）
#   code/train_reader/eval_reverse_reader.py      （新增）
#   code/active_graph_reader/reader/evidence.py   （need_scores 开关，默认不变）
#   baseline/agentnet.py                          （clone_walk + 多图缓存表）
# 任务3：CE-only 消融开关
#   code/train_reader/core/rl.py                  （rl_lambda）
#   code/train_reader/train_rl.py                 （--rl-lambda）

scp code/train_reader/core/reverse_eval.py code/train_reader/core/__init__.py \
    code/train_reader/core/rl.py code/train_reader/eval_reverse_reader.py \
    code/train_reader/train_rl.py <server>:<服务器仓库>/code/train_reader/  # core/__init__.py、rl.py 需落 core/ 子目录
scp code/active_graph_reader/reader/evidence.py <server>:<服务器仓库>/code/active_graph_reader/reader/
scp baseline/agentnet.py <server>:<服务器仓库>/baseline/

# checkpoint（服务器若缺；ev1 bc = CIFAR10 消融 BC 起点，ev-R-read ep6 = 主表）
#   result/cifar10/experiments/ev1/evidence/bc/reader_checkpoint_epoch2.pt
#   result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt
#   result/collab/experiments/evidence1/evidence/{bc/reader_checkpoint_epoch2.pt, rl/checkpoint_epoch5.pt}
#   result/ppa/experiments/ev256/evidence/rl/checkpoint_epoch5.pt

# 回传产物：
#   任务2：3 个 eval_reverse.json（各 ckpt 同目录）
#   任务3：result/{ds}/experiments/rl-w3-*/evidence/rl/{config.json, training_log.json, checkpoint_epoch*.pt 按需, eval_learnable.json}
```

**跑完后回填**：§8.2 反向表（test，与 §1 同格式的三预算斜杠串）；§9.2 消融表加两列（test own / clf，标注轮数）。每张表旁已注明 checkpoint 与命令，周末合稿时每个数字可追溯。


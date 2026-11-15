# ICLR Rebuttal 补充实验与数据核对清单（2026-11-15）

> **时间节点**：距离讨论截止还有约一周时间。请按下面 4 项优先级补齐实验和修正表格，目标在 **周四晚（11月19日）或周五（11月20日）15:00 例会前** 把更新后的表格数字、对应 checkpoint 路径及配置补入 `docs/rebuttal-prep/`（可直接更新 `rebuttal.md` 或新建结果汇总文档），留出周末两天合并理论部分与英文终稿。

---

## 1. 修正现有表格错漏 + 整理仓库已有关键数据（零训练成本，最优先）

### 1.1 修正 `rebuttal.md`「统一使用冻结分类器」表的笔误与口径错配
1. **数值笔误**：
   - `COLLAB` 的 `Random` 10% 预算写成了 `5.57`（应为 `15.57`）；
   - `CIFAR10` 的 `Soft-Mask` 10% 预算写成了 `18/58`（请核准小数点与对应评测日志）。
2. **valid / test 混用**：
   - `COLLAB` 的 `SAGPool / Soft-Mask / AgentNet` 三行填成了 `15.43`（这是 `docs/Summary of results.md` 里的 **valid** 集数字，请统一改为 **test** 集口径 `15.57`）。
3. **主表与冻结分类器表的 checkpoint 对齐**：
   - 正文主表 `CIFAR10` 汇报的 `BAGR-RL` 是 epoch 6（自带头 `27.79 / 33.84 / 36.55%`），`Summary of results.md` 中该 checkpoint 对应的冻结分类器 `RLref` 为 `15.50 / 16.76 / 18.66%`，而目前 `rebuttal.md` 填的 `16.68 / 19.78 / 22.98%` 来自 epoch 3 checkpoint（自带头 `25.32 / 30.03 / 32.68%`）。请核实并确保主表与同裁判表汇报**同一个 `.pt` checkpoint**（若同时保留 epoch 3 与 epoch 6，请分列注明轮数）。

### 1.2 将 `docs/Summary of results.md` 已有的 4 组现成证据整理入表
1. **`BAGR-Base`（无 Evidence 头，全程与基线共用同一冻结分类器训练和评测）—— 回应 `zJYV` W2**：
   - 补入 `CIFAR10`（`16.89 / 20.99 / 23.85%`）和 `OGBG-PPA` 5/10/15%（`30.53 / 41.42 / 46.21%`，对比最强基线 `SAGPool` 的 `27.28 / 37.92 / 45.08%`）的 test 结果。
   - **作用**：这是最严格“同一冻结预测器”下的受控对比，直接证明即使去掉循环分类头、完全用同一个分类器，BAGR 学到的选点策略在 CIFAR10 和 PPA 上依然全面超越所有启发式与可训练基线。
2. **可训练基线的双评测口径对照表（自带头 `judge=own` vs 共享冻结分类器 `judge=clf`）—— 回应 `4jJN` W1（消除 Soundness=1 疑虑）**：
   - 整理 `CIFAR10`、`COLLAB`、`OGBG-PPA` 上 `Random` 与 `SAGPool / Soft-Mask / AgentNet` 在两种评测头下的 test 准确率。
   - **作用**：直观展示在统一冻结分类器下，PPA（5%）上的 `Soft-Mask (21.88%)`、`AgentNet (23.43%)`、`SAGPool (27.28%)` 以及 CIFAR10（15%）上的 `Soft-Mask (21.07%)`、`AgentNet (21.58%)` **选出的子图其实全部优于 `Random`**（`20.16%` / `20.57%`）；主表中它们低于 Random 的真正主因是**全图训练的自带分类头在 5% 极稀疏子图上发生了严重的分布漂移**（加之未揭示前沿节点特征置零与单智能体步长约束），反向印证了 BAGR 联合训练局部序列预测器的必要性。
3. **`Scratch-RL`（无 BC 热启动）vs `BC + RL` 的测试集准确率与训练轮数表 —— 回应 `zJYV` W1**：
   - 正文图 2 上排只画了训练奖励，导致审稿人误以为无 BC 在测试集上性能完全相当。请把 `Summary of results.md` 中 `CIFAR10`（`27.11 / 32.30 / 34.79%` @ 15轮 vs 有 BC 的 `27.79 / 33.84 / 36.55%`）和 `COLLAB`（无 BC 在 test 上坍缩且随预算增大从 `52.25%` 倒退至 `48.05%`，而有 BC 稳步升至 `55.74%`）整理成正式对比表。
4. **预训练冻结分类器的完整配置与训练数据说明表 —— 回应 `phri`**：
   - 列出三个数据集冻结分类器的详细结构（5层 GINE、`hidden_dim=300`、`dropout=0.5`、readout 如 CIFAR10 的 `meanmax` + 残差 + FFN 头、COLLAB 的 balanced class weight 等）、**训练样本数**（明确严格只用各数据集官方/标准 train split，无任何外部预训练数据、无 valid/test 泄漏）及全图 test 准确率；
   - 同时附上 `CIFAR10` 使用次优分类器（全图 `55.73%` vs `60.50%`）做奖励源时 BAGR 仍显著超越基线的结果（`Summary of results.md` 中 `evidence-third`）。

---

## 2. 反向固定 `BAGR-RL` 预测头评测各基线选点（零训练成本，回应 `zJYV` W2）

- **实验做法**：
  - 加载主表训练好的 `BAGR-RL` checkpoint（参数完全冻结，不重训）；
  - 在 test 集 rollout 时，将每一步选点动作替换为 `Random`、`BFS`、`Max-Frontier-Degree`、`SAGPool`、`AgentNet` 选出的前沿节点，但每一步仍把当前揭示状态喂给 `BAGR-RL` 的 `forward_states_evidence` 更新其 LSTM 和 Attention 状态，读满预算后用 `BAGR-RL` 自带的 Evidence 头输出分类预测（跑 `CIFAR10`、`COLLAB`、`OGBG-PPA`）。
- **实验目的**：
  - 与「统一使用冻结全图分类器」构成**正反双向控制变量**——证明即便所有选点方法共享同一个专为局部序列训练的 `BAGR` 预测头，`BAGR` 选出的节点轨迹依然显著优于启发式和可训练基线。

---

## 3. 补齐两阶段 RL 受控消融实验（必补训练，回应 `zJYV` W3 & `phri`）

- **实验范围**：在 **`CIFAR10`** 和 **`COLLAB`** 上进行（无需重跑重度的 PPA 全量训练）。
- **控制条件**：固定**同一个 BC 起点 checkpoint**、**相同的总 RL 训练轮数**（对齐主表 `BAGR-RL` 的总 RL 轮数，如 CIFAR10 共 15 轮、COLLAB 共 14 轮）以及相同的 `graphs-per-batch` / `seeds-per-graph` / `lr` / `temperature`。
- **对比组（共 5 行，均汇报 test 集自带头 `own` 与冻结分类器 `clf` 双指标）**：
  1. **Classifier-only（仅参考分类器奖励）**：`train_rl.py --reward-source classifier --reward asymmetric` 跑满总轮数；
  2. **Reader-only（仅自身预测头奖励）**：`train_rl.py --reward-source reader --reward asymmetric` 从 RL 第 1 轮直接跑满总轮数（全程不加载冻结分类器，可同时回应 `phri` 对预训练分类器依赖的顾虑）；
  3. **Soft-both only（仅混合奖励、不切换）**：`train_rl.py --reward-source soft-both` 跑满总轮数；
  4. **CE-only（无策略梯度对照）**：在 `code/train_reader/core/rl.py` 中增加一个 `--rl-lambda 0` 开关（令 `batch_loss = 0 * rl_loss + evidence_lambda * ev_loss`），只在自采样轨迹上用交叉熵更新预测头、不给选点头传策略梯度，跑满相同轮数；
  5. **Two-stage (Ours)**：主表的 `Soft-both -> Reader` 两阶段结果。
- **附加配置核对**：
  - 请核准 **`OGBG-PPA`** 主表 `26.90%` 实际对应的 RL 奖励源与轮数（`Summary of results.md` 记为 `reward source: reader`，但论文附录 A.3 表格写成了 `10 soft-both + 5 reader`，需核准真实日志以便回复中统一口径）。

---

## 4. 修正推理开销统计口径并补齐三数据集（零训练成本，回应 `4jJN`）

请调整 `code/tools/benchmark_inference_cost.py`（及 `code/train_reader/core/eval_loop.py` 对应分支）后，跑齐 **`CIFAR10`、`COLLAB`、`OGBG-PPA`** 三个数据集：

1. **剥离 BAGR 部署时不需要的冻结分类器开销**：
   - `eval_loop.py` 的 `evaluate_reader` 在算出 `final_ev` 后仍调用了 `SubgraphBuilder` 和冻结 `classifier` 前向；测推理成本时请支持纯部署模式（如 `classifier=None` 时跳过终局子图构建与 `clf` 前向），因为实际部署 BAGR 只用自身 Evidence 头，不加载全图分类器。
2. **对齐单预算档计时口径**：
   - 注意此前 `evaluate_reader` / `evaluate_fixed` 默认一次跑 3 个预算档，`s_per_graph` 累加了 3 条轨迹的耗时，不能直接与 `full-graph` 的单次前向耗时相除；请指定**单预算档**（如各数据集主预算档 5% / 10%，`--n-seeds 1`）对比单图推理耗时 `ms/graph`。
3. **修正显存统计与展示方式**：
   - 取消 `chunk_size=1` 下仅对 BAGR 扣减 `~8.5 MiB` cuDNN LSTM workspace 再 `×16` 的换算，也不要使用左右双 Y 轴画显存曲线（双轴会导致曲线在视觉上高过 `full-graph` 虚线，易引起审稿人误判）；
   - 直接在统一真实批量（如 `batch/chunk=16` 或 `32`）下统计并填表：**`Params (M)`**、**`模型常驻显存 (MiB)`**、**`前向实际峰值瞬态显存 (MiB)`**（删去包含 Adam 优化器状态的磁盘文件大小 `ckpt MiB` 列）。
4. **补充硬件无关计算量指标（用于结合特征获取成本说明收益）**：
   - 在开销表中增加两列：每图平均**揭示节点占比 (%)**，以及整条轨迹逐步编码的**累计可见边数 $\sum_{t=1}^B |E_{\text{vis}}(t)|$ 占全图分类器消息传递边数 $L_{\text{clf}} \cdot |E|$ 的比例 (%)**。

---

## 交付格式提醒
- 每张补充表格旁请简要注明对应的 **checkpoint 路径 / 实验目录名 / 评测命令**，确保周末合稿时每个数字都可追溯、前后一致。

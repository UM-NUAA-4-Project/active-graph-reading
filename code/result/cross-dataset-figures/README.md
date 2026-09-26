# cross-dataset-figures/ — 跨数据集联合分析图

cifar10 + COLLAB 联合图（单数据集图仍在各自 `result/{ds}/` 下）。
按分析类型分子文件夹存放，生成脚本在 `code/comparison/`。
废弃/被替代版本统一归档在各子文件夹 `_legacy/`（不删除，留证）。

## behavior/ — 行为分析图（choice/ 子目录，1×1 单图）

| 面板 | 内容 | 1×1 文件（behavior/choice/） |
|---|---|---|
| a | CIFAR10 前沿度数：被选 vs 可选（逐步，1–10 + ≥11 桶） | `choice_behavior_cifar10_degree.*` |
| b | CIFAR10 chosen 节点信息增益（nats；0 线=揭示不带来熵减） | `choice_behavior_cifar10_ig.*` |
| c | COLLAB 前沿度数（同 a 编码） | `choice_behavior_collab_degree.*` |
| d | COLLAB chosen 节点信息增益（同 b 编码） | `choice_behavior_collab_ig.*` |

- 单图 89mm 单栏（pdf/png/svg），无面板字母；系列标注横放在标题行
  （数据集名右侧，彩色对应曲线），度数/IG 两类面板同式。
- 数据：CIFAR10 侧 `choice_behavior_b10_g100s4.json`（test 前 100 图 ×
  4 seeds 重采，**epoch6** checkpoint，同协议 argmax rollout）；COLLAB 侧
  `choice_behavior_b10.json`（test 501 图，epoch6，聚合同序）。均位于各自
  `behavior-analysis/choice/` 下。
- **IG 口径（2026-09-25 晚起）= reader 自身证据头 belief 熵的逐步下降**
  （沿实际轨迹相邻前向作差，零额外计算；旧冻结分类器判定口径 JSON 已加
  `_clfigain` 后缀留档在各自 behavior-analysis/ 下）。**x=1 位置记录种子
  信息增益 ln k − init_entropy**（G_0 仅种子时的 belief 熵相对均匀先验
  ln k 的下降；x=2 起为各步揭示的熵减），横坐标标注 `(1 = seed)`。
- **CIFAR10 口径注意**：epoch6 行为 = chosen 恒选前沿度 1 节点（轮廓行
  走），面板 a 说明文字已随之更新；旧 epoch3 口径（早期偏好高度数）已
  弃用，勿混引。
- 重新生成：`python code/comparison/plot_choice_behavior_1x1.py`（数据读
  behavior-analysis/choice/，产物写本目录 choice/）。
- `_legacy/`：美化前旧版单图（epoch3 口径 + 图例版），留证勿引用。
- mechanism 类图在各自 `behavior-analysis/mechanism/` 下，未入本目录。

## reward-curves/ — RL 奖励曲线（1×1 单图 + 2×2 合并）

| 面板 | 内容 | 1×1 文件 |
|---|---|---|
| a | CIFAR10 scratch-RL vs BC warm-start | `rl_reward_scratch_vs_evR_cifar10.*` |
| b | COLLAB 同上 | `rl_reward_scratch_vs_evR_collab.*` |
| c | COLLAB 奖励源切换链（仅 final accuracy） | `rl_reward_switch_collab.*` |
| d | CIFAR10 奖励源切换链（仅 final accuracy） | `rl_reward_switch_cifar10.*` |

- 单图均 89mm 单栏（pdf/png/svg），2×2 合并 `rl_reward_curves_2x2.svg`
  由四张单图 SVG 1:1 拼接（png 为 PyMuPDF 预览；论文引用 2×2）。
- **切换图规格（2026-09-25 定）**：只画最终判别精度 ev_acc_final（奖励源
  量纲不同不再同轴画奖励）；切换后区域浅蓝阴影（= blue 7% 白底预混实色
  #EEF2F8，视觉同 `_legacy/rl_reward_curves_1x4.svg` 存档版；不用 CSS
  opacity，PyMuPDF 渲染会坏）；切换点处前一阶段收敛末值以灰色水平虚线
  延伸到右端作对比基准。阶段标注：cifar10 = Soft-both→Reader；
  COLLAB = soft-both→Reader（第 7 轮起，epoch 连续无需 offset）。
- 重新生成：切换图 `plot_cifar10_reward_switch.py` / `plot_collab_reward_switch.py`
  （直接写入本目录）→ `merge_reward_svgs_1x4.py`（输出 2×2）。
- `_legacy/`（留证，勿引用）：`rl_reward_curves_1x4.*`（弃用的 1×4 合并版，
  缩放后字号不可读；亦是切换图阴影规格基准）、`rl_reward_curves_2x2_prev.svg`
  （换仅精度切换图前的 2×2）、`*_rl_reward_switch_dualaxis.*`（双轴奖励+
  精度过渡版，两数据集）、`collab_rl_reward_switch_1x4era.*`（1×4 时期
  collab 切换图，原 `collab/comparison/switch-rl-resource/` 已清空删除）。

## state-attention/ — attn_dist_*（state-attention 读出注意力分布，重绘版）

| 文件 | 内容 |
|---|---|
| attn_dist_heatmap.* | 预算 × 相对步位质量热图（渲染与原版一致） |
| attn_dist_curves.* | **结尾对齐**曲线：x = 0, …, last-15, …, last-1 + terminal 虚线；无 frozen decoder |
| attn_dist_combined.* | 2×2 合并（上热图下曲线） |

- 数据：`result/{ds}/experiments/state-attn/eval_state_attention.json`
  （100 图 × 4 seeds 重采，`eval_state_attention.py` 新增 `attn_dist_last`
  结尾对齐累计器；旧归一化分箱 JSON 与旧图 state-attn-figures/ 已清理）。
- 采样口径：100 图 × 4 seeds（2026-09-25，用户指定），非旧版全量 test。
- 重新生成：先跑 `code/train_reader/eval_state_attention.py`（两数据集），
  再 `python code/comparison/plot_attn_dist.py`。

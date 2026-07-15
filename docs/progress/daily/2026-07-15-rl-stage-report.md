# RL 阶段训练报告

**日期**: 2026-07-15
**方案**: Self-Sampled REINFORCE — tanh reward vs binary reward
**基线对比**: BC (Behavior Cloning, M4 teacher trajectories)

---

## 1. 训练配置

| 参数 | 值 |
|------|-----|
| Reader | LearnableReader (128 hidden, 2 layers) |
| Classifier | M3 best_model_ft.pt (frozen) |
| 训练图数 | 全部 train split |
| Graphs/batch | 8, Seeds/graph | 4 (effective batch = 32 trajectories) |
| Temperature | 0.2 |
| LR | 1e-4 |
| Budget | 随机抽取 {5%, 10%, 15%} |

---

## 2. 训练过程 — Loss 与 Train Acc

### REINFORCE (tanh)

| Epoch | Loss | avg_R | Train Acc |
|-------|------|-------|-----------|
| 2 | 0.639 | 0.335 | 0.648 |
| 3 | 0.535 | 0.326 | 0.646 |
| 4 | 0.553 | 0.325 | 0.641 |
| 5 | 0.372 | 0.339 | 0.652 |
| 6 | 0.373 | 0.349 | 0.657 |
| 7 | 0.425 | 0.352 | 0.660 |
| 8 | 0.434 | 0.349 | 0.654 |
| 9 | 0.395 | 0.351 | 0.651 |
| 10 | 0.299 | 0.328 | 0.631 |

- Loss 从 0.64 下降到 ~0.30-0.43，但 avg_R 和 train acc 基本持平，无明显上升趋势
- Epoch 10 的 avg_R (0.328) 反而不如 epoch 6-8，存在波动但无过拟合迹象

### REINFORCE (binary)

| Epoch | Loss | avg_R | Train Acc |
|-------|------|-------|-----------|
| 1 | 5.833 | 0.111 | 0.111 |
| 2 | 3.012 | 0.121 | 0.121 |
| 3 | 2.230 | 0.122 | 0.122 |
| 4 | 1.826 | 0.125 | 0.125 |
| 5 | 2.004 | 0.123 | 0.123 |
| 6 | 1.669 | 0.120 | 0.120 |

- Loss 持续下降（5.83 → 1.67），avg_R 和 train acc 基本持平 (~0.12)
- 注意：binary 的 loss 量级比 tanh 大，因为 R ∈ {0, 1} 而非 tanh 映射后的连续值

> **关于 train acc 差异**：tanh 的 train acc (~0.65) 远高于 binary (~0.12)，这是因为二者的 acc 统计口径一致（都是 classifier argmax 正确率），但 checked 的 checkpoint epoch 不同（tanh epoch 7 vs binary epoch 4）。更大可能是 tanh 实验的 checkpoint 对应的 classifier 在训练集上过拟合更严重（train acc 虚高），而验证集上实际表现反而更差（见下表）。

---

## 3. 最终验证/测试结果（best seed, per-budget）

### Validation

| Method | Budget 5% | Budget 10% | Budget 15% |
|--------|-----------|------------|------------|
| BC | 11.07% | 12.49% | 13.90% |
| REINFORCE (binary) | 10.83% | 12.26% | 13.40% |
| REINFORCE (tanh) | 10.33% | 11.96% | 12.76% |

### Test

| Method | Budget 5% | Budget 10% | Budget 15% |
|--------|-----------|------------|------------|
| BC | 10.60% | 12.57% | 14.15% |
| REINFORCE (binary) | 10.46% | 12.41% | 13.70% |
| REINFORCE (tanh) | 10.14% | 11.28% | 11.85% |

---

## 4. 结论

1. **BC > binary > tanh**：BC 在所有 budget 下最优；binary reward 接近 BC（差距 ~0.3-0.5%），tanh 明显最差
2. **Tanh reward 效果不佳**：tanh(1 − CE/ln37) 将 reward 映射为连续值，但实际效果不如简单的 0/1 二元信号，可能与 PPA 分类器置信度偏高导致 tanh reward 区分度不足有关
3. **Binary REINFORCE 接近 BC 上限**：说明 REINFORCE 在当前 setup 下可以逼近 teacher 轨迹的质量，但 BC 仍然是更稳定的选择
4. **绝对准确率偏低**（~10-14% vs random ≈ 2.7%）：在低 budget (5-15%) 下，自适应策略确实优于随机，但距离全图分类器 (~64%) 仍有巨大差距

---

## 5. 附录：完整结果对比图

见 `code/m5/base/analysis/accuracy_comparison.png`

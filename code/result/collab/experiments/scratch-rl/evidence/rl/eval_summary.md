# collab 从零直连 RL（scratch-rl）评估总结

- **日期**：2026-09-22（训练 2026-09-21 22:57 CST 启动，~03:15 完成 10 轮；评估 ~03:20–03:50 完成）
- **训练**：`train_rl.py --dataset collab --root /mnt/data/COLLAB --graphs-per-batch 4 --seeds-per-graph 4 --reader-kind evidence --reward-source reader --exp scratch-rl --from-scratch --epochs 10`（无 BC/bootstrap，奖励=reader 自身证据头终判）
- **最优轮选择**：`training_log.json` 中 `ev_acc_final` 最高 = **Epoch 5（0.6368）**；评估对象 `checkpoint_epoch5.pt`

## 逐轮 ev_acc_final（训练时自采样 rollout，reward-source=reader 故 avg_R≡acc≡ev_acc_final）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ev_acc_final | .5916 | .6155 | .6237 | .6282 | **.6368** | .6319 | .6323 | .6347 | .6288 | .6221 |
| 逐步 ev_acc | .6395 | .6726 | .6829 | .6930 | .6904 | .6901 | .6853 | .6919 | .6907 | .6844 |

## 评估结果（checkpoint_epoch5，eval_reader.py，ev 终判=自身证据头；n_seeds 默认，跨种子平均）

| split（图数） | 预算 B | learnable（ev 终判） | clf 参照（同子图） |
|---|---|---|---|
| valid（499） | 10% | 0.5205 | 0.1543 |
| valid（499） | 15% | 0.5165 | 0.1543 |
| valid（499） | 20% | 0.5050 | 0.1543 |
| **valid overall** | — | **0.5140** | 0.1543 |
| test（501） | 10% | 0.5225 | 0.1557 |
| test（501） | 15% | 0.5110 | 0.1557 |
| test（501） | 20% | 0.4805 | 0.1557 |
| **test overall** | — | **0.5047** | 0.1557 |

- 全图上界分类器参照（full-graph ref）：valid **0.7154**，test **0.6906**
- 预算 = 揭示节点比例（不含 seed）；种子 = 确定性公式（tools/seeds）

## 观察备注

- 从零直连 RL 十轮内收敛到 **52% 上下 = COLLAB 多数类水平**（多数类占比 ≈52%），与 AgentNet 自带头终判（52.1/51.9）同一水平档；冻结分类器对揭示子图的判定塌缩在 ~15.5（少数类水平），与既有 collab 口径一致。
- 预算增大反而略降（B20 最低），提示策略对读多读少不敏感、类水平判定为主。

## 产物

- 服务器：`/mnt/active-graph-reading/code/result/collab/experiments/scratch-rl/evidence/rl/`（checkpoint_epoch1–10.pt、training_log.json、eval_learnable.json、eval_best_epoch5.log）
- 本目录（已回传）：`eval_learnable.json`、`training_log.json`、`config.json`、本总结

## 评估命令

```
python code/train_reader/eval_reader.py --dataset collab --root /mnt/data/COLLAB \
  --checkpoint .../scratch-rl/evidence/rl/checkpoint_epoch5.pt \
  --classifier .../result/collab/classifier/finetune/best_collab_classifier.pt \
  --ref-batch-size 16
```

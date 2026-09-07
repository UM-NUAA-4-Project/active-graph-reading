# 实验 config 模板

用法：复制 `*.example.json` 改值，启动时 `--config <file>`：

```bash
python code/train_reader/train_bc.py --config configs/train_bc.example.json
# 临时覆盖一个值（CLI 显式传参 > config > 默认值）
python code/train_reader/train_bc.py --config configs/train_bc.example.json --lr 1e-4 --exp bc_lr1e4
```

三条规则：

1. **键名 = CLI 选项的下划线形式**（`--reader-kind` → `"reader_kind"`）；脚本 `--help` 列出的选项都可配置。
2. **优先级**：CLI 显式传参 > config JSON > argparse 默认值。config 只覆盖"仍是默认值"的参数——想让某键生效，命令行就不要重复传它。
3. config 里的多余键被忽略 → **训练产物的 `<output>/config.json` 可直接喂回**复现。
   每次训练把合并后的完整 config 写入 `<output>/config.json` 并内嵌每个 epoch
   checkpoint；`--resume` 与 `eval.py` 以 checkpoint 内嵌 config 为准。

`--bootstrap` / `--classifier` / `--resume` 这类逐次运行才确定的路径参数建议留在
命令行。相对路径解析：优先按 `code/`（`data/...`、`result/...`），兼容仓库根形式。

---

## train_classifier/train_full_graph.py（全图上界分类器）

| 键 | 类型/默认 | 说明 |
|---|---|---|
| dataset / root | str / 必填 | 数据集注册名（ppa/cifar10/collab）/ 数据根目录 |
| max_graphs | int, 0 | 每 split 截断（0=全部；**PPA 必配**） |
| epochs / batch_size / lr | int/float | 训练基本参数 |
| weight_decay | float, 0.0 | Adam weight decay |
| label_smoothing | float, 0.0 | 交叉熵标签平滑 |
| class_weight | str, "none" | 类别加权（cross_entropy weight）：`balanced`= w_c=N/(K·count_c)，用于多数类占比 >0.5 导致退化为预测多数类的数据集（如 collab）；`none`=不加权。finetune_subgraph.py 同名参数共用实现 |
| scheduler_patience | int, 0 | ReduceLROnPlateau patience（0=不用） |
| early_stop_patience | int, 0 | 连续 N 轮 test acc 无提升早停（0=不用） |
| eval_every | int, 5 | 每 N 轮 test 评估并保留 eval_epoch{N}.pt / best_*.pt |
| hidden_dim / n_layers / dropout | int/float | uniform 结构（300×5 层、dropout 0.5） |
| hidden_dims / dropouts | int[]/float[] | **逐层结构**（提供时覆盖 uniform，如 [300,256,128]） |
| backbone | str, gin | 分类器 backbone：`gin`（GINEEncoder）/ `gat`（GATEncoder 多头注意力含边特征） |
| n_heads | int/int[] | GAT 每层注意力头数（默认 4 头/层；要求 hidden_dims[i] 能被 n_heads[i] 整除，如 hidden 256 + heads 4） |
| head | str, linear | 分类头：`linear`（单层 Linear）/ `ffn`（两层 MLP，同宽隐层） |
| gnn_norm | str, null | GNN 层间归一化：null=各 backbone 原行为（GIN=BatchNorm、GAT=无）；可显式 `batchnorm` / `layernorm` / `none`（不含 GINELayer MLP 内部的 BN） |
| virtual_node / residual | bool, false | OGB gin-virtual / 残差（GAT 同样支持，要求各层同宽） |
| device / seed | - | 设备 / 随机种子 |
| exp | str | 实验名（默认输出 result/{dataset}/classifier/） |

## train_reader/train_bc.py（BC 行为克隆）

| 键 | 类型/默认 | 说明 |
|---|---|---|
| dataset / root / max_graphs | - | 同上 |
| data_dir | str | 轨迹目录（默认 data/{dataset}/trajectories；可指向 _filtered/_balanced） |
| teachers | str[] | 参与训练的教师键（需与轨迹文件内教师名一致） |
| reader_kind | base/attention | reader 类型 |
| hidden_dim / n_layers / dropout | - | uniform 结构（默认 128×2 层、dropout 0.1） |
| hidden_dims / encoder_dropouts | int[]/float[] | encoder GNN 逐层（attention 要求各层同宽） |
| scorer_hidden_dims | int[] | base reader 的 MLP scorer 逐层（默认 2 个隐层） |
| m / n_heads | int | belief token 数 / 注意力头数（attention） |
| temperature | float | 仅影响 RL/评估；BC 训练恒 1.0 |
| edge_feat_dim / node_feat_dim | int | 覆盖 DatasetSpec 维度（一般不用） |
| epochs / lr / batch_size | - | 10 / 1e-3 / 32 |
| device / seed / exp | - | 输出 result/{dataset}/experiments/{exp}/{kind}/bc/ |

## train_reader/train_rl.py（REINFORCE 微调）

| 键 | 类型/默认 | 说明 |
|---|---|---|
| dataset / root / max_graphs | - | 同上 |
| reader_kind + 结构键 | - | 同 BC（hidden_dim/n_layers/dropout/hidden_dims/m/n_heads/temperature） |
| reward | binary/tanh/asymmetric | 奖励类型（默认 binary） |
| reward_neg_scale | float, 0.1 | batch 优势归一化负侧降权 λ（1=无不对称） |
| freeze | bool, false | 冻结共享层，仅训特征投影层 |
| epochs / lr | - | 10 / 1e-4 |
| graphs_per_batch | int, 32 | 每批图数（×seeds_per_graph 个 rollout/图） |
| seeds_per_graph | int, 4 | 每图确定性 seed 数 |
| temperature | float, 0.2 | 采样温度（探索强度） |
| exp | str | 输出 result/{dataset}/experiments/{exp}/{kind}/rl/ |

`--bootstrap`（起点 reader ckpt，跨架构/跨数据集自动部分权重迁移）与
`--classifier`（冻结奖励分类器）传 CLI；`--resume` 与 bootstrap 互斥。

## trajectories/generate_trajectories.py（教师轨迹生成）

| 键 | 类型/默认 | 说明 |
|---|---|---|
| dataset / root / max_graphs | - | 同上 |
| classifier | str | 上界分类器 ckpt（也可 CLI 传；gradient 教师需要 autograd） |
| teachers | str[] | gradient / structural-degree / structural-pagerank / info-gain / hybrid-a{α} |
| budget_ratios | float[] | 预算档位（默认 spec.budget_options = 0.05/0.10/0.15） |
| n_seeds | int, 5 | 每图每教师 top-n seed |
| labels | int[]/null | 只生成指定类别（多机按类分摊）；null=全部 |
| k_shards | int, 1 | 分片数（配 --shard-idx 用；已有分片自动跳过=断点续跑） |
| graph_batch_size | int, 32 | 跨图批大小 |
| device / seed | - | 输出默认 data/{dataset}/trajectories/ |

## 数据集注册表（datasets/）

| 名 | 类数 | node_feat | edge_feat | 自环 | root |
|---|---|---|---|---|---|
| ppa | 37 | 0 | 7 | 加载器加（恰好一次） | D:\ogb-master\dataset |
| cifar10 | 10 | 5 | 1 | 数据自带 | D:\data\superpixels |
| collab | 3 | 0 | 1（零占位） | 加载器 remove+add（恰好一次） | D:\data\COLLAB |

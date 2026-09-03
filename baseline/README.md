# baseline/ — 基线复现

在当前三个数据集（cifar10 / collab / ppa）上复现与我们工作最相似的基线
**GAM**（Graph Classification Using Structural Attention, KDD 2018,
Lee/Rossi/Kong）。原实现见 `gam_src/`（逐字复制自
[benedekrozemberczki/GAM-master](https://github.com/benedekrozemberczki/GAM)，
MIT License，仅加装运行依赖，未改一行逻辑）。

## GAM 与本工作的对齐决策

GAM 原设定：RNN 智能体在图上自适应走 `time` 步收集证据后分类。与我们
的差异及对齐方式：

| 维度 | GAM 原设定 | 本复现 | 理由 |
|---|---|---|---|
| 智能体数 | 测试时 `repetitions=10` 次随机起点读取 + 多数投票（≈多智能体集成） | `--repetitions 1` | 对齐我们的单 reader 设定 |
| 预算 | 全数据集统一 `time` 步 | `round(ratio × n_nodes)` 逐图计算 | 对齐协议"预算 = 揭示节点数（不含 seed）的比例"语义；GAM 走 t 步访问 t+1 个节点（含随机起点），计数一致 |
| 节点标签 | 数据集自带离散标签（HIV/ER） | collab/ppa 无节点标签 → 每图度数 quantile 分桶合成（`--node-labels degree`，默认 4 桶）；`constant`=全 0（此时 step 网络失效，退化为随机游走 + LSTM 分类） | GAM 的 step 网络按"标签注意力"选步，必须有离散标签才能运作；度数是结构可见信息，不引入额外监督 |
| 边 | 原始边 | 剔除协议自环 | GAM 看自然图，reader 协议自环是它不需要的实现细节 |
| 评估 | 只评 test | 训练后分别评 valid 与 test | 与我们的评估口径一致 |
| 其余超参 | — | 与原默认一致（batch 32 / lr 1e-3 / γ 0.99 / step 32 / combined 64 / epochs 10 / wd 1e-5） | 忠实复现 |

## 结构

```
baseline/
├── gam_src/            # GAM 原实现逐字复制（gam/utils/param_parser/main + LICENSE）
├── convert_dataset.py  # registry 数据集 → GAM JSON（code/datasets 注册表驱动）
├── run_gam.py          # 单智能体 + 逐图预算比例训练/评估编排 + 记录
├── data/{dataset}/     # [gitignore] 转换产物 {train,valid,test}/*.json + meta.json
└── output/{dataset}/   # [gitignore] budget_{key}/{gam_logs,predictions_*,record}.json
                        #            + summary.json（全部预算档汇总）
```

## 用法

```bash
# 1. 数据集转换（collab；cifar10/ppa 转换待补，见 TODO）
python baseline/convert_dataset.py --dataset collab --root D:\\data\\collab

# 2. 训练 + 评估（单智能体，3 档预算）
python baseline/run_gam.py --dataset collab --data-dir baseline/data/collab \
    --budgets 0.10 0.15 0.20 --repetitions 1 --epochs 10

# 冒烟（小样本）
python baseline/convert_dataset.py --dataset collab --root D:\\data\\collab \
    --max-graphs 60 --output-dir baseline/data/collab_smoke
python baseline/run_gam.py --dataset collab --data-dir baseline/data/collab_smoke \
    --budgets 0.10 --epochs 2 --output-root baseline/output/collab_smoke
```

依赖：`texttable`（原仓库要求，`pip install texttable`）；其余
networkx/pandas/sklearn 环境已备。

## TODO

- [x] cifar10 转换：degree 分桶标签（`--max-graphs` 可采样）。注意 GAM 只
      消费离散节点标签，cifar10 的 5 维连续节点特征不进入 baseline；cifar10
      无 valid split，run_gam 自动跳过 valid、只评 test。
- [ ] ppa 转换：需走 `spec.load_graph_subset` 选择性加载（158k 全图不可行），
      节点标签同样用度数分桶。
- [ ] 其余基线（若有）按同一目录约定扩展。

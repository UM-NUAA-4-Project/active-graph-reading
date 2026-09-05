# baseline/ — 基线复现（AGR 协议适配版）

在 cifar10 / collab / ppa 三个数据集上与主方法对比的 baseline 复现。
**所有基线都是协议可行的 agent**：对每个基线从原文设定到协议可用形态的
改动与理由，见 [protocol-adaptation.md](protocol-adaptation.md)（先读它）。

当前基线：

| 基线 | 状态 | 训练 | `--strategy` 名 |
|---|---|---|---|
| random-walk（GraphSAGE 一族，深度型启发式） | ✅ 已接入 eval | 无需训练 | `random-walk` |
| SAGPool（ICML 2019，学习打分器） | ✅ 已接入 eval，scorer 待正式训练 | 完整图监督 | `sagpool` |
| SubgraphX（ICML 2021，测试时搜索） | ⬜ 待实现（适配预决已定，见 protocol-adaptation.md §5） | 无训练/搜索贵 | — |

主仓已有固定策略（random / bfs / max-frontier-degree）不在本目录，
见 `code/active_graph_reader/reader/fixed/`。

## 结构与产物约定

```
baseline/
├── README.md                  # 本文件：总览 + 用法
├── protocol-adaptation.md     # 对 AGR 协议的适配改动诠释（专门文档）
├── randwalk.py                # random-walk 策略（无训练）
├── sagpool.py                 # SAGPool 模型（复刻官方骨架）+ 协议合规选点 agent
├── run_sagpool.py             # SAGPool scorer 训练入口（full graph）
├── strategies.py              # 注册 random-walk 到 fixed strategy 注册表
├── artifacts/                 # [gitignore] 中间产物（按数据集分目录）
│   └── {dataset}/
│       ├── sagpool.pt               # 训练出的 scorer（eval.py 默认读取）
│       └── sagpool.history.json     # 训练曲线
└── output/                    # [gitignore] 最终结果（仅存放不走 eval.py 的
                               #   独立评估产物；目前为空）
```

约定：
- **代码提交**，`artifacts/` 与 `output/` gitignore（`.gitignore` 已配）；
- 走 `eval.py` 的基线评估，**最终结果在主仓** `code/result/{ds}/fixed/eval_fixed.json`
  （统一 schema，与 random/bfs/reader 同表），不在本目录重复落盘；
- 相对路径一律按仓库根解析（不要用 `tools.paths.resolve`，其 code/ 优先
  规则会把 baseline/ 误解析到 code/ 下）。

## 用法

```bash
# 1. SAGPool scorer 训练（中间产物 → baseline/artifacts/{ds}/sagpool.pt）
python baseline/run_sagpool.py --dataset cifar10 --root D:\\data\\superpixels \
    --epochs 30                        # PPA 必配 --max-graphs
# 冒烟（仅验证管线，精度无意义）
python baseline/run_sagpool.py --dataset cifar10 --root D:\\data\\superpixels \
    --max-graphs 2000 --epochs 5 --output baseline/artifacts/cifar10/sagpool_smoke.pt

# 2. 评估（与 random/bfs 同管线；sagpool 省略 --strategy-ckpt 时读
#    baseline/artifacts/{ds}/sagpool.pt）
python code/train_reader/eval.py --dataset cifar10 --root D:\\data\\superpixels \
    --strategy random-walk sagpool \
    --classifier result/cifar10/classifier/GIN/best_cifar10_classifier.pt
```

单测（协议合规性 / 确定性 / 预算约束 / 注册集成，已入 smoke_test）：

```bash
python code/tests/test_baseline_strategies.py
```

## TODO

- [ ] SAGPool scorer 正式训练（cifar10 全量 30k、更长 epoch；collab/ppa 各一）。
- [ ] SubgraphX（连通约束树搜索 + 无标签奖励 + 算力上限，见
      protocol-adaptation.md §5）。

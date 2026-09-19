# ICLR 2027 OpenReview 提交清单

> **✅ 摘要已提交（2026-09-19 14:13，早于 19:59 截止）**：Submission Number **55473**，
> forum: <https://openreview.net/forum?id=0B9JzwM8MR>，OpenReview 回执邮件已收到。
> 已锁死：作者名单（Zhihui Huang, Meng Qu，此后只能调序）、Reciprocal Reviewing Author（Meng Qu）、
> Exemption（We do not need an exemption）。AI Assistance 勾了 4 项（writing / retrieval /
> ideation+execution / draft sections）——**终稿 AI 披露节必须覆盖全部 4 项**。

> 表单：ICLR 2027 Conference Submission（OpenReview）
> 表单显示窗口：Submission start Aug 19, 2026 7:59 PM — Deadline Sep 19, 2026 7:59 PM
> 你确认的**摘要截止：今晚午夜**。带 `*` 为必填；⚠️ 标记的字段**摘要截止后锁死，今晚必须定稿**。
> 可直接粘贴的内容见下方 §2（均出自 `paper/iclr_draft/main-v3.tex` / `main_en-v3.md`，与定稿数据口径一致）。

---

## 1. 全字段速览

| #  | 字段                                  | 必填 |                 摘要截止后锁死？                 | 填什么                                                                                                     |
| -- | ------------------------------------- | :--: | :----------------------------------------------: | ---------------------------------------------------------------------------------------------------------- |
| 1  | Title                                 |  ✅  |                 可改(至全文截止)                 | §2.1 现成                                                                                                 |
| 2  | Authors                               |  ✅  | ⚠️**不能增删**（只能调顺序，至全文截止） | 你（Zhihui Huang，已加）+**导师（现在就加！）**                                                      |
| 3  | Keywords                              |  ✅  |                       可改                       | §2.3 现成                                                                                                 |
| 4  | TL;DR                                 |  ❌  |                       可改                       | §2.2 现成（建议填）                                                                                       |
| 5  | Abstract                              |  ✅  |                       可改                       | §2.4 现成（支持 TeX）                                                                                     |
| 6  | PDF                                   |  ❌  |                       可换                       | 摘要阶段**可不上传**（≤50MB）；全文截止前必须交，见 §4                                             |
| 7  | Supplementary Material                |  ❌  |                       可换                       | 建议摘要阶段不传；要传必须匿名、单 zip（≤100MB）、对公众可见                                              |
| 8  | Primary Area                          |  ✅  |                       可改                       | 见 §3 决策 D1                                                                                             |
| 9  | Code of Ethics                        |  ✅  |                        —                        | 勾选确认（导师也需知情同意）                                                                               |
| 10 | Paper Visibility                      |  ✅  |                        —                        | 勾选确认（**评审一开始论文即公开、不可撤回删除**；撤稿立即去匿名）                                   |
| 11 | Submission Requirements               |  ✅  |                        —                        | 勾选确认（互惠评审 + AI 披露 + 作者须知）                                                                  |
| 12 | Reciprocal Reviewing Author           |  ✅  |                ⚠️**锁死**                | 见 §3 决策 D2（填作者的 OpenReview profile）                                                              |
| 13 | Recent Qualifying Paper               |  ❌  |                ⚠️**锁死**                | 一般不勾；仅当指定评审人的唯一合格论文发表在 2026 版 CVPR/ECCV/EACL/IJCAI/IROS/RSS/CoRL/KDD/CoLT/AAAI 时勾 |
| 14 | Reciprocal Reviewing Exemption        |  ✅  |                ⚠️**锁死**                | 见 §3 决策 D2（不需要豁免就选 "We do not need an exemption"）                                             |
| 15 | Reciprocal Reviewing Exemption Reason |  ❌  |                ⚠️**锁死**                | 仅申请豁免时填理由（如均为首次投稿、无合格评审人）                                                         |
| 16 | AI Assistance                         |  ✅  |                       可改                       | 见 §3 决策 D3（多选，须与文中 AI 披露节一致）                                                             |

**硬性规则提醒**：

- 所有作者投稿前必须已有 OpenReview profile；每人每年最多 20 篇投稿，禁止建重复 profile（会 desk reject + 封禁）。
- 双盲：投稿期作者身份仅 chair 可见，评审结束去匿名。

---

## 2. 内容

### 2.1 Title

Learning Where to Read: Graph Classification under Observation Budgets

### 2.2 TL;DR

We introduce budget-limited active graph reading: an agent learns where to reveal nodes one by one under a fixed budget and classifies from the revealed subgraph alone.

### 2.3 Keywords（逗号分隔）

graph classification,reinforcement learning,behavior cloning,graph neural networks,budget-limited decision making

### 2.4 Abstract（与 `main-v3.tex` 一致，TeX 原样粘贴）

Graph-level prediction typically assumes access to the complete input graph. However, when node information must be acquired through external queries or costly measurements, a predictor must decide what to observe within a limited budget. We study budget-limited active graph reading, where an agent sequentially reveals nodes and predicts the graph label from acquired evidence. Learning such a policy is challenging because graph labels provide no direct supervision for exploration actions, and the combinatorial space of reading trajectories makes unguided exploration difficult. We introduce BAGR, a budget-aware graph reader that combines sequential node selection with recurrent evidence accumulation. To guide exploration, we generate privileged teacher trajectories and filter them according to a pretrained classifier’s predictive performance on the revealed subgraphs. Imitation learning transfers these demonstrations into an observation-constrained policy, providing a task-informed initialization. On-policy reinforcement learning then refines the policy using classification-based rewards on its own trajectories, enabling task-driven exploration beyond the teacher demonstrations. We evaluate BAGR on CIFAR10 superpixels, COLLAB, and OGBG-PPA under explicit observation and node-budget constraints, and experimental results show gains over the evaluated fixed heuristics.

### 2.5 Primary Area

learning on graphs and other geometries & topologies

### 2.6

Code Of Ethics\Paper Visibility\Submission Requirements 都勾选

### 2.7 Reciprocal Reviewing Author

Meng Qu

### 2.8 Reciprocal Reviewing Exemption

don't need

### 2.9 AI Assistance（必填多选，须与论文披露一致）

ICLR 2027 政策要求论文内含**强制性 AI 使用披露节**，漏写可 desk reject（政策：iclr.cc/Conferences/2027/AIPolicyForAuthors）。本项目的实际情况（写作/代码大量使用 LLM，草稿原拟有 LLM usage 声明），诚实勾选，典型组合：

- [X] Yes, to aid or polish writing.
- [X] Yes, to draft sections of the paper.
- [X] Yes, for retrieval and discovery (e.g., finding related work).
- [X] Yes, for research ideation or execution.（若实验代码也由 AI 辅助则勾）

⚠️ 对应动作：全文稿恢复/保留 **LLM usage 披露节**（README 待填清单第 3 条本来就计划"移入附录"），不勾没写的选项。

---

## 3. 提交前 30 秒自检

- [ ] 导师已加入 Authors（⚠️ 锁死项）
- [ ] Reciprocal Reviewing Author / Exemption 已填（⚠️ 锁死项）
- [ ] Title / Abstract / Keywords 已粘贴并 Preview 无乱码
- [ ] Primary Area 已选
- [ ] 三个声明框（Ethics / Visibility / Requirements）已勾
- [ ] AI Assistance 勾选与论文内披露节一致
- [ ] 点 Submit 后收到 OpenReview 确认邮件/页面状态变为 submitted

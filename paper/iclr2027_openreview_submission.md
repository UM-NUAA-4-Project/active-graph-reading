# ICLR 2027 OpenReview 提交清单（摘要截止：今晚）

> 表单：ICLR 2027 Conference Submission（OpenReview）
> 表单显示窗口：Submission start Aug 19, 2026 7:59 PM — Deadline Sep 19, 2026 7:59 PM
> 你确认的**摘要截止：今晚午夜**。带 `*` 为必填；⚠️ 标记的字段**摘要截止后锁死，今晚必须定稿**。
> 可直接粘贴的内容见下方 §2（均出自 `paper/iclr_draft/main-v3.tex` / `main_en-v3.md`，与定稿数据口径一致）。

---

## 1. 全字段速览

| # | 字段 | 必填 | 摘要截止后锁死？ | 填什么 |
|---|------|:----:|:----:|--------|
| 1 | Title | ✅ | 可改(至全文截止) | §2.1 现成 |
| 2 | Authors | ✅ | ⚠️ **不能增删**（只能调顺序，至全文截止） | 你（Zhihui Huang，已加）+ **导师（现在就加！）** |
| 3 | Keywords | ✅ | 可改 | §2.3 现成 |
| 4 | TL;DR | ❌ | 可改 | §2.2 现成（建议填） |
| 5 | Abstract | ✅ | 可改 | §2.4 现成（支持 TeX） |
| 6 | PDF | ❌ | 可换 | 摘要阶段**可不上传**（≤50MB）；全文截止前必须交，见 §4 |
| 7 | Supplementary Material | ❌ | 可换 | 建议摘要阶段不传；要传必须匿名、单 zip（≤100MB）、对公众可见 |
| 8 | Primary Area | ✅ | 可改 | 见 §3 决策 D1 |
| 9 | Code of Ethics | ✅ | — | 勾选确认（导师也需知情同意） |
| 10 | Paper Visibility | ✅ | — | 勾选确认（**评审一开始论文即公开、不可撤回删除**；撤稿立即去匿名） |
| 11 | Submission Requirements | ✅ | — | 勾选确认（互惠评审 + AI 披露 + 作者须知） |
| 12 | Reciprocal Reviewing Author | ✅ | ⚠️ **锁死** | 见 §3 决策 D2（填作者的 OpenReview profile） |
| 13 | Recent Qualifying Paper | ❌ | ⚠️ **锁死** | 一般不勾；仅当指定评审人的唯一合格论文发表在 2026 版 CVPR/ECCV/EACL/IJCAI/IROS/RSS/CoRL/KDD/CoLT/AAAI 时勾 |
| 14 | Reciprocal Reviewing Exemption | ✅ | ⚠️ **锁死** | 见 §3 决策 D2（不需要豁免就选 "We do not need an exemption"） |
| 15 | Reciprocal Reviewing Exemption Reason | ❌ | ⚠️ **锁死** | 仅申请豁免时填理由（如均为首次投稿、无合格评审人） |
| 16 | AI Assistance | ✅ | 可改 | 见 §3 决策 D3（多选，须与文中 AI 披露节一致） |

**硬性规则提醒**：
- 所有作者投稿前必须已有 OpenReview profile；每人每年最多 20 篇投稿，禁止建重复 profile（会 desk reject + 封禁）。
- 双盲：投稿期作者身份仅 chair 可见，评审结束去匿名。

---

## 2. 可直接粘贴的内容

### 2.1 Title

```
Learning Where to Read: Adaptive Node-Revealing Policies for Budget-Limited Graph Classification
```

### 2.2 TL;DR（两选一，均可再改）

方案 A（贴结论）：

```
We introduce budget-limited active graph reading: an agent learns where to reveal nodes one by one under a fixed budget and classifies from the revealed subgraph alone, recovering ~81% of full-graph accuracy on COLLAB with a 20% budget.
```

方案 B（更短）：

```
Learning where to read in a graph under a node budget: a leak-proof protocol plus three-stage training recovers ~81% of full-graph accuracy on COLLAB with only 20% of nodes revealed.
```

### 2.3 Keywords（逗号分隔）

```
graph classification, active graph reading, reinforcement learning, behavior cloning, graph neural networks, budget-limited decision making
```

### 2.4 Abstract（与 `main-v3.tex` 一致，TeX 原样粘贴）

```text
Graph classification methods almost universally assume that the entire graph is available at prediction time, yet on large graphs discriminative evidence is often concentrated in a small region while node-by-node inspection carries real cost. Existing pooling and explanation methods all operate under full observation and cannot answer, before the graph is visible, where reading should go; the direction has long lacked a leak-proof evaluation protocol, as well as evidence that a learned policy beats fixed ones at matched budgets. We therefore propose \emph{budget-limited active graph reading}: an agent starts from a single seed node, step by step selects frontier nodes to reveal under a fixed node budget, and once the budget is exhausted classifies only the revealed subgraph. Training proceeds in three stages, namely teacher-trajectory distillation, behavior cloning, and reinforcement fine-tuning with an asymmetric reward judged by a frozen classifier, while the evidence reader is decoupled from node selection and accumulates what has been read in a recurrent state, guaranteeing by construction that prediction depends only on the revealed set. Experiments show that the learned policy recovers about $81\%$ of full-graph test accuracy on COLLAB under a $20\%$ node budget, and exceeds the best fixed policy by up to $11.3$ percentage points on CIFAR10. These results indicate that task-aware adaptive reading can substantially recover full-graph accuracy under a very small inspection budget.
```

> 注：OpenReview 的 TeX 支持 `$...$`；`\emph{...}` 若预览异常，可替换为 `*budget-limited active graph reading*`（纯 markdown 斜体）。

---

## 3. 今晚必须由你决定的事项

### D1. Primary Area（必填下拉）

本工作是"图上的序贯决策/RL + 图分类"。建议在 "Select Primary Area" 下拉里按顺序搜：

1. **Agents**（ICLR 2026 起新增的一级领域，与"学会去哪里读"的 agent 叙事最贴）
2. **Reinforcement Learning and Decision Making**（稳妥、领域完全对口）
3. 含 **Graph** 的领域（如 Graph Learning / GNN 类条目）

以上以表单实际下拉项为准，选一个最贴近的即可（Primary Area 摘要截止后仍可改，不必纠结）。

### D2. 互惠评审三件套（⚠️ 摘要截止后全部锁死，今晚定）

- **Reciprocal Reviewing Author**：从已添加作者里选一人登记为评审人。先到
  <https://openreview.net/group?id=ICLR.cc/2027/Conference/Reciprocal_Reviewing_Check>
  查你（Zhihui Huang）是否合格；合格就填你自己的 profile，不合格通常填**导师**（导师资历一般稳过）。
  ⚠️ 若**没有任何作者**登记且无豁免，可能被 desk reject。
- **Recent Qualifying Paper**：默认不勾。仅当 D2 指定人的"唯一合格论文"是 2026 版 CVPR/ECCV/EACL/IJCAI/IROS/RSS/CoRL/KDD/CoLT/AAAI（自动检查器不索引、需线下核验）才勾。
- **Reciprocal Reviewing Exemption**：有合格评审人 → 选 "We do not need an exemption"；两人都不合格（如均为首次作者）→ 选对应豁免选项，并在 Reason 里写一句，例如：
  `Both authors are first-time ICLR authors and do not meet the reciprocal reviewing qualification.`

### D3. AI Assistance（必填多选，须与论文披露一致）

ICLR 2027 政策要求论文内含**强制性 AI 使用披露节**，漏写可 desk reject（政策：iclr.cc/Conferences/2027/AIPolicyForAuthors）。本项目的实际情况（写作/代码大量使用 LLM，草稿原拟有 LLM usage 声明），诚实勾选，典型组合：

- [x] Yes, to aid or polish writing.
- [x] Yes, to draft sections of the paper.
- [x] Yes, for retrieval and discovery (e.g., finding related work).
- [ ] Yes, for research ideation or execution.（若实验代码也由 AI 辅助则勾）

⚠️ 对应动作：全文稿恢复/保留 **LLM usage 披露节**（README 待填清单第 3 条本来就计划"移入附录"），不勾没写的选项。

---

## 4. 摘要提交前后路径（今晚 → 全文截止）

1. **今晚（摘要截止前）**：填 §2 全部内容 + 完成 D1–D3 + 三个确认框勾选。PDF/补充材料可留空，先点提交把摘要锁住。
2. **全文截止前（日期以官网为准，通常晚几天）**：
   - 上传最终 PDF（≤50MB）。⚠️ 当前 `main-v3.pdf` 仍有 17 处蓝色 `[REF: ...]` 占位引用，**未清完不要当作投稿版上传**（README 待填清单第 1、3 条：补引用、去匿名页眉确认）。
   - 可上传补充材料（单 zip、匿名、≤100MB；代码需先清理作者信息）。
   - 作者顺序仍可调整（增删不行）。
3. **注意**：Paper Visibility 条款——评审开始后论文即公开且**不可删除/撤回**；abstract 截止后作者名单不能再动。

---

## 5. 提交前 30 秒自检

- [ ] 导师已加入 Authors（⚠️ 锁死项）
- [ ] Reciprocal Reviewing Author / Exemption 已填（⚠️ 锁死项）
- [ ] Title / Abstract / Keywords 已粘贴并 Preview 无乱码
- [ ] Primary Area 已选
- [ ] 三个声明框（Ethics / Visibility / Requirements）已勾
- [ ] AI Assistance 勾选与论文内披露节一致
- [ ] 点 Submit 后收到 OpenReview 确认邮件/页面状态变为 submitted

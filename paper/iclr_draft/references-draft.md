# References 草稿（ICLR 引用格式）— v2 全量核实版

> 对应 `main-v3.tex` 中的蓝色占位 `\refadd{...}`。每条给出：REF 键、文献、ICLR bst 渲染预览、BibTeX 键与条目、核实状态。
> **v2（2026-09-21）**：全部条目已通过 Obsidian 知识库（Graph Learning vault）、Zotero「图学习」库、arXiv API、Crossref、OpenAlex、AAAI OJS 六路交叉核实。原 ❓/⚠️ 条目全部落定；同时纠正了 v1 中五处错误（见第二节）。
> 状态标记：✅ 已核实（元数据完整、来源交叉确认）｜⚠️ 已核实但有一处待你拍板｜❓ 非文献引用（交叉引用替换）。

---

## 一、ICLR 引用格式要点

1. **机制**：ICLR 模板使用 `natbib` 的**作者-年份**制。正文引用 `\citep{key}` 渲染为 `(Xu et al., 2019)`；行内主语用 `\citet{key}` 渲染为 `Xu et al. (2019)`。多项并列 `\citep{gin,diffpool}` 渲染为 `(Xu et al., 2019; Ying et al., 2018)`。
2. **样式文件**：`\bibliographystyle{iclr2027_conference}`（当前仓库暂以 `iclr2025_conference.bst` 改名顶替，官方 2027 版发布后替换文件名即可，bib 条目无需改动）。格式特征：作者名缩写在前（K. Xu, W. Hu, ...）、标题 sentence case、会议用 `In _会议名_`、年份收尾。
3. **落地方式**：新建 `references.bib`（第五节的全文即该文件内容），tex 末尾加两行：
   ```latex
   \bibliographystyle{iclr2027_conference}
   \bibliography{references}
   ```
   然后把全部 `\refadd{xxx}` 按第二节映射表替换为 `\citep{...}`，编译流程变为 pdflatex → bibtex → pdflatex ×2。

---

## 二、v1 → v2 纠错清单（重要）

以下五处是 v1 中标 ✅/⚠️ 但核实后**有误**的条目，投稿前务必以 v2 为准：

| # | REF 键 | v1 写法 | 核实结果 | 证据 |
|---|---|---|---|---|
| 1 | `graph-pooling` | 作者含 **Livi**，2022 | 第二作者应为 **Daniele Zambon**（无 Livi）；正式卷期为 TNNLS **35(2): 2708–2718, 2024**（2022 仅为在线早期访问） | Crossref DOI 10.1109/TNNLS.2022.3190922；arXiv:2110.05292 |
| 2 | `graph-explainability` | 第三作者 **J. Luo** | 应为 **Shurui Gui**；TPAMI **45(5): 5782–5799, 2023**（DOI 10.1109/TPAMI.2022.3204236） | arXiv:2012.15445 现行版作者表 + Crossref + OpenAlex 三方一致 |
| 3 | `subgraphx` | **ICML 2020**；作者 Yuan, **Yu Haitao**, **Gui**, Ji（4 人） | 应为 **ICML 2021**；正式作者 5 人：Yuan, **Yu Haiyang**, **Wang Jie**, **Li Kang**, Ji | arXiv:2102.05152（comment "Accepted by ICML 2021"）；知识库 SubgraphX 页 |
| 4 | `lrgb` | 作者 "Dwivedi, **Bresson**, et al."；且被用于 CIFAR10 | LRGB 正式作者 7 人**不含 Bresson**：Dwivedi, Rampášek, Galkin, Parviz, Wolf, Luu, Beaini；且 **LRGB 的 5 个数据集（Peptides-func/struct、PCQM-Contact、COCO-SP、VOC-SP）里没有 CIFAR10**——CIFAR10 超像素应引 Dwivedi et al., Benchmarking GNNs, JMLR 24(43):1–48, 2023（见 `cifar10` 条目） | arXiv:2206.08164；jmlr.org/papers/v24/22-0567.html |
| 5 | `rwesgnn` | 标题 "Self-Interpretable..."（AAAI 2026）vs 知识库/Zotero 的 "Self-supervised..."（arXiv 2025） | 两者是同一论文的两个版本：arXiv:2502.01809 旧题 Self-supervised，**AAAI-26 正式版改题为 Self-Interpretable**，引用以 AAAI 版为准：40(26): 21984–21993, DOI 10.1609/aaai.v40i26.39352 | AAAI OJS + Crossref |

另两处小坑：
- **RAW 第三作者姓名顺序**：arXiv 与 ACM/Crossref 均登记为 "Han Yufei"（姓 Yufei），但该作者通行署名为 **Yufei Han**；bib 按 `Han, Yufei` 写（渲染为 Y. Han，与 Google Scholar 一致）。Zotero 条目里该作者姓/名颠倒，已在本文件修正、Zotero 库未动。
- **正文 REF 覆盖**：`main_zh-v3.md` 实际出现 17 个 REF 键；**`soft-mask` 没有 REF 占位**（§5.1/A.2 提到 Soft-Mask 基线但未挂引用），建议在 §5.1 首次出现处补 `\citep{yang2021soft}`。

---

## 三、REF 键 → 文献映射表（v2 全量）

| REF 键 | 文献 | 正文渲染 | BibTeX 键 | 状态 |
|---|---|---|---|---|
| `gin` | Xu et al., How Powerful are Graph Neural Networks? ICLR 2019 | (Xu et al., 2019) | `xu2019how` | ✅ |
| `graph-pooling` | Grattarola, Zambon, Bianchi, Alippi, Understanding Pooling in GNNs, TNNLS 35(2):2708–2718, 2024 | (Grattarola et al., 2024) | `grattarola2024understanding` | ✅（渲染年份由 2022 变 2024） |
| `diffpool` | Ying et al., Hierarchical Graph Representation Learning with Differentiable Pooling, NeurIPS 2018 | (Ying et al., 2018) | `ying2018hierarchical` | ✅ |
| `sagpool` | Lee et al., Self-Attention Graph Pooling, ICML 2019 | (Lee et al., 2019) | `lee2019self` | ✅ |
| `evidence-concentration` | Ying et al., GNNExplainer, NeurIPS 2019 | (Ying et al., 2019) | `ying2019gnnexplainer` | ✅ |
| `subgraphx` | Yuan, Yu, Wang, Li, Ji, On Explainability of GNNs via Subgraph Explorations, **ICML 2021** | (Yuan et al., 2021) | `yuan2021subgraphx` | ✅（年份/作者已纠错） |
| `graph-explainability` | Yuan, Yu, **Gui**, Ji, Explainability in GNNs: A Taxonomic Survey, TPAMI 45(5):5782–5799, 2023 | (Yuan et al., 2023) | `yuan2023explainability` | ✅ |
| `active-learning-graphs` | **推荐 GPA**：Hu, Xiong, Qu, Yuan, Côté, Liu, Tang, Graph Policy Network for Transferable Active Learning on Graphs, NeurIPS 2020 | (Hu et al., 2020) | `hu2020graph` | ⚠️ 已核实（在 Zotero 库，条目 VIZTNYFG）；如更想要"宽松查询+软标签"的 ICLR 系代表作可换 IGP/GALAS（见逐条说明 8b） |
| `feature-acquisition` | Li & Oliva, Active Feature Acquisition with Generative Surrogate Models, ICML 2021, PMLR 139:6450–6459 | (Li and Oliva, 2021) | `li2021active` | ✅ 候选已核实存在 |
| `gam` | Lee, Rossi, Kong, Graph Classification Using Structural Attention, KDD 2018, 1666–1674 | (Lee et al., 2018) | `lee2018graph` | ✅（含 DOI 10.1145/3219819.3219980） |
| `rwesgnn` | Huang & Kasai, Self-Interpretable Subgraph Neural Network with Deep Reinforcement Walk Exploration, AAAI 2026, 40(26):21984–21993 | (Huang and Kasai, 2026) | `huang2026self` | ✅（以 AAAI 版为正题，注意非 arXiv 旧题） |
| `agentnet` | **Martinkus, Papp, Schesch, Wattenhofer**, Agent-based Graph Neural Networks, ICLR 2023 | (Martinkus et al., 2023) | `martinkus2023agent` | ✅ 原始论文已定位 |
| `raw` | Akujuobi, Zhang, Han, Zhang, Recurrent Attention Walk for Semi-supervised Classification, WSDM 2020, 16–24 | (Akujuobi et al., 2020) | `akujuobi2020recurrent` | ✅（正式发表 WSDM '20，不再是 arXiv） |
| `ogbg` | Hu et al., Open Graph Benchmark, NeurIPS 2020 | (Hu et al., 2020) | `hu2020open` | ✅ |
| `lrgb` → 建议改键 `cifar10` | **CIFAR10 超像素实际出处**：Dwivedi, Joshi, Luu, Laurent, Bengio, Bresson, Benchmarking Graph Neural Networks, JMLR 24(43):1–48, 2023 | (Dwivedi et al., 2023) | `dwivedi2023benchmarking` | ⚠️ 需你确认换引（LRGB 本身不含 CIFAR10；若正文其他处确需 LRGB，其条目在第五节备查） |
| `tudataset` | Morris et al., TUDataset, ICML 2020 GRL+ Workshop | (Morris et al., 2020) | `morris2020tudataset` | ✅（COLLAB 原始构造出自 Yanardag & Vishwanathan, KDD 2015，如审稿人较真可补引，条目备查见第五节） |
| `soft-mask` | **Yang, Shen, Qi, Yin**, Soft-mask: Adaptive Substructure Extractions for Graph Neural Networks, WWW 2021, 2058–2068 | (Yang et al., 2021) | `yang2021soft` | ✅ 原始论文已定位（v1 只说"SAGPool 软加权版"，实为独立的 SMG 论文） |
| `teacher-details-SI` | 非外部引用：教师细节在自家附录，tex 应改为 `Appendix~\ref{app:impl}` 一类交叉引用 | — | — | ❓（替换方式待你点头） |

---

## 四、逐条说明与渲染预览

1. **`gin`** — K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In _International Conference on Learning Representations_, 2019.
2. **`graph-pooling`** — D. Grattarola, D. Zambon, F. M. Bianchi, and C. Alippi. Understanding pooling in graph neural networks. _IEEE Transactions on Neural Networks and Learning Systems_, 35(2):2708–2718, 2024.
   纠错：v1 作者表第二位 Livi 错误，正确为 Zambon（Crossref 权威记录）。注意渲染年份变为 2024。
3. **`diffpool`** — R. Ying, J. You, C. Morris, X. Ren, W. L. Hamilton, and J. Leskovec. Hierarchical graph representation learning with differentiable pooling. In _Advances in Neural Information Processing Systems_, 2018.
4. **`sagpool`** — J. Lee, I. Lee, and J. Kang. Self-attention graph pooling. In _International Conference on Machine Learning_, 2019.
5. **`evidence-concentration`** — R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec. GNNExplainer: Generating explanations for graph neural networks. In _Advances in Neural Information Processing Systems_, 2019.
6. **`subgraphx`** — H. Yuan, H. Yu, J. Wang, K. Li, and S. Ji. On explainability of graph neural networks via subgraph explorations. In _International Conference on Machine Learning_, 2021.
   纠错：年份 ICML 2021（非 2020）；正式作者 5 人（v1 的 "Yu Haitao" 与 "Gui" 均误，Gui 在 TPAMI 综述而非本文）。
7. **`graph-explainability`** — H. Yuan, H. Yu, S. Gui, and S. Ji. Explainability in graph neural networks: A taxonomic survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 45(5):5782–5799, 2023.
8. **`active-learning-graphs`（推荐）** — S. Hu, Z. Xiong, M. Qu, X. Yuan, M.-A. Côté, Z. Liu, and J. Tang. Graph policy network for transferable active learning on graphs. In _Advances in Neural Information Processing Systems_, 2020.
   选材理由：GPA 是"图上主动学习=预算买训练标签"的最常引代表作，且 RL 做法与本文叙事同源；条目已在你的 Zotero 库（VIZTNYFG）。
   8b. 备选 **IGP/GALAS** — W. Zhang, Y. Wang, Z. You, M. Cao, P. Huang, J. Shan, Z. Yang, and B. Cui. Information gain propagation: A new way to graph active learning with soft labels. In _International Conference on Learning Representations_, 2022.（也在 Zotero 库；引用方法名以 **IGP** 为准，GALAS 是 Apple 博客别称。）
9. **`gam`** — J. B. Lee, R. Rossi, and X. Kong. Graph classification using structural attention. In _Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 2018.
10. **`rwesgnn`** — J. Huang and H. Kasai. Self-interpretable subgraph neural network with deep reinforcement walk exploration. _Proceedings of the AAAI Conference on Artificial Intelligence_, 40(26):21984–21993, 2026.
    说明：AAAI-26 正式版改题（arXiv:2502.01809 旧题 Self-supervised...）；作者仅 Huang 与 Kasai 两人（早稻田大学）。Zotero/知识库里存的还是 arXiv 版元数据，建议同步更新。
11. **`agentnet`** — K. Martinkus, P. A. Papp, B. Schesch, and R. Wattenhofer. Agent-based graph neural networks. In _International Conference on Learning Representations_, 2023.
    说明：即正文所说 AgentNet（论文方法名），ETH Zurich 组，arXiv:2206.11010。
12. **`raw`** — U. Akujuobi, Q. Zhang, Y. Han, and X. Zhang. Recurrent attention walk for semi-supervised classification. In _Proceedings of the 13th International Conference on Web Search and Data Mining_（WSDM '20）, 2020, pp. 16–24。
    说明：正式发表于 WSDM 2020（DOI 10.1145/3336191.3371853），不再是 arXiv preprint；第三作者通行署名 Yufei Han。
13. **`ogbg`** — W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In _Advances in Neural Information Processing Systems_, 2020.
14. **`lrgb` → `cifar10`（换引待确认）** — V. P. Dwivedi, C. K. Joshi, A. T. Luu, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. _Journal of Machine Learning Research_, 24(43):1–48, 2023.
    说明：CIFAR10 超像素图出自该基准（jmlr.org/papers/v24/22-0567.html），LRGB 不含 CIFAR10。tex 中把该处 `\refadd{lrgb}` 换成新键即可；LRGB 原条目（Dwivedi, Rampášek, Galkin, Parviz, Wolf, Luu, Beaini, NeurIPS 2022 D&B）已核实并附在第五节备查。
15. **`tudataset`** — C. Morris, N. M. Kriege, F. Bause, K. Kersting, and P. Mutzel. TUDataset: A collection of benchmark datasets for learning with graphs. In _ICML Workshop on Graph Representation Learning and Beyond (GRL+)_, 2020.
    备查：COLLAB 数据集原始出处为 B. Yanardag & S. V. N. Vishwanathan, Deep graph kernels, KDD 2015（如需可补引）。
16. **`soft-mask`** — M. Yang, Y. Shen, H. Qi, and B. Yin. Soft-mask: Adaptive substructure extractions for graph neural networks. In _Proceedings of the Web Conference 2021_, 2021, 2058–2068.
    说明：即你适配的 Soft-Mask（SMG 层：逐层连续掩码调制消息传递）原始论文，大连理工组，DOI 10.1145/3442381.3449929；arXiv:2206.05499 为 2022 年补挂的 post-print。正文 v3 尚未给该基线挂 REF 占位，建议在 §5.1 首次出现处补引。
17. **`feature-acquisition`** — Y. Li and J. B. Oliva. Active feature acquisition with generative surrogate models. In _International Conference on Machine Learning_, 2021.
    已核实存在（PMLR 139:6450–6459；arXiv:2010.02433）。AGR 索引中另有 AFA 综述（#2 Aronsson et al., arXiv:2502.11067）可作替换备选。

---

## 五、待你确认的三件事（v2 后仅剩）

1. **CIFAR10 换引**：正文 §5.1 的 `【REF：lrgb】` 改指 Benchmarking GNNs（JMLR 2023），REF 键建议同步改名为 `cifar10` 或 `benchmark-gnns`。若你实验实际用的是 LRGB 的 VOC/COCO 超像素（而非 CIFAR10），告诉我，引用回退为 LRGB 条目。
2. **`active-learning-graphs` 选型**：默认 GPA（Hu et al., NeurIPS 2020，已在 Zotero）；想换 ICLR 出身的 IGP/GALAS（Zhang et al., ICLR 2022，也在 Zotero）就说一声。
3. **`teacher-details-SI`**：建议把 tex 里这一处 `\refadd` 改成对附录 A.0/A.2 的交叉引用（不是文献），你确认后我一并替换。

---

## 六、references.bib 全文（v2 定稿草稿）

```bibtex
@inproceedings{xu2019how,
  title     = {How Powerful are Graph Neural Networks?},
  author    = {Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  booktitle = {International Conference on Learning Representations},
  year      = {2019}
}

@article{grattarola2024understanding,
  title   = {Understanding Pooling in Graph Neural Networks},
  author  = {Grattarola, Daniele and Zambon, Daniele and Bianchi, Filippo Maria and Alippi, Cesare},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {35},
  number  = {2},
  pages   = {2708--2718},
  year    = {2024}
}

@inproceedings{ying2018hierarchical,
  title     = {Hierarchical Graph Representation Learning with Differentiable Pooling},
  author    = {Ying, Rex and You, Jiaxuan and Morris, Christopher and Ren, Xiang and Hamilton, William L. and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2018}
}

@inproceedings{lee2019self,
  title     = {Self-Attention Graph Pooling},
  author    = {Lee, Junhyun and Lee, Inyeop and Kang, Jaewoo},
  booktitle = {International Conference on Machine Learning},
  year      = {2019}
}

@inproceedings{ying2019gnnexplainer,
  title     = {{GNNExplainer}: Generating Explanations for Graph Neural Networks},
  author    = {Ying, Rex and Bourgeois, Dylan and You, Jiaxuan and Zitnik, Marinka and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}

@inproceedings{yuan2021subgraphx,
  title     = {On Explainability of Graph Neural Networks via Subgraph Explorations},
  author    = {Yuan, Hao and Yu, Haiyang and Wang, Jie and Li, Kang and Ji, Shuiwang},
  booktitle = {International Conference on Machine Learning},
  year      = {2021}
}

@article{yuan2023explainability,
  title   = {Explainability in Graph Neural Networks: A Taxonomic Survey},
  author  = {Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {45},
  number  = {5},
  pages   = {5782--5799},
  year    = {2023}
}

@inproceedings{hu2020graph,
  title     = {Graph Policy Network for Transferable Active Learning on Graphs},
  author    = {Hu, Shengding and Xiong, Zheng and Qu, Meng and Yuan, Xingdi and C{\^o}t{\'e}, Marc-Alexandre and Liu, Zhiyuan and Tang, Jian},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}

@inproceedings{lee2018graph,
  title     = {Graph Classification Using Structural Attention},
  author    = {Lee, John Boaz and Rossi, Ryan and Kong, Xiangnan},
  booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {1666--1674},
  year      = {2018}
}

@inproceedings{huang2026self,
  title     = {Self-Interpretable Subgraph Neural Network with Deep Reinforcement Walk Exploration},
  author    = {Huang, Jianming and Kasai, Hiroyuki},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {26},
  pages     = {21984--21993},
  year      = {2026}
}

@inproceedings{martinkus2023agent,
  title     = {Agent-based Graph Neural Networks},
  author    = {Martinkus, Karolis and Papp, P{\'a}l Andr{\'a}s and Schesch, Benedikt and Wattenhofer, Roger},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{akujuobi2020recurrent,
  title     = {Recurrent Attention Walk for Semi-supervised Classification},
  author    = {Akujuobi, Uchenna and Zhang, Qiannan and Han, Yufei and Zhang, Xiangliang},
  booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages     = {16--24},
  year      = {2020}
}

@inproceedings{hu2020open,
  title     = {Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author    = {Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}

@article{dwivedi2023benchmarking,
  title   = {Benchmarking Graph Neural Networks},
  author  = {Dwivedi, Vijay Prakash and Joshi, Chaitanya K. and Luu, Anh Tuan and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  journal = {Journal of Machine Learning Research},
  volume  = {24},
  number  = {43},
  pages   = {1--48},
  year    = {2023}
}

@inproceedings{morris2020tudataset,
  title     = {{TUDataset}: A Collection of Benchmark Datasets for Learning with Graphs},
  author    = {Morris, Christopher and Kriege, Nils M. and Bause, Franka and Kersting, Kristian and Mutzel, Petra},
  booktitle = {ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+)},
  year      = {2020}
}

@inproceedings{yang2021soft,
  title     = {Soft-mask: Adaptive Substructure Extractions for Graph Neural Networks},
  author    = {Yang, Mingqi and Shen, Yanming and Qi, Heng and Yin, Baocai},
  booktitle = {Proceedings of the Web Conference 2021},
  pages     = {2058--2068},
  year      = {2021}
}

@inproceedings{li2021active,
  title     = {Active Feature Acquisition with Generative Surrogate Models},
  author    = {Li, Yang and Oliva, Junier B.},
  booktitle = {International Conference on Machine Learning},
  year      = {2021}
}

% ===== 备查（正文暂不引用，按需启用）=====

@inproceedings{dwivedi2022long,
  title     = {A Long Range Graph Benchmark},
  author    = {Dwivedi, Vijay Prakash and Ramp{\'a}{\v s}ek, Ladislav and Galkin, Mikhail and Parviz, Ali and Wolf, Guy and Luu, Anh Tuan and Beaini, Dominique},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2022}
}

@inproceedings{yanardag2015deep,
  title     = {Deep Graph Kernels},
  author    = {Yanardag, Pinar and Vishwanathan, S. V. N.},
  booktitle = {Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {1365--1374},
  year      = {2015}
}
```

---

## 检索来源（v2 核实记录，2026-09-21）

- 知识库（Graph Learning vault）：AgentNet / SoftMask / RecurrentAttentionWalk / RWE-SGNN / SubgraphX / SAGPool / GALAS / GCBR 论文页（含 Zotero 条目键）
- Zotero「图学习」库（MCP get_item_details）：3A62ABME（AgentNet）、4RFV7QHD（SoftMask）、RG9TDCAC（RAW）、67J6SV98（RWE-SGNN arXiv 版）、4R2LCAIZ（IGP）、VIZTNYFG（GPA）
- arXiv API：1910.10266、2502.01809、2206.05499、2012.15445、2102.05152、2003.00982
- Crossref：10.1109/TNNLS.2022.3190922、10.1109/TPAMI.2022.3204236、10.1145/3442381.3449929、10.1145/3219819.3219980、10.1609/aaai.v40i26.39352、10.1145/3336191.3371853
- OpenAlex：TPAMI/TNNLS/JMLR 卷期交叉验证、Li & Oliva 页码 6450–6459
- AAAI OJS：RWE-SGNN AAAI-26 正式版（Vol. 40 No. 26, 21984–21993）
- JMLR：Benchmarking GNNs = 24(43):1–48, 2023

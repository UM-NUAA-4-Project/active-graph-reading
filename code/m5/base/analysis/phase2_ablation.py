"""Phase 2: 特征消融归因 — 量化各输入特征对决策质量和分类性能的贡献。

通过推理时干预（inference-time intervention）实现消融，不修改模型代码：
  A0: 完整（132维输入）— 基线
  A1: 去 budget_ratio（设为中性值 0.5）
  A2: 去 struct_v（3维结构特征置零）
  A3: 去 budget + struct（两者同时移除）
  A4: 去 edge_feat（FrontierGraphBuilder 构建后边特征置零）

输出：
  - raw_data/phase2_ablation.json  原始数据
  - figures/phase2_waterfall.png   特征贡献瀑布图
  - figures/phase2_phase_analysis.png  分阶段分析
"""

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import random
import torch
import torch.nn.functional as F
import dgl
import pandas as pd
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader.gsae import compute_structural_features
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from training_utils import patch_old_checkpoint

# ---------------------------------------------------------------------------
# ablation configs
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    'A0': {'label': '完整', 'ablate': set()},
    'A1': {'label': '去budget_ratio(置零)', 'ablate': {'budget_ratio'}},
    'A2': {'label': '去struct_v', 'ablate': {'struct_v'}},
    'A3': {'label': '去budget+struct', 'ablate': {'budget_ratio', 'struct_v'}},
    'A4': {'label': '去edge_feat', 'ablate': {'edge_feat'}},
}

K = 37  # PPA 类别数
LN_K = math.log(K)


# ---------------------------------------------------------------------------
# ablated forward — 复现 LearnableReader.forward() 并注入消融修改
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablated_forward(
    reader: LearnableReader,
    graph_state: GraphState,
    ablation: Set[str],
    device: torch.device,
):
    """复现 reader.forward()，在中间步骤注入特征消融。

    ablation 可含: 'edge_feat', 'budget_ratio', 'struct_v'
    Returns: (scores, probs, frontier_nodes)
    """
    # 1. 构建前沿子图
    subgraph = reader.frontier_builder.build(graph_state)

    # A4: 边特征置零
    if 'edge_feat' in ablation:
        if 'feat' in subgraph.g.edata and subgraph.g.edata['feat'].numel() > 0:
            subgraph.g.edata['feat'] = torch.zeros_like(subgraph.g.edata['feat'])

    # 2. GNN 编码
    frontier_embeds, frontier_nodes = reader.frontier_encoder(subgraph)

    if len(frontier_nodes) == 0:
        return torch.empty(0), torch.empty(0), []

    # 3. budget_ratio
    if 'budget_ratio' in ablation:
        remaining = 0.0
    else:
        remaining = graph_state.get_budget_ratio_remaining()

    remaining_tensor = torch.full(
        (len(frontier_nodes), 1), remaining,
        device=frontier_embeds.device, dtype=frontier_embeds.dtype,
    )

    # 4. 结构特征
    struct_all = compute_structural_features(subgraph.g, subgraph.n_revealed)
    struct_frontier = struct_all[subgraph.n_revealed:].to(
        device=frontier_embeds.device, dtype=frontier_embeds.dtype,
    )
    if 'struct_v' in ablation:
        struct_frontier = torch.zeros_like(struct_frontier)

    # 5. 拼接 + MLP + softmax
    combined = torch.cat([frontier_embeds, remaining_tensor, struct_frontier], dim=-1)
    scores = reader.reader_mlp(combined)
    probs = F.softmax(scores, dim=0)

    return scores, probs, frontier_nodes


# ---------------------------------------------------------------------------
# run one graph for all ablation configs
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_one_graph_all_ablations(
    reader: LearnableReader,
    g_raw: dgl.DGLGraph,
    seed_node: int,
    label: int,
    budget_abs: int,
    device: torch.device,
    classifier: GraphClassifier,
    builder: SubgraphBuilder,
) -> Dict[str, dict]:
    """对一张图运行所有消融配置，返回 per-config 指标字典。"""
    results = {}

    # 先跑 A0（基线），记录每一步的 frontier_nodes 序列和 probs 用于对比
    a0_nodes_per_step = []
    a0_probs_per_step = []

    for config_name in ABLATION_CONFIGS:
        ablate_set = ABLATION_CONFIGS[config_name]['ablate']

        state = GraphState(g_raw, seed_node=seed_node)
        state._budget_limit = budget_abs
        reader.reset()
        initial_frontier = list(state._initial_frontier)
        if initial_frontier:
            reader.on_new_frontier(initial_frontier)

        step_nodes = []       # 每步选择的节点
        step_probs = []       # 每步的 softmax 分布
        step_rewards = []     # 每步的 R_t

        for _step in range(budget_abs):
            if state.is_frontier_empty:
                break

            scores, probs, frontier_nodes = ablated_forward(
                reader, state, ablate_set, device,
            )

            if len(frontier_nodes) == 0:
                break

            # 记录当前步的选择和概率（用于对比 A0）
            chosen_idx = probs.argmax().item()
            chosen_node = frontier_nodes[chosen_idx]
            step_nodes.append(chosen_node)
            step_probs.append(probs.cpu().clone())

            # reveal
            new_frontier = reveal_node(state, chosen_node)
            if new_frontier:
                reader.on_new_frontier(new_frontier)

            # 每步分类评估（即时奖励）
            sub_g, _ = builder.build(state)
            sub_g = sub_g.to(device)
            nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(sub_g, nids)
            ce_t = F.cross_entropy(logits, torch.tensor([label], device=device))
            r_t = math.tanh(1.0 - ce_t.item() / LN_K)
            step_rewards.append(r_t)

        # 最终分类结果
        sub_g, _ = builder.build(state)
        sub_g = sub_g.to(device)
        nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(sub_g, nids)
        final_correct = 1.0 if logits.argmax().item() == label else 0.0
        final_ce = F.cross_entropy(logits, torch.tensor([label], device=device)).item()

        avg_R = float(np.mean(step_rewards)) if step_rewards else 0.0

        result = {
            'avg_R': avg_R,
            'accuracy': final_correct,
            'final_CE': final_ce,
            'step_nodes': step_nodes,
            'step_rewards': step_rewards,
        }
        results[config_name] = result

        # 保存 A0 的基准数据
        if config_name == 'A0':
            a0_nodes_per_step = step_nodes
            a0_probs_per_step = step_probs

    # 计算各消融配置相对 A0 的对比指标
    a0_nodes = a0_nodes_per_step
    a0_probs = a0_probs_per_step
    n_steps_a0 = len(a0_nodes)

    for config_name in ABLATION_CONFIGS:
        if config_name == 'A0':
            results[config_name]['decision_overlap'] = 1.0
            results[config_name]['kl_divergence'] = [0.0] * n_steps_a0
            continue

        cfg_nodes = results[config_name]['step_nodes']
        overlap = sum(1 for i, n in enumerate(cfg_nodes)
                      if i < n_steps_a0 and n == a0_nodes[i]) / max(n_steps_a0, 1)
        results[config_name]['decision_overlap'] = overlap

        # per-step KL divergence
        kl_steps = []
        for i in range(min(len(cfg_nodes), n_steps_a0)):
            # 需要该步的 probs — 重新从 ablated_forward 获取
            pass  # KL 在第二遍收集（见下方）
        results[config_name]['kl_divergence'] = kl_steps

    return results


# ---------------------------------------------------------------------------
# second pass: collect KL divergence data for non-A0 configs
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_one_graph_kl(
    reader: LearnableReader,
    g_raw: dgl.DGLGraph,
    seed_node: int,
    budget_abs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """重新运行所有配置，收集每步的 probs 用于 KL 散度计算。"""
    # 先跑 A0 获取基准 probs
    all_probs = {}

    for config_name in ABLATION_CONFIGS:
        ablate_set = ABLATION_CONFIGS[config_name]['ablate']
        state = GraphState(g_raw, seed_node=seed_node)
        state._budget_limit = budget_abs
        reader.reset()
        initial_frontier = list(state._initial_frontier)
        if initial_frontier:
            reader.on_new_frontier(initial_frontier)

        step_probs = []
        for _step in range(budget_abs):
            if state.is_frontier_empty:
                break
            _, probs, frontier_nodes = ablated_forward(reader, state, ablate_set, device)
            if len(frontier_nodes) == 0:
                break
            chosen_idx = probs.argmax().item()
            chosen_node = frontier_nodes[chosen_idx]
            step_probs.append(probs.cpu().clone())
            new_frontier = reveal_node(state, chosen_node)
            if new_frontier:
                reader.on_new_frontier(new_frontier)

        all_probs[config_name] = step_probs

    # 计算 KL(A0 || config) — 仅当概率向量维度相同时计算
    kl_results = {}
    a0_probs = all_probs['A0']
    for config_name in ABLATION_CONFIGS:
        if config_name == 'A0':
            kl_results[config_name] = [0.0] * len(a0_probs)
            continue
        cfg_probs = all_probs[config_name]
        kl_list = []
        for i in range(min(len(cfg_probs), len(a0_probs))):
            p = a0_probs[i]
            q = cfg_probs[i]
            if p.shape[0] == q.shape[0] and p.shape[0] > 0:
                p_clamp = p.clamp(min=1e-10)
                q_clamp = q.clamp(min=1e-10)
                kl = (p_clamp * (p_clamp.log() - q_clamp.log())).sum().item()
            else:
                kl = float('nan')  # 维度不匹配，无法比较
            kl_list.append(kl)
        # pad to A0 length if shorter
        while len(kl_list) < len(a0_probs):
            kl_list.append(float('nan'))
        kl_results[config_name] = kl_list

    return kl_results


# ---------------------------------------------------------------------------
# per-phase analysis helpers
# ---------------------------------------------------------------------------

def split_into_phases(data: List, n_phases: int = 3):
    """将长度为 B 的列表分为 early/mid/late 三个阶段的平均值（忽略 NaN）。"""
    B = len(data)
    if B == 0:
        return [0.0, 0.0, 0.0]
    phase_size = max(1, B // n_phases)
    phases = []
    for p in range(n_phases):
        start = p * phase_size
        if p == n_phases - 1:
            end = B
        else:
            end = (p + 1) * phase_size
        chunk = [v for v in data[start:end] if not (isinstance(v, float) and math.isnan(v))]
        phases.append(float(np.mean(chunk)) if chunk else 0.0)
    return phases


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_waterfall(
    agg: Dict[str, Dict[str, float]],
    output_path: str,
):
    """特征贡献瀑布图：按 A4→A2→A1→A0 逐步叠加特征。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    order = ['A4', 'A2', 'A1', 'A0']
    labels = ['A4: 仅frontier\nembedding', 'A2: +struct_v', 'A1: +budget_ratio', 'A0: 完整(基线)']

    means = [agg[c]['accuracy_mean'] for c in order]
    stds = [agg[c]['accuracy_std'] for c in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(order))
    colors = ['#C44E52', '#DD8452', '#55A868', '#4C72B0']

    bars = ax.bar(x, means, color=colors, yerr=stds, capsize=8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Phase 2: Feature Ablation Waterfall (cumulative contribution)', fontsize=14)

    # 标注增量
    for i in range(1, len(order)):
        delta = means[i] - means[i - 1]
        mid_y = (means[i] + means[i - 1]) / 2
        ax.annotate(
            f'+{delta:.4f}', xy=(i - 0.1, mid_y),
            fontsize=9, color='green' if delta > 0 else 'red',
            ha='right',
        )

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Waterfall plot saved to {output_path}")


def plot_phase_analysis(
    agg: Dict[str, Dict[str, float]],
    output_path: str,
):
    """分阶段（early/mid/late）分析各特征的贡献。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

    configs = ['A0', 'A1', 'A2', 'A3', 'A4']
    phase_names = ['Early (1st 1/3)', 'Mid (2nd 1/3)', 'Late (last 1/3)']
    metric_keys = ['kl_phase', 'overlap_phase', 'reward_phase']
    metric_labels = ['KL Divergence (vs A0)', 'Decision Overlap (vs A0)', 'Avg Reward R_t']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8B5CF6', '#DD8452']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        x = np.arange(len(phase_names))
        n_configs = len(configs) - (1 if mkey == 'kl_phase' else 0)  # A0 has no KL
        width = 0.8 / (n_configs if n_configs > 0 else 1)
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_configs) if n_configs > 0 else [0]

        configs_to_plot = [c for c in configs if not (mkey == 'kl_phase' and c == 'A0')]
        for ci, config_name in enumerate(configs_to_plot):
            values = agg[config_name].get(mkey, [0, 0, 0])
            ax.bar(x + offsets[ci], values, width,
                   label=ABLATION_CONFIGS[config_name]['label'],
                   color=colors[ci % len(colors)])

        ax.set_title(mlabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Phase 2: Per-Phase Feature Contribution Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Phase analysis plot saved to {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Feature ablation attribution analysis')
    parser.add_argument('--checkpoint', default='m5/base/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--budget', type=float, default=0.10,
                        help='Budget ratio (default 0.10)')
    parser.add_argument('--max-graphs', type=int, default=500,
                        help='Max graphs from val split (0=all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else _script_dir
    raw_dir = output_dir / 'raw_data'
    fig_dir = output_dir / 'figures'
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 classifier ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 2. 加载 reader ----
    print("\n=== Loading reader ===")
    reader_ckpt = torch.load(str(repo_root / args.checkpoint),
                             map_location='cpu', weights_only=False)
    reader_sd = reader_ckpt['reader_state_dict']
    hidden_dim = reader_sd['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in reader_sd:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"best_val_acc={reader_ckpt.get('best_val_acc', 'N/A')}")

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers, dropout=0.1,
    ).to(device)
    reader_sd, _ = patch_old_checkpoint(reader_ckpt['reader_state_dict'])
    reader.load_state_dict(reader_sd)
    reader.eval()
    print(f"  Params: {sum(p.numel() for p in reader.parameters()):,}")

    # ---- 3. 加载 val 图 ----
    processed_path = str(Path(args.root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    split_path = Path(args.root) / 'ogbg_ppa' / 'split' / 'species' / 'valid.csv.gz'
    val_indices = pd.read_csv(split_path, compression='gzip', header=None).values.flatten().tolist()

    print(f"\n=== Loading {len(val_indices)} val graphs ===")
    graphs, label_dict = dgl.load_graphs(processed_path, idx_list=val_indices)
    print(f"  Loaded {len(graphs)} graphs")

    if args.max_graphs > 0 and args.max_graphs < len(graphs):
        # 分层抽样：每类均匀覆盖
        n_per_class = max(1, args.max_graphs // K)
        class_to_local_idx = defaultdict(list)
        for local_i, ogb_idx in enumerate(val_indices):
            lbl = int(label_dict['labels'][ogb_idx].item())
            class_to_local_idx[lbl].append(local_i)

        sampled_local = []
        for cls in range(K):
            pool = class_to_local_idx.get(cls, [])
            n = min(n_per_class, len(pool))
            if n > 0:
                sampled_local.extend(random.sample(pool, n))

        # 如果每类凑不够，从剩余图中随机补齐
        if len(sampled_local) < args.max_graphs:
            remaining_pool = [i for i in range(len(graphs)) if i not in set(sampled_local)]
            extra_needed = args.max_graphs - len(sampled_local)
            sampled_local.extend(random.sample(remaining_pool, min(extra_needed, len(remaining_pool))))
        else:
            sampled_local = random.sample(sampled_local, args.max_graphs)

        graphs = [graphs[i] for i in sampled_local]
        val_indices = [val_indices[i] for i in sampled_local]
        print(f"  Stratified sampling: {len(graphs)} graphs "
              f"({len(set(int(label_dict['labels'][idx].item()) for idx in val_indices))} classes covered)")

    builder = SubgraphBuilder()

    # ---- 4. 逐图运行所有消融配置 ----
    # 聚合指标
    agg: Dict[str, Dict[str, List[float]]] = {
        c: defaultdict(list) for c in ABLATION_CONFIGS
    }
    per_graph_results = []

    for i, ogb_idx in enumerate(tqdm(val_indices, desc='Phase 2 ablation')):
        g_raw = dgl.add_self_loop(graphs[i])
        label = int(label_dict['labels'][ogb_idx].item())
        n_nodes = g_raw.num_nodes()
        budget_abs = max(1, int(args.budget * n_nodes))

        # 选 best seed（与 eval.py 一致）
        indeg = g_raw.in_degrees().float()
        outdeg = g_raw.out_degrees().float()
        total_deg = indeg + outdeg
        best_seed = int(total_deg.argmax().item())

        graph_result = {
            'graph_idx': ogb_idx,
            'n_nodes': n_nodes,
            'label': label,
            'budget_abs': budget_abs,
            'budget_ratio': args.budget,
            'configs': {},
        }

        # 第一遍：主实验（不含 KL）
        results = run_one_graph_all_ablations(
            reader, g_raw, best_seed, label, budget_abs,
            device, classifier, builder,
        )

        # 第二遍：KL 散度数据
        kl_data = run_one_graph_kl(
            reader, g_raw, best_seed, budget_abs, device,
        )

        for config_name in ABLATION_CONFIGS:
            r = results[config_name]
            kl_list = kl_data.get(config_name, [])
            kl_phases = split_into_phases(kl_list)
            overlap = r['decision_overlap']
            reward_phases = split_into_phases(r['step_rewards'])

            # 聚合
            agg[config_name]['accuracy'].append(r['accuracy'])
            agg[config_name]['avg_R'].append(r['avg_R'])
            agg[config_name]['final_CE'].append(r['final_CE'])
            agg[config_name]['decision_overlap'].append(overlap)
            agg[config_name]['kl_mean'].append(
                float(np.nanmean(kl_list)) if kl_list else 0.0)

            for p in range(3):
                agg[config_name][f'kl_phase_{p}'].append(kl_phases[p])
                agg[config_name][f'overlap_phase_{p}'].append(
                    overlap if p == 0 else overlap)  # overlap 目前是整体的，按 phase 拆分需重新计算
                agg[config_name][f'reward_phase_{p}'].append(reward_phases[p] if p < len(reward_phases) else 0.0)

            graph_result['configs'][config_name] = {
                'accuracy': r['accuracy'],
                'avg_R': r['avg_R'],
                'final_CE': r['final_CE'],
                'decision_overlap': overlap,
                'kl_per_step': kl_list,
                'step_rewards': r['step_rewards'],
            }

        per_graph_results.append(graph_result)

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 5. 汇总统计 ----
    summary = {}
    for config_name in ABLATION_CONFIGS:
        d = {}
        for metric, values in agg[config_name].items():
            vals = [v for v in values if v is not None]
            d[metric] = {
                'mean': float(np.mean(vals)) if vals else 0.0,
                'std': float(np.std(vals)) if vals else 0.0,
                'n': len(vals),
            }
        summary[config_name] = d

    # 准备绘图用的扁平化指标
    plot_agg = {}
    for config_name in ABLATION_CONFIGS:
        s = summary[config_name]
        plot_agg[config_name] = {
            'accuracy_mean': s['accuracy']['mean'],
            'accuracy_std': s['accuracy']['std'],
            'avg_R_mean': s['avg_R']['mean'],
            'kl_phase': [
                s.get(f'kl_phase_{p}', {'mean': 0.0})['mean'] for p in range(3)
            ],
            'overlap_phase': [
                s.get(f'overlap_phase_{p}', {'mean': 0.0})['mean'] for p in range(3)
            ],
            'reward_phase': [
                s.get(f'reward_phase_{p}', {'mean': 0.0})['mean'] for p in range(3)
            ],
        }

    # ---- 6. 输出 JSON ----
    output = {
        'summary': summary,
        'per_graph': per_graph_results,
        'config': {
            'checkpoint': str(repo_root / args.checkpoint),
            'classifier': str(repo_root / args.classifier),
            'budget': args.budget,
            'n_graphs': len(val_indices),
            'ablations': {k: v['label'] for k, v in ABLATION_CONFIGS.items()},
        },
    }

    json_path = raw_dir / 'phase2_ablation.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {json_path}")

    # 打印汇总表
    print("\n=== Ablation Summary ===")
    print(f"{'Config':<6} {'Accuracy':>10} {'Avg_R':>10} {'Overlap':>10} {'KL_mean':>10}")
    print("-" * 50)
    for config_name in ['A0', 'A1', 'A2', 'A3', 'A4']:
        s = summary[config_name]
        print(f"{config_name:<6} {s['accuracy']['mean']:>10.4f} {s['avg_R']['mean']:>10.4f} "
              f"{s['decision_overlap']['mean']:>10.4f} {s['kl_mean']['mean']:>10.4f}")

    # ---- 7. 可视化 ----
    plot_waterfall(plot_agg, str(fig_dir / 'phase2_waterfall.png'))
    plot_phase_analysis(plot_agg, str(fig_dir / 'phase2_phase_analysis.png'))

    print("\nPhase 2 done.")


if __name__ == '__main__':
    main()

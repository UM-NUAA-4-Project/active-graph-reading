"""冻结 evidence reader + state-attention 聚合头（离线训练，参照口径）。

设计（2026-09-20 讨论，docs/decisions.md 同日条目）：
- 冻结已训 evidence reader，在其 rollout 上采集逐步证据状态
  x_t = [z_t ; h_t] ∈ R^{2d}（z_t = 历史引导池化读出，h_t = LSTM 积分器
  隐状态；经 forward_states_evidence(return_states=True) 只读获取），
  序列 = rollout 各步"揭示前"状态 + 终局补充前向状态（最后一位）。
- 聚合头：masked attention 池化——x_t ⊕ t/T（归一化时间步标量）→
  LayerNorm → 与零初始化可学习查询打分（初始 ≡ 均匀权重 = mean-pool
  下限）→ mask softmax（pad 位 -inf）→ 加权和 s → MLP → 类别。
  不拼接终局状态（终局状态已在序列内，attention 可自行加权）。
- 三口径对照（同一批 rollout）：learned（学 attention 权重）vs uniform
  （同 MLP、权重恒均匀——"学特化是否优于均匀"的直接对照）vs terminal-
  only（冻结 reader 自身终判，参照下限）。模型选择用 train 内 carve 的
  val（15%），eval split 只做最终报告（无选择泄漏）。
- 协议合法性：全部输入来自已揭示区表征 + 可观测标量（t/T），无前沿
  节点信息。

采集为跨 episode lockstep 批量前向（eval_loop 同款，--collect-batch 控制
块大小；逐 state 独立维护 hc/teleport，与串行版轨迹逐状态等价）。特征缓
存 --cache 落盘后消融变体可复用，无需重采集。

用法（cifar10 冒烟）:
  python code/train_reader/train_state_attention.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch3.pt \
      --max-graphs 300 --n-seeds 2 --epochs 40
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import (  # noqa: E402
    reveal_node, teleport_if_stuck)
from datasets import get_dataset, release_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import deterministic_seed, set_all_seeds  # noqa: E402
from train_reader.core.build import load_reader_from_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='state-attention 聚合头（冻结 evidence reader）')
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True, help='evidence reader checkpoint')
    p.add_argument('--splits', nargs='+', default=['train', 'test'],
                   help='采集（split[0]）与评测（其余）split')
    p.add_argument('--max-graphs', type=int, default=300, help='每 split 图数上限')
    p.add_argument('--stratified', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--n-seeds', type=int, default=2)
    p.add_argument('--budget-ratios', nargs='+', type=float, default=None,
                   help='缺省取 spec.budget_options；逐 (graph, seed) 均匀采样')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=777)
    # 头结构
    p.add_argument('--d-attn', type=int, default=128)
    p.add_argument('--mlp-hidden', type=int, default=128)
    # 训练
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='稠密梯度小头无坍缩风险，默认 0（agentnet 教训：wd 只该'
                        '用在输入路径梯度可能饥饿的端到端训练里）')
    p.add_argument('--cache', default='',
                   help='采集特征缓存 .pt（存在则跳过采集直接加载；消融复用）')
    p.add_argument('--collect-batch', type=int, default=256,
                   help='lockstep 采集的每批 state 数（显存不足可调小）')
    p.add_argument('--out', default='',
                   help='输出目录（默认 code/result/{ds}/experiments/state-attn）')
    return p


# ----------------------------------------------------------------------
# 采集（lockstep 批量：跨 episode 块对角批前向，eval_loop 同款）
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_split(reader, graphs, labels, budget_ratios, n_seeds, device,
                  split_name, graph_offset=0, collect_batch=256):
    """一个 split 的全部 episodes。graph_offset 保证两个 split 的种子公式
    图索引不重叠。

    逐 state 独立维护 hc / last_revealed / teleport rng（(seed_node, budget)
    播种，eval_loop 同式）；每步对全部 active state 一次批量前向，同一次
    结果既取 (z_t, h_t) 又 argmax 选点。与逐 state 串行版**预算/步数/hc
    语义等价但轨迹不逐位一致**：batch 形状不同的 GEMM 浮点噪声（~1e-7）
    会在 argmax 近平手处翻转选点并放大为轨迹分叉（cifar10 实测 120 episode
    中 30 条分叉、max|dX|~3.4，T 恒一致）。本 lockstep 版为**规范采集器**，
    固定批组成下 run-to-run 逐位可复现（实测 T/X/terminal 全零差异）。
    预算哨兵每轮全查（rollout_strategy 复合条件），预算耗尽的 state 停止、
    frontier 空且无法瞬移的 state 提前结束。循环后全量补一次终局前向（含
    最后一次 reveal 的证据）追加为序列末位。
    """
    states, metas, rngs = [], [], []
    for gi, (g, lbl) in enumerate(tqdm(zip(graphs, labels), total=len(graphs),
                                       desc=f'  [{split_name}] states',
                                       leave=False)):
        n = g.num_nodes()
        for seed_i in range(n_seeds):
            seed_node = deterministic_seed(graph_offset + gi, seed_i, n)
            ratio = random.choice(budget_ratios)
            budget = max(1, int(ratio * n))
            st = GraphState(g, seed_node=seed_node)
            st._budget_limit = budget
            states.append(st)
            rngs.append(random.Random(seed_node * 1_000_003 + budget))
            metas.append({'label': int(lbl.item()), 'budget': ratio, 'n': n,
                          'split': split_name, 'graph_idx': graph_offset + gi,
                          'seed_i': seed_i})

    n_total = len(states)
    all_hc = reader.init_evidence(n_total, device)
    last_revealed = [s.seed_node for s in states]
    # 逐步证据行缓冲：每 state 预分配 (budget+1, 2d)，行号 ev_len 追加。
    # 全量采集（~200k episode × ~13 步）若用逐张 append 会产生 ~2.6M 个小
    # tensor（对象开销 + 碎片），叠加上文 states 全程存活直接打爆内存。
    state_dim = 2 * reader.config.hidden_dim
    ev_buf = [torch.empty(s._budget_limit + 1, state_dim) for s in states]
    ev_len = [0] * n_total
    active = [True] * n_total
    max_budget = max(s._budget_limit for s in states)

    step_bar = tqdm(range(max_budget), desc=f'  [{split_name}] lockstep',
                    leave=False)
    for _step in step_bar:
        for j in range(n_total):
            if active[j] and states[j].is_frontier_empty:
                tp = teleport_if_stuck(states[j], rng=rngs[j])
                if tp is not None:
                    last_revealed[j] = tp
                    if (states[j].get_budget_used()
                            >= states[j]._budget_limit):
                        active[j] = False
                else:
                    active[j] = False
        active_idx = [j for j in range(n_total)
                      if active[j] and not states[j].is_frontier_empty]
        if not active_idx:
            break
        step_bar.set_postfix(active=len(active_idx), done=n_total - len(active_idx))
        for c0 in range(0, len(active_idx), collect_batch):
            chunk = active_idx[c0:c0 + collect_batch]
            results, new_hc, _, step_states = reader.forward_states_evidence(
                [states[j] for j in chunk], temperature=1.0,
                hc_list=[all_hc[j] for j in chunk],
                last_revealed=[last_revealed[j] for j in chunk],
                return_states=True)
            for k, j in enumerate(chunk):
                all_hc[j] = new_hc[k]
                z_t, h_t = step_states[k]
                ev_buf[j][ev_len[j]] = torch.cat([z_t, h_t]).float().cpu()
                ev_len[j] += 1
                _, probs, frontier_nodes = results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                chosen = frontier_nodes[int(probs.argmax().item())]
                new_frontier = reveal_node(states[j], chosen)
                if new_frontier:
                    last_revealed[j] = chosen
                if states[j].get_budget_used() >= states[j]._budget_limit:
                    active[j] = False
    step_bar.close()

    episodes = []
    for c0 in tqdm(range(0, n_total, collect_batch),
                   desc=f'  [{split_name}] terminal', leave=False):
        chunk = list(range(c0, min(c0 + collect_batch, n_total)))
        _, _, ev_term, step_states = reader.forward_states_evidence(
            [states[j] for j in chunk], temperature=1.0,
            hc_list=[all_hc[j] for j in chunk],
            last_revealed=[last_revealed[j] for j in chunk],
            return_states=True)
        for k, j in enumerate(chunk):
            z_T, h_T = step_states[k]
            ev_buf[j][ev_len[j]] = torch.cat([z_T, h_T]).float().cpu()
            ep = dict(metas[j])
            # clone 摘出定长前缀，原缓冲随即释放（states 同步释放）——
            # terminal 段内存单调下降，不再三方叠加换页
            ep['x'] = ev_buf[j][:ev_len[j] + 1].clone()
            ep['terminal_logits'] = ev_term[k].float().cpu()
            episodes.append(ep)
            ev_buf[j] = None
            states[j] = None
            all_hc[j] = None
        del step_states
    print(f'  [{split_name}] {n_total} episodes collected '
          f'(batch={collect_batch})', flush=True)
    return episodes


# ----------------------------------------------------------------------
# 聚合头
# ----------------------------------------------------------------------
class StateAttentionHead(nn.Module):
    """masked attention 池化（learned）与均匀池化（uniform）双变体共用
    MLP；attention 查询零初始化 ⇒ 初始权重 ≡ 均匀（mean-pool 下限起步）。"""

    def __init__(self, state_dim: int, n_classes: int,
                 d_attn: int = 128, mlp_hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(state_dim + 1)      # +1: t/T 标量
        self.proj = nn.Linear(state_dim + 1, d_attn)
        self.query = nn.Parameter(torch.zeros(d_attn))   # 均匀初始化关键
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + 1, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, n_classes))

    def forward(self, x, mask, learned: bool = True):
        """x [B,T,D+1]（特征 ⊕ t/T），mask [B,T]。返回 (logits, alpha)。"""
        h = self.norm(x)                                    # [B,T,D+1]
        if learned:
            e = (self.proj(h) @ self.query) / math.sqrt(self.proj.out_features)
            e = e.masked_fill(~mask, float('-inf'))
            alpha = torch.softmax(e, dim=-1)                # [B,T]
        else:
            alpha = mask.float() / mask.float().sum(dim=1, keepdim=True)
        s = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)     # [B,D+1]
        return self.mlp(s), alpha


def make_batches(episodes, batch_size, shuffle=True):
    idx = list(range(len(episodes)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [episodes[j] for j in idx[i:i + batch_size]]


def collate(episodes, device):
    """pad 到 T_max → (x [B,T,D+1], mask [B,T], y [B])。t/T = (t+1)/T，
    终局步（末位）恒为 1.0。"""
    B = len(episodes)
    T = max(ep['x'].shape[0] for ep in episodes)
    D = episodes[0]['x'].shape[1]
    x = torch.zeros(B, T, D + 1, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    y = torch.tensor([ep['label'] for ep in episodes], device=device)
    for b, ep in enumerate(episodes):
        t = ep['x'].shape[0]
        x[b, :t, :D] = ep['x'].to(device)
        x[b, :t, D] = torch.arange(1, t + 1, device=device,
                                   dtype=torch.float32) / t
        mask[b, :t] = True
    return x, mask, y


@torch.no_grad()
def evaluate_variant(head, episodes, device, learned: bool):
    """评测 + 诊断（attention 质量：末位质量 = 终局步权重和，归一化熵）。"""
    head.eval()
    correct = n = 0
    last_mass, ent = [], []
    rows = None
    for batch in make_batches(episodes, 512, shuffle=False):
        x, mask, yb = collate(batch, device)
        logits, alpha = head(x, mask, learned=learned)
        correct += int(logits.argmax(-1).eq(yb).sum().item())
        n += len(batch)
        if learned:
            if rows is None or rows.shape[0] != alpha.shape[0]:
                rows = torch.arange(alpha.shape[0], device=device)
            last = mask.float().cumsum(dim=1).argmax(dim=1)     # 各序列末位
            last_mass += alpha[rows, last].tolist()
            p = alpha.clamp_min(1e-9)
            ent += (-(p * p.log()).sum(dim=1)
                    / mask.float().sum(dim=1).clamp_min(1).log()).tolist()
    diag = {'alpha_last_mass': round(sum(last_mass) / max(len(last_mass), 1), 4),
            'alpha_norm_entropy': round(sum(ent) / max(len(ent), 1), 4)} \
        if learned else {}
    return correct / max(n, 1), diag


def train_variant(head, train_eps, val_eps, eval_eps, args, device,
                  learned: bool):
    """训练一个变体：train 拟合、val 选 best、eval 终报。"""
    opt = torch.optim.Adam(head.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = n = correct = 0
        for batch in make_batches(train_eps, args.batch_size):
            x, mask, yb = collate(batch, device)
            logits, _ = head(x, mask, learned=learned)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
            correct += int(logits.argmax(-1).eq(yb).sum().item())
            n += len(batch)
        val_acc, _ = evaluate_variant(head, val_eps, device, learned)
        history.append({'epoch': epoch,
                        'train_loss': round(total_loss / n, 4),
                        'train_acc': round(correct / n, 4),
                        'val_acc': round(val_acc, 4)})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    eval_acc, diag = evaluate_variant(head, eval_eps, device, learned)
    return best_acc, eval_acc, diag, history


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    reader = load_reader_from_checkpoint(args.checkpoint, device)
    reader.eval()
    for p_ in reader.parameters():
        p_.requires_grad_(False)
    d = reader.config.hidden_dim
    n_classes = reader.config.n_classes
    print(f'Frozen evidence reader loaded: hidden={d}, classes={n_classes}')

    budget_ratios = (list(args.budget_ratios) if args.budget_ratios else None)
    ds = get_dataset(args.dataset, root=args.root, splits=tuple(args.splits),
                     max_graphs=args.max_graphs, stratified=args.stratified)
    ds_name = ds.spec.name
    if budget_ratios is None:
        budget_ratios = list(ds.spec.budget_options)
    print(f'Dataset {ds_name}: '
          + ' | '.join(f'{s}={len(getattr(ds, f"{s}_graphs"))}'
                       for s in args.splits)
          + f' | budgets={budget_ratios} | seeds={args.n_seeds}')

    # ---- 采集（缓存命中则跳过）----
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None and cache_path.exists():
        print(f'Cache hit: {cache_path}')
        blob = torch.load(str(cache_path), weights_only=False)
        train_eps, eval_eps = blob['train'], blob['eval']
    else:
        train_eps = collect_split(reader, ds.train_graphs, ds.train_labels,
                                  budget_ratios, args.n_seeds, device,
                                  args.splits[0], graph_offset=0,
                                  collect_batch=args.collect_batch)
        eval_eps = collect_split(reader, ds.test_graphs, ds.test_labels,
                                 budget_ratios, args.n_seeds, device,
                                 args.splits[-1], graph_offset=100000,
                                 collect_batch=args.collect_batch)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'train': train_eps, 'eval': eval_eps,
                        'budget_ratios': budget_ratios,
                        'checkpoint': args.checkpoint},
                       str(cache_path))
            print(f'Cache saved: {cache_path}')
    release_dataset(ds)
    t_mean = sum(e['x'].shape[0] for e in train_eps) / max(len(train_eps), 1)
    print(f'Episodes: train={len(train_eps)} eval={len(eval_eps)} '
          f'(T mean={t_mean:.1f})')

    # ---- terminal-only 参照（冻结 reader 自身终判，同批 rollout）----
    acc_terminal = sum(int(ep['terminal_logits'].argmax().item() == ep['label'])
                       for ep in eval_eps) / max(len(eval_eps), 1)
    print(f'[terminal-only] eval acc = {acc_terminal:.4f}')

    # ---- train 内 carve val（15%，按 episode 分层无必要，随机即可）----
    random.shuffle(train_eps)
    n_val = max(1, int(len(train_eps) * 0.15))
    val_eps, fit_eps = train_eps[:n_val], train_eps[n_val:]
    print(f'Fit/val episodes: {len(fit_eps)}/{len(val_eps)}')

    # ---- 双变体训练 ----
    out_dir = (Path(args.out) if args.out else
               CODE_DIR / 'result' / ds_name / 'experiments' / 'state-attn')
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {'acc_terminal': round(acc_terminal, 4), 'config': vars(args),
              'n_fit': len(fit_eps), 'n_val': len(val_eps),
              'n_eval': len(eval_eps),
              'trained_at': datetime.now().isoformat(timespec='seconds')}
    for name, learned in (('learned', True), ('uniform', False)):
        head = StateAttentionHead(2 * d, n_classes, args.d_attn,
                                  args.mlp_hidden).to(device)
        val_acc, eval_acc, diag, hist = train_variant(
            head, fit_eps, val_eps, eval_eps, args, device, learned)
        print(f'[{name}] best val acc = {val_acc:.4f} | '
              f'eval acc = {eval_acc:.4f} {diag}')
        report[f'acc_{name}'] = round(eval_acc, 4)
        report[f'val_acc_{name}'] = round(val_acc, 4)
        if learned:
            report['attention_diag'] = diag
            torch.save({'state_dict': state_dict_of(head),
                        'config': vars(args), 'state_dim': 2 * d,
                        'n_classes': n_classes},
                       out_dir / 'state_attention_head.pt')
        report[f'history_{name}'] = hist

    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'Report → {out_dir / "report.json"}')
    print('=== SMOKE SUMMARY ===')
    print(f'terminal-only: {report["acc_terminal"]:.4f} | '
          f'uniform: {report["acc_uniform"]:.4f} | '
          f'learned: {report["acc_learned"]:.4f}')


def state_dict_of(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


if __name__ == '__main__':
    main()

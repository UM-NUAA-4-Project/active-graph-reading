"""state-attention 聚合头独立评估（全量 held-out，逐预算档细分报告）。

训练脚本（train_state_attention.py）为压缩训练时长只在终局报一行汇总
精度；本脚本做完整事后评估，全部口径基于同一批 episodes：
- 逐 split × 逐预算档：learned / uniform / terminal-only（冻结 reader
  证据头终判）精度（±95% 二项 CI）+ attention 诊断（末位质量、归一化
  熵）+ 平均步数与提前截断率（frontier 早于预算耗尽的比例）；
- 零成本第四对照：冻结证据头在存储逐步状态上的 logits 均值集成
  （per-step ensemble）；其末步判据与采集时存下的 terminal_logits 的
  不一致数作为 checkpoint 一致性告警（GEMM 噪声可在近平手翻转 argmax，
  只计数不中断）。
- 采集与训练脚本同 lockstep 采集器；--cache 只读复用训练缓存（按
  split 标签 + (graph_idx, seed_i) 覆盖集校验命中，不命中则现场重采，
  绝不写缓存）。eval split 图索引偏移：test=100000（与训练一致）、
  valid=200000、train=0。

用法（cifar10 全量 test，复用训练缓存，数分钟出全表）:
  python code/train_reader/eval_state_attention.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --checkpoint code/result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch3.pt \
      --cache baseline/artifacts/cifar10/state_attn_full_cache.pt \
      --max-graphs 0 --n-seeds 4
输出 JSON → 聚合头同目录 eval_state_attention.json（--out 可覆盖）。
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))

import torch  # noqa: E402

from datasets import get_dataset, release_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from train_reader.core.build import load_reader_from_checkpoint  # noqa: E402
from train_reader.train_state_attention import (  # noqa: E402
    StateAttentionHead, collate, collect_split, make_batches)

# eval split 图索引偏移：test 与训练脚本一致（固定 100000）→ 缓存可复用；
# valid/train 各自独立偏移，避免 (graph_idx, seed_i) 跨 split 撞车
SPLIT_OFFSETS = {'train': 0, 'valid': 200000, 'test': 100000}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='state-attention 聚合头评估')
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True,
                   help='evidence reader checkpoint（须与训练头时冻结的同一 ckpt）')
    p.add_argument('--head', default='',
                   help='聚合头 checkpoint（默认 code/result/{ds}/experiments/'
                        'state-attn/state_attention_head.pt）')
    p.add_argument('--split', nargs='+', default=['test'],
                   choices=['train', 'valid', 'test'], help='评估 split')
    p.add_argument('--max-graphs', type=int, default=0,
                   help='每 split 图数上限（0=全部；缓存命中时仍用于覆盖校验）')
    p.add_argument('--stratified', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--n-seeds', type=int, default=4,
                   help='论文口径 4；缓存命中时仍用于覆盖校验')
    p.add_argument('--budget-ratios', nargs='+', type=float, default=None,
                   help='缺省取 spec.budget_options；仅现场重采集时生效'
                        '（缓存复用时逐 episode 的 budget 以缓存为准）')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--cache', default='',
                   help='训练侧特征缓存 .pt（只读复用；缺失或不匹配则现场重采）')
    p.add_argument('--collect-batch', type=int, default=256)
    p.add_argument('--eval-batch', type=int, default=512)
    p.add_argument('--out', default='',
                   help='输出 JSON（默认与 --head 同目录 eval_state_attention.json）')
    return p


def load_head(head_path: Path, device):
    blob = torch.load(str(head_path), map_location='cpu', weights_only=False)
    cfg = blob.get('config', {})
    head = StateAttentionHead(blob['state_dim'], blob['n_classes'],
                              cfg.get('d_attn', 128),
                              cfg.get('mlp_hidden', 128))
    head.load_state_dict(blob['state_dict'])
    return head.to(device).eval(), blob


def try_cache_split(cache_blob, split: str, n_graphs: int, n_seeds: int):
    """缓存命中判定：episode 的 split 标签 + (graph_idx, seed_i) 覆盖集与
    当前请求完全一致才命中（训练缓存键为 train/eval 两半，按标签挑选）。
    """
    for key in ('eval', 'train'):
        eps = cache_blob.get(key)
        if not eps or any(ep.get('split') != split for ep in eps[:5]):
            continue
        offset = SPLIT_OFFSETS.get(split)
        expected = {(offset + gi, si) for gi in range(n_graphs)
                    for si in range(n_seeds)}
        actual = {(ep['graph_idx'], ep['seed_i']) for ep in eps}
        if actual == expected and len(eps) == len(expected):
            return eps
        return None    # 标签对上但覆盖不匹配（max-graphs/n-seeds 不同）→ 不命中
    return None


@torch.no_grad()
def head_records(head, episodes, device, learned: bool, batch_size: int):
    """逐 episode（learned/uniform）精度 + attention 诊断，与 episodes 同序。"""
    recs = []
    for batch in make_batches(episodes, batch_size, shuffle=False):
        x, mask, y = collate(batch, device)
        logits, alpha = head(x, mask, learned=learned)
        correct = logits.argmax(-1).eq(y).tolist()
        last_idx = mask.float().cumsum(dim=1).argmax(dim=1)   # 各序列末位
        for b in range(len(batch)):
            rec = {'correct': bool(correct[b])}
            if learned:
                a = alpha[b, mask[b]]
                rec['last_mass'] = float(a[last_idx[b]].item())
                p_ = a.clamp_min(1e-9)
                rec['entropy'] = float((-(p_ * p_.log()).sum()
                                        / math.log(a.numel())).item())
            recs.append(rec)
    return recs


@torch.no_grad()
def evidence_head_scores(reader, episodes, device, chunk_rows=65536):
    """冻结证据头零成本对照：逐步 logits 的末步判据与均值集成。
    末步判据应与采集存下的 terminal_logits 一致（同权重同输入），
    不一致仅计数（近平手 argmax 翻转）。返回与 episodes 同序的
    (last_ok, ens_ok, mismatch) 列表。"""
    head = reader.evidence_head
    recs = []
    buf, bounds, rows = [], [], 0

    def flush():
        nonlocal buf, bounds, rows
        if not buf:
            return
        logits = head(torch.cat(buf, dim=0).to(device))          # [R, C]
        o = 0
        for T, lab, term_pred in bounds:
            lg = logits[o:o + T]
            last_pred = int(lg[-1].argmax().item())
            ens_pred = int(lg.mean(dim=0).argmax().item())
            recs.append((last_pred == lab, ens_pred == lab,
                         last_pred != term_pred))
            o += T
        buf, bounds, rows = [], [], 0

    for ep in episodes:
        buf.append(ep['x'])
        bounds.append((ep['x'].shape[0], int(ep['label']),
                       int(ep['terminal_logits'].argmax().item())))
        rows += ep['x'].shape[0]
        if rows >= chunk_rows:
            flush()
    flush()
    return recs


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1.0 - p), 1e-12) / max(n, 1))


def build_rows(episodes, rec_learned, rec_uniform, rec_ev):
    rows = []
    for i, ep in enumerate(episodes):
        rows.append({
            'ratio': float(ep['budget']),
            'T': int(ep['x'].shape[0]),
            'n_nodes': ep.get('n'),   # 旧缓存无此键 → 提前截断率跳过
            'ok_learned': rec_learned[i]['correct'],
            'ok_uniform': rec_uniform[i]['correct'],
            'ok_terminal': int(ep['terminal_logits'].argmax().item())
            == int(ep['label']),
            'ok_ev_last': rec_ev[i][0],
            'ok_ev_ens': rec_ev[i][1],
            'last_mass': rec_learned[i].get('last_mass'),
            'entropy': rec_learned[i].get('entropy'),
        })
    return rows


def summarize(rows):
    s = {'n': len(rows)}
    for k in ('ok_learned', 'ok_uniform', 'ok_terminal',
              'ok_ev_last', 'ok_ev_ens'):
        name = k[3:]
        acc = sum(r[k] for r in rows) / max(len(rows), 1)
        s[f'acc_{name}'] = round(acc, 4)
        s[f'ci95_{name}'] = round(ci95(acc, len(rows)), 4)
    s['T_mean'] = round(sum(r['T'] for r in rows) / max(len(rows), 1), 1)
    with_n = [r for r in rows if r['n_nodes'] is not None]
    if with_n:
        early = sum(1 for r in with_n
                    if r['T'] < max(1, int(r['ratio'] * r['n_nodes'])) + 1)
        s['early_stop_rate'] = round(early / len(with_n), 4)
    lm = [r['last_mass'] for r in rows if r['last_mass'] is not None]
    he = [r['entropy'] for r in rows if r['entropy'] is not None]
    if lm:
        s['alpha_last_mass'] = round(sum(lm) / len(lm), 4)
        s['alpha_norm_entropy'] = round(sum(he) / len(he), 4)
    return s


def print_table(title: str, by_budget: dict, overall: dict):
    def cell(s, k):
        return f'{s[f"acc_{k}"]:.4f}±{s[f"ci95_{k}"]:.4f}'

    print(f'\n[{title}]')
    print(f'{"budget":>6} {"n":>8}  {"learned":<16} {"uniform":<16} '
          f'{"terminal":<16} {"ev-ens":<7} {"a_last":>6} {"H_norm":>6} {"T_mean":>7}')
    rows = list(by_budget.items()) + [('ALL', overall)]
    for key, s in rows:
        print(f'{key:>6} {s["n"]:>8}  {cell(s, "learned"):<16} '
              f'{cell(s, "uniform"):<16} {cell(s, "terminal"):<16} '
              f'{s["acc_ev_ens"]:<7.4f} '
              f'{s.get("alpha_last_mass", float("nan")):>6.3f} '
              f'{s.get("alpha_norm_entropy", float("nan")):>6.3f} '
              f'{s["T_mean"]:>7.1f}')


def main():
    args = build_parser().parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    head_path = (Path(args.head) if args.head else
                 CODE_DIR / 'result' / args.dataset / 'experiments'
                 / 'state-attn' / 'state_attention_head.pt')
    if not head_path.exists():
        raise SystemExit(f'聚合头不存在: {head_path}\n先训练: '
                         f'python code/train_reader/train_state_attention.py ...')
    head, head_blob = load_head(head_path, device)
    head_cfg = head_blob.get('config', {})

    reader = load_reader_from_checkpoint(args.checkpoint, device)
    reader.eval()
    for p_ in reader.parameters():
        p_.requires_grad_(False)
    if head_blob['state_dim'] != 2 * reader.config.hidden_dim:
        raise SystemExit(f'聚合头 state_dim={head_blob["state_dim"]} 与 reader '
                         f'hidden_dim={reader.config.hidden_dim}（2d='
                         f'{2 * reader.config.hidden_dim}）不匹配')
    if head_blob['n_classes'] != reader.config.n_classes:
        raise SystemExit('聚合头 n_classes 与 reader 不匹配')
    print(f'Head: {head_path} | frozen reader: '
          f'hidden={reader.config.hidden_dim}, classes={reader.config.n_classes}')

    ratios = (list(args.budget_ratios) if args.budget_ratios else None)
    ds = get_dataset(args.dataset, root=args.root,
                     splits=tuple(dict.fromkeys(args.split)),
                     max_graphs=args.max_graphs, stratified=args.stratified)
    if ratios is None:
        ratios = list(ds.spec.budget_options)
    print(f'Dataset {ds.spec.name}: budgets={ratios} | seeds={args.n_seeds} '
          f'| splits={" ".join(args.split)}')

    cache_blob = None
    if args.cache:
        cp = Path(args.cache)
        if cp.exists():
            cache_blob = torch.load(str(cp), weights_only=False)
            print(f'Cache loaded: {cp}')
        else:
            print(f'Cache not found, will collect fresh: {cp}')

    report = {'dataset': ds.spec.name, 'checkpoint': args.checkpoint,
              'head': str(head_path), 'head_config': head_cfg,
              'budget_ratios': ratios, 'n_seeds': args.n_seeds,
              'splits': {},
              'evaluated_at': datetime.now().isoformat(timespec='seconds')}

    for split in args.split:
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if not len(graphs):
            print(f'\n[{split}] 无数据，跳过')
            continue
        eps, used_cache = None, False
        if cache_blob is not None:
            eps = try_cache_split(cache_blob, split, len(graphs), args.n_seeds)
            used_cache = eps is not None
            tag = ('命中' if used_cache
                   else '未命中（覆盖集/split 标签不匹配）→ 现场重采集')
            print(f'\n[{split}] 缓存{tag}')
        if eps is None:
            eps = collect_split(reader, graphs, labels, ratios, args.n_seeds,
                                device, split,
                                graph_offset=SPLIT_OFFSETS[split],
                                collect_batch=args.collect_batch)
        rec_l = head_records(head, eps, device, learned=True,
                             batch_size=args.eval_batch)
        rec_u = head_records(head, eps, device, learned=False,
                             batch_size=args.eval_batch)
        rec_e = evidence_head_scores(reader, eps, device)
        by_budget, overall, mismatch = {}, {}, 0
        grouped = {}
        for i, ep in enumerate(eps):
            grouped.setdefault(f"{float(ep['budget']):.2f}", []).append(i)
        for key, idxs in sorted(grouped.items()):
            rows = build_rows([eps[i] for i in idxs],
                              [rec_l[i] for i in idxs],
                              [rec_u[i] for i in idxs],
                              [rec_e[i] for i in idxs])
            by_budget[key] = summarize(rows)
        overall = summarize(build_rows(eps, rec_l, rec_u, rec_e))
        mismatch = sum(1 for m in rec_e if m[2])
        print_table(split, by_budget, overall)
        if mismatch:
            print(f'[warn] 证据头末步与 terminal_logits 不一致 {mismatch}/'
                  f'{len(eps)}（reader ckpt 与采集时不一致或近平手翻转）')
        report['splits'][split] = {
            'n_graphs': len(graphs), 'n_episodes': len(eps),
            'cache_used': used_cache, 'terminal_mismatch': mismatch,
            'overall': overall, 'by_budget': by_budget}

    release_dataset(ds)
    out_path = (Path(args.out) if args.out else
                head_path.parent / 'eval_state_attention.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\nJSON → {out_path}')
    print('=== EVAL SUMMARY ===')
    for split, s in report['splits'].items():
        o = s['overall']
        print(f'[{split}] n={o["n"]}  learned {o["acc_learned"]:.4f} | '
              f'uniform {o["acc_uniform"]:.4f} | terminal '
              f'{o["acc_terminal"]:.4f} | ev-ensemble {o["acc_ev_ens"]:.4f}')


if __name__ == '__main__':
    main()

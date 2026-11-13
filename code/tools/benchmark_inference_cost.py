"""推理成本基准：全图分类器 / 协议训练基线 / BAGR 测试集运行时间+内存统计。

目的：量化预算感知采样（BAGR）相对全图分类与学习型基线的推理开销收益。
同一评估协议（eval_loop.evaluate_fixed / evaluate_reader，与既有 eval JSON
完全同口径）下逐系统计时 + 显存峰值采样，一次运行产出单 JSON：

  code/result/{ds}/cost/benchmark_cost.json
    = {'config': {...图规模/协议参数...},
       'systems': {mode: {split: {wall_total_s, s_per_graph, s_per_rollout,
                                  acc_*, gpu_peak_allocated_mib, rss_peak_mb,
                                  n_params, ckpt_size_mb, ...}}},
       'comparison': {mode: {split: {time_ratio_vs_full_graph,
                                     acc_delta_vs_full_graph, ...}}}}

覆盖五类系统（--modes 任选，建议按下列顺序跑）：
- full-graph  : 上界分类器整批前向（--classifier 即系统本体；全图上界口径）；
- sagpool/softmask/agentnet : baseline/ 训练基线（协议逐点揭示 +
  自带头终判 judge=own 默认，对齐 eval_baselines --judge own 现行口径）；
- bagr        : learnable reader（--bagr-ckpt，结构按 checkpoint 内嵌
  config 重建；evidence/base 自动识别）。

计时口径（各系统一致为端到端 wall time）：
- full-graph：dgl.batch + 传输 + 前向整循环（另记 forward_s 纯前向分解）；
- 协议系统（基线/BAGR）：整次 evaluate_* 调用（rollout/lockstep + 子图
  构建 + 终判前向；BAGR 侧含冻结分类器参照前向，与现行 eval 流程一致）。
内存口径：每系统重置 CUDA peak 统计后独立采样（allocated/reserved MiB）；
CPU RSS 用 psutil 后台线程 50ms 采样峰值（次要指标，缺 psutil 自动跳过）。

公平性注意：协议系统按协议跑 budgets × n_seeds 个 rollout/图（默认
3 × 4 = 12），s_per_rollout 已按 rollout 数归一，可跨系统直接比。

用法:
  # 冒烟（小样本 + 1 seed，验证管线）
  python code/tools/benchmark_inference_cost.py --dataset cifar10 \
      --root D:\\data\\superpixels --max-graphs 40 --n-seeds 1 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt>

  # 正式（cifar10 全量 test）
  python code/tools/benchmark_inference_cost.py --dataset cifar10 \
      --root D:\\data\\superpixels --classifier <cls.pt> --bagr-ckpt <rl.pt>

  # 正式（collab：本地 8GB 卡必须调小 ref-batch/chunk，参照既有 collab eval）
  python code/tools/benchmark_inference_cost.py --dataset collab \
      --root D:\\data\\COLLAB --ref-batch-size 4 --chunk-size 4 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt>

基线 scorer 缺省读 baseline/artifacts/{ds}/{mode}.pt（--sagpool-ckpt 等可覆盖）。
PPA 后续补跑需 --max-graphs（协议同 eval）。
"""

import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
import dgl  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset, release_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import CODE_DIR, REPO_ROOT, resolve  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (eval_resolve, load_classifier_frozen,  # noqa: E402
                               load_reader_from_checkpoint)
from train_reader.core.eval_loop import (evaluate_fixed,  # noqa: E402
                                         evaluate_reader)

_BASELINE_MODES = ('sagpool', 'softmask', 'agentnet')
_ALL_MODES = ('full-graph',) + _BASELINE_MODES + ('bagr',)
_DEVICE_STR = str


def add_args(parser):
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--classifier', required=True,
                        help='上界分类器 checkpoint（= full-graph 系统本体'
                             '+ 协议系统冻结裁判）')
    parser.add_argument('--bagr-ckpt', default='',
                        help='BAGR reader checkpoint（bagr 模式必需）')
    parser.add_argument('--sagpool-ckpt', default='',
                        help='sagpool scorer（缺省 baseline/artifacts/{ds}/sagpool.pt）')
    parser.add_argument('--softmask-ckpt', default='',
                        help='softmask scorer（缺省 baseline/artifacts/{ds}/softmask.pt）')
    parser.add_argument('--agentnet-ckpt', default='',
                        help='agentnet scorer（缺省 baseline/artifacts/{ds}/agentnet.pt）')
    parser.add_argument('--modes', nargs='+', default=list(_ALL_MODES),
                        choices=list(_ALL_MODES),
                        help='要统计的系统（默认全部五个，按给定顺序执行）')
    parser.add_argument('--judge', default='own', choices=['clf', 'own'],
                        help='基线终判口径（默认 own=自带头，对齐现行'
                             'eval_baselines 口径；协议 rollout 成本两种几乎相同）')
    parser.add_argument('--splits', nargs='+', default=['test'],
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 图数限制（0=全部；冒烟/PPA 用）')
    parser.add_argument('--stratified', action='store_true',
                        help='--max-graphs 按类别分层抽样')
    parser.add_argument('--n-seeds', type=int, default=1,
                        help='每图确定性 seed 数（协议系统 rollout 数 = 图数×预算档×seeds）')
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='预算比例档覆盖（默认数据集 spec.budget_options）')
    parser.add_argument('--ref-batch-size', type=int, default=256,
                        help='full-graph 批量 + 协议系统终判批量参照'
                             '（collab 本地 8GB 卡用 4）')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='BAGR lockstep 图批量（collab 本地用 4）')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='',
                        help='输出 JSON（缺省 code/result/{ds}/cost/benchmark_cost.json）')


# ---------------------------------------------------------------- 计时/内存

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _gpu_reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def _gpu_peak_mib(device):
    if device.type != 'cuda':
        return None, None
    return (torch.cuda.max_memory_allocated(device) / 2 ** 20,
            torch.cuda.max_memory_reserved(device) / 2 ** 20)


def _rss_now():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return None


class _RssPeakSampler:
    """后台线程低频采样进程 RSS 峰值（次要指标；psutil 缺失时静默降级）。"""

    def __init__(self, interval_s=0.05):
        self._interval = interval_s
        self._stop = None
        self._thread = None
        self._proc = None
        self.peak = None

    def start(self):
        try:
            import psutil
        except ImportError:
            return False
        self._proc = psutil.Process()
        self.peak = self._proc.memory_info().rss / 2 ** 20
        self._stop = threading.Event()

        def _loop():
            while not self._stop.wait(self._interval):
                rss = self._proc.memory_info().rss / 2 ** 20
                if rss > self.peak:
                    self.peak = rss

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if self._stop is None:
            return None
        self._stop.set()
        self._thread.join(timeout=1.0)
        return self.peak


def _ckpt_size_mb(path):
    try:
        return os.path.getsize(str(path)) / 2 ** 20
    except OSError:
        return None


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def _agent_model(agent):
    """策略实例的主 nn.Module（sagpool/softmask/agentnet 均为 _model）。"""
    m = getattr(agent, '_model', None)
    if m is None:
        mods = [v for v in agent.__dict__.values()
                if isinstance(v, torch.nn.Module)]
        m = mods[0] if mods else None
    return m


# ---------------------------------------------------------------- 各系统基准

@torch.no_grad()
def bench_full_graph(classifier, graphs, labels, device, batch_size):
    """全图分类器批量推理（口径同 evaluate_classifier_full + 分段计时）。

    返回 (acc, wall_total_s, forward_s, n_batches)：wall 含 dgl.batch/H2D
    传输（端到端），forward_s 为纯前向分解。
    """
    classifier.eval()
    correct = total = n_batches = 0
    forward_s = 0.0
    _sync(device)
    t0 = time.perf_counter()
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        batch_g = dgl.batch(chunk).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[start:start + batch_size].to(device)
        _sync(device)
        tf = time.perf_counter()
        logits = classifier(batch_g, nids)
        _sync(device)
        forward_s += time.perf_counter() - tf
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(y)
        n_batches += 1
        del batch_g, nids, y, logits
    _sync(device)
    return correct / max(total, 1), time.perf_counter() - t0, forward_s, n_batches


def load_baseline_agent(mode, ckpt_path, device):
    """按 mode 构造并加载 baseline 协议适配策略（接口同 eval_baselines）。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if mode == 'sagpool':
        from baseline.sagpool import SagpoolStrategy
        agent = SagpoolStrategy()
    elif mode == 'softmask':
        from baseline.softmask import SoftmaskStrategy
        agent = SoftmaskStrategy()
    elif mode == 'agentnet':
        from baseline.agentnet import AgentnetStrategy
        agent = AgentnetStrategy()
    else:
        raise ValueError(mode)
    if not Path(ckpt_path).exists():
        raise SystemExit(f"{mode} scorer checkpoint 不存在: {ckpt_path}\n"
                         f"先训练: python baseline/run_{mode}.py "
                         f"--dataset <ds> --root <root>")
    agent.load(str(ckpt_path), _DEVICE_STR(device))
    return agent


# ---------------------------------------------------------------- 主流程

def _cleanup(device, *handles):
    for h in handles:
        del h
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _graph_stats(graphs):
    ns = [g.num_nodes() for g in graphs]
    es = [g.num_edges() for g in graphs]
    ns_sorted = sorted(ns)
    mid = len(ns_sorted) // 2
    return {
        'n_nodes_mean': round(sum(ns) / len(ns), 1),
        'n_nodes_median': ns_sorted[mid],
        'n_nodes_max': max(ns),
        'n_edges_mean': round(sum(es) / len(es), 1),
        'n_edges_max': max(es),
    }


def main():
    parser = argparse.ArgumentParser(description='推理成本基准（时间+内存）')
    add_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})"
             if device.type == 'cuda' else ''))

    # spec 只读探针（不触发数据加载）
    spec = get_dataset(args.dataset, root=args.root, splits=(),
                       max_graphs=args.max_graphs,
                       stratified=args.stratified).spec
    budget_options = (list(args.budget_ratios) if args.budget_ratios
                      else list(spec.budget_options))
    if 'bagr' in args.modes and not args.bagr_ckpt:
        raise SystemExit("--modes 含 bagr 时 --bagr-ckpt 必填")
    if 'full-graph' not in args.modes and not args.classifier:
        raise SystemExit("--classifier 必填（协议系统冻结裁判）")

    classifier_path = eval_resolve(args.classifier)
    out_path = (eval_resolve(args.output) if args.output
                else CODE_DIR / 'result' / spec.name / 'cost'
                     / 'benchmark_cost.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'dataset': spec.name,
        'splits': list(args.splits),
        'budgets': budget_options,
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'judge': args.judge,
        'seed_method': SEED_METHOD_STR,
        'ref_batch_size': args.ref_batch_size,
        'chunk_size': args.chunk_size,
        'max_graphs': args.max_graphs,
        'stratified': args.stratified,
        'device': _DEVICE_STR(device),
        'device_name': (torch.cuda.get_device_name(device)
                        if device.type == 'cuda' else 'cpu'),
        'classifier': str(classifier_path),
        'bagr_ckpt': str(eval_resolve(args.bagr_ckpt)) if args.bagr_ckpt else '',
        'baseline_ckpts': {
            m: str(eval_resolve(getattr(args, f'{m}_ckpt')
                                or REPO_ROOT / 'baseline' / 'artifacts'
                                   / spec.name / f'{m}.pt'))
            for m in _BASELINE_MODES if m in args.modes},
        'timing_note': '协议系统 wall = 整次 evaluate_* 调用（rollout + 子图构建 '
                       '+ 终判前向，BAGR 含冻结分类器参照前向）；full-graph wall '
                       '= dgl.batch+传输+前向整循环（forward_s 为纯前向分解）',
    }
    json_out = {'config': config, 'systems': {}}

    builder = SubgraphBuilder()
    for split in list(dict.fromkeys(args.splits)):
        ds = get_dataset(args.dataset, root=args.root, splits=(split,),
                         max_graphs=args.max_graphs, stratified=args.stratified)
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if len(graphs) == 0:
            print(f"[skip] 数据集 '{args.dataset}' 无 {split} 划分（或为空），跳过")
            release_dataset(ds)
            continue
        config[f'graph_stats_{split}'] = _graph_stats(graphs)
        config.setdefault('n_graphs', {})[split] = len(graphs)
        print(f"\n[{split}] 已加载 {len(graphs)} 图 "
              f"{config[f'graph_stats_{split}']}")
        n_rollouts = len(graphs) * len(budget_options) * args.n_seeds

        for mode in args.modes:
            print(f"\n--- [{split}] {mode} ---")
            _gpu_reset_peak(device)
            rss_before = _rss_now()
            sampler = _RssPeakSampler()
            sampler_ok = sampler.start()

            metrics = {'n_graphs': len(graphs)}
            handles = []
            try:
                if mode == 'full-graph':
                    clf = load_classifier_frozen(classifier_path, device)
                    handles.append(clf)
                    metrics['n_params'] = _n_params(clf)
                    metrics['ckpt_size_mb'] = round(
                        _ckpt_size_mb(classifier_path) or 0, 2)
                    _sync(device)
                    t0 = time.perf_counter()
                    acc, wall, forward_s, _ = bench_full_graph(
                        clf, graphs, labels, device, args.ref_batch_size)
                    wall = time.perf_counter() - t0   # 含构建/传输的端到端
                    metrics.update({'acc': round(acc, 6),
                                    'wall_total_s': round(wall, 3),
                                    'forward_s': round(forward_s, 3),
                                    's_per_graph': wall / len(graphs),
                                    'n_rollouts': None, 's_per_rollout': None})
                elif mode == 'bagr':
                    clf = load_classifier_frozen(classifier_path, device)
                    reader = load_reader_from_checkpoint(
                        eval_resolve(args.bagr_ckpt), device,
                        stochastic=False, temperature=1.0)
                    handles += [clf, reader]
                    metrics['n_params'] = _n_params(reader)
                    metrics['ckpt_size_mb'] = round(
                        _ckpt_size_mb(eval_resolve(args.bagr_ckpt)) or 0, 2)
                    metrics['reader_kind'] = reader.config.kind
                    _sync(device)
                    t0 = time.perf_counter()
                    results = evaluate_reader(
                        reader, clf, builder, graphs, labels, budget_options,
                        device, args.argmax, chunk_size=args.chunk_size,
                        n_seeds=args.n_seeds)
                    wall = time.perf_counter() - t0
                    acc_keys = {k: v for k, v in results.items()
                                if k.startswith('acc_')}
                    metrics.update({'wall_total_s': round(wall, 3),
                                    's_per_graph': wall / len(graphs),
                                    'n_rollouts': n_rollouts,
                                    's_per_rollout': wall / n_rollouts,
                                    **{k: round(v, 6)
                                       for k, v in acc_keys.items()}})
                else:   # sagpool / softmask / agentnet
                    ckpt = (getattr(args, f'{mode}_ckpt')
                            or REPO_ROOT / 'baseline' / 'artifacts'
                               / spec.name / f'{mode}.pt')
                    clf = load_classifier_frozen(classifier_path, device)
                    agent = load_baseline_agent(mode, eval_resolve(ckpt), device)
                    handles += [clf, agent]
                    am = _agent_model(agent)
                    if am is not None:
                        metrics['n_params'] = _n_params(am)
                    metrics['ckpt_size_mb'] = round(_ckpt_size_mb(ckpt) or 0, 2)
                    _sync(device)
                    t0 = time.perf_counter()
                    results = evaluate_fixed(
                        None, clf, builder, graphs, labels, budget_options,
                        device, args.argmax, n_seeds=args.n_seeds,
                        agent=agent, judge=args.judge)
                    wall = time.perf_counter() - t0
                    acc_keys = {k: v for k, v in results.items()
                                if k.startswith('acc_')}
                    metrics.update({'wall_total_s': round(wall, 3),
                                    's_per_graph': wall / len(graphs),
                                    'n_rollouts': n_rollouts,
                                    's_per_rollout': wall / n_rollouts,
                                    **{k: round(v, 6)
                                       for k, v in acc_keys.items()}})

                wall = metrics['wall_total_s']
                alloc_mib, reserved_mib = _gpu_peak_mib(device)
                metrics['gpu_peak_allocated_mib'] = (round(alloc_mib, 1)
                                                     if alloc_mib else None)
                metrics['gpu_peak_reserved_mib'] = (round(reserved_mib, 1)
                                                    if reserved_mib else None)
                metrics['rss_before_mb'] = (round(rss_before, 1)
                                            if rss_before else None)
                rss_peak = sampler.stop()
                metrics['rss_peak_mb'] = (round(rss_peak, 1)
                                          if rss_peak else None)
                if not sampler_ok:
                    metrics['rss_peak_mb'] = None

                json_out['systems'].setdefault(mode, {})[split] = metrics
                save_json(json_out, out_path)   # 每系统增量落盘
                print(f"[{split}] {mode}: "
                      f"acc={metrics.get('acc', metrics.get('acc_overall'))} "
                      f"wall={wall:.2f}s "
                      f"({metrics['s_per_graph'] * 1000:.1f} ms/图"
                      + (f", {metrics['s_per_rollout'] * 1000:.1f} ms/rollout"
                         if metrics['s_per_rollout'] else '')
                      + f")  gpu_peak={metrics['gpu_peak_allocated_mib']}MiB"
                      f"  params={metrics.get('n_params')}")
            finally:
                _cleanup(device, *handles)
                sampler.stop()

        release_dataset(ds)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 汇总对比（vs full-graph，同 split）----
    comparison = {}
    for mode, per_split in json_out['systems'].items():
        if mode == 'full-graph':
            continue
        comparison[mode] = {}
        for split, m in per_split.items():
            ref = json_out['systems'].get('full-graph', {}).get(split)
            if not ref:
                continue
            acc_m = m.get('acc', m.get('acc_overall'))
            comparison[mode][split] = {
                'time_ratio_vs_full_graph': round(
                    m['s_per_graph'] / ref['s_per_graph'], 3),
                'acc_delta_vs_full_graph': round(acc_m - ref['acc'], 6),
                'gpu_mem_ratio_vs_full_graph': round(
                    m['gpu_peak_allocated_mib'] / ref['gpu_peak_allocated_mib'], 3)
                if m['gpu_peak_allocated_mib'] and ref['gpu_peak_allocated_mib']
                else None,
            }
    json_out['comparison'] = comparison
    save_json(json_out, out_path)

    # ---- 汇总表 ----
    for split in list(dict.fromkeys(args.splits)):
        rows = [(mode, m) for mode, per in json_out['systems'].items()
                for sp, m in per.items() if sp == split]
        if not rows:
            continue
        print(f"\n==== 推理成本汇总 [{split}] ====")
        print(f"{'mode':<12}{'acc':>8}{'ms/图':>10}{'ms/rollout':>12}"
              f"{'gpu_peak(MiB)':>15}{'params':>12}{'ckpt(MB)':>10}")
        for mode, m in rows:
            acc = m.get('acc', m.get('acc_overall'))
            acc_s = f"{acc:.4f}" if acc is not None else '-'
            msr = (f"{m['s_per_rollout'] * 1000:.1f}"
                   if m.get('s_per_rollout') else '-')
            params = m.get('n_params')
            params_s = f"{params / 1e6:.2f}M" if params else '-'
            print(f"{mode:<12}{acc_s:>8}{m['s_per_graph'] * 1000:>10.1f}"
                  f"{msr:>12}{str(m['gpu_peak_allocated_mib']):>15}"
                  f"{params_s:>12}{str(m.get('ckpt_size_mb')):>10}")
        if comparison:
            print("\n-- vs full-graph（同 split）--")
            for mode, per in comparison.items():
                if split in per:
                    c = per[split]
                    print(f"{mode:<12} 时间×{c['time_ratio_vs_full_graph']}"
                          f"  acc{c['acc_delta_vs_full_graph']:+.4f}"
                          f"  显存×{c['gpu_mem_ratio_vs_full_graph']}")

    print(f"\nJSON saved to {out_path}")


if __name__ == '__main__':
    main()

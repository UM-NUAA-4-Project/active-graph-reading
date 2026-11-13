"""推理成本基准：全图分类器 / 协议训练基线 / BAGR 测试集运行时间+内存统计。

目的：量化预算感知采样（BAGR）相对全图分类与学习型基线的推理开销收益。
同一评估协议（eval_loop.evaluate_fixed / evaluate_reader，与既有 eval JSON
完全同口径）下逐系统计时 + 显存分解统计，一次运行产出单 JSON：

  code/result/{ds}/cost/benchmark_cost.json
    = {'config': {...图规模/协议参数...},
       'systems': {mode: {split: {wall_total_s, s_per_graph, s_per_rollout,
                                  acc_*, mem_model_mib, mem_step_*,
                                  mem_judge_*, rss_peak_mb, n_params, ...}}},
       'comparison': {mode: {split: {time_ratio_vs_full_graph,
                                     acc_delta_vs_full_graph,
                                     mem_model_ratio, mem_step_mean_ratio}}}}

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
显存口径（分解式，不报运行期总量峰值）：
- mem_model_mib   : 模型参数+buffer 常驻显存（numel×element_size 精确累计；
  协议系统共用的冻结分类器裁判不计入各系统，其字节数记
  config.judge_model_mib）；
- mem_step_*      : 每步前沿子图瞬态显存——每次步进前向内
  (peak_allocated − 调用前已分配) 的均值/中位/最大（MiB）：基线 = select
  内可见子图 H2D+打分前向（agentnet 首步另含 per-graph 静态缓存构建，
  缓存之后常驻、见 mem_resident_nonmodel_mib）；BAGR = lockstep 批子图
  前向（≤chunk_size 个子图/次）；full-graph = 整批全图 H2D+前向
  （无前沿概念，步 = 批）；
- mem_judge_*     : 终局分类瞬态（批量前向：BAGR 终局证据前向 + 冻结
  分类器批；基线 = 冻结分类器批 + own 头批）；
- mem_resident_nonmodel_mib : 评估结束仍常驻的非模型显存（agentnet 的
  per-graph 缓存等）；
- mem_series      : 单图单步序列（步号 = 已揭示节点数+1，JSON per_step 给
  逐步跨图 mean/std/n）：基线在主趟 select 直接打标签；BAGR 另跑
  chunk_size=1 序列趟（lockstep 批的瞬态无法归到单图）；full-graph 无步
  概念，只有每图近似（整批瞬态÷批内图数）。绘图见
  code/comparison/plot_mem_series.py。
CPU RSS 用 psutil 后台线程 50ms 采样峰值（次要指标，缺 psutil 自动跳过）。

公平性注意：协议系统按协议跑 budgets × n_seeds 个 rollout/图，统计口径
3 预算档 × 1 seed；s_per_rollout 已按 rollout 数归一，可跨系统直接比。

用法:
  # 冒烟（小样本 + 1 seed，验证管线）
  python code/tools/benchmark_inference_cost.py --dataset cifar10 \
      --root D:\\data\\superpixels --max-graphs 40 --n-seeds 1 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt>

  # 统计口径（test 随机 1000 图 = 分层抽样、每图 1 seed）
  python code/tools/benchmark_inference_cost.py --dataset cifar10 \
      --root D:\\data\\superpixels --max-graphs 1000 --stratified \
      --n-seeds 1 --classifier <cls.pt> --bagr-ckpt <rl.pt>

  # collab（本地 8GB 卡必须调小 ref-batch/chunk，参照既有 collab eval）
  python code/tools/benchmark_inference_cost.py --dataset collab \
      --root D:\\data\\COLLAB --max-graphs 1000 --stratified --n-seeds 1 \
      --ref-batch-size 4 --chunk-size 4 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt>

基线 scorer 缺省读 baseline/artifacts/{ds}/{mode}.pt（--sagpool-ckpt 等可覆盖）。
PPA 后续补跑需 --max-graphs（协议同 eval）。
"""

import argparse
import gc
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
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


class _StepMemRecorder:
    """步进/终判前向的瞬态显存记录（分解式显存口径的核心）。

    wrap() 包住每次前向入口：调用前记录 memory_allocated 并重置峰值计数
    （计数器在调用间不残留，故本脚本不再输出总量峰值），调用后取
    (peak − before) 为该次前向的瞬态 MiB，按 bucket 分桶累积：
    - 'step'  : 每步前沿子图前向（基线 select / BAGR lockstep 批 /
      full-graph 整批）；
    - 'judge' : 终局分类批前向（冻结分类器批 / own 头批 / BAGR 终局证据
      前向）。
    big_threshold 不为 None 时按首个参数（states/subgraphs 列表）长度分桶
    （> 阈值 → 'judge'）：区分 BAGR 的 lockstep 批与终局全状态批。
    tag_fn(args, mib, rec) 不为 None 时逐调用回填单图单步序列（rec.series，
    元素 {'step','b','mib'}；步号 = 已揭示节点数+1，可跨图对齐求均值±方差；
    tag_fn 自行决定是否入序列，如 bagr 序列趟的终局前向不入）。
    CPU 设备为 no-op（样本恒空）。
    """

    def __init__(self, device):
        self.device = device
        self.enabled = device.type == 'cuda'
        self.samples = {'step': [], 'judge': []}
        self.series = []          # 单图单步序列（tag_fn 回填）
        self.terminal_n = 0       # 序列趟中判为终局的前向数（不入曲线）
        self.batch_samples = []   # full-graph 专用：(瞬态MiB, 批内图数)
        self.net_floor_mib = 0.0  # 序列趟固定底（如 cuDNN LSTM workspace）

    def wrap(self, fn, bucket='step', big_threshold=None, tag_fn=None):
        if not self.enabled:
            return fn
        device = self.device
        rec = self

        def _inner(*args, **kwargs):
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            try:
                return fn(*args, **kwargs)
            finally:
                peak = torch.cuda.max_memory_allocated(device)
                mib = (peak - before) / 2 ** 20
                b = bucket
                if big_threshold is not None and args:
                    first = args[0]
                    if hasattr(first, '__len__') and len(first) > big_threshold:
                        b = 'judge'
                rec.samples[b].append(mib)
                if tag_fn is not None:
                    tag_fn(args, mib, rec)
        return _inner

    def stats(self):
        out = {}
        for bucket, arr in self.samples.items():
            if not arr:
                continue
            s = sorted(arr)
            n = len(s)
            out[f'mem_{bucket}_mean_mib'] = round(sum(s) / n, 2)
            out[f'mem_{bucket}_median_mib'] = round(s[n // 2], 2)
            out[f'mem_{bucket}_max_mib'] = round(s[-1], 2)
            out[f'mem_{bucket}_n'] = n
        return out


def _select_series_tag(args, mib, rec):
    """基线 select 序列标签：args[0]=GraphState，步号 = 已揭示数+1。"""
    st = args[0]
    rec.series.append({'step': st.get_budget_used() + 1,
                       'b': st._budget_limit, 'mib': round(mib, 4)})


def _bagr_series_tag(args, mib, rec):
    """bagr 序列趟标签：states 长度须为 1（chunk=1）；终局前向不入曲线。

    mib_net = 总瞬态 − rec.net_floor_mib（实测固定底，如 cuDNN LSTM
    每调用 workspace——与输入无关的实现开销，2026-11-13 诊断：单步
    (1,1,hidden) 输入即占 ~8.5 MiB）。
    """
    states = args[0]
    if len(states) != 1:
        return
    st = states[0]
    used = st.get_budget_used()
    if used >= st._budget_limit:
        rec.terminal_n += 1
        return
    rec.series.append({'step': used + 1, 'b': st._budget_limit,
                       'mib': round(mib, 4),
                       'mib_net': round(max(mib - rec.net_floor_mib, 0), 4)})


def _measure_lstm_floor(reader, device):
    """实测证据 LSTM 的固定每调用 workspace（MiB；首调含 cuDNN 初始化，
    取预热后稳定值；no_lstm/v1 或无 LSTM 返回 0）。"""
    lstm = getattr(reader, 'evidence_lstm', None)
    if lstm is None or getattr(reader, 'evidence_variant', ''
                               ) in ('no_lstm', 'v1'):
        return 0.0
    d = reader.config.hidden_dim
    x = torch.zeros(1, 1, d, device=device)
    hx = (torch.zeros(1, 1, d, device=device),
          torch.zeros(1, 1, d, device=device))
    floors = []
    with torch.no_grad():
        for _ in range(4):
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            lstm(x, hx)
            floors.append((torch.cuda.max_memory_allocated(device)
                           - before) / 2 ** 20)
    return round(floors[-1], 3)


def _series_stats(series, terminal_n, net_floor_mib=None):
    """单图单步序列 → 逐步跨图统计（mean/std/n，含可选净口径）+ 预算分布。"""
    by_step = defaultdict(list)
    by_step_net = defaultdict(list)
    b_count = defaultdict(int)
    for r in series:
        by_step[r['step']].append(r['mib'])
        if 'mib_net' in r:
            by_step_net[r['step']].append(r['mib_net'])
        b_count[r['b']] += 1
    per_step = {}
    for t in sorted(by_step):
        v = by_step[t]
        e = {'mean': round(sum(v) / len(v), 3),
             'std': round(statistics.pstdev(v), 3),
             'n': len(v)}
        if by_step_net:
            vn = by_step_net[t]
            e['mean_net'] = round(sum(vn) / len(vn), 3)
            e['std_net'] = round(statistics.pstdev(vn), 3)
        per_step[str(t)] = e
    out = {'per_step': per_step,
           'b_distribution': {str(k): v
                              for k, v in sorted(b_count.items())},
           'n_samples': len(series),
           'terminal_forwards': terminal_n,
           'note': '单图单步瞬态显存（步号=已揭示节点数+1，跨图对齐）；'
                   'std 为跨图总体标准差'}
    if by_step_net:
        out['net_floor_mib'] = net_floor_mib
        out['net_note'] = ('mean_net/std_net = 总瞬态 − 实测固定底'
                           f'（cuDNN LSTM 每调用 workspace {net_floor_mib} MiB，'
                           '与输入无关的实现开销，见 '
                           'tools/benchmark_inference_cost._measure_lstm_floor）；'
                           '基线无此底，不加扣')
    return out


def _model_bytes(model):
    """模型参数+buffer 的精确字节数（常驻显存口径）。"""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


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
def bench_full_graph(classifier, graphs, labels, device, batch_size, rec):
    """全图分类器批量推理（口径同 evaluate_classifier_full + 分段计时）。

    返回 (acc, wall_total_s, forward_s, n_batches)：wall 含 dgl.batch/H2D
    传输（端到端），forward_s 为纯前向分解。每批的 H2D+前向瞬态记入 rec
    的 'step' 桶（full-graph 无前沿子图概念，步 = 批，与基线 select 同级）。
    """
    classifier.eval()
    correct = total = n_batches = 0
    forward_s = 0.0
    on_gpu = rec.enabled
    _sync(device)
    t0 = time.perf_counter()
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        before = torch.cuda.memory_allocated(device) if on_gpu else 0
        if on_gpu:
            torch.cuda.reset_peak_memory_stats(device)
        batch_g = dgl.batch(chunk).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[start:start + batch_size].to(device)
        _sync(device)
        tf = time.perf_counter()
        logits = classifier(batch_g, nids)
        _sync(device)
        forward_s += time.perf_counter() - tf
        if on_gpu:
            peak = torch.cuda.max_memory_allocated(device)
            rec.samples['step'].append((peak - before) / 2 ** 20)
            rec.batch_samples.append(((peak - before) / 2 ** 20, len(chunk)))
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

    # 冻结分类器裁判的常驻字节（各协议系统共用，单列不计入系统模型显存；
    # CPU 加载读 numel×element_size，与 GPU 常驻字节一致）
    _judge_clf_cpu = load_classifier_frozen(classifier_path,
                                            torch.device('cpu'))
    judge_model_mib = round(_model_bytes(_judge_clf_cpu) / 2 ** 20, 2)
    judge_model_bytes = judge_model_mib * 2 ** 20
    del _judge_clf_cpu
    gc.collect()

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
        'judge_model_mib': judge_model_mib,
        'memory_note': '显存分解口径（不报总量峰值）：mem_model_mib=系统自身模型'
                       '参数+buffer 常驻字节；mem_step_*＝每次步进前向的峰值瞬态'
                       '（基线 select 含可见子图 H2D，agentnet 首步含 per-graph '
                       '缓存构建；BAGR=lockstep 批 ≤chunk_size；full-graph 步=批）；'
                       'mem_judge_*＝终局分类批前向瞬态；'
                       'mem_resident_nonmodel_mib=结束时仍常驻的非模型显存'
                       '（已扣除系统模型与裁判分类器字节，agentnet 缓存落在'
                       '此列）',
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
            rss_before = _rss_now()
            sampler = _RssPeakSampler()
            sampler_ok = sampler.start()
            rec = _StepMemRecorder(device)

            metrics = {'n_graphs': len(graphs)}
            handles = []
            try:
                if mode == 'full-graph':
                    clf = load_classifier_frozen(classifier_path, device)
                    handles.append(clf)
                    metrics['n_params'] = _n_params(clf)
                    metrics['ckpt_size_mb'] = round(
                        _ckpt_size_mb(classifier_path) or 0, 2)
                    metrics['mem_model_mib'] = round(
                        _model_bytes(clf) / 2 ** 20, 2)
                    _sync(device)
                    t0 = time.perf_counter()
                    acc, wall, forward_s, _ = bench_full_graph(
                        clf, graphs, labels, device, args.ref_batch_size, rec)
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
                    # 步 = lockstep 批前向（≤chunk_size）；终局全状态批按
                    # 长度归 judge 桶；冻结分类器参照批 = judge
                    reader.forward_states_evidence = rec.wrap(
                        reader.forward_states_evidence,
                        big_threshold=args.chunk_size)
                    reader.forward_states = rec.wrap(
                        reader.forward_states, big_threshold=args.chunk_size)
                    clf.forward = rec.wrap(clf.forward, bucket='judge')
                    metrics['n_params'] = _n_params(reader)
                    metrics['ckpt_size_mb'] = round(
                        _ckpt_size_mb(eval_resolve(args.bagr_ckpt)) or 0, 2)
                    metrics['reader_kind'] = reader.config.kind
                    metrics['mem_model_mib'] = round(
                        _model_bytes(reader) / 2 ** 20, 2)
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
                    # 步 = select（含可见子图 H2D+打分前向，单图/次）→
                    # 顺带记逐步序列；终判批 = judge
                    agent.select = rec.wrap(agent.select, bucket='step',
                                            tag_fn=_select_series_tag)
                    clf.forward = rec.wrap(clf.forward, bucket='judge')
                    if hasattr(agent, 'own_predict_batch'):
                        agent.own_predict_batch = rec.wrap(
                            agent.own_predict_batch, bucket='judge')
                    elif hasattr(agent, 'own_predict'):
                        agent.own_predict = rec.wrap(
                            agent.own_predict, bucket='judge')
                    am = _agent_model(agent)
                    if am is not None:
                        metrics['n_params'] = _n_params(am)
                        metrics['mem_model_mib'] = round(
                            _model_bytes(am) / 2 ** 20, 2)
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
                if rec.enabled:
                    resident_after = torch.cuda.memory_allocated(device)
                    model_bytes = metrics.get('mem_model_mib', 0) * 2 ** 20
                    nonmodel = resident_after - model_bytes - judge_model_bytes
                    metrics['mem_resident_nonmodel_mib'] = round(
                        max(nonmodel, 0) / 2 ** 20, 2)
                metrics.update(rec.stats())
                if rec.series:
                    metrics['mem_series'] = _series_stats(rec.series,
                                                          rec.terminal_n)
                if rec.batch_samples:
                    per_graph = [m / n for m, n in rec.batch_samples]
                    metrics['mem_per_graph_batch_mean_mib'] = round(
                        sum(per_graph) / len(per_graph), 2)
                    metrics['mem_per_graph_batch_std_mib'] = round(
                        statistics.pstdev(per_graph), 2)
                    metrics['mem_per_graph_batch_n'] = len(per_graph)
                    metrics['mem_per_graph_batch_approx'] = (
                        '整批瞬态÷批内图数的每图近似（批前向含固定开销，'
                        '非严格线性）')
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
                      + f")  模型={metrics.get('mem_model_mib')}MiB"
                      f" 步均={metrics.get('mem_step_mean_mib')}MiB"
                      f" 终判均={metrics.get('mem_judge_mean_mib')}MiB"
                      f"  params={metrics.get('n_params')}")
            finally:
                _cleanup(device, *handles)
                sampler.stop()

        # ---- BAGR 单图序列趟（chunk=1）：每图逐步前沿子图显存曲线 ----
        # 时间/精度口径以主趟为准；本趟只补单图单步序列——lockstep 批的
        # 瞬态无法归到单图（各步批大小不同），须逐图前向重采
        if ('bagr' in args.modes
                and json_out['systems'].get('bagr', {}).get(split)
                and device.type == 'cuda'):
            print(f"\n--- [{split}] bagr 序列趟（chunk_size=1，单图逐步曲线）---")
            rec_s = _StepMemRecorder(device)
            clf = load_classifier_frozen(classifier_path, device)
            reader = load_reader_from_checkpoint(
                eval_resolve(args.bagr_ckpt), device,
                stochastic=False, temperature=1.0)
            reader.forward_states_evidence = rec_s.wrap(
                reader.forward_states_evidence, tag_fn=_bagr_series_tag)
            reader.forward_states = rec_s.wrap(
                reader.forward_states, tag_fn=_bagr_series_tag)
            rec_s.net_floor_mib = _measure_lstm_floor(reader, device)
            try:
                evaluate_reader(reader, clf, builder, graphs, labels,
                                budget_options, device, args.argmax,
                                chunk_size=1, n_seeds=args.n_seeds)
            finally:
                _cleanup(device, clf, reader)
            if rec_s.series:
                json_out['systems']['bagr'][split]['mem_series'] = \
                    _series_stats(rec_s.series, rec_s.terminal_n,
                                  rec_s.net_floor_mib)
                save_json(json_out, out_path)
                steps = sorted(
                    int(t) for t in
                    json_out['systems']['bagr'][split]['mem_series']['per_step'])
                print(f"[{split}] bagr 序列: {len(rec_s.series)} 样本, "
                      f"步号 1..{steps[-1]}, "
                      f"终局前向 {rec_s.terminal_n} 个（不入曲线）")

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
                'mem_model_ratio_vs_full_graph': round(
                    m['mem_model_mib'] / ref['mem_model_mib'], 3)
                if m.get('mem_model_mib') and ref.get('mem_model_mib')
                else None,
                'mem_step_mean_ratio_vs_full_graph': round(
                    m['mem_step_mean_mib'] / ref['mem_step_mean_mib'], 3)
                if m.get('mem_step_mean_mib') and ref.get('mem_step_mean_mib')
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
        print(f"{'mode':<12}{'acc':>8}{'ms/图':>9}{'ms/rollout':>12}"
              f"{'模型MiB':>9}{'步均MiB':>9}{'终判MiB':>9}"
              f"{'params':>10}{'ckpt(MB)':>10}")
        for mode, m in rows:
            acc = m.get('acc', m.get('acc_overall'))
            acc_s = f"{acc:.4f}" if acc is not None else '-'
            msr = (f"{m['s_per_rollout'] * 1000:.1f}"
                   if m.get('s_per_rollout') else '-')
            mm = m.get('mem_model_mib')
            mm_s = f"{mm:.1f}" if mm is not None else '-'
            ms_ = m.get('mem_step_mean_mib')
            ms_s = f"{ms_:.2f}" if ms_ is not None else '-'
            mj = m.get('mem_judge_mean_mib')
            mj_s = f"{mj:.2f}" if mj is not None else '-'
            params = m.get('n_params')
            params_s = f"{params / 1e6:.2f}M" if params else '-'
            ckpt = m.get('ckpt_size_mb')
            ckpt_s = f"{ckpt:.2f}" if ckpt is not None else '-'
            print(f"{mode:<12}{acc_s:>8}{m['s_per_graph'] * 1000:>9.1f}"
                  f"{msr:>12}{mm_s:>9}{ms_s:>9}{mj_s:>9}"
                  f"{params_s:>10}{ckpt_s:>10}")
        if comparison:
            print("\n-- vs full-graph（同 split）--")
            for mode, per in comparison.items():
                if split in per:
                    c = per[split]
                    print(f"{mode:<12} 时间×{c['time_ratio_vs_full_graph']}"
                          f"  acc{c['acc_delta_vs_full_graph']:+.4f}"
                          f"  模型显存×{c['mem_model_ratio_vs_full_graph']}"
                          f"  步均显存×{c['mem_step_mean_ratio_vs_full_graph']}")

    print(f"\nJSON saved to {out_path}")


if __name__ == '__main__':
    main()

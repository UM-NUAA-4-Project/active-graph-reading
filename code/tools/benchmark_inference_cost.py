"""推理成本基准：全图分类器 / 协议训练基线 / BAGR 测试集运行时间+内存统计。

目的：量化预算感知采样（BAGR）相对全图分类与学习型基线的推理开销收益。
同一评估协议（eval_loop.evaluate_fixed / evaluate_reader，与既有 eval JSON
完全同口径）下逐系统计时 + 显存分解统计，一次运行产出单 JSON：

  code/result/{ds}/cost/benchmark_cost.json
    = {'config': {...图规模/协议参数...},
       'systems': {mode: {split: {wall_total_s, s_per_graph, s_per_rollout,
                                  acc_*, mem_model_mib, mem_step_*,
                                  mem_judge_*, rss_peak_mb, n_params,
                                  cost_stats, ...}}},
       'comparison': {mode: {split: {time_ratio_vs_full_graph,
                                     acc_delta_vs_full_graph,
                                     mem_model_ratio, mem_step_mean_ratio}}}}

覆盖六类系统（--modes 任选，建议按下列顺序跑）：
- full-graph  : 上界分类器整批前向（--classifier 即系统本体；全图上界口径）；
- sagpool/softmask/agentnet : baseline/ 训练基线（协议逐点揭示 +
  自带头终判 judge=own 默认，对齐 eval_baselines --judge own 现行口径）；
- bagr        : learnable reader（--bagr-ckpt，结构按 checkpoint 内嵌
  config 重建；evidence/base 自动识别；评测口径 = 证据头终判 + 冻结
  分类器参照前向，与现行 eval 流程一致）；
- bagr-deploy : BAGR 纯部署口径（classifier=None）：跳过终局子图构建与
  冻结分类器前向——实际部署只用自身证据头，不加载全图分类器；该模式
  不需要 --classifier。

计时口径（各系统一致为端到端 wall time；跨系统比较请指定单预算档，
如 --budget-ratios 0.10 --n-seeds 1，避免 s_per_graph 累加多档轨迹）：
- full-graph：dgl.batch + 传输 + 前向整循环（另记 forward_s 纯前向分解）；
- 协议系统（基线/BAGR）：整次 evaluate_* 调用（rollout/lockstep + 子图
  构建 + 终判前向；bagr 含冻结分类器参照前向，bagr-deploy 不含）。
显存口径（分解式，不报运行期总量峰值；统一在真实批量 --chunk-size /
--ref-batch-size 下统计，不做 chunk=1 单图瞬态 ×批量的换算）：
- mem_model_mib   : 模型参数+buffer 常驻显存（numel×element_size 精确累计；
  协议系统共用的冻结分类器裁判不计入各系统，其字节数记
  config.judge_model_mib；bagr-deploy 无裁判）；
- mem_step_*      : 每步前沿子图瞬态显存——每次步进前向内
  (peak_allocated − 调用前已分配) 的均值/中位/最大（MiB）：基线 = select
  内可见子图 H2D+打分前向（agentnet 首步另含 per-graph 静态缓存构建，
  缓存之后常驻、见 mem_resident_nonmodel_mib）；BAGR = lockstep 批子图
  前向（≤chunk_size 个子图/次）；full-graph = 整批全图 H2D+前向
  （无前沿概念，步 = 批）；
- mem_judge_*     : 终局分类瞬态（批量前向：BAGR 终局证据前向 + 冻结
  分类器批；基线 = 冻结分类器批 + own 头批；bagr-deploy = 证据前向）；
- mem_resident_nonmodel_mib : 评估结束仍常驻的非模型显存（agentnet 的
  per-graph 缓存等）；
- mem_series      : 单图单步序列（基线主趟 select 直接打标签；BAGR 默认
  不跑序列趟——2026-11-15 口径修订：删除 cuDNN LSTM 固定底扣减与 ×批量
  换算，跨系统比较一律用真实批量下的 mem_step_*；如仍需单图逐步曲线，
  显式加 --mem-series-trip 以 chunk=1 重采，耗时≈主趟一倍）。
  绘图见 code/comparison/plot_mem_series.py。
cost_stats（硬件无关计算量，2026-11-15 新增；协议系统经 evaluate_* 的
cost_stats_out 逐步累计）：
- revealed_node_pct : 每图平均揭示节点占比（不含 seed，协议口径）；
- vis_edges_per_rollout : 单条轨迹逐步编码的累计可见边数
  Σ_{t=1}^{B}|E_vis(t)|（已揭示边 + frontier one-ended 边，不含构图自环；
  不含终局补充前向）；
- vis_edge_ratio_pct : Σ|E_vis(t)| 占全图分类器消息传递边数
  L_clf·|E| 的比例（L_clf 自动取 --classifier config 的 n_layers，
  可用 --clf-layers 覆盖；full-graph 恒为 100%）。
CPU RSS 用 psutil 后台线程 50ms 采样峰值（次要指标，缺 psutil 自动跳过）。

用法:
  # 冒烟（小样本 + 1 seed，验证管线）
  python code/tools/benchmark_inference_cost.py --dataset cifar10 \
      --root D:\\data\\superpixels --max-graphs 40 --n-seeds 1 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt>

  # 统计口径（test 随机 1000 图 = 分层抽样、单预算档 10%、每图 1 seed，
  # 统一真实批量 chunk=32；bagr-deploy 不加载冻结分类器）
  python code/tools/benchmark_inference_cost.py --dataset cifar10 \
      --root D:\\data\\superpixels --max-graphs 1000 --stratified \
      --budget-ratios 0.10 --n-seeds 1 --chunk-size 32 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt> \
      --modes full-graph sagpool softmask agentnet bagr bagr-deploy

  # collab（本地 8GB 卡必须调小 ref-batch/chunk，参照既有 collab eval）
  python code/tools/benchmark_inference_cost.py --dataset collab \
      --root D:\\data\\COLLAB --max-graphs 1000 --stratified --n-seeds 1 \
      --budget-ratios 0.10 --ref-batch-size 16 --chunk-size 16 \
      --classifier <cls.pt> --bagr-ckpt <rl.pt>

基线 scorer 缺省读 baseline/artifacts/{ds}/{mode}.pt（--sagpool-ckpt 等可覆盖）。
PPA 需 --max-graphs 抽样（协议同 eval；本机无 agentnet scorer 时用 --modes
剔除）。
"""

import argparse
import gc
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
_ALL_MODES = ('full-graph',) + _BASELINE_MODES + ('bagr', 'bagr-deploy')
_NEED_CLF_MODES = ('full-graph',) + _BASELINE_MODES + ('bagr',)
_DEVICE_STR = str


def add_args(parser):
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--classifier', default='',
                        help='上界分类器 checkpoint（= full-graph 系统本体'
                             '+ 协议系统冻结裁判；仅 bagr-deploy 可缺省）')
    parser.add_argument('--bagr-ckpt', default='',
                        help='BAGR reader checkpoint（bagr / bagr-deploy 模式必需）')
    parser.add_argument('--sagpool-ckpt', default='',
                        help='sagpool scorer（缺省 baseline/artifacts/{ds}/sagpool.pt）')
    parser.add_argument('--softmask-ckpt', default='',
                        help='softmask scorer（缺省 baseline/artifacts/{ds}/softmask.pt）')
    parser.add_argument('--agentnet-ckpt', default='',
                        help='agentnet scorer（缺省 baseline/artifacts/{ds}/agentnet.pt）')
    parser.add_argument('--modes', nargs='+', default=list(_ALL_MODES),
                        choices=list(_ALL_MODES),
                        help='要统计的系统（默认全部，按给定顺序执行；'
                             'bagr-deploy=纯部署口径，无冻结分类器）')
    parser.add_argument('--clf-layers', type=int, default=0,
                        help='全图分类器消息传递层数 L_clf（0=自动取 '
                             '--classifier 内嵌 config 的 n_layers）')
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
                             '（collab/ppa 本地 8GB 卡调小，如 16）')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='BAGR lockstep 图批量（统一真实批量口径；'
                             'collab/ppa 本地用 16）')
    parser.add_argument('--mem-series-trip', action='store_true',
                        help='额外跑 chunk=1 单图逐步显存序列趟（默认关闭；'
                             '2026-11-15 口径修订后跨系统比较用真实批量 '
                             'mem_step_*，序列趟仅供单图曲线绘制）')
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

    2026-11-15 口径修订：不再扣减 cuDNN LSTM 每调用 workspace 固定底、
    不再做 ×批量换算——跨系统比较一律用真实批量下的 mem_step_*，本序列
    仅供单图逐步曲线（--mem-series-trip 显式开启）。
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
                       'mib': round(mib, 4)})


def _series_stats(series, terminal_n):
    """单图单步序列 → 逐步跨图统计（mean/std/n）+ 预算分布。"""
    by_step = defaultdict(list)
    b_count = defaultdict(int)
    for r in series:
        by_step[r['step']].append(r['mib'])
        b_count[r['b']] += 1
    per_step = {}
    for t in sorted(by_step):
        v = by_step[t]
        per_step[str(t)] = {'mean': round(sum(v) / len(v), 3),
                            'std': round(statistics.pstdev(v), 3),
                            'n': len(v)}
    return {'per_step': per_step,
            'b_distribution': {str(k): v for k, v in sorted(b_count.items())},
            'n_samples': len(series),
            'terminal_forwards': terminal_n,
            'note': '单图单步瞬态显存（步号=已揭示节点数+1，跨图对齐）；'
                    'std 为跨图总体标准差'}


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


def _cost_metrics(cs, clf_layers):
    """cost_stats 累计容器 → 硬件无关计算量指标（按预算档 + overall）。

    口径：revealed_node_pct = Σ揭示节点 / Σ节点数（不含 seed）；
    vis_edges_per_rollout = Σ_t|E_vis(t)| / rollout 数；
    vis_edge_ratio_pct = Σ|E_vis(t)| / (L_clf·Σ|E|)。full-graph 不经
    evaluate_* 累计，由调用方直接填 100%（定义即全图）。
    """
    if not cs or not cs.get('rollouts'):
        return {}
    out = {}
    tot = [0, 0, 0, 0, 0]   # vis, L·E, revealed, n_nodes, rollouts
    for b, n in sorted(cs['rollouts'].items()):
        vis = cs['vis_edges'][b]
        den = clf_layers * cs['full_edges'][b]
        out[f'{int(b * 100)}%'] = {
            'revealed_node_pct': round(
                100 * cs['revealed'][b] / max(cs['n_nodes'][b], 1), 2),
            'vis_edges_per_rollout': round(vis / n, 1),
            'vis_edge_ratio_pct': round(100 * vis / den, 2) if den else None,
        }
        tot = [tot[0] + vis, tot[1] + den,
               tot[2] + cs['revealed'][b], tot[3] + cs['n_nodes'][b],
               tot[4] + n]
    out['overall'] = {
        'revealed_node_pct': round(100 * tot[2] / max(tot[3], 1), 2),
        'vis_edges_per_rollout': round(tot[0] / tot[4], 1),
        'vis_edge_ratio_pct': round(100 * tot[0] / tot[1], 2) if tot[1] else None,
    }
    return out


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
    for m in ('bagr', 'bagr-deploy'):
        if m in args.modes and not args.bagr_ckpt:
            raise SystemExit(f"--modes 含 {m} 时 --bagr-ckpt 必填")
    need_clf = [m for m in args.modes if m in _NEED_CLF_MODES]
    if need_clf and not args.classifier:
        raise SystemExit(f"--classifier 必填（{', '.join(need_clf)} 需要"
                         "冻结裁判/full-graph 本体；纯部署口径请只跑 "
                         "--modes bagr-deploy）")

    classifier_path = eval_resolve(args.classifier) if args.classifier else ''
    out_path = (eval_resolve(args.output) if args.output
                else CODE_DIR / 'result' / spec.name / 'cost'
                     / 'benchmark_cost.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 冻结分类器裁判的常驻字节（各协议系统共用，单列不计入系统模型显存；
    # CPU 加载读 numel×element_size，与 GPU 常驻字节一致）。
    # 同时取 L_clf（消息传递层数）供可见边比例分母
    judge_model_mib = 0.0
    judge_model_bytes = 0
    clf_layers = args.clf_layers
    if classifier_path:
        _judge_clf_cpu = load_classifier_frozen(classifier_path,
                                                torch.device('cpu'))
        judge_model_mib = round(_model_bytes(_judge_clf_cpu) / 2 ** 20, 2)
        judge_model_bytes = judge_model_mib * 2 ** 20
        if not clf_layers:
            clf_layers = getattr(getattr(_judge_clf_cpu, 'config', None),
                                 'n_layers', 5) or 5
        del _judge_clf_cpu
        gc.collect()
    if not clf_layers:
        clf_layers = 5   # 无分类器可查时的保守缺省（三数据集裁判均为 5 层）

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
        'classifier': str(classifier_path) if classifier_path else '',
        'bagr_ckpt': str(eval_resolve(args.bagr_ckpt)) if args.bagr_ckpt else '',
        'baseline_ckpts': {
            m: str(eval_resolve(getattr(args, f'{m}_ckpt')
                                or REPO_ROOT / 'baseline' / 'artifacts'
                                   / spec.name / f'{m}.pt'))
            for m in _BASELINE_MODES if m in args.modes},
        'timing_note': '协议系统 wall = 整次 evaluate_* 调用（rollout + 子图构建 '
                       '+ 终判前向，bagr 含冻结分类器参照前向、bagr-deploy 不含'
                       '——纯部署口径无终局子图构建与 clf 前向）；full-graph wall '
                       '= dgl.batch+传输+前向整循环（forward_s 为纯前向分解）',
        'judge_model_mib': judge_model_mib,
        'clf_layers': clf_layers,
        'memory_note': '显存分解口径（不报总量峰值；统一真实批量 chunk/ref-batch '
                       '下统计，无 chunk=1 换算）：mem_model_mib=系统自身模型'
                       '参数+buffer 常驻字节；mem_step_*＝每次步进前向的峰值瞬态'
                       '（基线 select 含可见子图 H2D，agentnet 首步含 per-graph '
                       '缓存构建；BAGR=lockstep 批 ≤chunk_size；full-graph 步=批）；'
                       'mem_judge_*＝终局分类批前向瞬态；'
                       'mem_resident_nonmodel_mib=结束时仍常驻的非模型显存'
                       '（已扣除系统模型与裁判分类器字节，agentnet 缓存落在'
                       '此列）',
        'cost_note': 'cost_stats=硬件无关计算量：revealed_node_pct=每图平均揭示'
                     '节点占比（不含 seed）；vis_edges_per_rollout=Σ_t|E_vis(t)|'
                     '（已揭示+frontier 边，不含构图自环与终局补充前向）；'
                     'vis_edge_ratio_pct=Σ|E_vis(t)|/(L_clf·|E|)，L_clf='
                     f'{clf_layers}（{ "自动取 classifier config" if classifier_path and not args.clf_layers else "命令行指定/缺省"}）',
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
                                    'n_rollouts': None, 's_per_rollout': None,
                                    'cost_stats': {
                                        'overall': {
                                            'revealed_node_pct': 100.0,
                                            'vis_edges_per_rollout': None,
                                            'vis_edge_ratio_pct': 100.0}}})
                elif mode in ('bagr', 'bagr-deploy'):
                    deploy = (mode == 'bagr-deploy')
                    clf = None
                    if not deploy:
                        clf = load_classifier_frozen(classifier_path, device)
                    reader = load_reader_from_checkpoint(
                        eval_resolve(args.bagr_ckpt), device,
                        stochastic=False, temperature=1.0)
                    handles += [h for h in (clf, reader) if h is not None]
                    # 步 = lockstep 批前向（≤chunk_size）；终局全状态批按
                    # 长度归 judge 桶；冻结分类器参照批 = judge（deploy 无）
                    reader.forward_states_evidence = rec.wrap(
                        reader.forward_states_evidence,
                        big_threshold=args.chunk_size)
                    reader.forward_states = rec.wrap(
                        reader.forward_states, big_threshold=args.chunk_size)
                    if clf is not None:
                        clf.forward = rec.wrap(clf.forward, bucket='judge')
                    metrics['n_params'] = _n_params(reader)
                    metrics['reader_kind'] = reader.config.kind
                    metrics['deploy'] = deploy
                    metrics['mem_model_mib'] = round(
                        _model_bytes(reader) / 2 ** 20, 2)
                    cost_stats = {}
                    _sync(device)
                    t0 = time.perf_counter()
                    results = evaluate_reader(
                        reader, clf, builder, graphs, labels, budget_options,
                        device, args.argmax, chunk_size=args.chunk_size,
                        n_seeds=args.n_seeds, cost_stats_out=cost_stats)
                    wall = time.perf_counter() - t0
                    acc_keys = {k: v for k, v in results.items()
                                if k.startswith('acc_')}
                    metrics.update({'wall_total_s': round(wall, 3),
                                    's_per_graph': wall / len(graphs),
                                    'n_rollouts': n_rollouts,
                                    's_per_rollout': wall / n_rollouts,
                                    'cost_stats': _cost_metrics(cost_stats,
                                                                clf_layers),
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
                    cost_stats = {}
                    _sync(device)
                    t0 = time.perf_counter()
                    results = evaluate_fixed(
                        None, clf, builder, graphs, labels, budget_options,
                        device, args.argmax, n_seeds=args.n_seeds,
                        agent=agent, judge=args.judge,
                        cost_stats_out=cost_stats)
                    wall = time.perf_counter() - t0
                    acc_keys = {k: v for k, v in results.items()
                                if k.startswith('acc_')}
                    metrics.update({'wall_total_s': round(wall, 3),
                                    's_per_graph': wall / len(graphs),
                                    'n_rollouts': n_rollouts,
                                    's_per_rollout': wall / n_rollouts,
                                    'cost_stats': _cost_metrics(cost_stats,
                                                                clf_layers),
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

        # ---- BAGR 单图序列趟（chunk=1，--mem-series-trip 显式开启）----
        # 时间/精度口径以主趟为准；本趟只补单图单步序列（绘图用）——
        # 2026-11-15 口径修订：跨系统比较不再用本趟（无固定底扣减、
        # 无 ×批量换算），统一用真实批量下的 mem_step_*
        if (args.mem_series_trip and 'bagr' in args.modes
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
            try:
                evaluate_reader(reader, clf, builder, graphs, labels,
                                budget_options, device, args.argmax,
                                chunk_size=1, n_seeds=args.n_seeds)
            finally:
                _cleanup(device, clf, reader)
            if rec_s.series:
                json_out['systems']['bagr'][split]['mem_series'] = \
                    _series_stats(rec_s.series, rec_s.terminal_n)
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
              f"{'模型MiB':>9}{'步峰MiB':>9}{'终判MiB':>9}"
              f"{'params':>10}{'揭示%':>8}{'边占比%':>9}")
        for mode, m in rows:
            acc = m.get('acc', m.get('acc_overall'))
            acc_s = f"{acc:.4f}" if acc is not None else '-'
            msr = (f"{m['s_per_rollout'] * 1000:.1f}"
                   if m.get('s_per_rollout') else '-')
            mm = m.get('mem_model_mib')
            mm_s = f"{mm:.1f}" if mm is not None else '-'
            ms_ = m.get('mem_step_max_mib', m.get('mem_step_mean_mib'))
            ms_s = f"{ms_:.2f}" if ms_ is not None else '-'
            mj = m.get('mem_judge_mean_mib')
            mj_s = f"{mj:.2f}" if mj is not None else '-'
            params = m.get('n_params')
            params_s = f"{params / 1e6:.2f}M" if params else '-'
            cstat = (m.get('cost_stats') or {}).get('overall', {})
            rev = cstat.get('revealed_node_pct')
            rev_s = f"{rev:.1f}" if rev is not None else '-'
            ver = cstat.get('vis_edge_ratio_pct')
            ver_s = f"{ver:.1f}" if ver is not None else '-'
            print(f"{mode:<12}{acc_s:>8}{m['s_per_graph'] * 1000:>9.1f}"
                  f"{msr:>12}{mm_s:>9}{ms_s:>9}{mj_s:>9}"
                  f"{params_s:>10}{rev_s:>8}{ver_s:>9}")
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

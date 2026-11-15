"""反向固定 BAGR 预测头评测各基线选点（rebuttal checklist 第 2 项）。

加载训练好的 BAGR-RL checkpoint（参数完全冻结，不重训），test rollout
时把每步选点替换为 Random / BFS / Max-Frontier-Degree / SAGPool /
AgentNet 选出的前沿节点，但每步仍经 forward_states_evidence 更新 BAGR
的 LSTM/Attention 证据状态，读满预算后用 BAGR 自带 Evidence 头终判——
与「统一使用冻结分类器」（eval_baselines clf 口径）构成正反双向控制
变量：所有选点方法共享同一个专为局部序列训练的预测头。

- 协议与 eval_reader 完全同构（lockstep / teleport / 确定性 seed /
  argmax）；无冻结分类器、无终局子图构建（证据头即终判，内存友好）；
- 每数据集一次运行评完全部 --agents，合并输出单个统一 schema JSON →
  checkpoint 同目录 eval_reverse.json（--output-dir 可覆盖）；
  各策略键 = reverse-{agent}，与主表 learnable 行（= BAGR 自己选点 +
  同一证据头）直接对照（eval_learnable.json 的 acc_*）；
- agentnet 需本机 baseline/artifacts/{ds}/agentnet.pt（ppa 缺省无此
  scorer，缺文件自动跳过并告警）。

用法（cifar10，4 seeds 与固定策略主表口径一致）:
  python code/train_reader/eval_reverse_reader.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --checkpoint result/cifar10/experiments/ev-R-read/evidence/rl/checkpoint_epoch6.pt

  # ppa（1 seed 对齐 ppa 主表；collab 加 --ref 无关，直接跑）
  python code/train_reader/eval_reverse_reader.py --dataset ppa \
      --root D:\\data\\ogbg_ppa --n-seeds 1 --chunk-size 16 \
      --checkpoint result/ppa/experiments/ev256/evidence/rl/checkpoint_epoch5.pt
"""

import argparse
import gc
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from datasets import get_dataset, release_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import CODE_DIR, REPO_ROOT, resolve  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (build_eval_json, evaluate_reverse,  # noqa: E402
                               load_reader_from_checkpoint, merge_eval_json)

_AGENTS = ('random', 'bfs', 'max-frontier-degree', 'sagpool', 'agentnet')


def _load_baseline_agent(mode, ckpt_path, device):
    """按 mode 构造协议适配基线（接口同 benchmark_inference_cost）。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if mode == 'sagpool':
        from baseline.sagpool import SagpoolStrategy
        agent = SagpoolStrategy()
    elif mode == 'agentnet':
        from baseline.agentnet import AgentnetStrategy
        agent = AgentnetStrategy()
    else:
        raise ValueError(mode)
    agent.load(str(ckpt_path), str(device))
    return agent


def main():
    parser = argparse.ArgumentParser(
        description='反向固定 BAGR 预测头评测各基线选点（证据头终判）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', required=True,
                        help='BAGR-RL reader checkpoint（冻结，结构按内嵌 config 重建）')
    parser.add_argument('--agents', nargs='+', default=list(_AGENTS),
                        choices=list(_AGENTS),
                        help='选点来源（默认全部五个；缺 scorer 的自动跳过）')
    parser.add_argument('--sagpool-ckpt', default='',
                        help='sagpool scorer（缺省 baseline/artifacts/{ds}/sagpool.pt）')
    parser.add_argument('--agentnet-ckpt', default='',
                        help='agentnet scorer（缺省 baseline/artifacts/{ds}/agentnet.pt）')
    parser.add_argument('--splits', nargs='+', default=['test'],
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 图数限制（0=全部）')
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--n-seeds', type=int, default=4,
                        help='每图确定性 seed 数（对齐各数据集主表：cifar10/collab=4，ppa=1）')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='跨图 lockstep 图批量（collab/ppa 建议 16）')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 checkpoint 同目录，文件名 eval_reverse.json）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    spec = get_dataset(args.dataset, root=args.root, splits=(),
                       max_graphs=args.max_graphs,
                       stratified=args.stratified).spec
    budget_options = spec.budget_options

    ckpt_path = resolve(args.checkpoint)
    reader = load_reader_from_checkpoint(ckpt_path, device,
                                         stochastic=False, temperature=1.0)
    if getattr(reader.config, 'kind', '') != 'evidence':
        raise SystemExit("反向评测需要 evidence reader checkpoint（终判 = 证据头）")
    print(f"Reader: kind=evidence hidden={reader.config.encoder_hidden_dims} "
          f"(frozen) <- {ckpt_path}")

    # ---- 策略工厂：per-rollout 独立实例（lockstep 交错推进） ----
    factories = {}
    for mode in args.agents:
        if mode in ('random', 'bfs', 'max-frontier-degree'):
            factories[mode] = (lambda m=mode: create_fixed_reader(m))
        else:
            ckpt = (getattr(args, f'{mode}_ckpt')
                    or REPO_ROOT / 'baseline' / 'artifacts' / spec.name
                       / f'{mode}.pt')
            if not Path(ckpt).exists():
                print(f"[skip] {mode} scorer 不存在: {ckpt}（跳过该选点来源）")
                continue
            base = _load_baseline_agent(mode, resolve(ckpt), device)
            if mode == 'sagpool':
                # select 无跨 rollout 状态：全部 rollout 共享同一实例
                factories[mode] = (lambda b=base: b)
            else:
                # agentnet 有 per-rollout 游走状态：clone_walk 共享模型/
                # per-graph 静态缓存，游走状态独立
                factories[mode] = (lambda b=base: b.clone_walk())
    if not factories:
        raise SystemExit("无可用选点来源（scorer 缺失且未指定 --*-ckpt）")

    out_dir = (resolve(args.output_dir) if args.output_dir
               else ckpt_path.parent)
    out_path = out_dir / 'eval_reverse.json'
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'dataset': spec.name,
        'mode': 'reverse-reader',
        'note': 'BAGR-RL evidence 头终判（冻结），选点动作替换为各策略；'
                '对照行 = 同 checkpoint 的 eval_learnable.json acc_*',
        'strategies': [f'reverse-{m}' for m in factories],
        'checkpoint': str(ckpt_path),
        'budgets': list(budget_options),
        'n_graphs': {},
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'chunk_size': args.chunk_size,
        'seed_method': SEED_METHOD_STR,
        'baseline_ckpts': {
            m: str(resolve(getattr(args, f'{m}_ckpt')
                           or REPO_ROOT / 'baseline' / 'artifacts'
                              / spec.name / f'{m}.pt'))
            for m in ('sagpool', 'agentnet') if m in factories},
    }

    parts = []
    evaluated = []
    for split in list(dict.fromkeys(args.splits)):
        ds = get_dataset(args.dataset, root=args.root, splits=(split,),
                         max_graphs=args.max_graphs, stratified=args.stratified)
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if len(graphs) == 0:
            print(f"[skip] 数据集 '{args.dataset}' 无 {split} 划分（或为空），跳过")
            release_dataset(ds)
            continue
        config['n_graphs'][split] = len(graphs)
        print(f"\n[{split}] 已加载 {len(graphs)} 图；"
              f"budgets={list(budget_options)} seeds={args.n_seeds}")

        for mode, factory in factories.items():
            results = evaluate_reverse(reader, factory, graphs, labels,
                                       budget_options, device,
                                       chunk_size=args.chunk_size,
                                       n_seeds=args.n_seeds)
            key = f'reverse-{mode}'
            parts.append(build_eval_json(results, split, key,
                                         args.n_seeds, budget_options,
                                         config={}))
            print(f"[{split}] {key}: "
                  + '  '.join(f"B{int(b * 100)}%="
                              f"{results[f'acc_{int(b * 100)}%']:.4f}"
                              for b in budget_options)
                  + f"  overall={results['acc_overall']:.4f}")
            # 增量落盘：后续策略/ split 崩溃不丢已完成结果
            json_out = merge_eval_json(parts)
            json_out['config'] = config
            save_json(json_out, out_path)

        evaluated.append(split)
        del graphs, labels
        release_dataset(ds)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not evaluated:
        raise SystemExit(f"数据集 '{args.dataset}' 所请求 split {args.splits} 全部为空")
    print(f"\nJSON saved to {out_path}")
    print(f"绘图: python code/comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()

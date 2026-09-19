"""固定策略 / baseline 基线评估（统一入口）。

自 train_reader/eval.py 拆分（2026-09-19）：baseline 评估与 reader 评估
各自独立（reader 走 eval_reader.py；旧 eval.py 保留为兼容分发 shim）。

覆盖两类策略：
- 固定策略：random / bfs / max-frontier-degree（reader/fixed/）；
- baseline/ 协议适配基线：random-walk / sagpool / subgraphx / softmask /
  agentnet（注册于 baseline/strategies.py；sagpool/softmask/agentnet 需
  scorer checkpoint，--strategy-ckpt 单值故每命令只跑一个）。

一次运行覆盖全部请求 split（默认 valid+test；无验证集的数据集如 cifar10
自动检测并跳过），结果合并为单个统一 schema JSON（comparison 脚本直接
可用）→ result/{ds}/fixed/eval_fixed.json（--output-dir 可覆盖）。

协议要点（AGENTS.md Protocol / baseline/protocol-adaptation.md）：
预算 = 揭示节点数（不含 seed），比例档取自 DatasetSpec（默认
0.05/0.10/0.15）；种子 = 确定性公式（tools/seeds）；前沿边特征揭示前
可见（one-ended，零预算）；frontier 节点特征置零（P4）；随机类基线
（random/random-walk）RNG 由 (seed_node, n_nodes) 播种，同一 (graph,
seed_i) 轨迹可复现（原则 4）；前沿空即结束（rollout 契约不瞬移，与
reader 侧跳转续读的口径差异见 protocol-adaptation.md）；协议终判 =
冻结上界分类器（P5），agentnet 自带头判定另记 acc_own_*（参照口径，
确定性 argmax 游走）。

用法:
  # 固定策略基线（一次跑 random+bfs × valid+test）
  python code/train_reader/eval_baselines.py --dataset cifar10 \
      --root D:\\data\\superpixels --strategy random bfs \
      --classifier <上界分类器>
  # 需 scorer 的基线（每命令一个）
  python code/train_reader/eval_baselines.py --dataset ppa \
      --root D:\\ogb-master\\dataset --strategy agentnet \
      --classifier <cls.pt> --max-graphs 2000
"""

import argparse
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (add_common_args, eval_resolve,  # noqa: E402
                               evaluate_fixed, load_classifier_frozen,
                               run_split_eval)


def main():
    parser = argparse.ArgumentParser(description='固定策略/baseline 基线评估')
    add_common_args(parser)
    parser.add_argument('--strategy', nargs='+', required=True,
                        choices=['random', 'bfs', 'max-frontier-degree',
                                 'random-walk', 'sagpool', 'subgraphx',
                                 'softmask', 'agentnet'],
                        help='固定策略名（可同时多个；'
                             'random-walk/sagpool/subgraphx/softmask/agentnet 为 '
                             'baseline/ 协议适配基线；softmask/agentnet 需 scorer '
                             'checkpoint，--strategy-ckpt 单值故每命令只跑一个）')
    parser.add_argument('--strategy-ckpt', default='',
                        help='sagpool scorer checkpoint'
                             '（默认 baseline/artifacts/{ds}/sagpool.pt）')
    parser.add_argument('--search-rollouts', type=int, default=20,
                        help='subgraphx MCTS rollout 数（对齐官方默认 20）')
    parser.add_argument('--search-expand', type=int, default=12,
                        help='subgraphx 每次扩展的子节点上限（官方 expand_atoms=12）')
    parser.add_argument('--search-max-evals', type=int, default=2048,
                        help='subgraphx 单次规划的唯一 coalition 评估数上限（算力上限）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # spec 只读探针：splits=() 的缓存项永不访问数据字段，不触发加载
    spec = get_dataset(args.dataset, root=args.root, splits=(),
                       max_graphs=args.max_graphs,
                       stratified=args.stratified).spec
    budget_options = spec.budget_options

    classifier = load_classifier_frozen(eval_resolve(args.classifier), device)
    print(f"Classifier: {args.classifier}")
    builder = SubgraphBuilder()

    print(f"Strategies: {' '.join(args.strategy)}")
    if any(s in ('random-walk', 'sagpool', 'subgraphx',
                 'softmask', 'agentnet') for s in args.strategy):
        # baseline/ 位于仓库根：注册协议适配基线策略
        import sys as _sys
        from tools.paths import REPO_ROOT
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))
        from baseline.strategies import register_baseline_strategies
        register_baseline_strategies()

    agents = []
    for strategy in args.strategy:
        if strategy == 'sagpool':
            from baseline.sagpool import SagpoolStrategy
            ckpt = (eval_resolve(args.strategy_ckpt) if args.strategy_ckpt
                    else CODE_DIR.parent / 'baseline' / 'artifacts'
                         / spec.name / 'sagpool.pt')
            if not ckpt.exists():
                raise SystemExit(
                    f"sagpool scorer checkpoint 不存在: {ckpt}\n"
                    f"先训练: python baseline/run_sagpool.py "
                    f"--dataset {spec.name} --root <root>")
            agent = SagpoolStrategy()
            agent.load(str(ckpt), device)
            print(f"SAGPool scorer: {ckpt}")
        elif strategy in ('softmask', 'agentnet'):
            if strategy == 'softmask':
                from baseline.softmask import SoftmaskStrategy
                agent = SoftmaskStrategy()
            else:
                from baseline.agentnet import AgentnetStrategy
                agent = AgentnetStrategy()
            ckpt = (eval_resolve(args.strategy_ckpt) if args.strategy_ckpt
                    else CODE_DIR.parent / 'baseline' / 'artifacts'
                         / spec.name / f'{strategy}.pt')
            if not ckpt.exists():
                raise SystemExit(
                    f"{strategy} scorer checkpoint 不存在: {ckpt}\n"
                    f"先训练: python baseline/run_{strategy}.py "
                    f"--dataset {spec.name} --root <root>")
            agent.load(str(ckpt), device)
            print(f"{strategy} scorer: {ckpt}")
        elif strategy == 'subgraphx':
            from baseline.subgraphx import SubgraphXStrategy
            agent = SubgraphXStrategy(
                n_rollout=args.search_rollouts,
                expand_atoms=args.search_expand,
                max_evals=args.search_max_evals)
            agent.configure(classifier, device)
            print(f"SubgraphX planner: rollouts={args.search_rollouts} "
                  f"expand={args.search_expand} "
                  f"max_evals={args.search_max_evals}（planner 口径，"
                  f"详见 baseline/protocol-adaptation.md §5）")
        else:
            agent = create_fixed_reader(strategy)
        agents.append(agent)

    out_dir = (eval_resolve(args.output_dir) if args.output_dir
               else CODE_DIR / 'result' / spec.name / 'fixed')
    out_path = out_dir / 'eval_fixed.json'
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'dataset': spec.name,
        'mode': 'fixed',
        'strategies': list(args.strategy),
        'strategy_or_checkpoint': ' '.join(args.strategy),
        'classifier': str(eval_resolve(args.classifier)),
        'budgets': list(budget_options),
        'n_graphs': {},          # 逐 split 累积
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'seed_method': SEED_METHOD_STR,
        'full_graph_ref': {},    # 逐 split 累积
    }

    def eval_split(split, graphs, labels):
        for strategy, agent in zip(args.strategy, agents):
            results = evaluate_fixed(None, classifier, builder, graphs, labels,
                                     budget_options, device, args.argmax,
                                     args.n_seeds, agent=agent)
            yield strategy, results

    run_split_eval(args, classifier, budget_options, config, out_path,
                   eval_split)
    print(f"JSON saved to {out_path}")
    print(f"绘图: python code/comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()

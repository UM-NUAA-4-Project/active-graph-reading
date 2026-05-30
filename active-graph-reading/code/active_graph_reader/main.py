"""
Active Graph Reader — 命令行入口
=================================

提供统一的 CLI 接口，支持命令行参数指定策略、成本模型、预算等。
也可作为 Python 模块导入，通过程序化 API 进行批量实验。

用法：
    # 命令行模式
    python -m active_graph_reader.main --strategy bfs --budget 50

    # 程序化模式
    from active_graph_reader.main import run_experiment
    stats = run_experiment(graph, strategy='bfs', budget=50)
"""

import argparse
from typing import Optional

import dgl

from .config import RevealConfig
from .dataloader.loader import PPADataset, get_graph_stats
from .engine.graph_state import GraphState
from .engine.reveal_loop import RevealLoop
from .strategies.registry import get_strategy, list_strategies
from .cost.registry import get_cost_model, list_cost_models
from .utils.logging import get_logger, set_verbosity

logger = get_logger()


# ============================================================================
# 实验运行（程序化 API）
# ============================================================================

def run_experiment(
    dgl_graph: dgl.DGLGraph,
    strategy: str = 'random',
    cost_model: str = 'unit',
    budget: Optional[int] = None,
    max_ratio: Optional[float] = None,
    seed_node: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """在单张图上运行揭示协议并返回统计信息。

    Parameters
    ----------
    dgl_graph : dgl.DGLGraph
        完整图（所有节点、边、特征已知）。
    strategy : str
        策略名称: 'random', 'bfs', 'max-frontier-degree'。
    cost_model : str
        成本模型名称: 'unit'（v0 默认）。
    budget : int or None
        揭示节点数上限。None 表示揭示所有可达节点。
    max_ratio : float or None
        揭示节点比例上限 (0~1)。
    seed_node : int or None
        种子节点 ID。None 随机选取。
    verbose : bool
        是否输出详细日志。

    Returns
    -------
    dict
        揭示统计信息。

    Raises
    ------
    ValueError
        如果参数不合法。
    """
    set_verbosity(verbose)

    # 构造配置并验证
    config = RevealConfig(
        strategy=strategy,
        cost_model=cost_model,
        budget=budget,
        max_ratio=max_ratio,
        seed_node=seed_node,
        verbose=verbose,
    )

    # 获取策略和成本模型实例（通过注册表）
    strat_instance = get_strategy(config.strategy)
    cost_instance = get_cost_model(config.cost_model)

    # 解析种子节点
    if config.seed_node is None:
        import random
        config.seed_node = random.randrange(dgl_graph.num_nodes())

    # 构建图状态（初始化 G_0）
    state = GraphState(dgl_graph, seed_node=config.seed_node)

    # 构建并运行揭示循环
    loop = RevealLoop(state, strat_instance, cost_instance)
    stats = loop.run(budget=config.budget, max_ratio=config.max_ratio)

    return stats


def run_benchmark(
    dataset: PPADataset,
    strategy: str = 'bfs',
    budget: Optional[int] = None,
    max_ratio: Optional[float] = None,
    num_graphs: int = 5,
    start_idx: Optional[int] = None,
    verbose: bool = True,
) -> list:
    """在多张图上批量运行揭示协议。

    Parameters
    ----------
    dataset : PPADataset
        OGBG-PPA 数据集。
    strategy : str
        策略名称。
    budget : int or None
        揭示节点数上限。
    max_ratio : float or None
        揭示节点比例上限。
    num_graphs : int
        运行的图数量（从 start_idx 开始）。
    start_idx : int or None
        起始图索引。None 使用训练集第 0 张。
    verbose : bool
        日志开关。

    Returns
    -------
    list of dict
        每张图的揭示统计数据。
    """
    if start_idx is None:
        indices = dataset.train_indices[:num_graphs]
    else:
        indices = list(range(start_idx, start_idx + num_graphs))

    results = []
    for idx in indices:
        g, label = dataset[idx]
        stats = run_experiment(
            g,
            strategy=strategy,
            budget=budget,
            max_ratio=max_ratio,
            verbose=verbose,
        )
        stats['graph_index'] = idx
        stats['label'] = label
        results.append(stats)

    return results


# ============================================================================
# 命令行入口
# ============================================================================

def demo_all_strategies(args):
    """演示模式：对一张图运行三种策略并打印对比。"""
    print("=" * 65)
    print("Active Graph Reader — M3 v0 Base 协议演示")
    print("数据集: OGBG-PPA")
    print("=" * 65)

    # ---- 加载数据集 --------------------------------------------------------
    print("\n[1] 加载 OGBG-PPA 数据集...")
    ds = PPADataset(root=args.root)

    # ---- 选取示例图 --------------------------------------------------------
    graph_idx = args.graph_index if args.graph_index is not None else ds.train_indices[0]
    g, label = ds[graph_idx]
    g_stats = get_graph_stats(g, label)

    print(f"\n[2] 选取图 #{graph_idx}")
    print(f"    节点数 |V|: {g_stats['n_nodes']}")
    print(f"    边数 |E|: {g_stats['n_edges']}")
    print(f"    标签: {g_stats['label']}")
    print(f"    边特征: {'有 (dim=' + str(g_stats.get('edge_feat_dim', '?')) + ')' if g_stats['has_edge_feat'] else '无'}")

    # ---- 对三种策略运行揭示循环 --------------------------------------------
    strategies = ['random', 'bfs', 'max-frontier-degree']
    seed = args.seed if args.seed is not None else 42

    budget_str = f"预算: {args.budget or '无限'}"
    if args.max_ratio:
        budget_str += f", 比例上限: {args.max_ratio}"
    print(f"\n[3] 揭示循环（种子节点: {seed}, {budget_str}）")
    print("-" * 65)

    results = {}
    for strat in strategies:
        stats = run_experiment(
            g,
            strategy=strat,
            budget=args.budget,
            max_ratio=args.max_ratio,
            seed_node=seed,
            verbose=args.verbose,
        )
        results[strat] = stats

        print(f"\n  策略: {strat}")
        print(f"    已揭示节点: {stats['n_revealed']}/{stats['n_total']} "
              f"({stats['reveal_ratio']:.2%})")
        print(f"    消耗预算: {stats['budget_used']}")
        print(f"    剩余前沿边: {stats['n_frontier_remaining']}")
        print(f"    已获边特征数: {stats['n_revealed_edge_feats']}")
        print(f"    揭示顺序 (前 10): {stats['reveal_history'][:10]}")

    # ---- 汇总对比 ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("[4] 汇总对比")
    print("-" * 65)
    print(f"  {'策略':<22} {'揭示节点':>10} {'揭示比例':>10} "
          f"{'前沿剩余':>10} {'边特征':>10}")
    print("  " + "-" * 62)
    for strat in strategies:
        s = results[strat]
        print(f"  {strat:<22} {s['n_revealed']:>10} "
              f"{s['reveal_ratio']:>9.2%} "
              f"{s['n_frontier_remaining']:>10} "
              f"{s['n_revealed_edge_feats']:>10}")

    print("\n完成。")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='Active Graph Reader — M3 v0 Base 协议',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 演示三种策略对比
  python -m active_graph_reader.main --demo

  # 指定策略 + 预算
  python -m active_graph_reader.main --strategy bfs --budget 50

  # 按比例揭示
  python -m active_graph_reader.main --strategy max-frontier-degree --max-ratio 0.10

  # 查看可用策略
  python -m active_graph_reader.main --list-strategies
        """,
    )

    # ---- 策略参数 ----------------------------------------------------------
    parser.add_argument('--strategy', type=str, default='random',
                        choices=['random', 'bfs', 'max-frontier-degree'],
                        help='前沿边揭示策略 (默认: random)')

    parser.add_argument('--cost-model', type=str, default='unit',
                        help='成本模型名称 (默认: unit)')

    # ---- 数据路径 ----------------------------------------------------------
    parser.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                        help='OGB 数据集根目录')

    # ---- 预算参数 ----------------------------------------------------------
    parser.add_argument('--budget', type=int, default=None,
                        help='揭示节点数上限（不含种子），默认无限制')

    parser.add_argument('--max-ratio', type=float, default=None,
                        help='揭示节点比例上限 (0~1)')

    # ---- 图与种子 ----------------------------------------------------------
    parser.add_argument('--graph-index', type=int, default=None,
                        help='OGBG-PPA 数据集中的图索引')

    parser.add_argument('--seed', type=int, default=None,
                        help='固定的种子节点 ID (默认: 42)')

    # ---- 运行模式 ----------------------------------------------------------
    parser.add_argument('--demo', action='store_true', default=False,
                        help='运行三种策略对比演示')

    parser.add_argument('--no-demo', action='store_true', default=False,
                        help='仅运行指定策略的单次揭示')

    parser.add_argument('--list-strategies', action='store_true',
                        help='列出所有可用的策略和成本模型')

    parser.add_argument('--verbose', action='store_true', default=True,
                        help='输出详细日志 (默认)')

    parser.add_argument('--quiet', action='store_true',
                        help='静默模式')

    return parser


def main():
    """主入口函数。"""
    parser = build_parser()
    args = parser.parse_args()

    # ---- 列出策略 / 成本模型 ------------------------------------------------
    if args.list_strategies:
        print("可用策略:")
        for name, cls in sorted(list_strategies().items()):
            print(f"  {name:<25} → {cls.__name__}")
        print("\n可用成本模型:")
        for name, cls in sorted(list_cost_models().items()):
            print(f"  {name:<25} → {cls.__name__}")
        return

    # ---- 静默模式 ----------------------------------------------------------
    if args.quiet:
        args.verbose = False

    # ---- 演示模式（三种策略对比）--------------------------------------------
    # 规则：显式 --demo 或未指定任何单策略参数 → 进入演示模式。
    # 若给出 --strategy（非默认）、--budget、--max-ratio 或 --no-demo，
    # 则进入单策略模式。
    has_explicit_action = (
        args.demo
        or args.no_demo
        or args.budget is not None
        or args.max_ratio is not None
        or args.list_strategies
    )
    if args.demo or not has_explicit_action:
        demo_all_strategies(args)
        return

    # ---- 单策略模式 --------------------------------------------------------
    ds = PPADataset(root=args.root)
    graph_idx = args.graph_index if args.graph_index is not None else ds.train_indices[0]
    g, label = ds[graph_idx]

    stats = run_experiment(
        g,
        strategy=args.strategy,
        cost_model=args.cost_model,
        budget=args.budget,
        max_ratio=args.max_ratio,
        seed_node=args.seed,
        verbose=args.verbose,
    )

    print(f"\n策略: {stats['strategy']}, 成本模型: {stats['cost_model']}")
    print(f"图 #{graph_idx}: |V|={stats['n_total']}, |E|={stats['n_edges_total']}")
    print(f"揭示: {stats['n_revealed']}/{stats['n_total']} "
          f"({stats['reveal_ratio']:.2%})")
    print(f"预算: {stats['budget_used']}/{stats['budget_limit']}")
    print(f"前沿剩余: {stats['n_frontier_remaining']}")
    print(f"边特征: {stats['n_revealed_edge_feats']}")
    print(f"揭示顺序: {stats['reveal_history']}")


if __name__ == '__main__':
    main()

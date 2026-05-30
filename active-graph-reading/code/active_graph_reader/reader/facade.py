"""Reader 工具函数：reveal_one / evaluate / create_dataloader。"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import dgl
from tqdm import tqdm

from .base_reader import BaseReader
from ..engine.graph_state import GraphState
from ..engine.reveal_loop import RevealLoop
from ..cost.registry import get_cost_model
from ..encoder.subgraph_builder import SubgraphBuilder


def run_reveal(
    reader: BaseReader,
    full_graph: dgl.DGLGraph,
    budget: int,
    seed_node: Optional[int] = None,
    cost_model: str = 'unit',
) -> GraphState:
    """Run the reveal protocol and return the final GraphState."""
    if seed_node is None:
        seed_node = random.randrange(full_graph.num_nodes())
    cost = get_cost_model(cost_model)
    reader.reset()
    state = GraphState(full_graph, seed_node=seed_node)
    loop = RevealLoop(state, reader, cost)
    loop.run(budget=budget)
    return state


def reveal_one(
    reader: BaseReader,
    full_graph: dgl.DGLGraph,
    budget: int,
    seed_node: Optional[int] = None,
    cost_model: str = 'unit',
) -> Tuple[dgl.DGLGraph, List[int]]:
    """单图 reveal + build 最终子图（仅 V_revealed + revealed edges）。"""
    builder = SubgraphBuilder()
    state = run_reveal(reader, full_graph, budget, seed_node, cost_model)
    return builder.build(state)


def reveal_batch(
    reader: BaseReader,
    full_graphs: List[dgl.DGLGraph],
    budgets: List[int],
    seed_nodes: Optional[List[int]] = None,
    cost_model: str = 'unit',
) -> Tuple[dgl.DGLGraph, List[List[int]]]:
    """多图各自 reveal 后 batch。"""
    if seed_nodes is None:
        seed_nodes = [random.randrange(g.num_nodes()) for g in full_graphs]

    builder = SubgraphBuilder()
    states = []
    for g, b, s in zip(full_graphs, budgets, seed_nodes):
        state = run_reveal(reader, g, b, s, cost_model)
        states.append(state)

    return builder.build_batch(states)


@torch.no_grad()
def evaluate(
    reader: BaseReader,
    dataset,
    indices: List[int],
    classifier,
    budgets_rel: List[float],
    seeds_per_graph: int,
    device: torch.device,
    batch_size: int = 8,
    cost_model: str = 'unit',
) -> Dict[float, List[float]]:
    """完整评测：增量 reveal + 批量 GPU 推理。"""
    budgets_rel_sorted = sorted(budgets_rel)
    results: Dict[float, List[float]] = defaultdict(list)

    cost = get_cost_model(cost_model)
    builder = SubgraphBuilder()

    for batch_start in tqdm(
        range(0, len(indices), batch_size),
        desc=reader.name, leave=False,
    ):
        batch_indices = indices[batch_start:batch_start + batch_size]

        subgraphs: List[dgl.DGLGraph] = []
        meta: List[Tuple[int, float, int]] = []

        for batch_i, idx in enumerate(batch_indices):
            g_raw, label = dataset[int(idx)]
            if hasattr(label, 'item'):
                label = label.item()
            n_nodes = g_raw.num_nodes()
            budgets_abs = [max(1, int(r * n_nodes)) for r in budgets_rel_sorted]

            for seed_i in range(seeds_per_graph):
                seed_node = (idx * 31 + seed_i * 17 + 7) % max(n_nodes, 1)

                state = GraphState(g_raw, seed_node=seed_node)
                loop = RevealLoop(state, reader, cost)

                for abs_b, rel_b in zip(budgets_abs, budgets_rel_sorted):
                    loop.run(budget=abs_b)
                    sub_g, _ = builder.build(state)
                    subgraphs.append(sub_g)
                    meta.append((batch_i, rel_b, label))

        batch_g = dgl.batch(subgraphs).to(device)
        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = classifier(batch_g, node_ids)
        preds = logits.argmax(dim=1).cpu().tolist()

        graph_budget_correct: Dict[Tuple[int, float], List[float]] = defaultdict(list)
        for flat_i, (batch_i, rel_b, label) in enumerate(meta):
            graph_budget_correct[(batch_i, rel_b)].append(
                1.0 if preds[flat_i] == label else 0.0
            )

        for (_batch_i, rel_b), corrects in graph_budget_correct.items():
            results[rel_b].append(np.mean(corrects))

    return dict(results)


def create_dataloader(
    reader_factory,
    dataset,
    indices: List[int],
    budget_ratios: Optional[List[float]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seeds_per_graph: int = 1,
    shuffle: bool = True,
    cost_model: str = 'unit',
) -> torch.utils.data.DataLoader:
    """创建训练用 DataLoader。reader_factory 为无参数 callable，返回 BaseReader 实例。"""
    from torch.utils.data import DataLoader
    from ..dataloader.reader_dataset import ReaderDataset

    ds = ReaderDataset(
        dataset=dataset,
        indices=indices,
        budget_ratios=budget_ratios or [0.05, 0.10, 0.20, 0.50],
        seeds_per_graph=seeds_per_graph,
        cost_model=cost_model,
        shuffle_indices=shuffle,
        reader_factory=reader_factory,
    )

    def _collate(batch):
        subgraphs, labels = zip(*batch)
        batched_g = dgl.batch(list(subgraphs))
        label_t = torch.tensor(labels, dtype=torch.long)
        return batched_g, label_t

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate, pin_memory=True,
    )

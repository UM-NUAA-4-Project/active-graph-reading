"""Reader Dataset（IterableDataset）：多进程 DataLoader 支持。"""

import random
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset

from ..reader.facade import run_reveal
from ..cost.registry import get_cost_model
from ..encoder.subgraph_builder import SubgraphBuilder


class ReaderDataset(IterableDataset):
    """Reader-based graph dataset for training with DataLoader.

    每个 worker 独立创建 reader/builder 实例。
    迭代时：随机选 budget ratio → 随机选 seed → reveal → build 最终子图 → yield。
    """

    def __init__(
        self,
        dataset,
        indices: List[int],
        budget_ratios: Optional[List[float]] = None,
        seeds_per_graph: int = 1,
        cost_model: str = 'unit',
        shuffle_indices: bool = True,
        seed: Optional[int] = None,
        reader_factory=None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.budget_ratios = budget_ratios or [0.05, 0.10, 0.20, 0.50]
        self.seeds_per_graph = seeds_per_graph
        self.cost_model_name = cost_model
        self.shuffle_indices = shuffle_indices
        self.base_seed = seed if seed is not None else 42
        self.reader_factory = reader_factory

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = max(1, len(self.indices) // num_workers)
            start = worker_id * per_worker
            end = start + per_worker if worker_id < num_workers - 1 else len(self.indices)
            worker_indices = self.indices[start:end]
            worker_seed = self.base_seed + worker_id * 10000
        else:
            worker_indices = self.indices
            worker_seed = self.base_seed

        if self.reader_factory is not None:
            reader = self.reader_factory()
        else:
            from ..reader import FixedStrategyReader
            reader = FixedStrategyReader('random')
        builder = SubgraphBuilder()

        rng = random.Random(worker_seed)

        epoch_indices = list(worker_indices)
        if self.shuffle_indices:
            rng.shuffle(epoch_indices)

        for idx in epoch_indices:
            full_g, label = self.dataset[int(idx)]
            if hasattr(label, 'item'):
                label = label.item()
            n_nodes = full_g.num_nodes()

            for _ in range(self.seeds_per_graph):
                ratio = rng.choice(self.budget_ratios)
                budget = max(1, int(ratio * n_nodes))
                seed_node = rng.randint(0, n_nodes - 1)

                state = run_reveal(reader, full_g, budget, seed_node, self.cost_model_name)
                sub_g, _ = builder.build(state)
                yield sub_g, label

    def __len__(self):
        return len(self.indices) * self.seeds_per_graph

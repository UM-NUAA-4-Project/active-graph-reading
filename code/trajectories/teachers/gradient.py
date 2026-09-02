"""T1: 梯度归因 Teacher — 冻结分类器反向梯度 → 节点重要性（自 m4 迁入）。"""

import logging
from typing import Optional

import torch

from .base import Teacher

logger = logging.getLogger(__name__)


class GradientTeacher(Teacher):
    """T1 (gradient): 选使 classifier loss 下降最快的 frontier node。

    在全图上做反向传播，计算每层 conv 输出对各节点 embedding 的梯度范数，
    跨层聚合得到 per-node importance score。
    """

    def __init__(self):
        self._hooks = []

    @property
    def name(self) -> str:
        return 'gradient'

    def score_nodes(
        self,
        graph,
        classifier,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        if label is None:
            raise ValueError("GradientTeacher 需要 label 以计算正确类别的梯度")

        device = next(classifier.parameters()).device
        graph = graph.to(device)

        # 捕获每层 GINELayer 的输出激活
        activations = []

        def make_hook(act_list):
            def hook(module, _input, output):
                output.retain_grad()
                act_list.append(output)
            return hook

        encoder = classifier.encoder
        hooks = []
        for conv in encoder.convs:
            hooks.append(conv.register_forward_hook(make_hook(activations)))

        try:
            # 前向
            node_ids = torch.zeros(graph.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(graph, node_ids)

            classifier.zero_grad()
            correct_logit = logits[0, label]
            correct_logit.backward()

            # 聚合每层梯度范数
            n_nodes = graph.num_nodes()
            node_scores = torch.zeros(n_nodes, device=device)
            for act in activations:
                grad = act.grad
                if grad is not None:
                    node_scores += grad.norm(dim=1)

            # 防止全零（所有节点分数相同 → uniform）
            score_sum = node_scores.sum()
            if score_sum == 0:
                node_scores = torch.ones(n_nodes, device=device) / n_nodes
        finally:
            for h in hooks:
                h.remove()

        return node_scores.detach().cpu()

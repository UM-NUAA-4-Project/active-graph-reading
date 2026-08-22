"""Shared utilities for m6/attention (AttentionReader) training loops."""

from typing import List

import torch

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


def strip_classifier_keys(state_dict: dict) -> dict:
    """Drop legacy belief_state.classifier.* keys so the headless reader can
    strict-load checkpoints trained under belief/ (which had a ClassificationHead)."""
    return {k: v for k, v in state_dict.items()
            if not k.startswith('belief_state.classifier.')}


def reveal_and_update_belief(
    reader,
    state: GraphState,
    chosen_node: int,
    all_beliefs: List[torch.Tensor],
    all_subgraphs: list,
    idx: int,
    device: torch.device,
):
    """Reveal chosen_node, build new subgraph, encode via NewInfoEncoder, update belief.

    The subgraph is stored in all_subgraphs[idx] so forward_states can reuse it
    in the next lockstep iteration (avoiding double construction).
    """
    reveal_node(state, chosen_node)
    subgraph = reader.frontier_builder.build(state)
    new_info = reader.encode_new_info(state, subgraph, chosen_node, device)
    all_beliefs[idx] = reader.update_belief(all_beliefs[idx], new_info)
    all_subgraphs[idx] = subgraph


def reveal_and_update_beliefs_batch(
    reader,
    states: list,
    chosen_nodes: list,
    beliefs: torch.Tensor,
    device: torch.device,
    state_src: list = None,
):
    """Batch version: reveal all, build all subgraphs, batch-encode, batch-update.

    Args:
        reader:       AttentionReader
        states:       List[GraphState] (N_active,)
        chosen_nodes: List[int] (N_active,)
        beliefs:      torch.Tensor (N_active, m, d)  already stacked
        device:       torch.device
        state_src:    Optional[(图 index, seed_i)] 列表，与 states 平行（诊断用）

    Returns:
        new_beliefs:   torch.Tensor (N_active, m, d)
        new_subgraphs: List[FrontierSubgraph] (N_active,)
    """
    for state, node in zip(states, chosen_nodes):
        reveal_node(state, node)

    new_subgraphs = [reader.frontier_builder.build(s) for s in states]

    new_infos = reader.encode_new_info_batch(
        states, new_subgraphs, chosen_nodes, device, state_src=state_src,
    )  # (N_active, 1, d)

    new_beliefs = reader.belief_state.update_fn(beliefs, new_infos)

    return new_beliefs, new_subgraphs

"""Shared utilities for belief-aware training loops."""

from typing import List

import torch

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


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
):
    """Batch version: reveal all, build all subgraphs, batch-encode, batch-update.

    Args:
        reader:       LearnableReader
        states:       List[GraphState] (N_active,)
        chosen_nodes: List[int] (N_active,)
        beliefs:      torch.Tensor (N_active, m, d)  already stacked
        device:       torch.device

    Returns:
        new_beliefs:   torch.Tensor (N_active, m, d)
        new_subgraphs: List[FrontierSubgraph] (N_active,)
    """
    # Step 1: Reveal all nodes
    for state, node in zip(states, chosen_nodes):
        reveal_node(state, node)

    # Step 2: Build all subgraphs
    new_subgraphs = [reader.frontier_builder.build(s) for s in states]

    # Step 3: Batch-encode all new info (one GNN invocation)
    new_infos = reader.encode_new_info_batch(
        states, new_subgraphs, chosen_nodes, device,
    )  # (N_active, 1, d)

    # Step 4: Batch-update all beliefs
    new_beliefs = reader.belief_state.update_fn(beliefs, new_infos)

    return new_beliefs, new_subgraphs

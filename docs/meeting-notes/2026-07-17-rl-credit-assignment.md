# Meeting Notes: 2026-07-17 - RL Credit Assignment

**Date**: 2026-07-17
**Participants**: Meng, Zhihui
**Topic**: Why binary REINFORCE remains below behavior cloning (BC), and the next bounded RL ablation.
**Source**: `260717_1101.vtt`, with `260717_1101_spk.txt` used for speaker attribution.

---

## Summary

Matched held-out results establish the current ordering:

```text
BC > binary REINFORCE > tanh REINFORCE
```

Binary reward recovers most of the gap to BC, but no RL setting has yet improved the Reader. At the 15% test budget, BC reaches `14.15%`, binary RL reaches `13.70%`, and tanh RL reaches `11.85%`.

The immediate problem is credit assignment and update stability, not model capacity. Binary reward gives successful trajectories `+1` and failed trajectories `0`; with about `12.5%` training success, roughly seven out of eight sampled trajectories contribute no policy-gradient signal. The next experiments must isolate reward contrast and the number of trajectories per optimizer update.

## Results Reviewed

### Held-out accuracy

| Method | Validation 5% | Validation 10% | Validation 15% | Test 5% | Test 10% | Test 15% |
|---|---:|---:|---:|---:|---:|---:|
| BC | 11.07% | 12.49% | 13.90% | 10.60% | 12.57% | 14.15% |
| Binary REINFORCE | 10.83% | 12.26% | 13.40% | 10.46% | 12.41% | 13.70% |
| Tanh REINFORCE | 10.33% | 11.96% | 12.76% | 10.14% | 11.28% | 11.85% |

All sparse-reading results remain far below the frozen full-graph classifier's `64.19%` validation reference. The current classifier's sparse-subgraph calibration and class-attractor behavior remain a separate risk; this meeting focuses on the immediate RL update path.

### Training dynamics

For binary REINFORCE, loss falls from about `5.83` to `1.67`, while reward and training accuracy remain near `0.11-0.125`. After roughly epoch 4, lower loss does not correspond to more successful trajectories. The checkpoint used for held-out comparison was selected around epoch 4.

The tanh run reports roughly `65%` training accuracy but only `10-13%` held-out accuracy. This discrepancy is unresolved and must be audited by checking the checkpoint, graph split, labels, budget sampling, and metric implementation. It should not be used to explain the binary result without evidence.

## Decisions

1. Keep BC as the current best validated Reader.
2. Treat binary REINFORCE as encouraging but not as a method improvement.
3. Compare binary reward `{+1, 0}` with asymmetric reward `{+1, -0.1}`.
4. Compare `32` and `128` trajectories per optimizer update without changing reward in the same ablation.
5. Record per-graph rollout success, normalized action entropy, trajectory similarity, and the actual training configuration with each run.
6. Select checkpoints on validation only; use the existing held-out test protocol only after a validation improvement.
7. Time-box RL tuning. If no matched held-out result beats BC, retain BC and move to the backup method rather than extending reward sweeps indefinitely.

## Proposed Ablations

### 1. Asymmetric reward

```text
binary:       R = +1.0 if correct,  0.0 if incorrect
asymmetric:   R = +1.0 if correct, -0.1 if incorrect
```

The asymmetric variant explicitly moves the policy away from failed sampled actions. It is a practical contrast ablation, not an exactly centered advantage. A reward baseline or within-graph advantage remains the more principled next step if this variant is inconclusive.

### 2. Optimizer update size

The desired comparison is:

```text
32 trajectories/update
128 trajectories/update
```

The 128 setting can be implemented as four 32-trajectory minibatches with gradient accumulation. The effective trajectory count, graphs per batch, rollouts per graph, accumulation steps, and `optimizer.step()` frequency must be saved with the checkpoint; `8 x 4`, `16 x 2`, and `32 x 4` have different within-graph rollout structures even when their total counts are related.

## Required Diagnostics

- Per-graph initial rollout success-rate distribution: easy, hard, and informative middle-difficulty graphs.
- Normalized action entropy, `H(pi) / log(|frontier|)`, rather than raw entropy alone.
- Fixed-start trajectory similarity: keep the initial node fixed and vary only action sampling randomness.
- Per-class accuracy and largest-class baseline to monitor sparse-subgraph attractor behavior.
- A configuration manifest for every result: code commit, data split, classifier checkpoint, Reader checkpoint, reward, temperature, batch construction, and validation checkpoint-selection rule.

## Post-meeting Repository Follow-up

As of the 2026-07-19 repository state:

- `--reward-type asymmetric` is implemented as `{+1, -0.1}`, but no asymmetric held-out result is committed.
- A binary-policy diversity analysis covers 1,000 train graphs, 10 rollouts per graph, 10% budget, and temperature `0.2`; it reports mean final-node-set Jaccard `0.311 +/- 0.274` over 45,000 pairs.
- That Jaccard result mixes distinct initial seed nodes with stochastic action choices, so it shows that final subgraphs differ but does not by itself establish a good exploration/exploitation balance. The fixed-start diagnostic above is still required.
- Batched GNN encoding and CPU-resident graph loading were added. The current training defaults are `32 graphs x 4 rollouts = 128 trajectories` per batch, but the code still calls `optimizer.step()` once per batch rather than implementing explicit multi-minibatch gradient accumulation. A matched result must state the actual command and update semantics.

## Next Gate

Produce a validation-selected comparison table:

```text
reward:       binary {+1, 0} vs asymmetric {+1, -0.1}
update size:  32 vs 128 trajectories per optimizer step
budgets:      5% / 10% / 15%
metrics:      validation accuracy, reward, loss, normalized entropy,
              fixed-start similarity, per-graph success, per-class accuracy
```

Run the held-out test only for settings that improve over BC on validation under the same budget and seed protocol.

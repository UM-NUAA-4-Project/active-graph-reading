# Meeting Notes: 2026-07-03 - Base Reader RL Sync

**Date**: 2026-07-03
**Participants**: 瞿锰 (Meng), 黄智辉 (Zhihui)
**Topic**: Base Reader RL training contract, reward design, sampling, and next evaluation gate.
**Source**: `260703_1607.vtt`, with `260703_1607_spk.txt` used for weak speaker attribution.

---

## Summary

This sync moved the project focus from "can the Base Reader / RL code run" to "does RL actually improve the Reader under held-out evaluation."

The Base Reader now has a usable RL training path and directionally reasonable train-side logs, but RL is not yet a validated method improvement because validation / test accuracy for RL checkpoints has not been compared against the BC Reader.

The main decision was:

```text
Finish the Base Reader RL loop first.
Only then return to more complex POMDPReader / LSTM architecture work.
```

## Current RL Status

The current RL objective uses an episode-level reward:

```text
R = tanh(1 - CE / log(37))
loss = -R * sum_t w_t * log pi_t(a_t)
```

Training-side signals are directionally reasonable:

- training loss decreases overall;
- average reward rises from roughly `0.3351` to `0.3516` in the early / middle epochs;
- training accuracy moves consistently with reward;
- loss, reward, and accuracy are not perfectly aligned, which is expected in REINFORCE-style optimization.

The missing piece is validation:

- the RL checkpoint has not yet been shown to beat the BC Reader on held-out graphs;
- training-side reward alone is not enough to claim RL improves the Reader;
- the next result must compare BC vs RL under the same budget and evaluation contract.

## Reward Design

The current reward is likely too smooth.

`tanh(1 - CE / log(37))` can compress meaningful CE differences into very small reward differences. In the meeting, Meng estimated that CE changes around `0.3 -> 0.6` only move reward from about `0.72 -> 0.68`.

This may make REINFORCE weak:

- good and bad trajectories receive similar rewards;
- sampled action differences are not strongly credited;
- policy gradients have low useful contrast.

Next reward tests:

```text
R = 1 if final classifier prediction is correct
R = 0 otherwise
```

Compare this binary final-correctness reward against the current CE-shaped reward. Keep the rest of the RL loss fixed while testing reward alternatives.

## Rollout Construction

The team clarified the difference between BC and RL.

Behavior cloning:

```text
teacher trajectory prefix -> Reader predicts teacher's next action
```

Reinforcement learning:

```text
Reader samples the full trajectory itself
final subgraph is scored by the classifier
reward is assigned to the sampled trajectory
policy is updated from sampled-action log probabilities
```

RL should not reuse teacher trajectories as the trajectory path. It should roll out the current Reader policy, score the final subgraph, and update from those sampled actions.

This matters because the current Base Reader already appears better than some older teacher / heuristic trajectories. If the student is better than the teacher, continued teacher imitation can become a bottleneck.

## Sampling Temperature

Zhihui reported that temperature sampling had been tried and produced negative rewards. The tested temperature was about `1`.

Meng's interpretation:

- `temperature = 1` makes the frontier-action distribution too flat;
- flat sampling explores too many low-quality actions;
- this can produce bad trajectories and negative rewards.

Next experiment:

- sweep lower temperatures in the `0.x` range;
- keep sampling near argmax but not fully deterministic;
- compare `argmax`, `0.1`, `0.3`, `0.5`, and `1.0`.

Desired behavior:

```text
low temperature -> trajectories mostly similar to argmax
small stochastic differences -> enough variation to learn which actions matter
```

If two sampled trajectories differ in one or two actions and receive different rewards, the model can learn which actions caused the reward difference.

## Batch Construction

Current batch construction appears close to:

```text
32 trajectories = 32 graphs * 1 sampled trajectory each
```

Suggested alternative:

```text
32 trajectories = 8 graphs * 4 sampled trajectories each
```

Why this may help:

- multiple rollouts for the same graph make action differences easier to compare;
- graph-level difficulty is partially controlled;
- this resembles RL / post-training practice where multiple rollouts are sampled per task or prompt;
- reward differences may become more informative.

## Sparse-Subgraph Classifier Bias

Zhihui showed a 5,000-graph validation subset analysis after RL training.

Observed issues:

- many prediction changes move toward wrong labels;
- only a small subset of graphs become stably correct after some budget;
- CE remains large and reward remains sparse / weak;
- the frozen classifier often maps sparse subgraphs into a few attractor classes, especially classes such as `6` and `17`;
- the confusion matrix is computed as true class vs classifier prediction from the revealed subgraph.

Meng requested a class-prior baseline:

```text
If we always predict the largest class, what accuracy do we get?
```

Zhihui checked that the largest training-set class is around `15%`, while current Reader results are around the `11-14%` range depending on setting. This means class-prior / attractor-class behavior is a real risk and should be measured explicitly.

This does not block the immediate RL work, but it should be part of the next diagnostic table.

## Decisions

1. Treat current RL logs as incomplete until held-out validation / test evaluation is done.
2. Tune reward design; the current CE-shaped reward is probably too smooth.
3. Add a binary final-correctness reward baseline.
4. Make RL rollouts self-generated by the Reader policy.
5. Sweep sampling temperatures below `1`.
6. Try multi-rollout-per-graph batches.
7. Measure always-predict-largest-class accuracy to quantify class-prior / attractor-class bias.
8. Finish Base Reader RL before returning to complex Reader architecture.

## Action Items

| Owner | Action | Timing |
|---|---|---|
| Zhihui | Compare RL checkpoints against the BC Reader on validation / test under matched budgets | Immediate |
| Zhihui | Implement at least one higher-contrast reward, including binary final-correctness reward | Immediate |
| Zhihui | Ensure RL trajectories are sampled by the current Reader policy, not reused from teacher trajectories | Immediate |
| Zhihui | Sweep temperature below `1`, starting with several `0.x` values | Next experiment |
| Zhihui | Try `8` graphs times `4` rollouts and compare with `32` graphs times `1` rollout | Next experiment |
| Zhihui | Compute always-predict-largest-class baseline and compare it with Reader accuracy | Next analysis |
| Meng | Review the next compact RL comparison table | Next sync |

## Next Gate

Produce a compact table with:

```text
BC Reader
RL with current CE-shaped reward
RL with binary final-correctness reward
temperature sweep: argmax / 0.1 / 0.3 / 0.5 / 1.0
batching: 32 graphs * 1 rollout vs 8 graphs * 4 rollouts
always-predict-largest-class baseline
```

Required metrics:

- validation accuracy at `5%`, `10%`, and `15%` budgets;
- test accuracy if validation improves;
- average reward;
- final classifier CE;
- class-prior / attractor-class diagnostics.

Do not treat RL as solved until the RL Reader beats the BC Reader on held-out graphs under matched evaluation.

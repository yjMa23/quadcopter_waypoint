# P8A Checkpoint Selection and Policy Drift Diagnosis

P8A keeps the P6C environment, reward, observation, action, termination, and contact semantics frozen. It does not retrain.

## Result

Existing checkpoint reselection produced a positive P8A result.

- BC epoch 0 settled landing: 86.20%
- Metric-selected BC+PPO settled landing: 91.67%
- Improvement over BC: 5.47 percentage points
- Metric-selected hard-contact change: 0.30 percentage points
- Reaches 90%: True
- Reaches 92%: False
- Reward-selected actor equals metric-selected actor for every train seed: False

## Protocol

1. Screening: eval seed 145, 64 episodes, all epoch-0/10/.../200 and reward-selected checkpoints.
2. Validation: seeds 145/146/147, 128 episodes per seed, screening Top-5 plus reward-selected and BC.
3. Independent test: seeds 245/246/247, 256 episodes per seed, teacher, BC, validation-selected, and reward-selected policies.
4. Drift: deterministic checkpoint-specific normalized actions on the frozen P7 test split.

## Interpretation

Policy-drift correlations are observational and are not causal claims. The measured association is moderate, mainly through lower settled landing and higher deck miss as action drift grows. The earliest saved PPO snapshot is epoch 10, so P8A can determine whether degradation is already visible by epoch 10, but cannot directly observe the first gradient update.

## P8B Recommendation

Checkpoint reselection is sufficient for the positive P8A result, but the selected policies still do not reach 92% and drift continues after early PPO updates. For further improvement, use actor-preserving PPO: separate actor/critic handling, critic warm-up, temporary actor freezing, a BC actor anchor (KL or L2), and validation settled landing for checkpoint selection. Do not modify the frozen environment reward merely to improve this benchmark.

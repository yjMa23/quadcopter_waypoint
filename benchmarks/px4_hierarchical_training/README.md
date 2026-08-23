# PX4-Compatible Hierarchical RL Training Evidence

This directory is the reproducible evidence package for M2 training and evaluation. It is separate from `benchmarks/px4_hierarchical_smoke/`, which only proves the action/controller interface.

## Frozen prerequisites

- M0/M1 Direct action semantics, rewards, success/contact contracts, checkpoints, and historical benchmarks remain unchanged.
- Action-interface baseline commit: `ca974ee5118f8742af69a698a1c47a96aa7d0a9f`.
- M0/M1 action space remains 4; M2 action space is 3.

## Stage 0 regression

Post-evaluator implementation regression:

- `python -m pytest -q tests`: `116 passed`, `21 subtests passed`.
- 1-env PX4 hierarchical smoke: PASS.
- 16-env PX4 hierarchical smoke: PASS.
- NaN/Inf: 0.
- deterministic basic smoke ground crash: 0.

## Stage 1 evaluator diagnostics

M2 now latches per-episode diagnostics before automatic reset:

- relative velocity reference norm mean/P95/max;
- reference saturation ratio;
- controller velocity tracking error mean/max;
- acceleration/tilt/thrust/body-rate/moment saturation ratios;
- max desired tilt, max body rate, max applied moment;
- synchronized evaluator controller runtime mean/P95/max;
- normalized action component mean/std/abs-max.

The evaluator activates these fields only when the complete optional M2 latch contract is present, preserving M0/M1 output compatibility.

## Stage 2 zero-relative-action baseline

`zero_action_16env.json` records 16 deterministic completed episodes for each of static deck, constant-XY deck, heave deck, and physical-deck-attitude.

Semantics:

```text
normalized action = [0, 0, 0]
-> deck-relative velocity reference = 0
-> world velocity reference = deck contact-point velocity
```

This is a deck contact-point velocity-following baseline, not zero thrust and not an RL method.

Observed in all four cases:

- timeout rate = 1.0;
- contact / settled landing / hard contact / ground crash / deck miss = 0;
- relative velocity reference norm = 0;
- reference saturation = 0;
- every controller saturation ratio = 0.

Therefore any later alignment, descent, touchdown timing, or landing success must be produced by the learned policy rather than by the zero-action baseline.

## Stage 3 PPO sanity

The required `seed=42 / num_envs=64 / max_iterations=30` sanity run completed, but the gate result is:

```text
SANITY FAIL
```

Key evidence:

- training reward worsened from `-2.9475` at iteration 1 to `-60.9177` at iteration 30;
- settled landing stayed at `0%` throughout training;
- fixed-seed-145 deterministic ep10/ep20/ep30 checkpoint evaluations all had `0%` settled landing and `100%` crash;
- hard contact and ground crash stayed at `0%`;
- all five controller saturation ratios stayed at `0%`;
- ep30 deterministic reference saturation reached `48.8%`, with all three normalized action axes reaching `abs(action)=1.0` during evaluation;
- the inherited `continuous_a2c_logstd` exploration parameter remained about one normalized action unit (`exp(sigma) ≈ 0.97–1.21`).

Diagnosis therefore stops at the prescribed **Case C (policy/action distribution saturation)** before auditing or changing reward.

See:

```text
sanity_result.md
sanity_ep10_seed145.csv
sanity_ep20_seed145.csv
sanity_ep30_seed145.csv
candidate_table.md
validation_result.md
checkpoint_hashes.txt
```

## Next gate

Do not run 100–200 iteration C0 training and do not enter PX4 SITL. The next one-variable-at-a-time experiment is a repeated 64-env / seed-42 / 30-iteration sanity with only M2 `sigma_init.val` changed from `0` to `-1.0`. Action bounds/scaling, controller, reward, success/contact semantics, network, learning rate, and M0/M1 remain frozen.

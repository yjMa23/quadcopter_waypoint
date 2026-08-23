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

## Next gate

Only after the above PASS results, run the seed-42, 64-env, 30-iteration PPO sanity experiment. If learning is not stable/interpretable, stop and diagnose before any 100-200 iteration candidate training.

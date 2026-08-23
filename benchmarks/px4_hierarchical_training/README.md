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

## Stage 4 S1 exploration calibration

S1 changed only the M2 exploration initialization from `sigma_init.val=0` to `-1.0` and repeated the same `seed=42 / 64 env / 30 iteration` sanity budget from scratch.

S1 removed the S0 action/reference saturation pathology: fixed-seed deterministic reference saturation became `0% / 0.47% / 0%` at ep10/20/30 and all controller saturation ratios remained zero. Deterministic crash/deck-miss also improved, but settled landing stayed at zero and length-normalized return still degraded to approximately `-0.4249` reward per episode step by ep30.

Therefore S1 remained `SANITY FAIL`; a further sigma reduction was not justified.

## Stage 5 D0 reward-compatibility audit

The preregistered D0 audit used existing S0/S1 TensorBoard and deterministic-evaluator evidence only. It supported:

```text
D0-B LATCHED DESCENT-PHASE REWARD MISMATCH
```

The key mechanism was that latched `can_land` permanently enabled descent/contact shaping even after instantaneous horizontal alignment was lost. At S1 ep20, phase-sensitive terms accounted for about `64.0%` of tracked absolute reward magnitude, with `predicted_pad_error` and `contact_clearance` each contributing about `26%`.

Evidence:

```text
docs/m2_reward_compatibility_audit.md
reward_compatibility_audit/
```

## Stage 6 D1 reward-only descent-phase gate

D1 was preregistered before implementation in:

```text
docs/m2_d1_descent_phase_reward_gating.md
```

The one conceptual change was to separate latched physical landing permission from instantaneous reward eligibility:

```text
can_land                 = frozen latched task/contact permission
descent_reward_active    = can_land AND instantaneous recovery-alignment gate
```

No success/contact/termination contract, reward weight, action interface, controller, PPO setting, or M0/M1 behavior was changed.

Post-implementation validation:

```text
python -m pytest -q tests
128 passed, 21 subtests passed

1-env smoke  = PASS
16-env smoke = PASS
```

The frozen S1 training budget was then repeated exactly (`seed=42`, `64 env`, `30 iterations`, `sigma_init=-1.0`). D1 strongly improved reward compatibility:

- ep30 approximate reward/episode-step improved from S1 `-0.4249` to D1 `-0.2133`;
- phase-sensitive absolute-magnitude share was `9.63%` at ep20 and `15.76%` at ep30;
- deterministic ep30 align/crash/deck-miss became `48.44% / 1.56% / 0%`;
- reference saturation stayed near zero and all five controller saturation ratios stayed zero.

However, the complete preregistered gate failed:

- deterministic controller tracking mean was `0.1775 / 0.2040 / 0.4425 m/s` at ep10/20/30, exceeding the frozen `0.10 m/s` threshold at every checkpoint;
- ep30 aligned episodes ended outside the alignment radius at `58.06%`, above the `50%` gate;
- ep30 aligned timeouts ended outside the alignment radius at `60.00%`, also above the `50%` gate;
- settled landing remained `0%`, while timeout rose to `98.44%`.

Decision:

```text
D1 REWARD-GATING SANITY FAIL
```

Evidence:

```text
sanity_d1_ep10_seed145.csv
sanity_d1_ep20_seed145.csv
sanity_d1_ep30_seed145.csv
d1_reward_gating_sanity/d1_result.md
d1_reward_gating_sanity/snapshot_comparison.csv
d1_reward_gating_sanity/reward_attribution.csv
d1_reward_gating_sanity/deterministic_gate_summary.csv
d1_reward_gating_sanity/training_gate_snapshot.csv
d1_reward_gating_sanity/checkpoint_hashes.txt
```

## Current gate

Do **not** run the 100–200 iteration small candidate, C0, another sigma reduction, or PX4 SITL. The next intervention must be theory-first and separately preregistered. The evidence now points to post-latch horizontal recovery, within-bound action/reference temporal variation, controller tracking degradation, and failure to complete descent before timeout—not the original S0 reference saturation or the D0 reward-attribution mismatch.

# M2 D1 Descent-Phase Reward-Gating Sanity Result

## Decision

```text
D1 REWARD-GATING SANITY FAIL
```

D1 successfully removes the specific reward-attribution pathology identified by D0: length-normalized return improves materially, and `predicted_pad_error` / `contact_clearance` no longer dominate the tracked reward magnitude after a latched alignment event. However, the complete preregistered D1 gate is not satisfied. The fixed deterministic checkpoints violate the controller-tracking stability threshold at ep10, ep20, and ep30, and ep30 still violates the post-latch recovery geometry thresholds. The policy increasingly survives to timeout with no settled landing while exhibiting large within-bound action variation and recurrent horizontal recovery-gate violations.

This result therefore supports a narrower conclusion:

> separating latched physical landing permission from instantaneous descent-reward eligibility improves reward compatibility, but that change alone does not produce a stable M2 sanity candidate.

The frozen decision thresholds are in `docs/m2_d1_descent_phase_reward_gating.md`. They were committed before implementation or D1 training.

## 1. Baseline

Starting commit:

```text
6d1fb42 analysis: record M2 reward compatibility evidence
```

Initial workspace:

```text
git status = clean
```

Frozen Stage-0 regression:

```text
122 passed
21 subtests passed
M0/M1 action_space = 4
M2 action_space = 3
```

D0 evidence was read from:

```text
docs/m2_reward_compatibility_audit.md
benchmarks/px4_hierarchical_training/reward_compatibility_audit/
```

D0 starting conclusion:

```text
D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED
```

## 2. D1 implementation

Old reward-phase condition:

```python
descent_reward_active = terms["can_land"]
```

`can_land` remains the existing latched physical/task permission:

```python
can_land = self._align_success | (
    self._align_hold_steps >= self.cfg.align_hold_steps
)
```

D1 reward-only condition:

```python
recovery_alignment_ok = (
    (horizontal_error < align_radius)
    & (horizontal_speed < align_max_horizontal_speed)
    & (body_deck_normal_angle < align_body_deck_angle)
    & (upright > align_upright)
    & (robot_height_above_pad < align_height_max)
)

descent_reward_active = can_land & recovery_alignment_ok
```

There is deliberately no lower-height check against `align_height_min`.

Implementation structure:

```text
QuadcopterShipLandingHeaveEnv
  _reward_descent_phase_active(...) -> can_land

QuadcopterShipLandingPx4HierarchicalEnv
  _reward_descent_phase_active(...) -> can_land & recovery_alignment_ok

pure helper
  quadcopter_waypoint.utils.m2_reward_gating.m2_descent_reward_active(...)
```

The inherited reward path was not copied into M2.

Only these reward eligibility semantics changed for M2:

```text
post_align_descent
height_tracking target selection
descent_horizontal_rel_vel
near_pad_horizontal_rel_vel
predicted_pad_error
contact_clearance
center_precision
center_precision_square
```

Frozen and unchanged:

```text
reward weights/scales
can_land
_align_success
landing_candidate
safe_contact
successful_settle
settled-landing contract
termination/failure taxonomy
observation
action dimension/bounds/scaling
reference adapter
PX4-like controller
sigma/PPO/network
M0/M1 effective reward behavior
```

## 3. Unit tests

All preregistered reward-gate contracts pass:

| Case | Contract | Result |
| --- | --- | --- |
| A | `can_land=False -> reward gate=False` | PASS |
| B | latched + centered/stable -> gate=True | PASS |
| C | `can_land=True` + horizontal drift -> gate=False while `can_land` remains true | PASS |
| D | stable descent below `align_height_min` -> gate=True | PASS |
| E | post-latch horizontal-speed violation -> gate=False | PASS |
| F | post-latch body/deck attitude violation -> gate=False | PASS |

Post-implementation full regression:

```text
128 passed
21 subtests passed
```

The increase from 122 to 128 is exactly the six D1 A-F tests.

## 4. Smoke

Existing M2 smoke was run after implementation for both supported sizes.

### 1 env

```text
no_nan_inf                  = true
basic_ground_crash_zero     = true
tracking_stable             = true
contact_safe                = true
reward_path_finite          = true
status                      = PASS
```

### 16 env GPU

```text
no_nan_inf                  = true
basic_ground_crash_zero     = true
tracking_stable             = true
contact_safe                = true
reward_path_finite          = true
status                      = PASS
```

In the 16-env physical-deck-attitude tracking case:

```text
max velocity tracking error ~= 0.00863 m/s
reference saturation        = 0
controller saturation       = 0
```

The explicit `reward_path_finite` check calls `_get_rewards()` inside the Isaac environment, so the new reward hook is covered by runtime smoke rather than only by pure unit tests.

## 5. Training

D1 used exactly the frozen S1 training protocol:

```text
seed           = 42
num_envs       = 64
max_iterations = 30
sigma_init     = -1.0
network        = [64, 64]
```

Run directory:

```text
logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_17-15-36
```

Saved checkpoints:

```text
ep10  last_quadcopter_ship_landing_px4_hierarchical_ep_10_rew_-23.4894.pth
ep20  last_quadcopter_ship_landing_px4_hierarchical_ep_20_rew_-38.442574.pth
ep30  last_quadcopter_ship_landing_px4_hierarchical_ep_30_rew_-40.85966.pth
```

Checkpoint files remain under `logs/` and are not committed. Their SHA256 values are recorded in `checkpoint_hashes.txt`.

For cross-run comparison below, `reward` is the TensorBoard `rewards/iter` scalar, matching the D0 analysis convention. The ep30 TensorBoard D1 scalar is `-41.0288`; the checkpoint filename uses RL-Games' save-time reward value `-40.85966`.

## 6. S0 vs S1 vs D1

All task-behavior/controller rows below are from the fixed deterministic evaluator (`seed=145`, `64 env`, `64 completed episodes`). Reward and episode length are TensorBoard training scalars.

| metric | S0 ep10 | S0 ep20 | S0 ep30 | S1 ep10 | S1 ep20 | S1 ep30 | D1 ep10 | D1 ep20 | D1 ep30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reward | -37.4504 | -50.5182 | -60.9177 | -28.9095 | -53.9612 | -73.8551 | -23.4894 | -38.4426 | -41.0288 |
| episode length (steps) | 111.57 | 144.72 | 161.93 | 131.90 | 151.36 | 173.82 | 130.34 | 165.13 | 192.37 |
| reward / step | -0.3357 | -0.3491 | -0.3762 | -0.2192 | -0.3565 | -0.4249 | **-0.1802** | **-0.2328** | **-0.2133** |
| align | 25.00% | 26.56% | 28.12% | 7.81% | 15.62% | 31.25% | 10.94% | 25.00% | **48.44%** |
| settled landing | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| crash | 100% | 100% | 100% | 100% | 64.06% | 51.56% | 96.88% | 39.06% | **1.56%** |
| deck miss | 48.44% | 26.56% | 89.06% | 75.00% | 45.31% | 28.12% | 56.25% | 10.94% | **0%** |
| timeout | 0% | 0% | 0% | 0% | 35.94% | 51.56% | 3.12% | 60.94% | **98.44%** |
| reference saturation | 25.20% | 17.30% | 48.80% | 0% | 0.47% | 0% | 0.10% | 0.52% | 0.01% |
| controller tracking mean (m/s) | 0.3298 | 0.1407 | 0.2042 | 0.0974 | 0.0756 | 0.0763 | **0.1775** | **0.2040** | **0.4425** |

For D1, all five deterministic controller saturation ratios remain exactly zero at ep10/ep20/ep30, and ground crash/hard contact remain zero.

The D1 action distribution changes substantially by ep30 despite almost zero reference saturation:

```text
S1 ep30 action std [t1,t2,n] = [0.0885, 0.0620, 0.0955]
D1 ep30 action std [t1,t2,n] = [0.3853, 0.4858, 0.1845]
```

This is not a return to S0-style action-bound saturation. It is a within-bound high-variation policy accompanied by much larger controller velocity-tracking error.

## 7. Reward attribution

| run/checkpoint | phase-sensitive magnitude share | predicted_pad_error contribution / share | contact_clearance contribution / share | height_tracking contribution / share |
| --- | ---: | ---: | ---: | ---: |
| S1 ep20 | 64.02% | -28.615 / 25.99% | -28.595 / 25.97% | -9.149 / 8.31% |
| S1 ep30 | 29.11% | -2.706 / 5.44% | -3.307 / 6.65% | -6.139 / 12.35% |
| D1 ep20 | **9.63%** | **-0.730 / 1.23%** | **-1.245 / 2.10%** | -3.460 / 5.83% |
| D1 ep30 | **15.76%** | **-0.501 / 1.74%** | **-1.248 / 4.33%** | -2.644 / 9.18% |

The preregistered reward-compatibility gate is satisfied:

```text
D1 ep30 reward/step = -0.2133 >= -0.361
```

The S1 monotonic normalized-return failure pattern is not repeated: D1 worsens from ep10 to ep20, but ep30 improves from `-0.2328` to `-0.2133` while episode length continues to rise.

At D1 ep20 and ep30 deterministic align is >=15%, yet:

```text
phase-sensitive magnitude share <= 50%
predicted_pad_error individual share < 25%
contact_clearance individual share < 25%
```

Therefore the central D0 reward-attribution mismatch is materially improved.

## 8. Reward-gate diagnostics

The fixed deterministic evaluator provides per-completed-episode terminal-latched M2 diagnostics. These are preferred for exploitation analysis because all three checkpoints use the same seed145/64-episode protocol.

| metric | D1 ep10 | D1 ep20 | D1 ep30 |
| --- | ---: | ---: | ---: |
| reward gate active ratio, all episodes | 3.46% | 5.83% | 17.19% |
| reward gate active ratio, aligned episodes | 31.64% | 23.30% | 35.49% |
| `can_land` but gate inactive ratio, all episodes | 7.07% | 16.63% | 21.98% |
| `can_land` but gate inactive ratio, aligned episodes | 64.66% | 66.53% | 45.38% |
| aligned terminal outside `align_radius` | 85.71% | 93.75% | **58.06%** |
| aligned timeout episodes | 2 | 9 | 30 |
| aligned-timeout terminal outside `align_radius` | 50.00% | 88.89% | **60.00%** |
| transition count, all completed episodes | 0.67 | 0.64 | 2.67 |
| transition count, aligned episodes | 6.14 | 2.56 | 5.52 |
| horizontal-error violation ratio, aligned | 56.57% | 61.64% | **41.01%** |
| horizontal-speed violation ratio, aligned | 17.46% | 9.43% | 8.21% |
| attitude violation ratio, aligned | 0% | 0% | 0% |
| too-high violation ratio, aligned | 33.29% | 8.71% | 0% |

At ep30, `54.84%` of aligned episodes spend more than half of their `can_land` steps with the reward gate inactive. Only `3.23%` exceed 80%, so the failure is not that every aligned trajectory permanently disables the gate. Instead, the dominant pattern is recurrent horizontal loss of recovery alignment.

Training reset-cohort scalars are also preserved in `training_gate_snapshot.csv`. They are not substituted for the fixed evaluator because their cohort composition changes by iteration.

## 9. Exploitation / recovery audit

### Recovery gate closes when recovery alignment is lost

PASS as a mechanism. The gate is observably inactive after a latched `can_land`, and horizontal-error violations dominate the cause. This is consistent with the intended D1 separation of physical permission and reward eligibility.

### Normal low-height descent is not accidentally disabled

PASS at the contract level. Unit Case D explicitly proves there is no `align_height_min` lower bound and stable low-height descent remains reward-eligible.

No D1 deterministic checkpoint produces a settled landing, so there is no empirical successful-descent trajectory from this run that can establish final nominal landing behavior.

### Persistent gate avoidance / post-latch drift

FAIL under the frozen D1 thresholds.

At ep30:

```text
aligned terminal outside align_radius = 58.06% > 50%
aligned timeout outside align_radius  = 60.00% > 50%
```

The direct cause signal is primarily horizontal error:

```text
aligned can_land-step horizontal-error violation ratio = 41.01%
attitude violation ratio                               = 0%
too-high violation ratio                               = 0%
```

This is a clear gate-avoidance-compatible failure signature: once landing permission is latched, many trajectories repeatedly leave the horizontal recovery envelope and survive there until timeout. The evidence does not prove conscious/intended exploitation by the optimizer; it does prove that the resulting policy behavior violates the preregistered no-exploitation recovery contract.

### Gate chattering

The preregistered aggregate numeric transition threshold passes:

```text
mean transitions / completed episode = 2.67 <= 4
```

However, the aligned-only diagnostic is `5.52` transitions/episode at ep30. This was not the frozen numeric denominator and is therefore not used retroactively as an additional formal FAIL threshold, but it supports the qualitative concern that trajectories actually entering the landing phase repeatedly cross the recovery gate.

### Descent avoidance / timeout

The D1 policy does not convert improved alignment and crash avoidance into landing:

```text
ep30 timeout        = 98.44%
ep30 settled landing = 0%
```

Aligned ep30 episodes have mean descent speed about `0.0471 m/s`, and aligned timeouts about `0.0470 m/s`: they are not simply frozen at zero vertical speed, but the descent/recovery behavior is insufficient to complete the settled-landing contract.

## 10. Preregistered decision gate

| Gate | Frozen requirement | D1 evidence | Result |
| --- | --- | --- | --- |
| ep30 normalized return | `>= -0.361` | `-0.2133` | PASS |
| avoid same S1 monotonic reward/step collapse | not both ep10->20 and ep20->30 worse by >=0.05 | ep30 improves vs ep20 | PASS |
| phase-sensitive share at substantial align | `<=50%` | ep20 9.63%, ep30 15.76% | PASS |
| predicted/contact individual share | each `<=25%` | ep30 1.74% / 4.33% | PASS |
| ep30 align | `>=25%` | 48.44% | PASS |
| ep30 crash | `<=60%` | 1.56% | PASS |
| ep30 deck miss | `<=35%` | 0% | PASS |
| ep30 ground crash | `=0%` | 0% | PASS |
| ep30 hard contact | `=0%` | 0% | PASS |
| reference saturation, every checkpoint | `<=1%` | 0.10% / 0.52% / 0.01% | PASS |
| controller saturation, every checkpoint | all zero | all zero | PASS |
| controller tracking mean, every checkpoint | `<=0.10 m/s` | **0.1775 / 0.2040 / 0.4425** | **FAIL** |
| ep30 latched-but-gate-inactive ratio | `<=0.50` | 0.2198 all episodes; 0.4538 aligned | PASS |
| ep30 aligned terminal outside radius | `<=50%` | **58.06%** | **FAIL** |
| ep30 aligned-timeout outside radius | `<=50%` | **60.00%** | **FAIL** |
| transition count | `<=4` / completed episode | 2.67 | PASS |
| no clear gate-avoidance failure signature | required | recurrent horizontal post-latch drift + 98.44% timeout | **FAIL** |

Because complete comparable evidence exists and multiple frozen PASS requirements fail, this is not `INCONCLUSIVE`.

Final decision:

```text
D1 REWARD-GATING SANITY FAIL
```

## 11. Next recommendation

Because D1 fails, **do not** run the preregistered future M2 small-candidate training (`64/256 env`, `100-200 iterations`) and do not enter PX4 SITL.

The next work should be theory/diagnosis first, not longer training. The evidence now isolates a new question:

```text
Why does D1 produce much better reward compatibility and crash/deck-miss behavior,
while the learned within-bound reference becomes high-variation,
controller velocity tracking degrades,
and post-latch horizontal recovery still ends in timeout?
```

A future intervention must be separately preregistered and should distinguish at least:

```text
reward-gate exploitation vs incomplete positive incentive to re-enter the gate
reference temporal variation / action chattering vs controller capacity
horizontal recovery objective vs descent completion incentive
```

No such follow-up intervention is executed in D1.

## 12. Evidence inventory

```text
snapshot_comparison.csv
reward_attribution.csv
reward_terms_s0_s1_d1.csv
training_gate_snapshot.csv
deterministic_gate_summary.csv
commands.txt
checkpoint_hashes.txt

../sanity_d1_ep10_seed145.csv
../sanity_d1_ep20_seed145.csv
../sanity_d1_ep30_seed145.csv
```

Implementation/preregistration commits before the experiment evidence commit:

```text
ac10f4a docs: preregister M2 D1 reward-gating sanity
da07921 feat: gate M2 descent shaping on recoverable alignment
```

The experiment evidence commit is the commit containing this result file and the generated D1 evidence package.

# M2 D1 Descent-Phase Reward Gating Sanity Preregistration

## 1. Purpose and frozen hypothesis

D1 is the direct follow-up to the completed D0 reward compatibility audit. D0 concluded:

```text
D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED
```

The one hypothesis tested by D1 is:

> Separating latched physical landing permission (`can_land`) from instantaneous descent-reward eligibility removes the reward/behavior ranking mismatch observed in S1, without changing the physical task contract or creating a policy incentive to deliberately stay outside the recovery-alignment envelope.

D1 is a reward-phase eligibility experiment, not a nominal-capability or PX4-SITL gate. A settled-landing rate above 95% is not required in this 30-iteration sanity run.

## 2. Baseline and D0 evidence frozen before implementation

Starting commit:

```text
6d1fb42 analysis: record M2 reward compatibility evidence
```

Stage-0 regression at this commit:

```text
122 passed
21 subtests passed
M0/M1 action_space = 4
M2 action_space = 3
```

S1 used `seed=42`, `num_envs=64`, `max_iterations=30`, `sigma_init=-1.0`.

Training reward/step evidence:

| checkpoint | S1 reward | S1 episode length | S1 reward/step |
| --- | ---: | ---: | ---: |
| ep10 | -28.9095 | 131.904 | -0.2192 |
| ep20 | -53.9612 | 151.364 | -0.3565 |
| ep30 | -73.8551 | 173.825* | -0.4249 |

`*` ep30 uses the latest available episode-length scalar at iteration 29, matching the D0 audit.

Fixed deterministic S1 seed145 / 64-episode evidence:

| metric | ep10 | ep20 | ep30 |
| --- | ---: | ---: | ---: |
| align | 7.81% | 15.62% | 31.25% |
| settled landing | 0% | 0% | 0% |
| crash | 100% | 64.06% | 51.56% |
| deck miss | 75.00% | 45.31% | 28.12% |
| timeout | 0% | 35.94% | 51.56% |
| reference saturation | 0% | 0.47% | 0% |
| controller tracking mean | 0.0974 | 0.0756 | 0.0763 m/s |
| all controller saturation metrics | 0 | 0 | 0 |

Post-latch drift evidence:

```text
S1 ep20: aligned episodes ending outside align_radius = 10/10
S1 ep30: aligned episodes ending outside align_radius = 19/20
S1 ep20: aligned timeouts outside align_radius = 3/3
S1 ep30: aligned timeouts outside align_radius = 10/10
```

At high-align S1 iterations, phase-sensitive reward magnitude share reached approximately 58-70%. Representative direct contributors include:

```text
S1 ep20 predicted_pad_error = -28.615
S1 ep20 contact_clearance    = -28.595
S1 ep24 predicted_pad_error = -44.514
S1 ep24 contact_clearance    = -42.313
```

D0 also showed that aggregate reward/step still worsened as episode length increased. Therefore pure survival/episode-length bias is not the primary D1 hypothesis.

## 3. Old reward-phase condition

The physical/task phase is latched:

```python
can_land = (
    self._align_success
    | (self._align_hold_steps >= self.cfg.align_hold_steps)
)
```

Current descent shaping is effectively eligible whenever:

```python
descent_reward_active = can_land
```

Once `_align_success` becomes true, `can_land` stays true until reset even if the vehicle later loses horizontal, velocity, or attitude alignment.

## 4. D1 reward-only condition

`can_land` keeps exactly its current latched physical/task semantics.

D1 introduces a separate instantaneous reward-only condition:

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

There is deliberately **no** lower-height condition:

```text
robot_height_above_pad > align_height_min
```

Normal descent below `align_height_min` must keep descent shaping active as long as horizontal, velocity, attitude, upright, and maximum-height recovery conditions remain satisfied.

Interpretation:

```text
physical task permission       = can_land (latched)
reward shaping eligibility     = descent_reward_active (instantaneous)
```

## 5. Reward terms allowed to change phase eligibility

Only these terms may depend on `descent_reward_active` instead of `can_land`:

```text
post_align_descent
height_tracking
  - approach_target_height vs landing_target_height selection
descent_horizontal_rel_vel
near_pad_horizontal_rel_vel
predicted_pad_error
contact_clearance
center_precision
center_precision_square
```

These reward terms remain semantically unchanged:

```text
align_bonus
align_hold
lin_vel
ang_vel
horizontal_error
rel_vel
tilt
descent_vel
progress_to_pad
off_center_contact
landing_bonus
crash_penalty
```

No reward weight, scale, or reward term is added, removed, or retuned.

## 6. Frozen physical/task contracts

D1 must not change:

```text
can_land semantics
_align_success semantics
align hold semantics
landing_candidate
safe_contact
successful_settle
settled landing success contract
contact thresholds
termination
failure taxonomy
observation
3-D M2 action contract
action bounds / scaling
reference adapter
PX4-like controller
sigma_init
PPO hyperparameters
network
M0/M1 numerical reward behavior
```

`descent_reward_active` must never be substituted for `can_land` in task physics, contact success, termination, or failure classification.

## 7. Required implementation structure

Prefer one small reward-gating hook used by the inherited reward path rather than copying `_get_rewards()` into M2.

Target architecture:

```python
# parent default: preserves M0/M1
_reward_descent_phase_active(...) -> can_land

# M2 override
_reward_descent_phase_active(...) -> can_land & recovery_alignment_ok
```

The M2 recovery predicate should be small and pure enough to test without running Isaac Sim.

## 8. Unit-test contracts frozen before implementation

### Case A — never aligned

```text
can_land = False
=> descent_reward_active = False
```

### Case B — aligned and stable

```text
can_land = True
horizontal_error < align_radius
horizontal_speed < align_max_horizontal_speed
body_deck_normal_angle < align_body_deck_angle
upright > align_upright
robot_height_above_pad < align_height_max
=> descent_reward_active = True
```

### Case C — latched once, then horizontal drift

```text
can_land = True
horizontal_error > align_radius
=> descent_reward_active = False
AND can_land remains True
```

This is the core D1 contract.

### Case D — stable descent below align_height_min

```text
can_land = True
robot_height_above_pad < align_height_min
all recovery stability checks valid
=> descent_reward_active = True
```

### Case E — velocity recovery

```text
can_land = True
horizontal_speed > align_max_horizontal_speed
=> descent_reward_active = False
```

### Case F — attitude recovery

```text
can_land = True
body_deck_normal_angle > align_body_deck_angle
=> descent_reward_active = False
```

## 9. Required M2-only diagnostics

At minimum record terminal-latched, vectorized episode diagnostics:

```text
Episode/Metrics/m2_reward_descent_phase_active_ratio
Episode/Metrics/m2_can_land_but_reward_gate_inactive_ratio
```

Definitions are frozen as:

```text
m2_reward_descent_phase_active_ratio
= active reward-gate policy steps / all policy steps in the completed episode

m2_can_land_but_reward_gate_inactive_ratio
= (can_land & !descent_reward_active) policy steps / can_land policy steps
= 0 when an episode has zero can_land policy steps
```

If implementation remains small, also record:

```text
m2_reward_gate_transition_count
```

where a transition is any Boolean change of `descent_reward_active` between consecutive policy steps after episode start.

Existing terminal horizontal-error / align-latch data remain the source for terminal and timeout post-latch drift checks. No separate logging framework is introduced.

## 10. Smoke and training protocol

After implementation:

```text
1. full pytest regression
2. existing M2 1-env deterministic smoke
3. existing M2 16-env GPU smoke
```

Smoke must show:

```text
no NaN/Inf
no shape mismatch
no reward-path exception
finite reference/controller diagnostics
all controller saturation metrics = 0
```

Only then run D1 PPO sanity with exactly:

```text
seed           = 42
num_envs       = 64
max_iterations = 30
sigma_init     = -1.0
```

No other PPO or task parameter changes are allowed.

Save ep10 / ep20 / ep30 checkpoints in a distinct run directory.

Fixed validation for each checkpoint:

```text
seed          = 145
episodes      = 64
num_envs      = 64
deterministic = true
```

The existing evaluator success/contact contract stays frozen.

## 11. Reward-compatibility PASS / FAIL / INCONCLUSIVE gate

The following thresholds are frozen before D1 implementation/training.

### 11.1 Primary reward-compatibility requirements

All are required for PASS:

1. **ep30 normalized return improves materially over S1 ep30**:

```text
D1 ep30 reward/step >= -0.361
```

This is approximately a >=15% reduction in negative magnitude relative to S1 ep30 `-0.4249`, and returns ep30 to at least the S1-ep20 regime.

2. **The same strong monotonic S1 degradation must not persist.** D1 fails this item only if both are true:

```text
D1 ep20 reward/step <= D1 ep10 reward/step - 0.05
AND
D1 ep30 reward/step <= D1 ep20 reward/step - 0.05
```

3. In iterations/checkpoints with substantial alignment, `predicted_pad_error` and `contact_clearance` must no longer show the same latched-phase dominance pattern. For PASS, at ep20 or ep30 where deterministic align is >=15%:

```text
phase-sensitive magnitude share <= 50%
```

and neither `predicted_pad_error` nor `contact_clearance` may individually exceed 25% of total absolute tracked reward-term magnitude.

### 11.2 Task-behavior non-regression requirements at deterministic ep30

Relative to S1 ep30 (align 31.25%, crash 51.56%, deck miss 28.12%), PASS requires:

```text
align >= 25.00%
crash <= 60.00%
deck miss <= 35.00%
ground crash = 0%
hard contact = 0%
```

Settled landing is reported but is not a mandatory D1-PASS threshold.

Timeout may rise when immediate crash falls, so timeout alone is not a failure. Its post-latch geometry must instead pass the exploitation/recovery audit below.

### 11.3 Controller/reference stability requirements

At ep10, ep20, and ep30 deterministic validation:

```text
reference saturation <= 1.0%
controller velocity tracking mean <= 0.10 m/s
controller acceleration saturation = 0
controller tilt saturation = 0
controller thrust saturation = 0
controller body-rate saturation = 0
controller moment saturation = 0
```

### 11.4 Reward-gate recovery / exploitation requirements

At deterministic ep30, PASS requires:

```text
m2_can_land_but_reward_gate_inactive_ratio <= 0.50
```

and, among aligned completed episodes:

```text
terminal outside align_radius after latch <= 50%
aligned timeout outside align_radius after latch <= 50%
```

If there are zero aligned timeouts, the timeout-specific condition is N/A rather than failure.

If transition-count logging is implemented:

```text
mean m2_reward_gate_transition_count <= 4 per completed episode
```

In addition to numeric thresholds, trajectory/evaluator evidence must show no clear persistent strategy of:

```text
hovering outside align_radius to suppress shaping
persistent horizontal-speed violation
persistent body/deck attitude violation
descent avoidance after can_land
rapid reward-gate chattering
```

A clear gate-avoidance strategy overrides a numerically improved return and forces FAIL.

## 12. Decision rules

### D1 REWARD-GATING SANITY PASS

Only if **all** primary reward-compatibility, task non-regression, controller/reference stability, and exploitation requirements above are satisfied with complete evidence.

### D1 REWARD-GATING SANITY FAIL

Use FAIL if evidence is complete and any of the following occurs:

```text
primary reward compatibility thresholds fail;
material deterministic task regression crosses a frozen threshold;
controller/reference stability threshold fails;
reward-gate exploitation is clearly observed;
smoke or regression fails after implementation.
```

If regression or smoke fails, training must not be run.

### D1 INCONCLUSIVE

Use INCONCLUSIVE only when the prescribed evidence is incomplete or technically invalid, for example:

```text
training/evaluation cannot complete under the frozen protocol;
required diagnostics are missing/corrupt;
checkpoint or evaluator evidence is not comparable to S0/S1;
there is insufficient aligned evidence to judge the central recovery mechanism and the other thresholds do not justify PASS/FAIL.
```

Thresholds must not be changed after observing D1 training results.

## 13. Failure risks to audit

1. **Gate avoidance:** policy intentionally violates recovery alignment to turn off landing penalties.
2. **Gate chatter:** threshold crossings produce repeated phase switching and a noisy objective.
3. **Incorrect low-height cutoff:** accidentally using `align_height_min` would disable legitimate late descent.
4. **Task-contract leakage:** using reward gate for contact/success/termination would change the experiment variable.
5. **M0/M1 regression:** shared hook changes parent reward behavior instead of remaining identity-equivalent to `can_land`.
6. **Diagnostic reset race:** DirectRLEnv auto-reset could report post-reset values unless metrics are latched before reset.

## 14. Scope after D1

Even if D1 passes, this task stops after the 30-iteration sanity and fixed evaluation. The next recommended experiment may be a small M2 candidate run using the same D1 reward semantics:

```text
64 or 256 env
100-200 iterations
```

It must not be executed as part of D1.

PX4 SITL remains blocked until a future candidate demonstrates nominal settled landing near the existing M1 reference while preserving ground/contact/controller safety gates.

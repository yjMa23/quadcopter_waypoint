# M2 Reward Compatibility Decision

## D0 decision

```text
D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED
```

The evidence does **not** support a pure episode-length/survival-bias diagnosis. Longer S1 episodes do make raw return more negative, but the degradation remains after aggregate episode-length normalization. The stronger mechanism is that a one-time alignment success permanently latches the landing reward phase, while deterministic episodes frequently lose horizontal alignment afterwards. During those post-latch episodes, descent/contact shaping becomes a dominant negative contribution even though the vehicle is no longer inside the horizontal alignment envelope.

No reward scale, reward term, `can_land` semantic, success/contact/failure contract, controller, action, observation, PPO hyperparameter, or training run was modified in D0.

## 1. Baseline and evidence sources

Starting baseline:

```text
branch: feat/stochastic-sea-state
starting HEAD: 4ebdd16 docs: record M2 S1 sanity result
starting worktree: clean
```

Frozen regression before D0 work:

```text
116 passed, 1 warning, 21 subtests passed
```

Existing offline evidence only:

```text
S0 TensorBoard:
logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-00-25/summaries

S1 TensorBoard:
logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-17-58/summaries

S0 deterministic seed145 evaluator CSVs:
benchmarks/px4_hierarchical_training/sanity_ep{10,20,30}_seed145.csv

S1 deterministic seed145 evaluator CSVs:
benchmarks/px4_hierarchical_training/sanity_s1_ep{10,20,30}_seed145.csv
```

No S0/S1 retraining and no simulator run were performed.

## 2. Effective M2 reward inheritance

```text
QuadcopterShipLandingEnv
  -> QuadcopterShipLandingHeaveEnv
    -> QuadcopterShipLandingPhysicalDeckEnv
      -> QuadcopterShipLandingPhysicalDeckAttitudeEnv
        -> QuadcopterShipLandingPx4HierarchicalEnv
```

M2 does not override `_get_rewards()` or `_compute_landing_terms()`.

Effective reward behavior is therefore:

- Heave `_get_rewards()` supplies the main reward equation including center-precision terms;
- PhysicalDeck overrides reward scales and adds `off_center_contact`;
- PhysicalDeckAttitude supplies deck-frame kinematics, physical contact terms, and the instantaneous alignment predicate;
- M2 replaces only the 3-D action/reference/controller path and changes environment decimation to 4.

With simulator `dt=0.01 s` and M2 `decimation=4`:

```text
step_dt = 0.04 s
reward/control policy rate = 25 Hz
max episode = 10 s ~= 250 M2 steps
```

## 3. Effective reward scales and phase gates

| term | scale | main gate | dt-scaled | role in D0 |
| --- | ---: | --- | --- | --- |
| lin_vel | -0.05 | always | yes | general dense cost |
| ang_vel | -0.03 | always | yes | general dense cost |
| progress_to_pad | +5.0 | always | no | XY progress delta |
| post_align_descent | +6.0 | `can_land` | no | descent delta after latch |
| horizontal_error | -2.5 | always | yes | XY centering |
| height_tracking | -2.0 | target switches on `can_land` | yes | 0.75 m -> 0.055 m target |
| rel_vel | -1.0 | always | yes | surface-relative speed |
| tilt | -1.0 | always | yes | uprightness |
| descent_vel | -6.0 | excess descent speed | yes | impact/descent safety |
| descent_horizontal_rel_vel | -3.0 | `can_land` | yes | suppress tangential motion during descent |
| near_pad_horizontal_rel_vel | -7.0 | `can_land`, below 0.45 m | yes | near-pad tracking |
| predicted_pad_error | -8.0 | `can_land` | yes | predicted contact XY error |
| contact_clearance | -8.0 | `can_land` | yes | drive clearance to 0.005 m |
| center_precision | -30.0 | `can_land`, below 0.50 m | yes | near-contact precision |
| center_precision_square | -80.0 | `can_land`, below 0.50 m | yes | strong near-contact precision |
| off_center_contact | -25.0 | physical deck contact outside radius | yes | contact precision |
| align_bonus | +1.0 | instantaneous `align_candidate` | yes | reward current alignment |
| align_hold | +0.5 | latched `_align_success` | yes | reward prior alignment success |
| landing_bonus | +80 | settled landing terminal | no | terminal success |
| crash_penalty | -30 | crash terminal | no | terminal failure |

The full formulas and physical-contract definitions are preregistered in `docs/m2_reward_compatibility_audit.md`.

## 4. S0 vs S1 comparison

Training reward and episode length below come from TensorBoard. Deterministic behavior metrics are from the fixed seed145 / 64-episode evaluator. `ep30` episode length uses the latest available `episode_lengths/iter` event at iteration 29 because no iteration-30 length scalar exists.

| metric | S0 ep10 | S0 ep20 | S0 ep30 | S1 ep10 | S1 ep20 | S1 ep30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training reward | -37.45 | -50.52 | -60.92 | -28.91 | -53.96 | -73.86 |
| episode length, steps | 111.57 | 144.72 | 161.93* | 131.90 | 151.36 | 173.82* |
| approx reward / episode step | -0.3357 | -0.3491 | -0.3762* | -0.2192 | -0.3565 | -0.4249* |
| deterministic align | 25.00% | 26.56% | 28.12% | 7.81% | 15.62% | 31.25% |
| deterministic settled landing | 0% | 0% | 0% | 0% | 0% | 0% |
| deterministic crash | 100% | 100% | 100% | 100% | 64.06% | 51.56% |
| deterministic deck miss | 48.44% | 26.56% | 89.06% | 75.00% | 45.31% | 28.12% |
| deterministic timeout | 0% | 0% | 0% | 0% | 35.94% | 51.56% |
| deterministic ref saturation | 25.20% | 17.30% | 48.80% | 0% | 0.47% | 0% |
| deterministic controller tracking mean | 0.3298 | 0.1407 | 0.2042 m/s | 0.0974 | 0.0756 | 0.0763 m/s |
| controller saturation metrics | all 0 | all 0 | all 0 | all 0 | all 0 | all 0 |

`*` Aggregate diagnostic uses iteration-29 episode length with iteration-30 reward.

The S1 task behavior clearly improves by ep30 while raw return and approximate per-step return both become worse.

## 5. Dominant reward attribution

`Episode/Episode_Reward/*` is logged by the environment as:

```text
mean episodic term sum / max_episode_length_s
```

With `max_episode_length_s=10`, the audit multiplies the logged value by 10 to recover the logger's approximate mean episodic contribution for that reset cohort. These term contributions and `rewards/iter` are not guaranteed to use the same aggregation cohort, so they are used for decomposition/ranking rather than exact per-episode causal reconstruction.

### Snapshot top negatives

| run/iter | top negative contributors, approximate episodic contribution |
| --- | --- |
| S0 ep10 | crash -28.75; horizontal error -9.30; center square -3.28; height -3.20; progress -3.01 |
| S0 ep20 | horizontal error -32.09; crash -16.25; progress -11.28; height -8.18; rel vel -3.48 |
| S0 ep30 | horizontal error -16.77; crash -11.25; height -4.34; rel vel -3.23; progress -2.52 |
| S1 ep10 | horizontal error -16.15; crash -8.12; height -4.09; progress -3.07; rel vel -1.55 |
| S1 ep20 | **predicted pad -28.62; contact clearance -28.60**; crash -20.00; horizontal -12.40; height -9.15 |
| S1 ep30 | crash -17.50; horizontal -13.10; height -6.14; contact clearance -3.31; predicted pad -2.71 |

The ep20 values are not an isolated single point. In S1 iterations 17-21, 24, and 27-29, direct `can_land`-gated negative reward repeatedly becomes large. Examples:

```text
S1 iter17: predicted_pad_error ~= -26.62, contact_clearance ~= -21.79
S1 iter20: predicted_pad_error ~= -28.62, contact_clearance ~= -28.60
S1 iter21: predicted_pad_error ~= -37.03, contact_clearance ~= -45.87
S1 iter24: predicted_pad_error ~= -44.51, contact_clearance ~= -42.31
S1 iter28: predicted_pad_error ~= -33.37, contact_clearance ~= -22.21
```

At high-alignment S1 reset batches, phase-sensitive terms account for roughly 58-70% of total absolute reward-term magnitude.

Across all 30 S1 iterations:

```text
direct can_land negative magnitude vs align:
Pearson  = +0.809
Spearman = +0.866

phase-sensitive magnitude share vs align:
Pearson  = +0.832
Spearman = +0.897
```

These correlations are diagnostic, not significance claims, but the direction is strong and consistent with phase activation after alignment.

## 6. can_land latch audit

### Is `can_land` latched?

Yes.

`_get_dones()` uses:

```python
self._align_success |= self._align_hold_steps >= self.cfg.align_hold_steps
```

and `_compute_landing_terms()` uses:

```python
can_land = self._align_success | (self._align_hold_steps >= self.cfg.align_hold_steps)
```

`_align_success` is cleared only on episode reset. Therefore one successful alignment permanently keeps `can_land=True` for the rest of that episode.

### Which rewards are affected?

Directly zero-gated or target/weight-switched by `can_land`:

```text
post_align_descent
height_tracking
descent_horizontal_rel_vel
near_pad_horizontal_rel_vel
predicted_pad_error
contact_clearance
center_precision
center_precision_square
```

`align_hold` also remains active because it directly uses latched `_align_success`.

### Does transient alignment actually precede later loss of alignment?

Yes, strongly, in the existing deterministic evaluator.

Because `align_success` is latched, any episode with `align_success=True` but terminal horizontal error >= `align_radius=0.25 m` proves that the episode later ends outside at least the horizontal part of the instantaneous alignment envelope.

| run | iter | aligned episodes | aligned ending outside 0.25 m | aligned timeouts | aligned timeouts outside 0.25 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 10 | 16 | 12/16 = 75% | 0 | n/a |
| S0 | 20 | 17 | 17/17 = 100% | 0 | n/a |
| S0 | 30 | 18 | 18/18 = 100% | 0 | n/a |
| S1 | 10 | 5 | 5/5 = 100% | 0 | n/a |
| S1 | 20 | 10 | **10/10 = 100%** | 3 | **3/3 = 100%** |
| S1 | 30 | 20 | **19/20 = 95%** | 10 | **10/10 = 100%** |

For S1 aligned timeouts:

```text
ep20 mean terminal XY error      ~= 2.484 m
 ep20 mean terminal clearance     ~= 1.818 m
 ep30 mean terminal XY error      ~= 0.710 m
 ep30 mean terminal clearance     ~= 0.425 m
```

These are not near-contact states. They are episodes that had already latched alignment but later timed out substantially displaced from the landing geometry while the reward phase remained permanently switched to descent/contact shaping.

`post_align_descent` is also negative in the high-alignment S1 training batches 17-21 and 24, meaning the gated height-delta term is, on aggregate, penalizing motion away from the deck after latch rather than crediting a clean monotonic descent.

### Does the evidence support a failure mechanism?

Yes. The code proves permanent phase activation; the reward decomposition shows the phase terms become dominant when more episodes align; and the deterministic evaluator proves that almost all aligned S1 episodes later end outside the horizontal alignment envelope, including every aligned timeout. This is sufficient to support D0-B rather than treating the phase-cost spikes as merely the intended cost of a continuously aligned descent.

## 7. Survival-bias audit

Longer survival mechanically increases exposure to dt-scaled dense negative reward. This effect is real but is not sufficient to explain S1.

Raw reward versus episode length over common S1 iterations:

```text
Pearson  ~= -0.910
Spearman ~= -0.997
```

However, after aggregate episode-length normalization, reward still worsens strongly with episode length:

```text
S1 ep10 ~= -0.219 reward/step
S1 ep20 ~= -0.357 reward/step
S1 ep30 ~= -0.425 reward/step (using ep29 length)

length-normalized reward vs episode length:
Pearson  ~= -0.877
Spearman ~= -0.984
```

Therefore D0-A's required signature is absent: the degradation does not disappear or become stable after normalization. Longer episodes are not simply accumulating an unchanged per-step cost; the per-step reward environment becomes more punitive as the policy enters the latched phase.

Conclusion:

```text
survival bias exists as a secondary mechanical effect,
but D0-A is NOT supported as the primary diagnosis.
```

## 8. Terminal / survival reward budget

Exact terminal terms:

```text
settled landing bonus = +80
crash penalty         = -30
timeout terminal term = 0
```

A useful order-of-magnitude state is a level, centered, stationary vehicle that has already latched alignment but is still at the 0.75 m approach height.

Approximate level-deck geometry:

```text
robot root deck height          ~= 0.750 m
robot landing-surface offset    = 0.035 m
deck half thickness             = 0.020 m
surface clearance               ~= 0.695 m
clearance target                = 0.005 m
clearance error                 ~= 0.690 m
landing target height           = 0.055 m
height error after latch        ~= 0.695 m
```

Approximate reward rate if it remains instantaneously aligned and stationary:

```text
height_tracking     ~= -2.0 * 0.695 = -1.39 /s
contact_clearance   ~= -8.0 * 0.690 = -5.52 /s
align_bonus         ~= +1.00 /s
align_hold          ~= +0.50 /s
------------------------------------------------
subtotal            ~= -5.41 /s
```

This excludes other negative terms and assumes predicted XY error is zero. Roughly:

```text
30 / 5.41 ~= 5.5 s
```

of safe post-latch hover already costs about as much as the one-off crash penalty.

If the vehicle later drifts to about 0.25 m horizontal error while remaining high, then even with negligible velocity:

```text
horizontal_error      ~= -2.5 * 0.25 = -0.625 /s
predicted_pad_error   ~= -8.0 * 0.25 = -2.0 /s
height_tracking       ~= -1.39 /s
contact_clearance     ~= -5.52 /s
align_hold            ~= +0.50 /s
align_bonus           ~= 0 once outside the align radius
---------------------------------------------------
subtotal              ~= -9.04 /s
```

Recovery translation adds `rel_vel` and `descent_horizontal_rel_vel` costs. Thus a latched-but-drifted episode can consume the equivalent of the -30 crash penalty in only a few seconds while still physically recoverable and without hard/ground contact.

This budget explains why a safer, longer S1 policy can obtain worse return than an earlier crash-heavy policy, but the mechanism is specifically **phase-dependent** rather than a uniform survival tax.

### Approximate task-ordering implication

The observed reward can therefore violate the intended diagnostic ordering:

```text
safe aligned / recoverable behavior
> timeout without dangerous contact
> deck/workspace crash
```

because post-latch recoverable hovering/drift can accumulate more than -30 before timeout, while an early crash pays the terminal penalty once and stops accumulating shaping cost.

## 9. D0 verdict

```text
D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED
```

Why not D0-A:

- episode length increases, but normalized reward also degrades strongly;
- phase-sensitive cost share, not just duration, rises when alignment is latched.

Why D0-B is supported:

1. `_align_success` and therefore `can_land` are episode-latched;
2. the landing reward target immediately and permanently switches after the latch;
3. direct can_land-gated negative magnitude is strongly associated with alignment in S1;
4. predicted-pad and clearance terms repeatedly become the largest negative contributors;
5. fixed-seed deterministic evaluation proves that 95-100% of aligned S1 episodes later end outside the 0.25 m horizontal alignment radius, and 100% of aligned timeouts are outside it;
6. despite this loss of instantaneous alignment, the landing-phase reward remains active and can penalize lateral/climb recovery.

This does not prove that reward gating is the only limitation of M2. It is sufficient to identify it as the D0 failure mechanism that should be isolated next.

## 10. Exactly one D1 recommendation — DO NOT IMPLEMENT HERE

### Single next variable

```text
M2 descent-phase reward gate
```

This is one conceptual variable: the condition deciding whether landing/descent shaping is active in M2. No reward scale is changed and no new reward term is added.

### Old behavior

Current inherited reward effectively uses the episode-latched condition:

```python
descent_reward_active = terms["can_land"]
```

Once alignment succeeds once, this remains true even if the vehicle later drifts laterally, moves too fast tangentially, or loses deck-relative attitude alignment.

### Proposed behavior

Keep `can_land` itself unchanged for compatibility, but in an **M2-only reward override** use a recoverability-aware shaping gate built entirely from existing instantaneous thresholds:

```python
recovery_alignment_ok = (
    (terms["horizontal_error"] < self.cfg.align_radius)
    & (terms["robot_height_above_pad"] < self.cfg.align_height_max)
    & (terms["horizontal_speed"] < self.cfg.align_max_horizontal_speed)
    & (terms["body_deck_normal_angle"] < self.cfg.align_body_deck_angle)
    & (terms["upright"] > self.cfg.align_upright)
)

descent_reward_active = terms["can_land"] & recovery_alignment_ok
```

The lower `align_height_min` bound is intentionally **not** included in the proposed reward gate; otherwise the descent shaping would turn off automatically once a correct descent passes below 0.50 m. The existing upper bound is retained so a post-latch climb above the approach envelope can return to approach/recovery shaping.

Use `descent_reward_active` only in place of the current reward-phase `can_land` gating/target switch for:

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

Do **not** change:

```text
_align_success
can_land
align criterion
landing success
settled landing
safe/hard contact
failure taxonomy
reward scales
observation/action/controller
M0/M1
PPO settings
```

### Theoretical rationale

The latch still records that the policy once completed the approach alignment phase, but shaping no longer forces descent/contact precision while the instantaneous state has clearly left the recoverable alignment envelope. When lateral/attitude recovery succeeds, the same descent/contact shaping automatically reactivates without introducing a new threshold or curriculum variable.

This directly targets the D0-B mechanism while preserving the physical task contract.

### Expected effect

Expected directional changes:

- aligned episodes that drift should receive approach/recovery pressure rather than continued contact-clearance pressure;
- `predicted_pad_error` and `contact_clearance` should stop dominating during large post-latch displacement;
- terminal-outside-align fraction among previously aligned episodes should decrease;
- timeout should become less attractive than successful recovery/descent;
- settled landing should become learnable without changing the physical success gate.

### Main failure risk

The policy may learn a **gate-avoidance exploit**: deliberately move just outside the recovery alignment envelope to disable descent/contact shaping and hover there. Threshold chattering could also make the height target switch between approach and landing targets.

The unchanged global `horizontal_error`, progress, velocity, and safety terms partially oppose this exploit, but D1 must explicitly measure it.

### Future D1 sanity gate

Use the same small protocol only after separately preregistering D1:

```text
training: seed=42, num_envs=64, max_iterations=30
validation: deterministic seed=145, episodes=64
```

A D1 candidate should not advance unless all of the following directional gates are satisfied at the selected checkpoint:

1. the primary mechanism improves: aligned episodes ending outside `align_radius` falls materially from S1 ep30's 95%, with a preregistered target of <=50%;
2. aligned timeouts outside `align_radius` falls from S1's 100%, target <=50%;
3. `m2_settled_landing_rate` becomes non-zero in deterministic validation;
4. ground crash and hard contact remain 0;
5. controller saturation metrics remain 0 and reference saturation remains in the low S1 regime;
6. no new hover/gate-avoidance mode appears in the per-episode evidence.

Only after such a D1 sanity pass should the project consider a longer candidate run. D1 is **not implemented or trained in D0**.

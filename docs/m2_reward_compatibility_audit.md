# M2 Reward Compatibility Audit (D0)

## Scope and freeze boundary

This document preregisters the D0 reward compatibility audit for the independent PX4-compatible hierarchical task (`Quadcopter-Ship-Landing-Px4-Hierarchical-v0`). It is a diagnostic audit, not reward tuning.

Frozen for this audit:

- observation contract;
- 3-D M2 action contract and reference adapter;
- PX4-like controller;
- reward scales and reward terms;
- `can_land` semantics;
- success/contact/failure contracts;
- M0/M1 behavior;
- PPO hyperparameters and exploration settings.

No S0/S1 retraining and no D1 training are permitted. D0 must end in exactly one of:

- `D0-A EPISODE-LENGTH / SURVIVAL BIAS SUPPORTED`;
- `D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED`;
- `D0 INCONCLUSIVE`.

A D1 change may be recommended, but must not be implemented in this audit.

## Baseline verification

Starting HEAD before D0 work: `4ebdd16 docs: record M2 S1 sanity result` on branch `feat/stochastic-sea-state`.

Initial worktree state: clean.

Frozen regression command:

```bash
source /home/j/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=source/quadcopter_waypoint
python -m pytest -q tests
```

Observed before any D0 modification:

```text
116 passed, 1 warning, 21 subtests passed
```

Task contract remains:

- M0/M1 parent landing task action space: 4;
- M2 PX4 hierarchical task action space: 3.

## Reward inheritance chain

The class/config chain is:

```text
QuadcopterShipLandingEnv
  -> QuadcopterShipLandingHeaveEnv
    -> QuadcopterShipLandingPhysicalDeckEnv
      -> QuadcopterShipLandingPhysicalDeckAttitudeEnv
        -> QuadcopterShipLandingPx4HierarchicalEnv
```

The effective M2 reward implementation is not defined in the M2 class itself.

1. `QuadcopterShipLandingEnv` introduces the original reward vocabulary and base phase logic.
2. `QuadcopterShipLandingHeaveEnv` overrides `_get_rewards()` and keeps the base terms while adding `center_precision` and `center_precision_square`.
3. `QuadcopterShipLandingPhysicalDeckEnv` inherits the Heave reward, overrides several reward scales, and adds `off_center_contact` after calling `super()._get_rewards()`.
4. `QuadcopterShipLandingPhysicalDeckAttitudeEnv` does not override `_get_rewards()`. It overrides `_compute_landing_terms()` so the inherited reward is evaluated using deck-frame horizontal error, signed deck-surface clearance, rigid-surface relative velocity, deck-normal attitude, and physical contact semantics.
5. `QuadcopterShipLandingPx4HierarchicalEnv` does not override `_get_rewards()` or `_compute_landing_terms()`. Its stated behavior is to replace only the action/control path. Therefore M2 receives the PhysicalDeck reward scales, Heave reward equation, PhysicalDeck `off_center_contact`, and Attitude landing/contact kinematics unchanged at the Python contract level.

M2 changes `decimation` from the parent value 2 to 4 while simulator `dt=0.01 s` remains unchanged. Therefore:

```text
M2 step_dt = 0.01 * 4 = 0.04 s
M2 policy/reward frequency = 25 Hz
max_episode_length_s = 10 s
approximate full episode = 250 M2 steps
```

Most dense terms multiply by `step_dt`, so their cumulative magnitude approximates a time integral. `progress_to_pad` and `post_align_descent` do not multiply by `step_dt`; they are delta-progress terms and are approximately telescoping only when their gate/measurement remains consistent. Terminal `landing_bonus` and `crash_penalty` are not `step_dt` scaled.

### TensorBoard logging normalization caveat

At reset, every entry in `_episode_sums` is logged as:

```text
Episode/Episode_Reward/<term>
= mean(episodic_sum_for_reset_envs) / max_episode_length_s
```

With `max_episode_length_s = 10`, TensorBoard `Episode/Episode_Reward/*` is therefore a per-max-episode-second reporting normalization, not the literal raw episodic contribution. For the D0 offline audit:

```text
approx mean episodic contribution = TensorBoard Episode_Reward value * 10 s
```

This inversion restores the environment's logged mean episodic sum for that reset batch, but the aggregate still is not a one-to-one per-episode sample. Any division by aggregate episode length is an **aggregate diagnostic approximation**, not an exact causal per-episode estimate.

## Effective M2 task constants relevant to reward

### Phase/target constants

| quantity | effective M2 value | role |
| --- | ---: | --- |
| `approach_target_height` | 0.75 m | height target before `can_land` |
| `landing_target_height` | 0.055 m | height target after `can_land` |
| `align_radius` | 0.25 m | alignment candidate horizontal threshold |
| `align_height_min` | 0.50 m | lower deck-frame alignment height |
| `align_height_max` | 1.00 m | upper deck-frame alignment height |
| `align_max_horizontal_speed` | 0.30 m/s | max tangential speed for alignment |
| `align_upright` | 0.92 | world-upright threshold for alignment |
| `align_body_deck_angle` | 20 deg | max body/deck-normal angle for alignment |
| `align_hold_steps` | 8 | consecutive candidate steps to latch alignment |
| `landing_success_radius` | 0.12 m | physical contact precision radius |
| `near_pad_track_height` | 0.45 m | near-pad horizontal tracking activation height |
| `near_center_height` | 0.50 m | center-precision activation height |
| `descent_speed_limit` | 0.18 m/s | excess descent penalty threshold |
| `settle_hold_steps` | 3 | consecutive safe-contact steps for settled landing |
| `episode_length_s` | 10 s | timeout horizon |

Because M2 `step_dt=0.04 s`, the configured counters correspond to approximately 0.32 s of continuous `align_candidate` for the align latch and 0.12 s of continuous safe contact for settle. D0 does not change these contracts.

### Instantaneous alignment criterion

In `PhysicalDeckAttitude`, `align_candidate` is instantaneous and requires all of:

```text
horizontal_error < align_radius
align_height_min < robot_height_above_pad < align_height_max
horizontal_speed < align_max_horizontal_speed
body_deck_angle < align_body_deck_angle
world_upright > align_upright
```

`_align_hold_steps` increments only while this instantaneous predicate is true and resets to zero when it is false.

### Latched phase criterion

In `_get_dones()`:

```python
self._align_success |= self._align_hold_steps >= self.cfg.align_hold_steps
```

Therefore `_align_success` is latched for the remainder of the episode once the hold threshold is reached. It is cleared only in `_reset_idx()`.

The reward/target phase predicate is:

```python
can_land = self._align_success | (self._align_hold_steps >= self.cfg.align_hold_steps)
```

Since `_align_success` latches, `can_land` also remains true for the rest of the episode after alignment has once succeeded, even if the instantaneous `align_candidate` later becomes false.

## Physical success/contact/failure contract versus shaping

These concepts must not be conflated during D0.

### Physical success contract

`PhysicalDeckAttitude` defines `safe_contact` using physical deck contact plus precision and kinematic/attitude conditions. It requires, among other conditions:

- deck contact and no ground contact;
- contact point inside the effective deck;
- `horizontal_error < landing_success_radius`;
- no hard contact;
- bounded normal and tangential relative speed;
- bounded angular velocity;
- bounded body/deck-normal angle and sufficient world uprightness;
- penetration not exceeding the success limit.

`PhysicalDeckEnv._get_dones()` then requires `settle_hold_steps` consecutive `safe_contact` steps and that the first deck contact was precise. `_successful_settle` becomes `_landing_success`.

Thus **settled landing** is a physical terminal contract, not simply a high reward or a clearance proxy.

### Failure taxonomy

The physical task terminates on crash or settled landing. Crash is the union of the following physical/failure states, excluding successful settle:

- `hard_contact`: deck contact with excessive force/impulse, normal speed, or penetration;
- `deck_miss`: geometric miss below the deck surface, or first physical deck contact outside the precision region;
- `ground_crash`: ground contact or robot below minimum crash height;
- `workspace_crash`: excessive altitude or XY distance from environment origin.

Timeout occurs when the episode horizon is reached and carries no dedicated timeout reward term.

### Reward proxy versus task objective

The task objective is successful settled landing with safe physical contact. Reward terms are proxies intended to guide the policy toward this objective. A proxy can be numerically dominant or phase-misaligned without changing the physical task contract. D0 evaluates compatibility of those proxies with the M2 action/control path; it does not redefine the task objective.

## Effective M2 reward table

Notation:

- `dt = step_dt = 0.04 s`;
- `e_xy = horizontal_error` in deck coordinates;
- `e_h = height_tracking_error`;
- `v_rel = ||relative_velocity||` at the relevant rigid surface point;
- `v_xy = tangential_relative_speed`;
- `u = world_upright`;
- `v_down = max(-normal_relative_speed, 0)`;
- `v_excess = max(v_down - descent_speed_limit, 0)`;
- `w_pad` and `w_center` are the clamped near-pad/near-center weights;
- `I(x)` is an indicator.

| reward term | effective formula in M2 | scale | active phase / gate | `step_dt` scaled? | terminal/dense | `can_land` dependent? | intended purpose | possible M2 incompatibility to audit |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `lin_vel` | `||root_lin_vel_b||^2 * scale * dt` | -0.05 | always | yes | dense | no | discourage high translational speed | may accumulate with survival; root speed is not exactly deck-relative task quality |
| `ang_vel` | `||root_ang_vel_b||^2 * scale * dt` | -0.03 | always | yes | dense | no | discourage body rotation | survival accumulation; generally small |
| `progress_to_pad` | `(prev_e_xy - e_xy) * scale` | +5.0 | always | no | dense delta | no | reward horizontal progress | approximately telescoping, not directly time-biased; can reverse sign during recovery/drift |
| `post_align_descent` | `I(can_land) * (prev_height_error - height_error) * scale` | +6.0 | after `can_land` | no | dense delta | **yes** | reward continued descent after alignment | phase latch can keep descent incentive active after alignment is lost |
| `horizontal_error` | `e_xy * scale * dt` | -2.5 | always | yes | dense | no | stay centered over deck | survival accumulation, but recovery-compatible |
| `height_tracking` | `e_h * scale * dt` | -2.0 | always; target switches 0.75 -> 0.055 m | yes | dense | **yes, target switch** | stage-dependent altitude tracking | after latch, recovery at approach height is penalized as failure to descend |
| `rel_vel` | `v_rel * scale * dt` | -1.0 | always | yes | dense | no | match deck/surface motion | survival accumulation; physically relevant to safe contact |
| `tilt` | `(1-u) * scale * dt` | -1.0 | always | yes | dense | no | remain upright | survival accumulation; not deck-normal-specific but safe-contact also checks deck-normal angle |
| `descent_vel` | `v_excess^2 * scale * dt` | -6.0 | whenever descending faster than 0.18 m/s | yes | dense | no | limit impact/descent speed | not phase gated; may discourage aggressive descent before/after alignment but is safety-aligned |
| `descent_horizontal_rel_vel` | `I(can_land) * v_xy * scale * dt` | -3.0 | after `can_land` | yes | dense | **yes** | suppress tangential slip during descent | latch can penalize lateral recovery motion after transient alignment |
| `near_pad_horizontal_rel_vel` | `w_pad * v_xy * scale * dt` | -7.0 | after `can_land` and below 0.45 m, increasingly near deck | yes | dense | **yes** | low-slip near-pad tracking | latch plus low altitude can penalize recovery motion |
| `predicted_pad_error` | `I(can_land) * predicted_horizontal_error * scale * dt` | -8.0 | after `can_land` | yes | dense | **yes** | anticipate future contact location under relative motion | strong scale; can dominate after phase latch and penalize recoverable drift |
| `contact_clearance` | `I(can_land) * |clearance-0.005| * scale * dt` | -8.0 | after `can_land` | yes | dense | **yes** | drive toward desired contact clearance | after transient align, staying safely high for recovery incurs persistent penalty |
| `center_precision` | `w_center * e_xy * scale * dt` | -30.0 | after `can_land`, increasingly below 0.50 m | yes | dense | **yes** | precision near contact | large scale; recovery motion/temporary off-center state can be expensive near deck |
| `center_precision_square` | `w_center * e_xy^2 * scale * dt` | -80.0 | after `can_land`, increasingly below 0.50 m | yes | dense | **yes** | strongly penalize large near-contact XY error | very large nominal scale but weighted by height/error; possible dominant phase pressure |
| `off_center_contact` | `I(deck_contact) * max(e_xy-landing_success_radius,0) * scale * dt` | -25.0 | only during physical deck contact outside precision radius | yes | dense/contact | no | punish off-center physical contact | contact-specific; may be sparse before crash/termination |
| `align_bonus` | `I(align_candidate) * scale * dt` | +1.0 | instantaneous alignment only | yes | dense | no | reward staying in current alignment envelope | small positive shaping relative to some post-latch penalties |
| `align_hold` | `I(_align_success) * scale * dt` | +0.5 | after latched alignment | yes | dense | indirectly via same latch state | retain credit after alignment achievement | positive latch credit may be much smaller than post-latch descent/contact costs |
| `landing_bonus` | `I(_landing_success) * scale` | +80.0 | settled landing terminal step | no | terminal/sparse | no | make successful settled landing preferred | must compete with all accumulated dense negative terms |
| `crash_penalty` | `I(_crash) * scale` | -30.0 | crash terminal step | no | terminal/sparse | no | penalize unsafe terminal failure | a long safe timeout can accumulate more dense cost than an early crash, creating a ranking paradox |

### Terms whose behavior changes after latched `can_land`

Directly or through a switched target/weight:

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

`align_hold` is not written using `can_land`, but it depends on the same latched `_align_success` state and therefore also remains active after transient successful alignment.

## Three can_land trajectory cases

### Case A: never aligned

If `align_candidate` never holds for eight consecutive M2 steps:

```text
_align_success = false
can_land = false
desired_height = approach_target_height = 0.75 m
```

Consequences:

- no `post_align_descent`;
- no `descent_horizontal_rel_vel`;
- no `near_pad_horizontal_rel_vel`;
- no `predicted_pad_error`;
- no `contact_clearance`;
- no `center_precision` / `center_precision_square`;
- `height_tracking` targets 0.75 m;
- `align_bonus` can still appear on isolated candidate steps;
- no latched `align_hold` until alignment succeeds.

The reward primarily asks the policy to approach/center, track approach height, control motion, and avoid unsafe behavior.

### Case B: aligns and remains aligned, then descends

After the align hold threshold:

```text
_align_success = true (latched)
can_land = true
desired_height = landing_target_height = 0.055 m
```

The descent/contact shaping becomes active. This is the intended phase transition: horizontal relative speed should fall, projected pad error should be minimized, clearance should approach contact target, and near-center terms increasingly enforce precision as the vehicle approaches the deck.

If the policy remains aligned and descends safely, these terms are directionally compatible with the physical settled-landing objective.

### Case C: aligns briefly, latch occurs, then drifts and attempts recovery

After `_align_success` latches, a later failure of the instantaneous `align_candidate` resets `_align_hold_steps` but does **not** clear `_align_success`. Therefore:

```text
align_candidate = false
_align_success = true
can_land = true
```

The desired height remains 0.055 m and all post-latch terms listed above remain eligible. A policy that wants to climb/hold near the 0.75 m approach height or translate laterally to regain alignment can therefore face simultaneous pressure to:

- descend toward 0.055 m (`height_tracking`, positive descent progress);
- keep tangential speed low (`descent_horizontal_rel_vel`, near-pad relative velocity);
- reduce predicted contact error immediately (`predicted_pad_error`);
- move clearance toward 0.005 m (`contact_clearance`);
- stay very close to center when low (`center_precision`, square term).

This is a **theoretical phase-credit conflict candidate**, not yet a D0-B conclusion. It becomes D0-B only if offline evidence shows that one or more of these post-latch terms become dominant in S1 and quantitatively explain the return/behavior mismatch better than episode-length accumulation alone.

## D0 hypotheses and falsifiable evidence

### D-H1: episode-length / survival bias

S1 deterministic evaluation already shows lower crash/deck miss and higher timeout than S0/S1 earlier checkpoints, while training return degrades. If a policy survives longer, every time-integrated negative shaping term has more time to accumulate, whereas `crash_penalty` is paid once and timeout has no terminal penalty/reward.

D-H1 is supported only if the offline S0/S1 evidence shows both:

1. worse raw/aggregate return is strongly associated with longer episodes / more timeout;
2. after approximate episode-length normalization, return or dominant dense-term quality is stable or improving rather than comparably worsening.

If length-normalized return degrades similarly, survival bias alone is not supported.

### D-H2: latched descent-phase reward pressure

D-H2 is supported only if:

1. the code-level latch described above is confirmed (it is);
2. S1 evidence shows `can_land`-dependent descent/contact terms become dominant as alignment improves;
3. their magnitude/trend is consistent with Case C recovery conflict rather than merely longer survival.

The known observation that `predicted_pad_error` and `contact_clearance` are approximately -2.86 around S1 ep20 is a lead, not a conclusion. The TensorBoard series must be decomposed and normalized before D0-B can be selected.

### D-H3: terminal versus dense ranking mismatch

The audit must compare approximate return budgets for:

1. early workspace/deck-miss crash;
2. long safe but unlanded timeout;
3. aligned then drift/recovery;
4. aligned controlled descent;
5. settled landing.

Desired theoretical ordering is:

```text
settled landing
> safe aligned / recoverable behavior
> timeout without dangerous contact
> deck miss / workspace crash
> ground crash / hard contact
```

The current reward is not assumed to satisfy this ordering. D0 will estimate whether the combination of dense accumulation, `landing_bonus=+80`, `crash_penalty=-30`, and no timeout terminal term creates a survival paradox or a phase-credit mismatch.

## Offline evidence protocol

Use existing logs only:

```text
S0 = logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-00-25
S1 = logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-17-58
```

No simulator is needed. Extract all available iteration series for:

- training return (`rewards/iter` or actual matching TensorBoard scalar tag);
- episode length (`episode_lengths/iter` or actual matching tag);
- every `Episode/Episode_Reward/*` term;
- align / landing / M2 settled landing rates;
- hard contact, ground crash, deck miss;
- crash and timeout termination metrics;
- M2 reference norm/saturation;
- controller tracking and all controller saturation metrics;
- M2 action mean/std metrics.

At minimum compare iterations 10, 20, 30 and analyze the full 1..30 trends where available.

### Reward attribution calculations

For each reward term at each available iteration:

```text
logged_rate = TensorBoard Episode/Episode_Reward/<term>
approx_episode_contribution = logged_rate * max_episode_length_s
absolute_magnitude = abs(approx_episode_contribution)
negative_rank = rank among negative approximate episode contributions
relative_total_magnitude = abs(contribution) / sum(abs(all term contributions))
```

The term-sum should also be checked against the available aggregate episode reward series, while documenting any logger-window mismatch.

### Length-normalized diagnostic

For aggregate diagnostic purposes:

```text
normalized_return ~= aggregate episode return / aggregate episode length
normalized_term ~= approx_episode_contribution / aggregate episode length
```

If episode length is logged in seconds rather than steps, the script must detect/document the unit rather than silently assuming steps. The ratio is a window-level approximation only because TensorBoard aggregates reset cohorts and PPO statistics with potentially different windows.

### Trend diagnostics

Compute Pearson and Spearman correlations over common iterations when enough samples exist:

```text
reward vs align
reward vs crash
reward vs deck miss
reward vs timeout
reward vs episode length
```

These correlations are descriptive diagnostics with approximately 30 iterations, not statistical proof.

## Reward budget method

Use three complementary scales:

1. exact terminal values: `landing_bonus=+80`, `crash_penalty=-30`, timeout terminal reward = 0;
2. configured dense scales and `dt=0.04 s`;
3. observed mean episode contributions from TensorBoard, including their relationship to episode length.

For a representative dense penalty rate `c < 0` reward/s, a full 10 s episode contributes approximately `10*c`. The audit should compare this with the one-off terminal values to identify whether simply surviving from an early crash to timeout can make total return lower despite safer physical behavior.

No false precision is permitted: budgets built from aggregate TensorBoard means are order-of-magnitude diagnostics, not reconstructed individual trajectories.

## D0 decision rule

### D0-A EPISODE-LENGTH / SURVIVAL BIAS SUPPORTED

Select only if longer survival/timeout clearly increases cumulative negative reward while length-normalized return or dominant term quality is stable/improving.

### D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED

Select only if, after `can_land` latches, one or more descent/contact shaping terms become quantitatively dominant and the evidence supports a conflict with alignment recovery/stable approach behavior that cannot be explained primarily by episode length.

### D0 INCONCLUSIVE

Select if existing aggregate logs cannot separate survival bias, latch-phase mismatch, and general insufficient learning. Do not force a root cause solely to justify a next experiment.

## D1 recommendation rule (recommend only)

The final audit may recommend exactly one M2-only conceptual variable. It must leave observations, actions, controller, physical success/contact/failure contracts, M0/M1, and PPO hyperparameters unchanged.

- If D0-A is supported, select the smallest reward-ordering variable justified by the measured budget; do not automatically increase `crash_penalty`. Explicitly analyze timeout/hover/descent-avoidance exploitation.
- If D0-B is supported, a candidate is a single conceptual change to M2 descent-phase reward gating, but the final recommendation must state the exact old condition, exact proposed new condition, and affected reward terms.
- If D0 is inconclusive, recommend a single diagnostic variable/instrumentation step rather than silently tuning multiple reward weights.

No D1 implementation or training belongs in this task.

## D0 evidence result

The preregistered audit was completed using only the existing S0/S1 TensorBoard event files and fixed-seed deterministic evaluator CSVs. The detailed generated evidence is under:

```text
benchmarks/px4_hierarchical_training/reward_compatibility_audit/
```

The final decision is:

```text
D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED
```

The decisive evidence is:

1. Survival duration is not sufficient: S1 raw reward is strongly anti-correlated with episode length, but aggregate length-normalized reward also degrades from approximately `-0.219` reward/step at ep10 to `-0.357` at ep20 and `-0.425` at ep30 (ep30 uses the latest available length scalar from iteration 29).
2. `can_land`-dependent negative magnitude is strongly associated with alignment in S1 (`Pearson=+0.809`, `Spearman=+0.866`), and the broader phase-sensitive magnitude share versus alignment is `+0.832/+0.897`.
3. The phase costs repeatedly dominate high-alignment S1 reset cohorts. For example, restored approximate mean episodic contributions are about `-28.62` for `predicted_pad_error` and `-28.60` for `contact_clearance` at iteration 20; they reach approximately `-44.51` and `-42.31` at iteration 24.
4. The fixed seed145 evaluator shows that latched alignment is commonly followed by loss of instantaneous horizontal alignment: S1 ep20 has `10/10` aligned episodes ending outside the `0.25 m` align radius, and S1 ep30 has `19/20`. Every aligned timeout is outside the radius (`3/3` at ep20 and `10/10` at ep30).
5. Those aligned timeouts are not near-contact states: their mean terminal XY error / clearance are approximately `2.484 / 1.818 m` at ep20 and `0.710 / 0.425 m` at ep30. Yet the inherited reward phase remains permanently switched to landing/descent shaping because `_align_success` is latched.

The reward-budget estimate further shows the ordering risk. A centered, level vehicle that has latched alignment but remains near the `0.75 m` approach height receives approximately `-5.41 reward/s` from `height_tracking + contact_clearance - align_bonus - align_hold` even before other terms. A latched vehicle displaced to roughly the `0.25 m` horizontal boundary can reach approximately `-9.04 reward/s` before adding recovery-velocity costs. Thus several seconds of physically recoverable post-latch behavior can exceed the one-off `crash_penalty=-30`.

### D1 recommendation only

Recommend exactly one future M2-only conceptual variable: **the descent-phase reward gate**. Keep `can_land` itself and all physical success/contact/failure semantics unchanged, but activate descent/contact shaping only when the episode has latched `can_land` **and** the current state remains inside an instantaneous recoverability envelope based on the already-existing horizontal, upper-height, tangential-speed, body/deck-angle, and upright thresholds. The lower alignment-height bound must not be reused for this reward gate because a correct descent naturally passes below it.

The affected reward-phase terms would be only:

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

This is a recommendation, not an implementation. D0 made no reward-scale change, no new reward term, no `can_land` semantic change, no task-contract change, and performed no D1 training.

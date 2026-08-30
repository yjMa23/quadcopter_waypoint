# Continuous-Stage PX4-Compatible Landing Task Contract

> Scope: S3 independent task contract plus S4 1-env deterministic validation evidence. It does not claim PPO, 16-env GPU smoke, formal S6/S11 rotating-deck benchmarks, SITL, HIL, or real-vehicle evidence.

## Task identity and inheritance boundary

```text
directory: quadrotor_ship_landing_px4_continuous_stage
class: QuadcopterShipLandingPx4ContinuousStageEnv
cfg: QuadcopterShipLandingPx4ContinuousStageEnvCfg
task ID: Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
```

The task inherits the frozen `QuadcopterShipLandingPx4HierarchicalEnv` only to reuse the physical deck entity, PX4-like control backend, logging infrastructure, and historical deterministic failure taxonomy. The old M2 source is not modified. S3 overrides only the contracts that intentionally change: 4-D high-level action/reference construction, observation index 15, continuous reward shaping, terminal attitude reference, and the new task's relative-angular-velocity success predicate.

## Action semantics

```text
a[..., 0:3] = normalized deck-relative XYZ action
a[..., 3]   = normalized continuous landing-stage action
```

The XYZ action is mapped by `map_stage_conditioned_relative_velocity()` and then limited by `limit_stage_conditioned_reference_slew()`. The fourth action is never passed to the old `normalized_action_to_relative_velocity()` adapter and never represents roll/pitch/yaw, body rate, torque, thrust, or motor command.

## Observation semantics

The observation remains 22-D:

```text
0:3    root linear velocity, body
3:6    root angular velocity, body
6:9    projected gravity
9:12   deck relative position, body
12:15  deck surface velocity world - UAV root velocity world
15     current filtered landing stage
16:19  deck normal, body
19:22  deck angular velocity - UAV angular velocity, body frame
```

At policy inference time `o_t[15]` is the filtered stage produced by the previous policy step, i.e. `o_{t+1}[15] = s_t`.

## Caller-owned reset and update state

The new task explicitly owns:

```text
_landing_stage
_previous_relative_velocity_ref_d
_previous_attitude_reference_wxyz
_previous_deck_heading_w
```

Episode reset is deterministic:

```text
landing_stage = 0
previous_relative_velocity_ref_d = 0
previous_attitude_reference_wxyz = current UAV attitude
previous_deck_heading_w = current valid deck heading, otherwise world +x
```

Stage is updated once per policy step at 25 Hz (`decimation=4`, `step_dt=0.04 s`) using the S2 low-pass plus explicit rate limiter with `tau=0.20 s` and `rate=2.0 1/s`. No hard stage/FSM threshold is introduced.

## Relative-reference construction

```text
4-D action
-> normalized_stage_action
-> filter_landing_stage
-> map_stage_conditioned_relative_velocity
-> limit_stage_conditioned_reference_slew
-> deck_contact_point_velocity
-> deck_relative_to_world_velocity
-> world_to_ned_velocity
```

The slew limiter acts only on the policy-relative component. Exact rigid-body `v_deck + omega x r` feedforward is reused from `px4_reference_adapter.py` and is not duplicated or delayed by the relative-reference slew limiter.

## Terminal-attitude construction

Signed clearance comes from `_contact_kinematics()["surface_clearance"]`. Deterministic yaw/heading comes from `deck_heading_world()` and remains caller-owned.

At the 25 Hz reference layer:

```text
velocity reference + state + deterministic deck heading
-> q_vel from VectorizedPx4LikeController velocity-control math
-> alpha = terminal_alignment_weight(stage, signed clearance)
-> shortest_quaternion_slerp(q_vel, q_deck, alpha)
-> limit_attitude_tilt(..., 35 deg)
-> limit_attitude_reference_rate(..., dt=0.04, max_rate=(2.0,2.0,1.5))
-> q_ref
```

The 100 Hz controller receives optional `q_ref` for its existing attitude/rate/moment loop. RL never outputs attitude.

## Controller integration and backward compatibility

`VectorizedPx4LikeController` may gain additive APIs to expose the normal velocity-controller attitude and to accept an optional external attitude reference. Calling existing `compute(...)` without the optional reference must preserve old numerical thrust/moment/body-rate behavior. Existing M2 does not pass the new argument.

## Success migration

Only this new task replaces the old absolute UAV angular-speed safe-contact metric with:

```text
omega_rel^W = omega_uav^W - omega_deck^W
relative_ang_vel_norm = ||omega_rel^W||
relative_ang_vel_norm < 1.50 rad/s
```

All other frozen first-version safe-contact terms remain unchanged: deck contact, no ground contact, effective deck inclusion, horizontal error < 0.12 m, no hard contact, normal relative speed < 0.55 m/s, tangential relative speed < 0.30 m/s, body/deck normal angle < 12 deg, world upright > 0.90, penetration <= 0.025 m, 3-step hold, and first-contact precision. Stage never enters success/failure predicates.

## Reward migration

The new task does not call the inherited hard-`can_land` reward implementation. Its S3 reward structure is continuous:

- always active: horizontal/deck tracking, relative-velocity matching, flight attitude feasibility, progress/time efficiency, physical safety shaping;
- stage/alpha weighted: descent progress, terminal tangential speed, contact precision, terminal attitude alignment, relative angular alignment;
- smoothness metrics: `delta_stage`, `delta_relative_velocity_reference`;
- terminal: settled landing bonus, hard-contact / ground-crash / deck-miss penalties.

No `if can_land`, `torch.where(can_land, ...)`, or `align_success` binary decision gate controls descent/terminal reward. Existing inherited coefficients with matching physical meaning are reused. New coefficients without preregistered values (`delta_stage`, terminal attitude alignment, relative angular alignment) are explicit and default to zero until preregistered before S7 PPO sanity.

## Diagnostics

Step/reference diagnostics include:

```text
stage_raw, stage_filtered, delta_stage
V_t, V_down, V_up
relative_velocity_target_d, relative_velocity_reference_d, relative_reference_delta
deck_contact_velocity_w, velocity_reference_w
terminal_alpha, deck_heading_w
q_vel, q_ref, q_deck
terminal_attitude_conflict_angle, terminal_attitude_tilt_saturated, attitude_reference_rate
relative_angular_velocity, relative_angular_speed
```

Episode buffers/logging reserve stage mean/std/min/max, stage variation/saturation ratio, reference variation, velocity tracking error, alpha mean/max, tilt saturation ratio, attitude conflict mean/max, attitude-reference-rate max, and relative angular speed at contact/terminal. All operations remain vectorized over environments.

## PPO configuration

The new task has a separate RL-Games config with:

```text
load_checkpoint = false
load_path = ""
network = [64, 64], ELU
run name = quadcopter_ship_landing_px4_continuous_stage
```

The old 22D->3D M2 checkpoint is never loaded or semantically reinterpreted for the new 22D->4D policy.

## S3 PASS gate

S3 passes only if the independent task is registered with `action_space=4`, `observation_space=22`, `decimation=4`; all action/stage/reference/attitude/success/reward boundaries above are implemented; old M0/M1, Fixed-Stage M2 and historical PhysicalDeckAttitude source semantics remain unchanged; controller default behavior is numerically compatible; targeted tests, full regression, and `git diff --check` pass.

## S4 deterministic smoke evidence

S4 was executed after S3 PASS with:

```text
script   = scripts/rl_games/check_px4_continuous_stage_smoke.py
task ID  = Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
num_envs = 1
seed     = 42
physics  = 100 Hz
policy   = 25 Hz
output   = logs/rl_games/quadcopter_ship_landing_px4_continuous_stage/s4_deterministic_smoke_seed42.json
```

Execution command:

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python \
scripts/rl_games/check_px4_continuous_stage_smoke.py \
--num_envs 1 --headless \
--output logs/rl_games/quadcopter_ship_landing_px4_continuous_stage/s4_deterministic_smoke_seed42.json
```

All nine scripted cases passed: static hover, stage ramp, constant XY deck, heave tracking, normal descent stage ramp, roll/pitch terminal-attitude blend, static yaw heading, off-center contact-point interface, and recovery. Global gates were `NaN/Inf=0`, `ground crash=0`, finite reward path, controller saturation ratio 0 in every case, and `max |delta_stage| = 0.08`, matching the frozen `2.0 1/s * 0.04 s` policy-step limit.

Interface evidence includes: low-stage `V_down=0`; high-stage relative normal reference `-0.2370 m/s`; recovery stage `0.9880 -> 0.00158` with positive relative normal reference `+0.2100 m/s`; maximum terminal alpha `0.8693`; maximum q_ref tilt `5.276 deg`; maximum attitude-reference rate approximately `[0.2255, 0.1803, 0.0089] rad/s`; static deck/q_vel/q_ref heading approximately `+15 deg`; off-center contact-point feedforward correction `0.00536 m/s`. The largest velocity-tracking error occurred in the terminal-attitude blend case at approximately `0.968 m/s`; it remained finite with zero controller saturation and no hard/ground contact, so S4 basic-stability gate passes while this diagnostic is carried forward to S5/S6 rather than tuned away in S4.

```text
targeted S4 regression = 71 passed
full regression        = 171 passed + 21 subtests
pre-S4 baseline        = 167 passed + 21 subtests
added S4 tests         = 4
git diff --check       = PASS
frozen historical source diff = 0
```

This is only interface/smoke evidence. It does not establish center-vs-contact-point superiority or rotating-yaw performance; those remain S6 and S11. The next gate is S5 16-env GPU Continuous-Stage smoke. PPO, checkpoint training/loading, SITL, ROS2, HIL, and real-vehicle work remain out of scope until their later gates.

## Supplementary deterministic full-landing evidence

This is supplementary demonstration evidence only, not a new formal gate. The scenario is a static level deck with `num_envs=1`, `seed=42`, `physics=100 Hz`, `policy=25 Hz`, and a frozen initial landing-surface clearance of `0.25 m`. The high-level commands are deterministic scripted actions; no PPO runner, checkpoint, or learned policy is used.

The demo script is `scripts/rl_games/run_px4_continuous_stage_full_landing_demo.py`. Its `first-contact latch + contact_settle seating reference` is demonstration-side scripted action logic only. It is not part of the production policy, Reference Adapter, PX4-like controller, contact detector, or landing-success contract. Production safety/success thresholds remain unchanged.

Two repeated `seed=42` runs of the same `0.25 m` scenario produced identical first-contact and final terminal metrics. First safe contact occurred at policy step 104 with horizontal error `5.07e-6 m`, normal relative speed `0.00812 m/s`, tangential relative speed `0.00501 m/s`, relative angular speed `0.12173 rad/s`, `safe_contact=true`, `hard_contact=false`, and `ground_crash=false`. Settled landing occurred at policy step 119 with horizontal error `5.14e-5 m`, normal relative speed `0.000641 m/s`, tangential relative speed `0.000791 m/s`, relative angular speed `0.03062 rad/s`, `settle_hold=3/3`, and `landing_success=true`.

Reproduction note: the previously recorded `v6` evidence also used `initial_clearance_m=0.25`. An invocation with `0.45 m` was therefore a different initial condition and terminated without a detected deck contact before episode end; it is not counted as a repeat of the `v6` scenario. The demo default is now frozen to the actually validated `0.25 m` case so no-argument reproduction matches the recorded evidence.

```text
full-landing contract tests = PASS
deterministic repeatability = PASS (two identical terminal metric sets)
S4 no-video regression      = 9/9 cases PASS
full regression             = 176 passed + 21 subtests
git diff --check            = PASS
```

This supplementary evidence does not replace S5 16-env vectorization validation, S6 moving/rotating contact-point benchmark, S11 rotating-yaw benchmark, PPO evidence, or real-world evidence.

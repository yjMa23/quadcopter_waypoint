# Quadcopter Ship Landing Heave

## Stage

```text
Phase 6A-Heave-Precision
```

This task is a new independent task forked from the frozen Phase 5D-DeckContact baseline.

It does **not** modify `quadrotor_ship_landing/`.

## Task ID

```text
Isaac-Quadcopter-ShipLanding-Heave-Direct-v0
```

## Freeze status

Phase 6A is frozen as the last marker / geometric-contact-proxy heave baseline. It is not a physical deck task and must not be used as evidence for real collision or contact sensing.

The terminal-state evaluator was repaired before freezing. `DirectRLEnv.step()` resets completed environments before returning, so the environment now latches terminal robot and pad state before reset. The evaluator reads that exact latch rather than reset tensors or a pre-step approximation.

Three-seed formal evaluation (`seed=42,43,44`, 256 episodes each):

| Metric | Aggregate |
| --- | ---: |
| episodes | 768 |
| landing success | 98.83% |
| crash | 0.00% |
| timeout | 1.17% |
| touchdown distance mean | 0.0661 m |
| touchdown distance P95 | 0.0970 m |
| touchdown relative speed mean | 0.1323 m/s |
| terminal horizontal relative speed mean | 0.0908 m/s |
| terminal vertical relative speed mean | 0.0114 m/s |

Evidence:

```text
benchmarks/phase6a_heave_precision/summary.json
```

## Goal

Add visible vertical deck heave while improving landing accuracy near the pad center.

```text
moving pad xy translation + visible deck z sinusoidal heave + DeckContact success proxy + tighter xy landing radius
```

## Heave model

```text
z_pad(t) = pad_base_height + A sin(phase)
vz_pad(t) = A omega cos(phase)
```

Current visible-heave curriculum:

```text
pad_base_height = 0.16 m
pad_heave_amplitude_min = 0.08 m
pad_heave_amplitude_max = 0.12 m
pad_heave_frequency_min = 0.18 Hz
pad_heave_frequency_max = 0.30 Hz
```

This means the deck has a peak-to-peak vertical motion of about `0.16–0.24 m`, which is visible in GUI playback.

The deck still has no roll or pitch in this stage.

## Key parameters

```text
align_radius = 0.25
align_height_min = 0.50
align_height_max = 1.00
align_max_horizontal_speed = 0.30
align_hold_steps = 8

landing_success_radius = 0.10
landing_contact_clearance = 0.060
max_landing_surface_penetration = 0.010
landing_contact_target_clearance = 0.005
landing_success_horizontal_rel_vel = 0.16
```

Center-precision reward hooks are implemented in code for future experiments:

```text
near_center_height = 0.35
center_precision_reward_scale = -8.0
center_precision_square_reward_scale = -20.0
```

However, the current recommended checkpoint is **not** the fine-tuned center-reward checkpoint. See the failed fine-tuning record below.

## Current recommended checkpoint

The current recommended checkpoint remains the Phase 5D checkpoint:

```text
logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

This checkpoint is used with the current Heave-Precision task condition:

```text
landing_success_radius = 0.10
```

## Evaluation command

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Heave-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth \
  --num_envs=64 \
  --episodes=256 \
  --seed=42 \
  --csv logs/rl_games/quadcopter_ship_landing_heave/p6a_freeze_seed42.csv \
  --headless
```

## Seed-42 evaluation detail

The table below is the repaired terminal-latch evaluation for seed 42. The three-seed aggregate above is the frozen acceptance result.

256-episode evaluation:

| Metric | Phase 6A-Heave-Precision |
| --- | ---: |
| landing success rate | 98.44% |
| align success rate | 99.61% |
| crash rate | 0% |
| timeout rate | 1.56% |
| mean touchdown distance | 0.0677 m |
| touchdown distance P50 | 0.0717 m |
| touchdown distance P90 | 0.0955 m |
| touchdown distance P95 | 0.0966 m |
| mean touchdown rel vel | 0.1357 m/s |
| mean final horizontal relative speed | 0.0919 m/s |
| mean terminal vertical relative speed | 0.0113 m/s |
| mean landing time | 3.8572 s |
| high-speed pad bucket success rate | 99.32% |
| high-speed pad bucket touchdown distance | 0.0705 m |

Pad-speed bucket detail:

| pad speed bucket | n | success rate | touchdown distance mean |
| --- | ---: | ---: | ---: |
| 0.00-0.05 | 8 | 87.50% | 0.0704 m |
| 0.05-0.10 | 36 | 97.22% | 0.0703 m |
| 0.10-0.15 | 65 | 98.46% | 0.0597 m |
| >=0.15 | 147 | 99.32% | 0.0705 m |

## Comparison with visible-heave baseline

| Metric | Heave visible baseline, radius 0.16 | Heave precision, radius 0.10 |
| --- | ---: | ---: |
| landing success rate | 98.83% | 98.44% |
| crash rate | 0% | 0% |
| timeout rate | 1.17% | 1.56% |
| mean touchdown distance | 0.0859 m | 0.0677 m |
| touchdown distance P90 | 0.1489 m | 0.0955 m |
| touchdown distance P95 | 0.1523 m | 0.0966 m |
| mean touchdown rel vel | 0.1434 m/s | 0.1357 m/s |
| mean final horizontal speed | 0.1102 m/s | 0.0954 m/s |

Conclusion:

```text
landing_success_radius = 0.10 is the current best adopted balance:
  - visibly closer to pad center
  - all successful landings are within 10 cm xy error
  - success rate remains above 98%
  - crash rate remains 0%
```

## Center-reward fine-tuning attempt

A Precision v2 fine-tuning run was executed with center-precision rewards added to the Heave task.

Training command:

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-Heave-Direct-v0 \
  --num_envs=1024 \
  --headless \
  --max_iterations=700 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

Training directory:

```text
logs/rl_games/quadcopter_ship_landing_heave/2026-06-30_22-10-25
```

Evaluated checkpoints:

| Checkpoint | Success | Crash | Timeout | Mean touchdown distance | Mean touchdown rel vel | Adopted? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `nn/quadcopter_ship_landing_heave.pth` | 97.27% | 0.78% | 1.95% | 0.0729 m | 0.1464 m/s | No |
| `nn/last_quadcopter_ship_landing_heave_ep_675_rew_29.709484.pth` | 97.66% | 1.17% | 1.17% | 0.0706 m | 0.1584 m/s | No |
| `nn/last_quadcopter_ship_landing_heave_ep_700_rew_31.02413.pth` | 97.27% | 1.95% | 0.78% | 0.0789 m | 0.1413 m/s | No |

Conclusion:

```text
The center reward direction is reasonable, but this fine-tuning run caused policy drift:
  - success rate dropped below the adopted baseline
  - crash rate became non-zero
  - relative touchdown velocity increased in some checkpoints

Therefore, no Precision v2 trained checkpoint is adopted.
```

The current main version remains:

```text
Heave task code + landing_success_radius = 0.10 + Phase 5D checkpoint
```

## Tried but not selected

A stricter precision setting was tested without fine-tuning:

```text
landing_success_radius = 0.08
```

Result:

```text
landing_success_rate: 93.75%
timeout_rate: 5.86%
crash_rate: 0.39%
mean_touchdown_distance: 0.0556 m
```

Although the mean touchdown distance improved to about `5.6 cm`, the success rate dropped too much and a small crash rate appeared, so `0.08 m` is not used as the main version.

The earlier visible-heave version used:

```text
landing_success_radius = 0.16
```

It was more permissive but visually not centered enough, with mean touchdown distance about `8.6 cm` and P95 about `15.2 cm`.

## Important heave iteration note

The first heave version used a very mild setting:

```text
pad_base_height = 0.05 m
pad_heave_amplitude_max = 0.04 m
```

This was numerically valid, but the GUI looked almost static.

Then the amplitude was increased to `0.08–0.12 m` while keeping `pad_base_height = 0.05 m`. This made the heave visible, but the deck could move below the ground plane, which caused many crashes.

The current version fixes that by setting:

```text
pad_base_height = 0.16 m
pad_heave_amplitude = 0.08–0.12 m
```

This keeps the deck above the ground while preserving visible vertical motion.

## Playback command

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-Heave-Direct-v0 \
  --num_envs=1 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

## Current conclusion

```text
Phase 6A-Heave-Precision adopted version:
  landing_success_radius = 0.10
  checkpoint = Phase 5D ep650 checkpoint
  success rate = 98.44%
  crash rate = 0%
  timeout rate = 1.56%
  touchdown distance mean = 0.0677 m
  touchdown distance P95 = 0.0966 m
  GUI-visible heave = yes
```

This is the current precision-focused heave baseline. If further center improvement is required, the next attempt should use a gentler curriculum or lower center-reward weight, not the rejected Precision v2 checkpoints.

## Next stage

Do not add roll / pitch to this directory. Create a new task again, for example:

```text
quadrotor_ship_landing_deck_motion
Isaac-Quadcopter-ShipLanding-DeckMotion-Direct-v0
```

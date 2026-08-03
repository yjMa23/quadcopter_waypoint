# Phase 5D-DeckContact Historical Archive

> Historical record only. This file preserves the state known at the Phase 5D freeze and is not the current project plan.

This backup records the current usable ShipLanding version before moving on to deck heave / roll / pitch.

## Date

2026-06-30

## Stable task

```text
Isaac-Quadcopter-ShipLanding-Direct-v0
```

## Current stage

```text
Phase 5D-DeckContact: moving landing pad with deck-contact proxy success condition
```

The task is still a uniformly translating pad. No heave, roll, or pitch has been added yet.

## Recommended checkpoint

```text
logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

## Recommended eval CSV

```text
logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/eval_metrics_deck_contact_v4_ep650.csv
```

## Key result

256-episode evaluation:

```text
landing_success_rate: 99.22%
align_success_rate:   99.22%
crash_rate:           0%
timeout_rate:         0.78%
```

Quality metrics:

```text
mean_touchdown_distance:        0.0888 m
mean_touchdown_rel_vel:         0.1376 m/s
mean_final_horizontal_speed:    0.1295 m/s
mean_landing_time:              2.9768 s
```

High-speed pad bucket:

```text
pad_speed >= 0.15 m/s:
success_rate: 100%
touchdown_distance_mean: 0.0936 m
```

Deck-contact proxy clearance:

```text
landing_surface_clearance mean:   0.0116 m
landing_surface_clearance median: 0.0088 m
landing_surface_clearance P95:    0.0403 m
```

## Core parameters

```text
pad_velocity_xy_range = 0.20

align_radius = 0.25
align_max_horizontal_speed = 0.30
align_hold_steps = 8

landing_success_radius = 0.16
landing_success_rel_vel = 0.32
landing_success_horizontal_rel_vel = 0.16
landing_success_ang_vel = 0.9
landing_success_upright = 0.93
landing_success_hold_steps = 4

pad_thickness = 0.03
robot_landing_surface_offset = 0.035
landing_contact_clearance = 0.060
max_landing_surface_penetration = 0.010
landing_contact_target_clearance = 0.005
contact_clearance_reward_scale = -8.0
```

## Deck-contact proxy definition

```text
pad_surface_height = pad_z + pad_thickness / 2
robot_bottom_height = root_z - robot_landing_surface_offset
landing_surface_clearance = robot_bottom_height - pad_surface_height
```

Landing success requires:

```text
horizontal_error < landing_success_radius
landing_surface_clearance < landing_contact_clearance
landing_surface_clearance > -max_landing_surface_penetration
rel_vel < landing_success_rel_vel
horizontal_speed < landing_success_horizontal_rel_vel
ang_vel_norm < landing_success_ang_vel
upright > landing_success_upright
```

## Training source

This version was fine-tuned from the Phase 5C+ stable checkpoint:

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

Training command used for Phase 5D v1:

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --num_envs=1024 \
  --headless \
  --max_iterations=850 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

## Evaluation command

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/eval_metrics_deck_contact_v4_ep650.csv \
  --headless
```

## Playback command

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --num_envs=1 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

## Tried but not selected

```text
DeckContact v1 best:
  checkpoint: logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/quadcopter_ship_landing.pth
  success: 92.58%
  reason: timeout 7.42%, not stable enough.

DeckContact v1 ep850:
  checkpoint: logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_850_rew_36.64187.pth
  success: 82.03%
  reason: later training drifted / degraded.

DeckContact v2 ep650:
  success: 98.83%
  reason: close, but v4 has better success rate.

DeckContact v5 ep650:
  success: 99.22%
  reason: success same as v4, but touchdown distance was slightly worse.
```

## Why this version matters

Phase 5D upgrades the task from a simple geometric height threshold to a deck-contact proxy. This is a better baseline before adding deck heave, roll, and pitch.

Remaining limitation:

```text
The landing pad is still a marker/visual target, not a rigid deck with collision/contact sensors.
```

Recommended next stage:

```text
Historical plan at the time of this archive:

```text
Phase 6A: add deck heave / vertical sinusoidal motion while keeping roll and pitch fixed.
```

Actual later progression:

```text
Phase 6A: Heave Precision marker/contact-proxy baseline, completed and frozen.
Phase 6B: horizontal PhysicalDeck with rigid/kinematic deck and contact sensing, completed.
Next independent stage: add small roll/pitch using deck-frame landing conditions.
```
```

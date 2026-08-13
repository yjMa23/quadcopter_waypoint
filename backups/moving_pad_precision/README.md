# Moving-Pad Precision Baseline Historical Archive

> Historical record only. This file is not a current task README and must not override the frozen Deck-Contact Proxy Baseline, Heave-Precision Proxy, or Physical Deck documentation.

This archive records the stable ShipLanding state before starting Smooth-Control Attempt.

## Date

2026-06-29

## Stable task

```text
Isaac-Quadcopter-ShipLanding-Direct-v0
```

## Stable checkpoint

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

## Stable eval CSV

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/eval_metrics_moving_pad_precision.csv
```

## Key environment parameters

```text
pad_velocity_xy_range = 0.20
pad initial x/y range = [-0.5, 0.5] m

landing_success_radius = 0.16
landing_success_height = 0.10
landing_success_rel_vel = 0.30
landing_success_ang_vel = 0.9
landing_success_hold_steps = 4

descent_speed_limit = 0.22
progress_reward_scale = 5.0
horizontal_error_reward_scale = -2.5
rel_vel_reward_scale = -0.6
descent_vel_reward_scale = -3.0
landing_bonus = 40.0
post_align_descent_reward_scale = 6.0
```

## Moving-Pad Precision Baseline evaluation summary

```text
align_success_rate: 100%
landing_success_rate: 100%
crash_rate: 0%
timeout_rate: 0%
mean_touchdown_distance: 0.1002 m
touchdown_distance P95: 0.1533 m
mean_touchdown_rel_vel: 0.1453 m/s
touchdown_rel_vel P95: 0.2646 m/s
mean_landing_time: 3.6684 s
mean_max_descent_speed: 0.8263 m/s
mean_descent_speed: 0.2769 m/s
mean_pad_speed: 0.1618 m/s
```

## Files represented by this backup

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/quadrotor_ship_landing_env.py
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/README.md
scripts/rl_games/eval_metrics.py
scripts/rl_games/README.md
README.md
```

## Restore note

To recover the trained behavior, use the stable checkpoint above with the parameters listed here. This directory is intentionally outside the task package tree so it will not be imported by Isaac Lab task registration.

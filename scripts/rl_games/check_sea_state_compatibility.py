#!/usr/bin/env python3
"""Regression-check SeaState compatibility mode against PhysicalDeckAttitude motion."""

from __future__ import annotations

import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16, choices=(1, 16))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_physical_deck_attitude.quadrotor_ship_landing_physical_deck_attitude_env import (
    QuadcopterShipLandingPhysicalDeckAttitudeEnv,
)
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_sea_state.quadrotor_ship_landing_sea_state_env import (
    QuadcopterShipLandingSeaStateEnvCfg,
)

TASK_ID = "Isaac-Quadcopter-ShipLanding-SeaState-Direct-v0"


def main() -> None:
    cfg = QuadcopterShipLandingSeaStateEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.sea_state_mode = "compatibility"
    env = gym.make(TASK_ID, cfg=cfg)
    task = env.unwrapped
    env.reset(seed=31415)
    env_ids = task._robot._ALL_INDICES

    # Force deterministic parent parameters and compare the child method with the exact frozen parent implementation.
    task._deck_motion_time[:] = torch.linspace(0.0, 3.0, task.num_envs, device=task.device)
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w[:, 0] = 0.10
    task._deck_xy_velocity_w[:, 1] = -0.05
    task._pad_heave_amp[:] = 0.10
    task._pad_heave_omega[:] = 2.0 * math.pi * 0.20
    task._deck_heave_phase0[:] = 0.31
    task._deck_roll_amp[:] = math.radians(4.0)
    task._deck_roll_omega[:] = 2.0 * math.pi * 0.11
    task._deck_roll_phase0[:] = 0.73
    task._deck_pitch_amp[:] = math.radians(3.0)
    task._deck_pitch_omega[:] = 2.0 * math.pi * 0.13
    task._deck_pitch_phase0[:] = 1.17

    child_pose, child_velocity, child_roll, child_pitch = task._compute_absolute_deck_state(env_ids)
    parent_pose, parent_velocity, parent_roll, parent_pitch = (
        QuadcopterShipLandingPhysicalDeckAttitudeEnv._compute_absolute_deck_state(task, env_ids)
    )
    pose_error = torch.max(torch.abs(child_pose - parent_pose))
    velocity_error = torch.max(torch.abs(child_velocity - parent_velocity))
    roll_error = torch.max(torch.abs(child_roll - parent_roll))
    pitch_error = torch.max(torch.abs(child_pitch - parent_pitch))
    torch.testing.assert_close(child_pose, parent_pose, atol=0.0, rtol=0.0)
    torch.testing.assert_close(child_velocity, parent_velocity, atol=0.0, rtol=0.0)
    torch.testing.assert_close(child_roll, parent_roll, atol=0.0, rtol=0.0)
    torch.testing.assert_close(child_pitch, parent_pitch, atol=0.0, rtol=0.0)

    report = {
        "task_id": TASK_ID,
        "mode": "compatibility",
        "num_envs": task.num_envs,
        "observation_dimension": task.single_observation_space["policy"].shape[0],
        "action_dimension": task.single_action_space.shape[0],
        "max_pose_abs_error": float(pose_error),
        "max_velocity_abs_error": float(velocity_error),
        "max_roll_abs_error": float(roll_error),
        "max_pitch_abs_error": float(pitch_error),
        "status": "PASS",
    }
    print(json.dumps(report, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

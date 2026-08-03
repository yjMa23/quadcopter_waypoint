#!/usr/bin/env python3
"""Validate Phase-6C deck motion and independent deck/ground physical contacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, choices=(1, 16))
parser.add_argument("--motion_steps", type=int, default=500)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_physical_deck_attitude.quadrotor_ship_landing_physical_deck_attitude_env import (
    QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import local_to_world_position

TASK_ID = "Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0"


def advance_physics(task, control_actions: torch.Tensor | None = None, control_steps: int = 1) -> None:
    """Advance using DirectRLEnv's exact decimation order without running done/reset logic."""
    for _ in range(control_steps):
        if control_actions is not None:
            task._pre_physics_step(control_actions)
        for _ in range(task.cfg.decimation):
            task._apply_action()
            task.scene.write_data_to_sim()
            task.sim.step(render=False)
            task.scene.update(dt=task.physics_dt)


def filtered_forces(task) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        task._filtered_contact_force(task._deck_contact_sensor),
        task._filtered_contact_force(task._ground_contact_sensor),
    )


def place_robot(task, position_w: torch.Tensor, quat_wxyz: torch.Tensor) -> None:
    pose = torch.cat((position_w, quat_wxyz), dim=-1)
    velocity = torch.zeros(task.num_envs, 6, device=task.device)
    task._robot.write_root_pose_to_sim(pose)
    task._robot.write_root_velocity_to_sim(velocity)
    task._thrust.zero_()
    task._moment.zero_()


def main() -> None:
    cfg = QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.deck_roll_amplitude_min_deg = 5.0
    cfg.deck_roll_amplitude_max_deg = 5.0
    cfg.deck_pitch_amplitude_min_deg = 5.0
    cfg.deck_pitch_amplitude_max_deg = 5.0
    cfg.deck_roll_frequency_min = 0.10
    cfg.deck_roll_frequency_max = 0.10
    cfg.deck_pitch_frequency_min = 0.12
    cfg.deck_pitch_frequency_max = 0.12
    cfg.strict_deck_motion_consistency = True

    env = gym.make(TASK_ID, cfg=cfg)
    task = env.unwrapped
    env.reset(seed=42)
    env_ids = task._robot._ALL_INDICES

    # Replace random episode parameters with a deterministic absolute-time trajectory so direction,
    # range, drift, and pose/velocity consistency are reproducible in both 1-env and 16-env checks.
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w[:, 0] = 0.10
    task._deck_xy_velocity_w[:, 1] = -0.05
    task._pad_heave_amp[:] = 0.10
    task._pad_heave_omega[:] = 2.0 * math.pi * 0.10
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp[:] = math.radians(5.0)
    task._deck_roll_omega[:] = 2.0 * math.pi * 0.10
    task._deck_roll_phase0.zero_()
    task._deck_pitch_amp[:] = math.radians(5.0)
    task._deck_pitch_omega[:] = 2.0 * math.pi * 0.12
    task._deck_pitch_phase0[:] = 0.5 * math.pi
    task._write_absolute_deck_state(env_ids)

    hover_action = 2.0 / task.cfg.thrust_to_weight - 1.0
    actions = torch.zeros(task.num_envs, 4, device=task.device)
    actions[:, 0] = hover_action
    traces = {name: [] for name in ("x", "y", "z", "roll", "pitch", "angular_speed", "min_bottom")}
    bottom_corners_local = torch.tensor(
        [
            [x, y, -0.5 * task.cfg.pad_thickness]
            for x in (-0.5 * task.cfg.deck_size_x, 0.5 * task.cfg.deck_size_x)
            for y in (-0.5 * task.cfg.deck_size_y, 0.5 * task.cfg.deck_size_y)
        ],
        dtype=torch.float32,
        device=task.device,
    )
    for _ in range(args.motion_steps):
        advance_physics(task, actions)
        deck_pos = task._deck.data.root_pos_w[0]
        deck_quat = task._deck.data.root_quat_w[0]
        corners_w = local_to_world_position(
            deck_pos.unsqueeze(0).expand(4, -1), deck_quat.unsqueeze(0).expand(4, -1), bottom_corners_local
        )
        traces["x"].append(float(deck_pos[0]))
        traces["y"].append(float(deck_pos[1]))
        traces["z"].append(float(deck_pos[2]))
        traces["roll"].append(float(task._deck_roll[0]))
        traces["pitch"].append(float(task._deck_pitch[0]))
        traces["angular_speed"].append(float(torch.linalg.norm(task._deck.data.root_ang_vel_w[0])))
        traces["min_bottom"].append(float(torch.min(corners_w[:, 2])))

    ground_required = task.cfg.ground_slab_top_height + task.cfg.deck_ground_safety_margin
    motion = {
        "x_start": traces["x"][0],
        "x_end": traces["x"][-1],
        "y_start": traces["y"][0],
        "y_end": traces["y"][-1],
        "z_min": min(traces["z"]),
        "z_max": max(traces["z"]),
        "roll_min_deg": math.degrees(min(traces["roll"])),
        "roll_max_deg": math.degrees(max(traces["roll"])),
        "pitch_min_deg": math.degrees(min(traces["pitch"])),
        "pitch_max_deg": math.degrees(max(traces["pitch"])),
        "max_angular_speed": max(traces["angular_speed"]),
        "minimum_deck_bottom_corner_height": min(traces["min_bottom"]),
        "required_minimum_height": ground_required,
        "max_position_consistency_error": float(torch.max(task._max_deck_position_consistency_error)),
        "max_orientation_consistency_error_deg": math.degrees(
            float(torch.max(task._max_deck_orientation_consistency_error))
        ),
        "max_linear_velocity_consistency_error": float(
            torch.max(task._max_deck_linear_velocity_consistency_error)
        ),
        "max_angular_velocity_consistency_error": float(
            torch.max(task._max_deck_angular_velocity_consistency_error)
        ),
    }
    assert motion["x_end"] > motion["x_start"]
    assert motion["y_end"] < motion["y_start"]
    assert motion["roll_min_deg"] < -4.9 and motion["roll_max_deg"] > 4.9
    assert motion["pitch_min_deg"] < -4.9 and motion["pitch_max_deg"] > 4.9
    assert motion["minimum_deck_bottom_corner_height"] > ground_required

    # Freeze the deck flat before controlled contact placement.
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w.zero_()
    task._pad_heave_amp.zero_()
    task._deck_roll_amp.zero_()
    task._deck_pitch_amp.zero_()
    task._deck_heave_phase0.zero_()
    task._deck_roll_phase0.zero_()
    task._deck_pitch_phase0.zero_()
    task._write_absolute_deck_state(env_ids)
    identity = torch.zeros(task.num_envs, 4, device=task.device)
    identity[:, 0] = 1.0

    deck_position = task._deck.data.root_pos_w.clone()
    deck_position[:, 2] += 0.5 * task.cfg.pad_thickness + task.cfg.robot_landing_surface_offset - 0.003
    place_robot(task, deck_position, identity)
    advance_physics(task, control_steps=30)
    deck_force_on_deck, ground_force_on_deck = filtered_forces(task)

    clear_position = task._terrain.env_origins.clone()
    clear_position[:, 2] = 1.5
    place_robot(task, clear_position, identity)
    advance_physics(task, control_steps=20)

    ground_position = task._terrain.env_origins.clone()
    ground_position[:, 0] += 0.70
    ground_position[:, 2] = (
        task.cfg.ground_slab_top_height + task.cfg.robot_landing_surface_offset - 0.003
    )
    place_robot(task, ground_position, identity)
    advance_physics(task, control_steps=30)
    deck_force_on_ground, ground_force_on_ground = filtered_forces(task)

    contacts = {
        "deck_test_deck_force_min": float(torch.min(deck_force_on_deck)),
        "deck_test_ground_force_max": float(torch.max(ground_force_on_deck)),
        "ground_test_ground_force_min": float(torch.min(ground_force_on_ground)),
        "ground_test_deck_force_max": float(torch.max(deck_force_on_ground)),
        "contact_force_threshold": task.cfg.contact_force_threshold,
    }
    assert contacts["deck_test_deck_force_min"] > task.cfg.contact_force_threshold
    assert contacts["deck_test_ground_force_max"] <= task.cfg.contact_force_threshold
    assert contacts["ground_test_ground_force_min"] > task.cfg.contact_force_threshold
    assert contacts["ground_test_deck_force_max"] <= task.cfg.contact_force_threshold

    report = {
        "task_id": TASK_ID,
        "num_envs": task.num_envs,
        "motion_steps": args.motion_steps,
        "step_dt": task.step_dt,
        "motion": motion,
        "contacts": contacts,
        "status": "PASS",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"[INFO] Saved report to: {output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

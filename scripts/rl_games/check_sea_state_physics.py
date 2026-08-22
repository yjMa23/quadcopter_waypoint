#!/usr/bin/env python3
"""Validate stochastic SeaState deck motion, safety envelopes, and pose/velocity consistency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, choices=(1, 16))
parser.add_argument("--motion_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--profile", type=str, default=None)
parser.add_argument("--profiles", type=Path, default=Path("benchmarks/sea_state/profiles.yaml"))
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_sea_state.quadrotor_ship_landing_sea_state_env import (
    QuadcopterShipLandingSeaStateEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import local_to_world_position
from quadcopter_waypoint.utils.sea_state_profiles import apply_sea_state_profile, load_sea_state_profiles

TASK_ID = "Isaac-Quadcopter-ShipLanding-SeaState-Direct-v0"


def advance_physics(task, actions: torch.Tensor) -> None:
    task._pre_physics_step(actions)
    for _ in range(task.cfg.decimation):
        task._apply_action()
        task.scene.write_data_to_sim()
        task.sim.step(render=False)
        task.scene.update(dt=task.physics_dt)


def main() -> None:
    cfg = QuadcopterShipLandingSeaStateEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.strict_deck_motion_consistency = True
    if args.profile is not None:
        profiles = load_sea_state_profiles(args.profiles)
        if args.profile not in profiles:
            raise ValueError(f"Unknown Sea-State profile {args.profile!r}")
        apply_sea_state_profile(cfg, args.profile, profiles[args.profile])
    env = gym.make(TASK_ID, cfg=cfg)
    task = env.unwrapped
    env.reset(seed=args.seed)

    assert task.single_observation_space["policy"].shape == (22,)
    assert task.single_action_space.shape == (4,)
    assert task.cfg.sea_state_mode == "stochastic"

    hover_action = 2.0 / task.cfg.thrust_to_weight - 1.0
    actions = torch.zeros(task.num_envs, 4, device=task.device)
    actions[:, 0] = hover_action
    corners_local = torch.tensor(
        [
            [x, y, -0.5 * task.cfg.pad_thickness]
            for x in (-0.5 * task.cfg.deck_size_x, 0.5 * task.cfg.deck_size_x)
            for y in (-0.5 * task.cfg.deck_size_y, 0.5 * task.cfg.deck_size_y)
        ],
        dtype=torch.float32,
        device=task.device,
    )

    min_bottom = float("inf")
    max_step_position_delta = 0.0
    max_step_quaternion_angle = 0.0
    max_linear_speed = 0.0
    previous_pose = task._deck.data.root_pose_w.clone()
    for _ in range(args.motion_steps):
        advance_physics(task, actions)
        pose = task._deck.data.root_pose_w
        velocity = torch.cat((task._deck.data.root_lin_vel_w, task._deck.data.root_ang_vel_w), dim=-1)
        if not torch.all(torch.isfinite(pose)) or not torch.all(torch.isfinite(velocity)):
            raise RuntimeError("SeaState deck produced NaN/Inf pose or velocity")
        max_linear_speed = max(max_linear_speed, float(torch.max(torch.linalg.norm(velocity[:, :3], dim=-1))))

        corner_count = corners_local.shape[0]
        deck_pos = pose[:, :3].unsqueeze(1).expand(-1, corner_count, -1).reshape(-1, 3)
        deck_quat = pose[:, 3:7].unsqueeze(1).expand(-1, corner_count, -1).reshape(-1, 4)
        corners = corners_local.unsqueeze(0).expand(task.num_envs, -1, -1).reshape(-1, 3)
        corners_w = local_to_world_position(deck_pos, deck_quat, corners)
        min_bottom = min(min_bottom, float(torch.min(corners_w[:, 2])))

        position_delta = torch.linalg.norm(pose[:, :3] - previous_pose[:, :3], dim=-1)
        max_step_position_delta = max(max_step_position_delta, float(torch.max(position_delta)))
        quat_dot = torch.abs(torch.sum(pose[:, 3:7] * previous_pose[:, 3:7], dim=-1)).clamp(0.0, 1.0)
        quaternion_angle = 2.0 * torch.acos(quat_dot)
        max_step_quaternion_angle = max(max_step_quaternion_angle, float(torch.max(quaternion_angle)))
        previous_pose = pose.clone()

    count = task._sea_motion_sample_count.clamp_min(1.0)
    heave_rms = torch.sqrt(task._sea_heave_sq_sum / count)
    roll_rms = torch.sqrt(task._sea_roll_sq_sum / count)
    pitch_rms = torch.sqrt(task._sea_pitch_sq_sum / count)
    required_bottom = task.cfg.ground_slab_top_height + task.cfg.deck_ground_safety_margin

    report = {
        "task_id": TASK_ID,
        "seed": args.seed,
        "profile": task.cfg.sea_state_benchmark_profile,
        "num_envs": task.num_envs,
        "motion_steps": args.motion_steps,
        "step_dt": task.step_dt,
        "sea_state": {
            "hs_min": float(torch.min(task._sea_hs)),
            "hs_max": float(torch.max(task._sea_hs)),
            "tp_min": float(torch.min(task._sea_tp)),
            "tp_max": float(torch.max(task._sea_tp)),
            "gamma_min": float(torch.min(task._sea_gamma)),
            "gamma_max": float(torch.max(task._sea_gamma)),
            "heading_min_deg": math.degrees(float(torch.min(task._sea_heading))),
            "heading_max_deg": math.degrees(float(torch.max(task._sea_heading))),
            "heave_rms_max": float(torch.max(heave_rms)),
            "heave_max_abs": float(torch.max(task._sea_heave_max_abs)),
            "roll_rms_max_deg": math.degrees(float(torch.max(roll_rms))),
            "roll_max_abs_deg": math.degrees(float(torch.max(task._sea_roll_max_abs))),
            "pitch_rms_max_deg": math.degrees(float(torch.max(pitch_rms))),
            "pitch_max_abs_deg": math.degrees(float(torch.max(task._sea_pitch_max_abs))),
            "linear_velocity_max": max_linear_speed,
            "heave_velocity_max_abs": float(torch.max(task._sea_heave_velocity_max_abs)),
            "roll_rate_max_abs": float(torch.max(task._sea_roll_rate_max_abs)),
            "pitch_rate_max_abs": float(torch.max(task._sea_pitch_rate_max_abs)),
            "deck_angular_speed_max": float(torch.max(task._sea_deck_angular_speed_max)),
            "heave_scale_min": float(torch.min(task._sea_heave_scale)),
            "roll_scale_min": float(torch.min(task._sea_roll_scale)),
            "pitch_scale_min": float(torch.min(task._sea_pitch_scale)),
        },
        "safety": {
            "minimum_deck_bottom_corner_height": min_bottom,
            "required_minimum_height": required_bottom,
            "max_step_position_delta": max_step_position_delta,
            "max_step_orientation_delta_deg": math.degrees(max_step_quaternion_angle),
        },
        "consistency": {
            "max_position_error": float(torch.max(task._max_deck_position_consistency_error)),
            "max_orientation_error_deg": math.degrees(float(torch.max(task._max_deck_orientation_consistency_error))),
            "max_linear_velocity_error": float(torch.max(task._max_deck_linear_velocity_consistency_error)),
            "max_angular_velocity_error": float(torch.max(task._max_deck_angular_velocity_consistency_error)),
        },
        "status": "PASS",
    }

    assert min_bottom > required_bottom
    assert float(torch.max(task._sea_heave_max_abs)) <= task.cfg.sea_state_max_heave_m + 1.0e-5
    assert float(torch.max(task._sea_roll_max_abs)) <= math.radians(task.cfg.sea_state_max_roll_deg) + 1.0e-5
    assert float(torch.max(task._sea_pitch_max_abs)) <= math.radians(task.cfg.sea_state_max_pitch_deg) + 1.0e-5
    # A control-step jump larger than these limits would indicate an unintended reset/teleport rather than the
    # configured continuous finite-spectrum motion. The default benchmark remains well below both thresholds.
    assert max_step_position_delta < 0.10
    assert max_step_quaternion_angle < math.radians(15.0)

    text = json.dumps(report, indent=2)
    print(text, flush=True)
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

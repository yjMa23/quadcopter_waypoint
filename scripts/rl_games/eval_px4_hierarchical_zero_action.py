#!/usr/bin/env python3
"""Evaluate the deterministic zero-relative-action baseline for the M2 hierarchical task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes_per_case", type=int, default=16)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_px4_hierarchical.quadrotor_ship_landing_px4_hierarchical_env import (
    QuadcopterShipLandingPx4HierarchicalEnvCfg,
)

TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0"


def _identity_quat(task) -> torch.Tensor:
    quat = torch.zeros(task.num_envs, 4, device=task.device)
    quat[:, 0] = 1.0
    return quat


def _set_flat_deck(task, velocity_xy=(0.0, 0.0), heave_amplitude=0.0, heave_frequency=0.10) -> None:
    env_ids = task._robot._ALL_INDICES
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w[:, 0] = velocity_xy[0]
    task._deck_xy_velocity_w[:, 1] = velocity_xy[1]
    task._pad_heave_amp[:] = heave_amplitude
    task._pad_heave_omega[:] = 2.0 * math.pi * heave_frequency
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp.zero_()
    task._deck_pitch_amp.zero_()
    task._deck_roll_omega.zero_()
    task._deck_pitch_omega.zero_()
    task._deck_roll_phase0.zero_()
    task._deck_pitch_phase0.zero_()
    task._write_absolute_deck_state(env_ids)


def _set_attitude_deck(task) -> None:
    env_ids = task._robot._ALL_INDICES
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w[:, 0] = 0.12
    task._deck_xy_velocity_w[:, 1] = -0.06
    task._pad_heave_amp[:] = 0.06
    task._pad_heave_omega[:] = 2.0 * math.pi * 0.10
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp[:] = math.radians(5.0)
    task._deck_pitch_amp[:] = math.radians(4.0)
    task._deck_roll_omega[:] = 2.0 * math.pi * 0.10
    task._deck_pitch_omega[:] = 2.0 * math.pi * 0.12
    task._deck_roll_phase0.zero_()
    task._deck_pitch_phase0[:] = 0.5 * math.pi
    task._write_absolute_deck_state(env_ids)


def _place_robot_above_deck(task, height_above_deck: float = 0.60) -> None:
    robot_pos = task._deck_pose_command_w[:, :3].clone()
    robot_pos[:, 2] += 0.5 * task.cfg.pad_thickness + task.cfg.robot_landing_surface_offset + height_above_deck
    pose = torch.cat((robot_pos, _identity_quat(task)), dim=-1)
    velocity = torch.zeros(task.num_envs, 6, device=task.device)
    velocity[:, :3] = task._deck_velocity_command_w[:, :3]
    task._robot.write_root_pose_to_sim(pose)
    task._robot.write_root_velocity_to_sim(velocity)
    task._thrust.zero_()
    task._moment.zero_()
    task._previous_relative_velocity_ref_d.zero_()
    task._relative_velocity_ref_d.zero_()
    task._velocity_reference_w.zero_()
    task._velocity_reference_ned.zero_()
    task.episode_length_buf.zero_()
    task.scene.write_data_to_sim()
    task.sim.step(render=False)
    task.scene.update(dt=task.physics_dt)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_case(env, task, name: str, setup) -> dict:
    env.reset(seed=42)
    setup()
    _place_robot_above_deck(task)
    zero_action = torch.zeros(task.num_envs, 3, device=task.device)
    completed: list[dict[str, float | bool]] = []
    nonfinite = False

    for _ in range(task.max_episode_length + 5):
        _, _, terminated, truncated, _ = env.step(zero_action)
        finite = (
            torch.all(torch.isfinite(task._robot.data.root_pos_w))
            & torch.all(torch.isfinite(task._robot.data.root_lin_vel_w))
            & torch.all(torch.isfinite(task._thrust))
            & torch.all(torch.isfinite(task._moment))
        )
        nonfinite = nonfinite or not bool(finite)
        dones = torch.as_tensor(terminated, dtype=torch.bool, device=task.device) | torch.as_tensor(
            truncated, dtype=torch.bool, device=task.device
        )
        done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        for env_id in done_ids.detach().cpu().tolist():
            completed.append(
                {
                    "time_out": bool(task._terminal_time_out[env_id]),
                    "contact": bool(task._last_deck_contact[env_id]),
                    "settled_landing": bool(task._last_successful_settle[env_id]),
                    "hard_contact": bool(task._last_hard_contact[env_id]),
                    "ground_crash": bool(task._last_ground_crash[env_id]),
                    "deck_miss": bool(task._last_deck_miss[env_id]),
                    "relative_velocity_reference_norm_mean": float(
                        task._last_relative_velocity_reference_norm_mean[env_id]
                    ),
                    "relative_velocity_reference_norm_p95": float(
                        task._last_relative_velocity_reference_norm_p95[env_id]
                    ),
                    "relative_velocity_reference_norm_max": float(
                        task._last_relative_velocity_reference_norm_max[env_id]
                    ),
                    "reference_saturation_ratio": float(task._last_reference_saturation_ratio[env_id]),
                    "controller_velocity_tracking_error_mean": float(
                        task._last_controller_velocity_tracking_error_mean[env_id]
                    ),
                    "controller_velocity_tracking_error_max": float(
                        task._last_controller_velocity_tracking_error_max[env_id]
                    ),
                    "controller_acceleration_saturation_ratio": float(
                        task._last_controller_acceleration_saturation_ratio[env_id]
                    ),
                    "controller_tilt_saturation_ratio": float(task._last_controller_tilt_saturation_ratio[env_id]),
                    "controller_thrust_saturation_ratio": float(task._last_controller_thrust_saturation_ratio[env_id]),
                    "controller_body_rate_saturation_ratio": float(
                        task._last_controller_body_rate_saturation_ratio[env_id]
                    ),
                    "controller_moment_saturation_ratio": float(task._last_controller_moment_saturation_ratio[env_id]),
                    "max_desired_tilt_deg": math.degrees(float(task._last_max_desired_tilt[env_id])),
                    "max_body_rate_radps": float(task._last_max_body_rate[env_id]),
                    "max_moment_nm": float(task._last_max_moment[env_id]),
                    "controller_runtime_ms_mean": float(task._last_controller_runtime_ms_mean[env_id]),
                    "controller_runtime_ms_p95": float(task._last_controller_runtime_ms_p95[env_id]),
                    "controller_runtime_ms_max": float(task._last_controller_runtime_ms_max[env_id]),
                }
            )
            if len(completed) >= args.episodes_per_case:
                break
        if len(completed) >= args.episodes_per_case:
            break

    if len(completed) < args.episodes_per_case:
        raise RuntimeError(f"{name}: collected only {len(completed)} completed episodes")

    scalar_mean_keys = (
        "relative_velocity_reference_norm_mean",
        "relative_velocity_reference_norm_p95",
        "reference_saturation_ratio",
        "controller_velocity_tracking_error_mean",
        "controller_acceleration_saturation_ratio",
        "controller_tilt_saturation_ratio",
        "controller_thrust_saturation_ratio",
        "controller_body_rate_saturation_ratio",
        "controller_moment_saturation_ratio",
        "controller_runtime_ms_mean",
        "controller_runtime_ms_p95",
    )
    result: dict[str, float | bool | str | int] = {
        "name": name,
        "episodes": len(completed),
        "nonfinite": nonfinite,
        "timeout_rate": _mean([float(ep["time_out"]) for ep in completed]),
        "contact_rate": _mean([float(ep["contact"]) for ep in completed]),
        "settled_landing_rate": _mean([float(ep["settled_landing"]) for ep in completed]),
        "hard_contact_rate": _mean([float(ep["hard_contact"]) for ep in completed]),
        "ground_crash_rate": _mean([float(ep["ground_crash"]) for ep in completed]),
        "deck_miss_rate": _mean([float(ep["deck_miss"]) for ep in completed]),
    }
    for key in scalar_mean_keys:
        result[key] = _mean([float(ep[key]) for ep in completed])
    for key in (
        "relative_velocity_reference_norm_max",
        "controller_velocity_tracking_error_max",
        "max_desired_tilt_deg",
        "max_body_rate_radps",
        "max_moment_nm",
        "controller_runtime_ms_max",
    ):
        result[key] = max(float(ep[key]) for ep in completed)
    return result


def main() -> None:
    if args.episodes_per_case <= 0 or args.episodes_per_case > args.num_envs:
        raise ValueError("episodes_per_case must be in [1, num_envs] so every case remains deterministic after one reset")

    cfg = QuadcopterShipLandingPx4HierarchicalEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.strict_deck_motion_consistency = False
    cfg.controller_runtime_sync = True
    env = gym.make(TASK_ID, cfg=cfg)
    task = env.unwrapped

    cases = [
        _run_case(env, task, "static_deck", lambda: _set_flat_deck(task)),
        _run_case(env, task, "constant_xy_deck", lambda: _set_flat_deck(task, velocity_xy=(0.20, -0.10))),
        _run_case(env, task, "heave_deck", lambda: _set_flat_deck(task, heave_amplitude=0.08, heave_frequency=0.10)),
        _run_case(env, task, "physical_deck_attitude", lambda: _set_attitude_deck(task)),
    ]

    gates = {
        "zero_action_maps_to_zero_relative_reference": all(
            float(case["relative_velocity_reference_norm_max"]) < 1.0e-6 for case in cases
        ),
        "no_nan_inf": all(not bool(case["nonfinite"]) for case in cases),
        "ground_crash_zero": all(float(case["ground_crash_rate"]) == 0.0 for case in cases),
        "hard_contact_zero": all(float(case["hard_contact_rate"]) == 0.0 for case in cases),
        "reference_not_saturated": all(float(case["reference_saturation_ratio"]) == 0.0 for case in cases),
        "controller_not_persistently_saturated": all(
            max(
                float(case["controller_acceleration_saturation_ratio"]),
                float(case["controller_tilt_saturation_ratio"]),
                float(case["controller_thrust_saturation_ratio"]),
                float(case["controller_body_rate_saturation_ratio"]),
                float(case["controller_moment_saturation_ratio"]),
            )
            < 0.95
            for case in cases
        ),
    }
    report = {
        "task_id": TASK_ID,
        "seed": 42,
        "num_envs": task.num_envs,
        "episodes_per_case": args.episodes_per_case,
        "semantics": "normalized action = [0,0,0] -> deck-relative velocity reference = 0 -> world velocity reference = deck contact-point velocity",
        "interpretation": "deck contact-point velocity following baseline; not zero thrust and not an RL method",
        "cases": cases,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"[INFO] Saved report to: {output}")
    env.close()
    app.close()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

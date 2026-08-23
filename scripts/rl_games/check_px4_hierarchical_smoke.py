#!/usr/bin/env python3
"""Run deterministic controller/reference smoke checks for the PX4-compatible hierarchical task."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, choices=(1, 16))
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


def _place_robot_above_deck(task, height_above_deck: float, match_deck_velocity: bool = False) -> None:
    deck_pos = task._deck_pose_command_w[:, :3].clone()
    robot_pos = deck_pos.clone()
    robot_pos[:, 2] += 0.5 * task.cfg.pad_thickness + task.cfg.robot_landing_surface_offset + height_above_deck
    pose = torch.cat((robot_pos, _identity_quat(task)), dim=-1)
    velocity = torch.zeros(task.num_envs, 6, device=task.device)
    if match_deck_velocity:
        velocity[:, :3] = task._deck_velocity_command_w[:, :3]
    task._robot.write_root_pose_to_sim(pose)
    task._robot.write_root_velocity_to_sim(velocity)
    task._thrust.zero_()
    task._moment.zero_()
    task._previous_relative_velocity_ref_d.zero_()
    task._relative_velocity_ref_d.zero_()
    task._velocity_reference_w.zero_()
    task._velocity_reference_ned.zero_()
    task.scene.write_data_to_sim()
    task.sim.step(render=False)
    task.scene.update(dt=task.physics_dt)


def _advance_control_step(task, action: torch.Tensor, runtime_samples: list[float]) -> None:
    task._pre_physics_step(action)
    for _ in range(task.cfg.decimation):
        if task.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        task._apply_action()
        if task.device.startswith("cuda"):
            torch.cuda.synchronize()
        runtime_samples.append(time.perf_counter() - start)
        task.scene.write_data_to_sim()
        task.sim.step(render=False)
        task.scene.update(dt=task.physics_dt)


def _run_tracking_case(task, name: str, setup, steps: int = 100) -> dict:
    setup()
    _place_robot_above_deck(task, 0.60, match_deck_velocity=True)
    action = torch.zeros(task.num_envs, 3, device=task.device)
    runtime_samples: list[float] = []
    initial_rel = task._robot.data.root_pos_w - task._deck.data.root_pos_w
    max_velocity_error = 0.0
    max_position_drift = 0.0
    max_tilt = 0.0
    max_body_rate = 0.0
    max_moment = 0.0
    any_ground = False
    any_nonfinite = False
    ref_saturated_steps = 0
    controller_saturated_steps = 0

    for _ in range(steps):
        _advance_control_step(task, action, runtime_samples)
        relative_position = task._robot.data.root_pos_w - task._deck.data.root_pos_w
        position_drift = torch.linalg.norm(relative_position - initial_rel, dim=-1)
        velocity_error = torch.linalg.norm(task._velocity_reference_w - task._robot.data.root_lin_vel_w, dim=-1)
        max_position_drift = max(max_position_drift, float(position_drift.max()))
        max_velocity_error = max(max_velocity_error, float(velocity_error.max()))
        diagnostics = task._last_controller_diagnostics
        max_tilt = max(max_tilt, float(diagnostics["desired_tilt_rad"].max()))
        max_body_rate = max(max_body_rate, float(torch.linalg.norm(task._robot.data.root_ang_vel_b, dim=-1).max()))
        max_moment = max(max_moment, float(torch.abs(task._moment[:, 0, :]).max()))
        any_ground = any_ground or bool(torch.any(task._filtered_contact_force(task._ground_contact_sensor) > task.cfg.contact_force_threshold))
        finite = (
            torch.all(torch.isfinite(task._robot.data.root_pos_w))
            & torch.all(torch.isfinite(task._velocity_reference_w))
            & torch.all(torch.isfinite(task._thrust))
            & torch.all(torch.isfinite(task._moment))
        )
        any_nonfinite = any_nonfinite or not bool(finite)
        ref_saturated_steps += int(torch.count_nonzero(task._reference_saturated))
        controller_saturated = (
            diagnostics["acceleration_saturated"]
            | diagnostics["tilt_saturated"]
            | diagnostics["thrust_saturated"]
            | diagnostics["body_rate_saturated"]
            | diagnostics["moment_saturated"]
        )
        controller_saturated_steps += int(torch.count_nonzero(controller_saturated))

    total_env_steps = steps * task.num_envs
    return {
        "name": name,
        "steps": steps,
        "max_relative_position_drift_m": max_position_drift,
        "max_velocity_tracking_error_mps": max_velocity_error,
        "max_desired_tilt_deg": math.degrees(max_tilt),
        "max_body_rate_radps": max_body_rate,
        "max_moment_nm": max_moment,
        "reference_saturation_ratio": ref_saturated_steps / total_env_steps,
        "controller_saturation_ratio": controller_saturated_steps / total_env_steps,
        "ground_crash": any_ground,
        "nonfinite": any_nonfinite,
        "controller_runtime_ms_mean": 1000.0 * sum(runtime_samples) / len(runtime_samples),
        "controller_runtime_ms_max": 1000.0 * max(runtime_samples),
    }


def _run_contact_case(task) -> dict:
    _set_flat_deck(task)
    _place_robot_above_deck(task, 0.35, match_deck_velocity=False)
    descent_action = torch.zeros(task.num_envs, 3, device=task.device)
    descent_action[:, 2] = -0.50  # -0.20 m/s deck-normal relative reference.
    hold_action = torch.zeros_like(descent_action)
    runtime_samples: list[float] = []
    first_contact_normal_speed = None
    first_contact_tangential_speed = None
    hard_contact = False
    ground_crash = False
    nonfinite = False
    contact_seen = False

    for _ in range(250):
        action = hold_action if contact_seen else descent_action
        _advance_control_step(task, action, runtime_samples)
        terms = task._compute_landing_terms()
        deck_contact = terms["deck_contact"]
        if bool(torch.any(deck_contact)) and not contact_seen:
            contact_seen = True
            first_contact_normal_speed = float(torch.abs(terms["normal_rel_speed"][deck_contact]).max())
            first_contact_tangential_speed = float(terms["tangential_rel_speed"][deck_contact].max())
        hard_contact = hard_contact or bool(torch.any(terms["hard_contact"]))
        ground_crash = ground_crash or bool(torch.any(terms["ground_crash"]))
        finite = (
            torch.all(torch.isfinite(task._robot.data.root_pos_w))
            & torch.all(torch.isfinite(task._thrust))
            & torch.all(torch.isfinite(task._moment))
        )
        nonfinite = nonfinite or not bool(finite)
        if contact_seen and len(runtime_samples) >= 40:
            # Keep several low-level updates after first contact, but do not force a full RL episode.
            break

    terms = task._compute_landing_terms()
    return {
        "name": "slow_normal_descent_contact",
        "contact_seen": contact_seen,
        "first_contact_normal_speed_mps": first_contact_normal_speed,
        "first_contact_tangential_speed_mps": first_contact_tangential_speed,
        "hard_contact": hard_contact,
        "ground_crash": ground_crash,
        "nonfinite": nonfinite,
        "final_surface_clearance_m": float(terms["landing_surface_clearance"].min()),
        "controller_runtime_ms_mean": 1000.0 * sum(runtime_samples) / len(runtime_samples),
    }


def main() -> None:
    cfg = QuadcopterShipLandingPx4HierarchicalEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.strict_deck_motion_consistency = False
    env = gym.make(TASK_ID, cfg=cfg)
    task = env.unwrapped
    env.reset(seed=42)

    cases = [
        _run_tracking_case(task, "static_hover", lambda: _set_flat_deck(task)),
        _run_tracking_case(task, "constant_xy_deck", lambda: _set_flat_deck(task, velocity_xy=(0.20, -0.10))),
        _run_tracking_case(task, "heave_deck", lambda: _set_flat_deck(task, heave_amplitude=0.08, heave_frequency=0.10)),
        _run_tracking_case(task, "physical_deck_attitude", lambda: _set_attitude_deck(task)),
    ]
    contact = _run_contact_case(task)
    reward = task._get_rewards()
    reward_path_finite = bool(torch.all(torch.isfinite(reward)))

    tracking_pass = all(
        (not case["nonfinite"])
        and (not case["ground_crash"])
        and case["max_relative_position_drift_m"] < 0.25
        and case["controller_saturation_ratio"] < 0.95
        for case in cases
    )
    contact_pass = (
        contact["contact_seen"]
        and not contact["nonfinite"]
        and not contact["ground_crash"]
        and not contact["hard_contact"]
        and contact["first_contact_normal_speed_mps"] is not None
        and contact["first_contact_normal_speed_mps"] < task.cfg.hard_contact_normal_speed
    )
    report = {
        "task_id": TASK_ID,
        "num_envs": task.num_envs,
        "physics_hz": round(1.0 / task.physics_dt),
        "policy_hz": round(1.0 / task.step_dt),
        "cases": cases,
        "contact": contact,
        "gates": {
            "no_nan_inf": all(not case["nonfinite"] for case in cases) and not contact["nonfinite"],
            "basic_ground_crash_zero": all(not case["ground_crash"] for case in cases) and not contact["ground_crash"],
            "tracking_stable": tracking_pass,
            "contact_safe": contact_pass,
            "reward_path_finite": reward_path_finite,
        },
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
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

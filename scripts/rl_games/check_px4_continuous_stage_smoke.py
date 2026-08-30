#!/usr/bin/env python3
"""Run the S4 deterministic 1-env smoke gate for the Continuous-Stage PX4 landing task."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, choices=(1,))
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--video", action="store_true", help="Record all nine scripted cases into one MP4.")
parser.add_argument("--video_output", type=Path, default=None)
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_px4_continuous_stage.quadrotor_ship_landing_px4_continuous_stage_env import (
    QuadcopterShipLandingPx4ContinuousStageEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import quat_apply, quat_from_euler_xyz


TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"
SEED = 42
STAGE_RATE_TOL = 1.0e-5
QUAT_NORM_TOL = 2.0e-4
ATTITUDE_RATE_TOL = 2.0e-3
VIDEO_WRITER = None
VIDEO_ENV = None
VIDEO_FRAME_COUNT = 0


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


def _set_tilted_deck(task, roll_deg: float = 5.0, pitch_deg: float = 4.0) -> None:
    env_ids = task._robot._ALL_INDICES
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w.zero_()
    task._pad_heave_amp.zero_()
    task._pad_heave_omega.zero_()
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp[:] = math.radians(roll_deg)
    task._deck_pitch_amp[:] = math.radians(pitch_deg)
    task._deck_roll_omega.zero_()
    task._deck_pitch_omega.zero_()
    task._deck_roll_phase0[:] = 0.5 * math.pi
    task._deck_pitch_phase0[:] = 0.5 * math.pi
    task._write_absolute_deck_state(env_ids)


def _set_rotating_tilt_deck(task) -> None:
    env_ids = task._robot._ALL_INDICES
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w.zero_()
    task._pad_heave_amp.zero_()
    task._pad_heave_omega.zero_()
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp[:] = math.radians(5.0)
    task._deck_pitch_amp[:] = math.radians(4.0)
    task._deck_roll_omega[:] = 2.0 * math.pi * 0.10
    task._deck_pitch_omega[:] = 2.0 * math.pi * 0.12
    task._deck_roll_phase0.zero_()
    task._deck_pitch_phase0[:] = 0.5 * math.pi
    task._write_absolute_deck_state(env_ids)


def _make_fixed_yaw_update(task, yaw_deg: float):
    yaw = torch.full((task.num_envs,), math.radians(yaw_deg), device=task.device)
    zero = torch.zeros_like(yaw)
    quat = quat_from_euler_xyz(zero, zero, yaw)

    def _update(self) -> None:
        env_ids = self._robot._ALL_INDICES
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, :2] = self._terrain.env_origins[:, :2]
        pose[:, 2] = self.cfg.pad_base_height
        pose[:, 3:7] = quat
        velocity = torch.zeros(self.num_envs, 6, device=self.device)
        self._deck.write_root_pose_to_sim(pose, env_ids)
        self._deck.write_root_velocity_to_sim(velocity, env_ids)
        self._deck_pose_command_w.copy_(pose)
        self._deck_velocity_command_w.copy_(velocity)
        self._deck_command_valid[:] = True
        self._deck_roll.zero_()
        self._deck_pitch.zero_()
        self._sync_pad_state_from_deck()

    return MethodType(_update, task)


def _place_robot_above_target(task, clearance: float, match_deck_velocity: bool = False) -> None:
    env_ids = task._robot._ALL_INDICES
    deck_pose = task._deck_pose_command_w.clone()
    local_offset = task._target_contact_point_d.clone()
    local_offset[:, 2] += task.cfg.robot_landing_surface_offset + clearance
    robot_pos = deck_pose[:, :3] + quat_apply(deck_pose[:, 3:7], local_offset)
    pose = torch.cat((robot_pos, _identity_quat(task)), dim=-1)
    velocity = torch.zeros(task.num_envs, 6, device=task.device)
    if match_deck_velocity:
        velocity[:, :3] = task._deck_velocity_command_w[:, :3]
    task._robot.write_root_pose_to_sim(pose)
    task._robot.write_root_velocity_to_sim(velocity)
    task._thrust.zero_()
    task._moment.zero_()
    task.scene.write_data_to_sim()
    task.sim.step(render=False)
    task.scene.update(dt=task.physics_dt)
    task._reset_continuous_reference_state(env_ids)
    task.episode_length_buf[:] = 1


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


def _scalar(tensor: torch.Tensor) -> float:
    return float(tensor.reshape(-1)[0].detach().cpu())


def _vector(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.reshape(-1, tensor.shape[-1])[0].detach().cpu().tolist()]


def _heading_deg(vector_w: torch.Tensor) -> float:
    vector = vector_w.reshape(-1, 3)[0]
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _reference_tilt_rad(q_wxyz: torch.Tensor) -> torch.Tensor:
    local_z = q_wxyz.new_tensor([0.0, 0.0, 1.0]).expand(q_wxyz.shape[0], 3)
    body_z_w = quat_apply(q_wxyz, local_z)
    return torch.acos(body_z_w[:, 2].clamp(-1.0, 1.0))


def _snapshot(task) -> dict:
    terms = task._compute_landing_terms()
    diagnostics = task._last_controller_diagnostics
    q_deck = task._deck_pose_command_w[:, 3:7]
    finite_tensors = (
        task._robot.data.root_pos_w,
        task._robot.data.root_lin_vel_w,
        task._robot.data.root_quat_w,
        task._landing_stage,
        task._relative_velocity_target_d,
        task._relative_velocity_ref_d,
        task._velocity_reference_w,
        task._velocity_reference_ned,
        task._velocity_attitude_reference_wxyz,
        task._attitude_reference_wxyz,
        q_deck,
        task._terminal_alpha,
        task._thrust,
        task._moment,
    )
    finite = all(bool(torch.all(torch.isfinite(tensor))) for tensor in finite_tensors)
    controller_saturated = (
        diagnostics["acceleration_saturated"]
        | diagnostics["tilt_saturated"]
        | diagnostics["thrust_saturated"]
        | diagnostics["body_rate_saturated"]
        | diagnostics["moment_saturated"]
    )
    velocity_error = torch.linalg.norm(task._velocity_reference_w - task._robot.data.root_lin_vel_w, dim=-1)
    return {
        "stage_raw": _scalar(task._stage_raw),
        "stage_filtered": _scalar(task._landing_stage),
        "delta_stage": _scalar(task._delta_stage),
        "V_t": _scalar(task._stage_tangential_limit),
        "V_down": _scalar(task._stage_descent_limit),
        "V_up": _scalar(task._stage_ascent_limit),
        "relative_velocity_target_d": _vector(task._relative_velocity_target_d),
        "relative_velocity_reference_d": _vector(task._relative_velocity_ref_d),
        "relative_reference_delta": _vector(task._relative_reference_delta),
        "deck_center_velocity_w": _vector(task._deck_velocity_command_w[:, :3]),
        "deck_contact_velocity_w": _vector(task._deck_contact_velocity_ref_w),
        "velocity_reference_w": _vector(task._velocity_reference_w),
        "velocity_reference_ned": _vector(task._velocity_reference_ned),
        "terminal_alpha": _scalar(task._terminal_alpha),
        "deck_heading_w": _vector(task._deck_heading_w),
        "deck_heading_deg": _heading_deg(task._deck_heading_w),
        "q_vel": _vector(task._velocity_attitude_reference_wxyz),
        "q_ref": _vector(task._attitude_reference_wxyz),
        "q_deck": _vector(q_deck),
        "q_ref_norm": _scalar(torch.linalg.norm(task._attitude_reference_wxyz, dim=-1)),
        "q_ref_tilt_deg": math.degrees(_scalar(_reference_tilt_rad(task._attitude_reference_wxyz))),
        "terminal_attitude_conflict_angle_deg": math.degrees(_scalar(task._terminal_attitude_conflict_angle)),
        "terminal_attitude_tilt_saturated": bool(task._terminal_attitude_tilt_saturated[0]),
        "attitude_reference_rate": _vector(task._attitude_reference_rate),
        "relative_angular_velocity": _vector(task._relative_angular_velocity_w),
        "relative_angular_speed": _scalar(task._relative_angular_speed),
        "velocity_tracking_error": _scalar(velocity_error),
        "acceleration_saturated": bool(diagnostics["acceleration_saturated"][0]),
        "tilt_saturated": bool(diagnostics["tilt_saturated"][0]),
        "thrust_saturated": bool(diagnostics["thrust_saturated"][0]),
        "body_rate_saturated": bool(diagnostics["body_rate_saturated"][0]),
        "moment_saturated": bool(diagnostics["moment_saturated"][0]),
        "controller_saturated": bool(controller_saturated[0]),
        "ground_contact": bool(terms["ground_contact"][0]),
        "deck_contact": bool(terms["deck_contact"][0]),
        "hard_contact": bool(terms["hard_contact"][0]),
        "ground_crash": bool(terms["ground_crash"][0]),
        "surface_clearance": _scalar(terms["landing_surface_clearance"]),
        "normal_relative_speed": _scalar(terms["normal_rel_speed"]),
        "tangential_relative_speed": _scalar(terms["tangential_rel_speed"]),
        "contact_force": _scalar(terms["deck_force"]),
        "finite": finite,
    }


def _capture_video_frame(case_name: str, sample: dict) -> None:
    global VIDEO_FRAME_COUNT
    if VIDEO_WRITER is None or VIDEO_ENV is None:
        return
    rendered = VIDEO_ENV.render()
    if rendered is None:
        raise RuntimeError("headless render returned no frame while video recording is enabled")
    frame = np.asarray(rendered)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f"unexpected render frame shape: {frame.shape}")
    frame = frame[:, :, :3].astype(np.uint8, copy=False)
    image = Image.fromarray(frame).resize((args.video_width, args.video_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    lines = (
        f"S4 Continuous-Stage Smoke | {case_name}",
        f"stage={sample['stage_filtered']:.3f}  alpha={sample['terminal_alpha']:.3f}  clearance={sample['surface_clearance']:.3f} m",
        "v_rel_ref_D=[" + ", ".join(f"{value:+.2f}" for value in sample["relative_velocity_reference_d"]) + "] m/s",
        f"tracking_error={sample['velocity_tracking_error']:.3f} m/s  saturated={sample['controller_saturated']}",
    )
    box_height = 20 + 24 * len(lines)
    draw.rectangle((8, 8, min(args.video_width - 8, 760), box_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((18, 16 + 24 * index), line, fill=(255, 255, 255))
    VIDEO_WRITER.append_data(np.asarray(image, dtype=np.uint8))
    VIDEO_FRAME_COUNT += 1


def _run_actions(task, actions: list[torch.Tensor], case_name: str) -> tuple[list[dict], list[float]]:
    samples: list[dict] = []
    runtime_samples: list[float] = []
    for action in actions:
        _advance_control_step(task, action, runtime_samples)
        sample = _snapshot(task)
        samples.append(sample)
        _capture_video_frame(case_name, sample)
    return samples, runtime_samples


def _case_summary(name: str, samples: list[dict], runtime_samples: list[float]) -> dict:
    saturation_ratio = sum(sample["controller_saturated"] for sample in samples) / len(samples)
    max_rate = [max(abs(sample["attitude_reference_rate"][axis]) for sample in samples) for axis in range(3)]
    return {
        "name": name,
        "steps": len(samples),
        "stage_raw_range": [min(sample["stage_raw"] for sample in samples), max(sample["stage_raw"] for sample in samples)],
        "stage_filtered_range": [
            min(sample["stage_filtered"] for sample in samples),
            max(sample["stage_filtered"] for sample in samples),
        ],
        "max_abs_delta_stage": max(abs(sample["delta_stage"]) for sample in samples),
        "max_velocity_tracking_error_mps": max(sample["velocity_tracking_error"] for sample in samples),
        "controller_saturation_ratio": saturation_ratio,
        "max_attitude_reference_rate_radps": max_rate,
        "max_q_ref_tilt_deg": max(sample["q_ref_tilt_deg"] for sample in samples),
        "max_relative_angular_speed_radps": max(sample["relative_angular_speed"] for sample in samples),
        "min_surface_clearance_m": min(sample["surface_clearance"] for sample in samples),
        "deck_contact_seen": any(sample["deck_contact"] for sample in samples),
        "hard_contact_seen": any(sample["hard_contact"] for sample in samples),
        "ground_crash": any(sample["ground_crash"] for sample in samples),
        "nonfinite": any(not sample["finite"] for sample in samples),
        "controller_runtime_ms_mean": 1000.0 * sum(runtime_samples) / len(runtime_samples),
        "controller_runtime_ms_max": 1000.0 * max(runtime_samples),
        "first": samples[0],
        "last": samples[-1],
    }


def _action(task, xyz=(0.0, 0.0, 0.0), stage=-1.0) -> torch.Tensor:
    action = torch.zeros(task.num_envs, 4, device=task.device)
    action[:, :3] = torch.tensor(xyz, device=task.device)
    action[:, 3] = stage
    return action


def _run_static_hover(task) -> dict:
    _set_flat_deck(task)
    _place_robot_above_target(task, 0.60)
    initial_pos = task._robot.data.root_pos_w.clone()
    samples, runtimes = _run_actions(task, [_action(task) for _ in range(80)], "static_hover")
    summary = _case_summary("static_hover", samples, runtimes)
    drift = torch.linalg.norm(task._robot.data.root_pos_w - initial_pos, dim=-1)
    summary["max_position_drift_m"] = _scalar(drift)
    summary["q_ref_q_vel_final_l2"] = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(summary["last"]["q_ref"], summary["last"]["q_vel"], strict=True))
    )
    summary["gate"] = (
        not summary["nonfinite"]
        and not summary["ground_crash"]
        and summary["controller_saturation_ratio"] < 0.95
        and abs(summary["last"]["stage_filtered"]) < 1.0e-5
        and abs(summary["last"]["terminal_alpha"]) < 1.0e-5
    )
    return summary


def _run_stage_ramp(task) -> dict:
    _set_flat_deck(task)
    _place_robot_above_target(task, 0.70)
    stages = torch.linspace(-1.0, 1.0, 61).tolist()
    samples, runtimes = _run_actions(task, [_action(task, stage=value) for value in stages], "stage_ramp")
    summary = _case_summary("stage_ramp", samples, runtimes)
    summary["V_t_monotonic_nonincreasing"] = all(
        nxt["V_t"] <= prev["V_t"] + 1.0e-6 for prev, nxt in zip(samples, samples[1:])
    )
    summary["V_down_monotonic_nondecreasing"] = all(
        nxt["V_down"] + 1.0e-6 >= prev["V_down"] for prev, nxt in zip(samples, samples[1:])
    )
    summary["V_up_monotonic_nonincreasing"] = all(
        nxt["V_up"] <= prev["V_up"] + 1.0e-6 for prev, nxt in zip(samples, samples[1:])
    )
    summary["gate"] = (
        not summary["nonfinite"]
        and summary["stage_filtered_range"][0] >= -1.0e-6
        and summary["stage_filtered_range"][1] <= 1.0 + 1.0e-6
        and summary["max_abs_delta_stage"] <= task.cfg.stage_rate_limit * task.step_dt + STAGE_RATE_TOL
        and summary["V_t_monotonic_nonincreasing"]
        and summary["V_down_monotonic_nondecreasing"]
        and summary["V_up_monotonic_nonincreasing"]
    )
    return summary


def _run_tracking_case(task, name: str, setup, steps: int = 100) -> dict:
    setup()
    _place_robot_above_target(task, 0.65, match_deck_velocity=True)
    samples, runtimes = _run_actions(task, [_action(task) for _ in range(steps)], name)
    summary = _case_summary(name, samples, runtimes)
    summary["gate"] = (
        not summary["nonfinite"]
        and not summary["ground_crash"]
        and summary["controller_saturation_ratio"] < 0.95
    )
    return summary


def _run_normal_descent(task) -> dict:
    _set_flat_deck(task)
    _place_robot_above_target(task, 0.90)
    actions = [_action(task, xyz=(0.0, 0.0, -0.8), stage=-1.0) for _ in range(10)]
    actions.extend(
        _action(task, xyz=(0.0, 0.0, -0.8), stage=value) for value in torch.linspace(-1.0, 1.0, 70).tolist()
    )
    samples, runtimes = _run_actions(task, actions, "normal_descent_stage_ramp")
    summary = _case_summary("normal_descent_stage_ramp", samples, runtimes)
    low = samples[:10]
    high = samples[-10:]
    summary["low_stage_max_V_down_mps"] = max(sample["V_down"] for sample in low)
    summary["low_stage_min_target_normal_mps"] = min(sample["relative_velocity_target_d"][2] for sample in low)
    summary["high_stage_min_target_normal_mps"] = min(sample["relative_velocity_target_d"][2] for sample in high)
    summary["high_stage_min_reference_normal_mps"] = min(sample["relative_velocity_reference_d"][2] for sample in high)
    summary["gate"] = (
        not summary["nonfinite"]
        and not summary["ground_crash"]
        and summary["low_stage_max_V_down_mps"] < 1.0e-5
        and summary["low_stage_min_target_normal_mps"] > -1.0e-5
        and summary["high_stage_min_target_normal_mps"] < -0.02
        and summary["high_stage_min_reference_normal_mps"] < -0.02
    )
    return summary


def _run_terminal_attitude(task) -> dict:
    _set_tilted_deck(task)
    _place_robot_above_target(task, 0.20)
    samples, runtimes = _run_actions(task, [_action(task, stage=1.0) for _ in range(45)], "terminal_attitude_blend")
    summary = _case_summary("terminal_attitude_blend", samples, runtimes)
    alpha_values = [sample["terminal_alpha"] for sample in samples]
    summary["alpha_initial"] = alpha_values[0]
    summary["alpha_max"] = max(alpha_values)
    summary["q_ref_norm_error_max"] = max(abs(sample["q_ref_norm"] - 1.0) for sample in samples)
    summary["gate"] = (
        not summary["nonfinite"]
        and not summary["ground_crash"]
        and summary["alpha_max"] > summary["alpha_initial"] + 0.05
        and summary["q_ref_norm_error_max"] <= QUAT_NORM_TOL
        and summary["max_q_ref_tilt_deg"] <= task.cfg.terminal_attitude_max_tilt_deg + 0.05
        and all(
            value <= limit + ATTITUDE_RATE_TOL
            for value, limit in zip(summary["max_attitude_reference_rate_radps"], task.cfg.terminal_reference_max_rate, strict=True)
        )
    )
    return summary


def _run_static_yaw(task) -> dict:
    _set_flat_deck(task)
    original_update = task._update_pad_motion
    task._update_pad_motion = _make_fixed_yaw_update(task, 15.0)
    try:
        task._update_pad_motion()
        _place_robot_above_target(task, 0.60)
        samples, runtimes = _run_actions(task, [_action(task) for _ in range(20)], "static_yaw_heading")
    finally:
        task._update_pad_motion = original_update
    summary = _case_summary("static_yaw_heading", samples, runtimes)
    final_heading = summary["last"]["deck_heading_deg"]
    q_vel = torch.tensor([summary["last"]["q_vel"]], dtype=torch.float32)
    q_ref = torch.tensor([summary["last"]["q_ref"]], dtype=torch.float32)
    local_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    summary["q_vel_heading_deg"] = _heading_deg(quat_apply(q_vel, local_x))
    summary["q_ref_heading_deg"] = _heading_deg(quat_apply(q_ref, local_x))
    summary["gate"] = (
        not summary["nonfinite"]
        and abs(final_heading - 15.0) < 0.2
        and abs(summary["q_vel_heading_deg"] - 15.0) < 1.0
        and abs(summary["q_ref_heading_deg"] - 15.0) < 1.0
    )
    return summary


def _run_off_center_contact_point(task) -> dict:
    original_target = task._target_contact_point_d.clone()
    try:
        task._target_contact_point_d[:, 0] = 0.15
        task._target_contact_point_d[:, 1] = -0.10
        task._target_contact_point_d[:, 2] = 0.5 * task.cfg.pad_thickness
        _set_rotating_tilt_deck(task)
        _place_robot_above_target(task, 0.65, match_deck_velocity=True)
        samples, runtimes = _run_actions(task, [_action(task) for _ in range(25)], "off_center_contact_point")
    finally:
        task._target_contact_point_d.copy_(original_target)
    summary = _case_summary("off_center_contact_point", samples, runtimes)
    correction_norms = []
    for sample in samples:
        correction_norms.append(
            math.sqrt(
                sum(
                    (contact - center) ** 2
                    for contact, center in zip(
                        sample["deck_contact_velocity_w"], sample["deck_center_velocity_w"], strict=True
                    )
                )
            )
        )
    summary["max_contact_point_velocity_correction_mps"] = max(correction_norms)
    summary["gate"] = (
        not summary["nonfinite"]
        and not summary["ground_crash"]
        and summary["max_contact_point_velocity_correction_mps"] > 1.0e-4
    )
    return summary


def _run_recovery(task) -> dict:
    _set_flat_deck(task)
    _place_robot_above_target(task, 0.55)
    actions = [_action(task, stage=1.0) for _ in range(25)]
    actions.extend(_action(task, xyz=(0.0, 0.0, 0.7), stage=-1.0) for _ in range(35))
    samples, runtimes = _run_actions(task, actions, "recovery")
    summary = _case_summary("recovery", samples, runtimes)
    peak_stage = max(sample["stage_filtered"] for sample in samples[:25])
    final_stage = samples[-1]["stage_filtered"]
    recovery_samples = samples[25:]
    summary["stage_before_recovery"] = peak_stage
    summary["stage_after_recovery"] = final_stage
    summary["recovery_max_V_up_mps"] = max(sample["V_up"] for sample in recovery_samples)
    summary["recovery_max_target_normal_mps"] = max(
        sample["relative_velocity_target_d"][2] for sample in recovery_samples
    )
    summary["recovery_max_reference_normal_mps"] = max(
        sample["relative_velocity_reference_d"][2] for sample in recovery_samples
    )
    summary["gate"] = (
        not summary["nonfinite"]
        and not summary["ground_crash"]
        and peak_stage > 0.6
        and final_stage < peak_stage - 0.1
        and summary["recovery_max_V_up_mps"] > 0.0
        and summary["recovery_max_target_normal_mps"] > 0.02
        and summary["recovery_max_reference_normal_mps"] > 0.02
    )
    return summary


def main() -> None:
    global VIDEO_WRITER, VIDEO_ENV, VIDEO_FRAME_COUNT
    if args.video_fps <= 0 or args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("video fps and dimensions must be positive")

    cfg = QuadcopterShipLandingPx4ContinuousStageEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = SEED
    cfg.debug_vis = False
    cfg.strict_deck_motion_consistency = False
    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array" if args.video else None)
    task = env.unwrapped
    env.reset(seed=42)

    video_output = None
    if args.video:
        video_output = (
            args.video_output
            if args.video_output is not None
            else Path("logs/rl_games/quadcopter_ship_landing_px4_continuous_stage/s4_deterministic_smoke_seed42.mp4")
        ).expanduser().resolve()
        video_output.parent.mkdir(parents=True, exist_ok=True)
        VIDEO_ENV = env
        VIDEO_FRAME_COUNT = 0
        VIDEO_WRITER = imageio.get_writer(
            video_output,
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )

    cases = [
        _run_static_hover(task),
        _run_stage_ramp(task),
        _run_tracking_case(task, "constant_xy_deck", lambda: _set_flat_deck(task, velocity_xy=(0.20, -0.10))),
        _run_tracking_case(
            task,
            "heave_tracking",
            lambda: _set_flat_deck(task, heave_amplitude=0.08, heave_frequency=0.10),
        ),
        _run_normal_descent(task),
        _run_terminal_attitude(task),
        _run_static_yaw(task),
        _run_off_center_contact_point(task),
        _run_recovery(task),
    ]
    if VIDEO_WRITER is not None:
        VIDEO_WRITER.close()
        VIDEO_WRITER = None
        VIDEO_ENV = None
        if video_output is None or not video_output.is_file() or video_output.stat().st_size <= 0:
            raise RuntimeError("video encoder produced an empty file")
    reward_path_finite = bool(torch.all(torch.isfinite(task._get_rewards())))

    no_nan_inf = all(not case["nonfinite"] for case in cases)
    basic_ground_crash_zero = all(not case["ground_crash"] for case in cases)
    stage_rate_ok = all(
        case["max_abs_delta_stage"] <= task.cfg.stage_rate_limit * task.step_dt + STAGE_RATE_TOL for case in cases
    )
    controller_stable = all(case["controller_saturation_ratio"] < 0.95 for case in cases)
    cases_pass = all(case["gate"] for case in cases)
    report = {
        "task_id": TASK_ID,
        "num_envs": task.num_envs,
        "seed": SEED,
        "physics_hz": round(1.0 / task.physics_dt),
        "policy_hz": round(1.0 / task.step_dt),
        "stage_rate_limit_per_second": task.cfg.stage_rate_limit,
        "stage_rate_limit_per_policy_step": task.cfg.stage_rate_limit * task.step_dt,
        "video": {
            "generated": bool(args.video),
            "path": str(video_output) if video_output is not None else None,
            "fps": args.video_fps if args.video else None,
            "resolution": [args.video_width, args.video_height] if args.video else None,
            "frames": VIDEO_FRAME_COUNT if args.video else 0,
            "duration_seconds": VIDEO_FRAME_COUNT / args.video_fps if args.video else 0.0,
        },
        "cases": cases,
        "gates": {
            "all_scripted_cases": len(cases) == 9,
            "no_nan_inf": no_nan_inf,
            "basic_ground_crash_zero": basic_ground_crash_zero,
            "stage_rate_limit": stage_rate_ok,
            "controller_stable": controller_stable,
            "reward_path_finite": reward_path_finite,
            "case_specific_contracts": cases_pass,
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

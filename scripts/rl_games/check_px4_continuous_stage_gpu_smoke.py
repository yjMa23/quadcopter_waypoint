#!/usr/bin/env python3
"""Run the S5 deterministic 16-env GPU smoke gate for the Continuous-Stage PX4 task."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16, choices=(16,))
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_px4_continuous_stage.quadrotor_ship_landing_px4_continuous_stage_env import (
    QuadcopterShipLandingPx4ContinuousStageEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import quat_apply, quat_from_euler_xyz


TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"
SEED = 42
EXPECTED_DEVICE = "cuda"
EXPECTED_DTYPE = torch.float32
STAGE_RATE_TOL = 1.0e-5
QUAT_NORM_TOL = 2.0e-4
ATTITUDE_RATE_TOL = 2.0e-3
BASIC_SATURATION_RATIO_LIMIT = 0.95
MAIN_POLICY_STEPS = 45

SCENARIOS = (
    "static_hover_a",
    "static_hover_b",
    "stage_ramp_a",
    "stage_ramp_b",
    "constant_xy_deck_x",
    "constant_xy_deck_y",
    "heave_positive_phase",
    "heave_negative_phase",
    "normal_descent_ramp",
    "normal_descent_terminal",
    "terminal_attitude_roll",
    "terminal_attitude_pitch",
    "static_yaw_15deg",
    "off_center_contact_point_rotation",
    "recovery_high_to_low",
    "mixed_velocity_heave_tilt",
)

TENSOR_CONTRACT = (
    "_landing_stage",
    "_stage_raw",
    "_delta_stage",
    "_relative_velocity_target_d",
    "_relative_velocity_ref_d",
    "_deck_contact_velocity_ref_w",
    "_velocity_reference_w",
    "_velocity_reference_ned",
    "_terminal_alpha",
    "_deck_heading_w",
    "_velocity_attitude_reference_wxyz",
    "_attitude_reference_wxyz",
    "_attitude_reference_rate",
    "_relative_angular_velocity_w",
    "_relative_angular_speed",
)


def _identity_quat(task) -> torch.Tensor:
    quat = torch.zeros(task.num_envs, 4, device=task.device)
    quat[:, 0] = 1.0
    return quat


def _zero_deck_profiles(task) -> None:
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w.zero_()
    task._pad_heave_amp.zero_()
    task._pad_heave_omega.zero_()
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp.zero_()
    task._deck_pitch_amp.zero_()
    task._deck_roll_omega.zero_()
    task._deck_pitch_omega.zero_()
    task._deck_roll_phase0.zero_()
    task._deck_pitch_phase0.zero_()


def _set_flat_deck(task) -> None:
    _zero_deck_profiles(task)
    task._write_absolute_deck_state(task._robot._ALL_INDICES)


def _set_terminal_isolation_decks(task) -> None:
    _zero_deck_profiles(task)
    task._deck_roll_amp[10] = math.radians(5.0)
    task._deck_roll_phase0[10] = 0.5 * math.pi
    task._deck_pitch_amp[11] = math.radians(4.0)
    task._deck_pitch_phase0[11] = -0.5 * math.pi
    task._write_absolute_deck_state(task._robot._ALL_INDICES)


def _set_main_deck_profiles(task) -> None:
    _zero_deck_profiles(task)
    task._deck_xy_velocity_w[4, 0] = 0.20
    task._deck_xy_velocity_w[5, 1] = -0.15

    task._pad_heave_amp[6:8] = 0.08
    task._pad_heave_omega[6:8] = 2.0 * math.pi * 0.20
    task._deck_heave_phase0[6] = 0.0
    task._deck_heave_phase0[7] = math.pi

    task._deck_roll_amp[10] = math.radians(5.0)
    task._deck_roll_phase0[10] = 0.5 * math.pi
    task._deck_pitch_amp[11] = math.radians(4.0)
    task._deck_pitch_phase0[11] = -0.5 * math.pi

    task._deck_roll_amp[13] = math.radians(5.0)
    task._deck_pitch_amp[13] = math.radians(4.0)
    task._deck_roll_omega[13] = 2.0 * math.pi * 0.10
    task._deck_pitch_omega[13] = 2.0 * math.pi * 0.12
    task._deck_pitch_phase0[13] = 0.5 * math.pi

    task._deck_xy_velocity_w[15] = torch.tensor([0.12, -0.06], device=task.device)
    task._pad_heave_amp[15] = 0.06
    task._pad_heave_omega[15] = 2.0 * math.pi * 0.16
    task._deck_roll_amp[15] = math.radians(4.0)
    task._deck_pitch_amp[15] = math.radians(3.0)
    task._deck_roll_omega[15] = 2.0 * math.pi * 0.10
    task._deck_pitch_omega[15] = 2.0 * math.pi * 0.12
    task._deck_pitch_phase0[15] = 0.5 * math.pi
    task._write_absolute_deck_state(task._robot._ALL_INDICES)


def _install_static_yaw_overlay(task):
    original_update = task._update_pad_motion
    yaw_ids = torch.tensor([12], dtype=torch.long, device=task.device)
    yaw = torch.tensor([math.radians(15.0)], device=task.device)
    zero = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zero, zero, yaw)

    def _update(self) -> None:
        original_update()
        pose = self._deck_pose_command_w[yaw_ids].clone()
        velocity = self._deck_velocity_command_w[yaw_ids].clone()
        pose[:, 3:7] = yaw_quat
        velocity.zero_()
        self._deck.write_root_pose_to_sim(pose, yaw_ids)
        self._deck.write_root_velocity_to_sim(velocity, yaw_ids)
        self._deck_pose_command_w[yaw_ids] = pose
        self._deck_velocity_command_w[yaw_ids] = velocity
        self._deck_command_valid[yaw_ids] = True
        self._sync_pad_state_from_deck()

    task._update_pad_motion = MethodType(_update, task)
    return original_update


def _place_robots(task, clearance: torch.Tensor, match_deck_velocity: bool = True) -> None:
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
    task._reset_continuous_reference_state(task._robot._ALL_INDICES)
    task.episode_length_buf[:] = 1


def _tensor_contract_snapshot(task) -> tuple[dict, bool]:
    metadata = {}
    valid = True
    for name in TENSOR_CONTRACT:
        tensor = getattr(task, name)
        finite = bool(torch.all(torch.isfinite(tensor)))
        entry = {
            "shape": list(tensor.shape),
            "device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "finite": finite,
        }
        metadata[name] = entry
        valid = (
            valid
            and tensor.shape[0] == task.num_envs
            and tensor.device.type == EXPECTED_DEVICE
            and tensor.dtype == EXPECTED_DTYPE
            and finite
        )
    return metadata, valid


def _probe_recovery_cross_env_isolation(task) -> dict:
    _set_flat_deck(task)
    clearance = torch.full((task.num_envs,), 0.60, device=task.device)
    _place_robots(task, clearance)
    task._reset_continuous_reference_state(task._robot._ALL_INDICES)
    action = torch.zeros(task.num_envs, 4, device=task.device)
    action[:, 3] = -1.0
    action[14, 3] = 1.0
    for _ in range(20):
        task._pre_physics_step(action)
    stage_before = task._landing_stage.clone()

    recovery = torch.zeros_like(action)
    recovery[:, 3] = -1.0
    recovery[14, 2] = 0.7
    task._pre_physics_step(recovery)
    other_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    other_mask[14] = False
    other_stage_zero = bool(torch.all(torch.abs(task._landing_stage[other_mask]) < 1.0e-7))
    other_reference_zero = bool(torch.all(torch.abs(task._relative_velocity_ref_d[other_mask]) < 1.0e-7))
    env14_recovered = bool(
        (task._landing_stage[14] < stage_before[14])
        and (task._relative_velocity_target_d[14, 2] > 0.02)
        and (task._relative_velocity_ref_d[14, 2] > 0.0)
    )
    return {
        "other_stage_zero": other_stage_zero,
        "other_reference_zero": other_reference_zero,
        "env14_stage_before": float(stage_before[14]),
        "env14_stage_after": float(task._landing_stage[14]),
        "env14_target_normal_mps": float(task._relative_velocity_target_d[14, 2]),
        "env14_reference_normal_mps": float(task._relative_velocity_ref_d[14, 2]),
        "isolated": other_stage_zero and other_reference_zero and env14_recovered,
    }


def _probe_terminal_attitude_cross_env_isolation(task) -> dict:
    _set_terminal_isolation_decks(task)
    clearance = torch.full((task.num_envs,), 0.20, device=task.device)
    _place_robots(task, clearance, match_deck_velocity=False)
    action = torch.zeros(task.num_envs, 4, device=task.device)
    action[:, 3] = 1.0
    for _ in range(10):
        task._pre_physics_step(action)
    q_ref = task._attitude_reference_wxyz
    non_tilt_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    non_tilt_mask[10:12] = False
    baseline = q_ref[0].expand(task.num_envs - 2, 4)
    non_tilt_same = bool(torch.allclose(q_ref[non_tilt_mask], baseline, atol=1.0e-6, rtol=0.0))
    roll_changed = bool(torch.linalg.norm(q_ref[10] - q_ref[0]) > 1.0e-4)
    pitch_changed = bool(torch.linalg.norm(q_ref[11] - q_ref[0]) > 1.0e-4)
    return {
        "non_tilt_references_match": non_tilt_same,
        "roll_env_changed": roll_changed,
        "pitch_env_changed": pitch_changed,
        "isolated": non_tilt_same and roll_changed and pitch_changed,
    }


def _main_action(task, step: int) -> torch.Tensor:
    action = torch.zeros(task.num_envs, 4, device=task.device)
    action[:, 3] = -1.0
    ramp = -1.0 + 2.0 * step / (MAIN_POLICY_STEPS - 1)
    action[2, 3] = ramp
    action[3, 3] = ramp
    action[8, 2] = -0.8
    action[8, 3] = ramp
    action[9, 2] = -0.7
    action[9, 3] = 1.0
    action[10:12, 3] = 1.0
    action[14, 3] = 1.0 if step < 20 else -1.0
    action[14, 2] = 0.0 if step < 20 else 0.7
    action[15, :3] = torch.tensor([0.3, -0.2, -0.4], device=task.device)
    action[15, 3] = 0.6
    return action


def _advance_policy_step(task, action: torch.Tensor, saturation_counts: torch.Tensor) -> tuple[float, float, bool]:
    policy_start = time.perf_counter()
    task._pre_physics_step(action)
    controller_runtime_ms = 0.0
    controller_finite = True
    for _ in range(task.cfg.decimation):
        if task.device.startswith("cuda"):
            torch.cuda.synchronize()
        controller_start = time.perf_counter()
        task._apply_action()
        if task.device.startswith("cuda"):
            torch.cuda.synchronize()
        runtime_ms = 1000.0 * (time.perf_counter() - controller_start)
        controller_runtime_ms += runtime_ms
        diagnostics = task._last_controller_diagnostics
        saturated = (
            diagnostics["acceleration_saturated"]
            | diagnostics["tilt_saturated"]
            | diagnostics["thrust_saturated"]
            | diagnostics["body_rate_saturated"]
            | diagnostics["moment_saturated"]
        )
        saturation_counts += saturated.long()
        controller_finite = controller_finite and bool(
            torch.all(torch.isfinite(task._thrust))
            and torch.all(torch.isfinite(task._moment))
            and torch.all(torch.isfinite(diagnostics["velocity_error_w"]))
        )
        task.scene.write_data_to_sim()
        task.sim.step(render=False)
        task.scene.update(dt=task.physics_dt)
    task.episode_length_buf += 1
    task._get_dones()
    reward = task._get_rewards()
    reward_finite = bool(torch.all(torch.isfinite(reward)))
    if task.device.startswith("cuda"):
        torch.cuda.synchronize()
    policy_wall_ms = 1000.0 * (time.perf_counter() - policy_start)
    return controller_runtime_ms, policy_wall_ms, controller_finite and reward_finite


def _partial_reset_probe(task) -> dict:
    reset_id = torch.tensor([3], dtype=torch.long, device=task.device)
    non_reset = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    non_reset[3] = False
    names = (
        "_landing_stage",
        "_stage_raw",
        "_delta_stage",
        "_previous_relative_velocity_ref_d",
        "_relative_velocity_ref_d",
        "_relative_velocity_target_d",
        "_relative_reference_delta",
        "_previous_attitude_reference_wxyz",
        "_attitude_reference_wxyz",
        "_previous_deck_heading_w",
        "_deck_heading_w",
        "_first_contact_seen",
        "_settle_hold_steps",
        "_deck_contact",
        "_safe_contact",
        "_hard_contact",
        "_ground_crash",
        "_episode_controller_step_count",
        "_episode_velocity_tracking_error_sum",
        "_episode_continuous_step_count",
    )
    before = {name: getattr(task, name).clone() for name in names}
    task._reset_idx(reset_id)
    unchanged = all(torch.equal(getattr(task, name)[non_reset], before[name][non_reset]) for name in names)
    reset_state = bool(
        torch.abs(task._landing_stage[3]) < 1.0e-7
        and torch.all(torch.abs(task._relative_velocity_ref_d[3]) < 1.0e-7)
        and torch.all(torch.abs(task._relative_velocity_target_d[3]) < 1.0e-7)
        and (task._episode_controller_step_count[3] == 0)
        and (task._episode_continuous_step_count[3] == 0)
        and (task._settle_hold_steps[3] == 0)
        and (~task._first_contact_seen[3])
    )
    attitude_reset = bool(
        torch.allclose(task._previous_attitude_reference_wxyz[3], task._robot.data.root_quat_w[3], atol=1.0e-6, rtol=0.0)
        and torch.allclose(task._attitude_reference_wxyz[3], task._robot.data.root_quat_w[3], atol=1.0e-6, rtol=0.0)
    )
    return {
        "reset_env": 3,
        "non_reset_buffers_unchanged": unchanged,
        "reset_env_state_cleared": reset_state,
        "terminal_attitude_state_reset": attitude_reset,
        "partial_reset_isolated": unchanged and reset_state and attitude_reset,
    }


def _reference_tilt_rad(q_wxyz: torch.Tensor) -> torch.Tensor:
    local_z = q_wxyz.new_tensor([0.0, 0.0, 1.0]).expand(q_wxyz.shape[0], 3)
    body_z_w = quat_apply(q_wxyz, local_z)
    return torch.acos(body_z_w[:, 2].clamp(-1.0, 1.0))


def main() -> None:
    cfg = QuadcopterShipLandingPx4ContinuousStageEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = SEED
    cfg.debug_vis = False
    cfg.strict_deck_motion_consistency = False
    env = gym.make(TASK_ID, cfg=cfg)
    task = env.unwrapped
    env.reset(seed=SEED)

    if task.num_envs != 16:
        raise RuntimeError(f"S5 requires exactly 16 envs, got {task.num_envs}")
    if task.device != "cuda:0" and not task.device.startswith("cuda"):
        raise RuntimeError(f"S5 requires CUDA execution, got {task.device}")

    recovery_isolation = _probe_recovery_cross_env_isolation(task)
    env.reset(seed=SEED)
    terminal_isolation = _probe_terminal_attitude_cross_env_isolation(task)
    env.reset(seed=SEED)

    _set_main_deck_profiles(task)
    target_before = task._target_contact_point_d.clone()
    task._target_contact_point_d[13, 0] = 0.15
    task._target_contact_point_d[13, 1] = -0.10
    target_other_mask = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    target_other_mask[13] = False
    target_contact_point_isolated = bool(
        torch.equal(task._target_contact_point_d[target_other_mask], target_before[target_other_mask])
        and torch.linalg.norm(task._target_contact_point_d[13] - target_before[13]) > 1.0e-4
    )

    original_update = _install_static_yaw_overlay(task)
    task._update_pad_motion()
    clearance = torch.full((task.num_envs,), 0.65, device=task.device)
    clearance[8:10] = 0.85
    clearance[10:12] = 0.20
    clearance[12] = 0.60
    clearance[14] = 0.55
    clearance[15] = 0.45
    _place_robots(task, clearance)

    max_velocity_error = torch.zeros(task.num_envs, device=task.device)
    max_attitude_rate = torch.zeros(task.num_envs, 3, device=task.device)
    max_stage_delta = torch.zeros(task.num_envs, device=task.device)
    max_q_norm_error = torch.zeros(task.num_envs, device=task.device)
    max_q_tilt = torch.zeros(task.num_envs, device=task.device)
    saturation_counts = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
    ground_crash_seen = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    per_env_nonfinite = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    min_stage = torch.ones(task.num_envs, device=task.device)
    max_stage = torch.zeros(task.num_envs, device=task.device)
    frame_error_max = 0.0
    controller_runtime_ms = []
    policy_wall_ms = []
    controller_and_reward_finite = True
    normal_descent_seen = torch.zeros(2, dtype=torch.bool, device=task.device)
    recovery_positive_seen = False
    xy_motion_seen = torch.zeros(2, dtype=torch.bool, device=task.device)
    heave_sign_seen = torch.zeros(2, dtype=torch.bool, device=task.device)
    tensor_metadata = {}
    shape_device_finite = True

    for step in range(MAIN_POLICY_STEPS):
        batch_controller_ms, batch_wall_ms, finite = _advance_policy_step(task, _main_action(task, step), saturation_counts)
        controller_runtime_ms.append(batch_controller_ms / task.cfg.decimation)
        policy_wall_ms.append(batch_wall_ms)
        controller_and_reward_finite = controller_and_reward_finite and finite

        tensor_metadata, tensor_ok = _tensor_contract_snapshot(task)
        shape_device_finite = shape_device_finite and tensor_ok
        velocity_error = torch.linalg.norm(task._velocity_reference_w - task._robot.data.root_lin_vel_w, dim=-1)
        max_velocity_error = torch.maximum(max_velocity_error, velocity_error)
        max_attitude_rate = torch.maximum(max_attitude_rate, torch.abs(task._attitude_reference_rate))
        max_stage_delta = torch.maximum(max_stage_delta, torch.abs(task._delta_stage))
        q_norm_error = torch.abs(torch.linalg.norm(task._attitude_reference_wxyz, dim=-1) - 1.0)
        max_q_norm_error = torch.maximum(max_q_norm_error, q_norm_error)
        max_q_tilt = torch.maximum(max_q_tilt, _reference_tilt_rad(task._attitude_reference_wxyz))
        min_stage = torch.minimum(min_stage, task._landing_stage)
        max_stage = torch.maximum(max_stage, task._landing_stage)
        ground_crash_seen |= task._ground_crash

        finite_env = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        for name in TENSOR_CONTRACT:
            tensor = getattr(task, name).reshape(task.num_envs, -1)
            finite_env &= torch.all(torch.isfinite(tensor), dim=-1)
        finite_env &= torch.all(torch.isfinite(task._thrust.reshape(task.num_envs, -1)), dim=-1)
        finite_env &= torch.all(torch.isfinite(task._moment.reshape(task.num_envs, -1)), dim=-1)
        per_env_nonfinite |= ~finite_env

        ned_expected = torch.stack(
            (task._velocity_reference_w[:, 1], task._velocity_reference_w[:, 0], -task._velocity_reference_w[:, 2]),
            dim=-1,
        )
        frame_error_max = max(frame_error_max, float(torch.max(torch.abs(task._velocity_reference_ned - ned_expected))))
        normal_descent_seen |= task._relative_velocity_ref_d[8:10, 2] < -0.02
        recovery_positive_seen = recovery_positive_seen or bool(
            step >= 20 and task._relative_velocity_ref_d[14, 2] > 0.02
        )
        xy_motion_seen[0] |= task._deck_contact_velocity_ref_w[4, 0] > 0.05
        xy_motion_seen[1] |= task._deck_contact_velocity_ref_w[5, 1] < -0.05
        heave_sign_seen[0] |= task._deck_contact_velocity_ref_w[6, 2] > 0.01
        heave_sign_seen[1] |= task._deck_contact_velocity_ref_w[7, 2] < -0.01

    reward_path_finite = controller_and_reward_finite
    saturation_ratio = saturation_counts.float() / float(MAIN_POLICY_STEPS * task.cfg.decimation)
    global_saturation_ratio = float(saturation_counts.sum()) / float(MAIN_POLICY_STEPS * task.cfg.decimation * task.num_envs)
    max_rate_limits = torch.tensor(task.cfg.terminal_reference_max_rate, device=task.device)

    stage_bounds = bool(torch.all(min_stage >= -1.0e-6) and torch.all(max_stage <= 1.0 + 1.0e-6))
    stage_rate = bool(torch.all(max_stage_delta <= task.cfg.stage_rate_limit * task.step_dt + STAGE_RATE_TOL))
    reference_frame_signs = bool(
        frame_error_max <= 1.0e-6
        and torch.all(normal_descent_seen)
        and recovery_positive_seen
        and torch.all(xy_motion_seen)
        and torch.all(heave_sign_seen)
    )
    attitude_reference = bool(
        torch.all(max_q_norm_error <= QUAT_NORM_TOL)
        and torch.all(max_attitude_rate <= max_rate_limits + ATTITUDE_RATE_TOL)
        and torch.all(max_q_tilt <= math.radians(task.cfg.terminal_attitude_max_tilt_deg + 0.05))
    )
    controller_finite = controller_and_reward_finite and not bool(torch.any(per_env_nonfinite))
    controller_stable = bool(torch.all(saturation_ratio < BASIC_SATURATION_RATIO_LIMIT))
    ground_crash_zero = not bool(torch.any(ground_crash_seen))

    partial_reset = _partial_reset_probe(task)
    task._update_pad_motion = original_update

    cross_env_isolation = bool(
        recovery_isolation["isolated"] and terminal_isolation["isolated"] and target_contact_point_isolated
    )
    cross_env_contamination = not cross_env_isolation

    max_velocity_value = float(torch.max(max_velocity_error))
    max_velocity_env = int(torch.argmax(max_velocity_error))
    per_env_rate_peak = torch.max(max_attitude_rate, dim=-1).values
    max_attitude_rate_value = float(torch.max(per_env_rate_peak))
    max_attitude_rate_env = int(torch.argmax(per_env_rate_peak))

    per_env = []
    stage_min_list = min_stage.detach().cpu().tolist()
    stage_max_list = max_stage.detach().cpu().tolist()
    stage_delta_list = max_stage_delta.detach().cpu().tolist()
    velocity_error_list = max_velocity_error.detach().cpu().tolist()
    attitude_rate_list = max_attitude_rate.detach().cpu().tolist()
    q_norm_error_list = max_q_norm_error.detach().cpu().tolist()
    q_tilt_list = torch.rad2deg(max_q_tilt).detach().cpu().tolist()
    saturation_ratio_list = saturation_ratio.detach().cpu().tolist()
    nonfinite_list = per_env_nonfinite.detach().cpu().tolist()
    ground_crash_list = ground_crash_seen.detach().cpu().tolist()
    for index, scenario in enumerate(SCENARIOS):
        per_env.append(
            {
                "env_id": index,
                "scenario": scenario,
                "stage_min": stage_min_list[index],
                "stage_max": stage_max_list[index],
                "max_abs_delta_stage": stage_delta_list[index],
                "max_velocity_tracking_error_mps": velocity_error_list[index],
                "max_abs_attitude_reference_rate_radps": attitude_rate_list[index],
                "max_q_ref_norm_error": q_norm_error_list[index],
                "max_q_ref_tilt_deg": q_tilt_list[index],
                "controller_saturation_ratio": saturation_ratio_list[index],
                "nonfinite": nonfinite_list[index],
                "ground_crash": ground_crash_list[index],
            }
        )

    gates = {
        "shape_device_finite": shape_device_finite,
        "stage_bounds": stage_bounds,
        "stage_rate": stage_rate,
        "reference_frame_signs": reference_frame_signs,
        "attitude_reference": attitude_reference,
        "cross_env_isolation": cross_env_isolation,
        "partial_reset_isolated": partial_reset["partial_reset_isolated"],
        "controller_finite": controller_finite,
        "controller_stable": controller_stable,
        "ground_crash_zero": ground_crash_zero,
        "reward_path_finite": reward_path_finite,
    }
    report = {
        "task_id": TASK_ID,
        "seed": SEED,
        "device": task.device,
        "num_envs": task.num_envs,
        "physics_hz": round(1.0 / task.physics_dt),
        "policy_hz": round(1.0 / task.step_dt),
        "policy_steps": MAIN_POLICY_STEPS,
        "tensor_contract": tensor_metadata,
        "per_env": per_env,
        "global_max_stage_delta": float(torch.max(max_stage_delta)),
        "global_max_velocity_tracking_error_mps": max_velocity_value,
        "global_max_velocity_tracking_error_env": max_velocity_env,
        "global_max_velocity_tracking_error_scenario": SCENARIOS[max_velocity_env],
        "global_max_attitude_reference_rate_radps": max_attitude_rate_value,
        "global_max_attitude_reference_rate_env": max_attitude_rate_env,
        "global_max_attitude_reference_rate_scenario": SCENARIOS[max_attitude_rate_env],
        "global_controller_saturation_ratio": global_saturation_ratio,
        "controller_runtime_ms_mean_per_low_level_batch": sum(controller_runtime_ms) / len(controller_runtime_ms),
        "controller_runtime_ms_max_per_low_level_batch": max(controller_runtime_ms),
        "policy_step_wall_ms_mean": sum(policy_wall_ms) / len(policy_wall_ms),
        "policy_step_wall_ms_max": max(policy_wall_ms),
        "reference_frame_max_abs_error": frame_error_max,
        "cross_env_contamination": cross_env_contamination,
        "cross_env_isolation_details": {
            "recovery": recovery_isolation,
            "terminal_attitude": terminal_isolation,
            "target_contact_point_isolated": target_contact_point_isolated,
        },
        "partial_reset": partial_reset,
        "NaN/Inf": bool(torch.any(per_env_nonfinite)),
        "ground_crash": bool(torch.any(ground_crash_seen)),
        "gates": gates,
    }
    report["status"] = "PASS" if all(gates.values()) else "FAIL"
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Saved report to: {output}")

    env.close()
    app.close()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

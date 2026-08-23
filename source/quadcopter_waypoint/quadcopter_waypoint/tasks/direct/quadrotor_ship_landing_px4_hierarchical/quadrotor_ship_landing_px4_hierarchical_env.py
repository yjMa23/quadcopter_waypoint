# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import time

import torch

from isaaclab.utils import configclass

from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_physical_deck_attitude.quadrotor_ship_landing_physical_deck_attitude_env import (
    QuadcopterShipLandingPhysicalDeckAttitudeEnv,
    QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import local_to_world_position
from quadcopter_waypoint.utils.px4_reference_adapter import (
    Px4ReferenceAdapterConfig,
    build_velocity_reference,
)
from quadcopter_waypoint.utils.vectorized_px4_like_controller import (
    VectorizedPx4LikeController,
    VectorizedPx4LikeControllerConfig,
)


@configclass
class QuadcopterShipLandingPx4HierarchicalEnvCfg(QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg):
    """Independent 3-D deck-relative velocity task with a 100 Hz PX4-like training controller."""

    # Keep physics at 100 Hz, but update the learned high-level velocity reference at 25 Hz.
    decimation = 4
    action_space = 3
    observation_space = 22

    # Deployable action contract: [deck t1, deck t2, deck normal] relative velocity in m/s.
    relative_velocity_min = (-0.80, -0.80, -0.40)
    relative_velocity_max = (0.80, 0.80, 0.30)
    max_horizontal_relative_speed = 0.80
    max_relative_reference_acceleration = 2.0

    # Training-only PX4-like controller. These are engineering initial values, not PX4 defaults or
    # real-vehicle identification results. Mass, inertia, and gravity are read from the simulator.
    controller_velocity_gain = (2.0, 2.0, 2.5)
    controller_velocity_integral_gain = (0.0, 0.0, 0.0)
    controller_velocity_derivative_gain = (0.0, 0.0, 0.0)
    controller_max_acceleration = 5.0
    controller_max_tilt_deg = 35.0
    controller_min_thrust = 0.0
    controller_max_thrust = 1.9
    controller_attitude_gain = (6.0, 6.0, 4.0)
    controller_max_body_rate = (6.0, 6.0, 4.0)
    controller_rate_gain = (12.0, 12.0, 8.0)
    controller_max_moment = (0.01, 0.01, 0.01)
    controller_yaw_ref_enu = 0.0
    # Formal evaluator/benchmark can enable synchronized wall-time measurement. Keep it disabled for
    # training so diagnostics do not serialize every CUDA physics substep.
    controller_runtime_sync = False


class QuadcopterShipLandingPx4HierarchicalEnv(QuadcopterShipLandingPhysicalDeckAttitudeEnv):
    """Physical-deck landing task whose policy commands PX4-deployable relative velocity references.

    Observation, reward, contact semantics, success contract, and failure taxonomy are inherited from
    the frozen PhysicalDeckAttitude task. Only the action/control path is replaced inside this new task.
    """

    cfg: QuadcopterShipLandingPx4HierarchicalEnvCfg

    def __init__(
        self,
        cfg: QuadcopterShipLandingPx4HierarchicalEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # The inherited SimulationCfg was authored for the 50 Hz Direct task. Keep rendering aligned
        # with this task's 25 Hz environment step without changing the frozen parent configuration.
        cfg.sim.render_interval = cfg.decimation
        super().__init__(cfg, render_mode, **kwargs)

        self._reference_adapter_config = Px4ReferenceAdapterConfig(
            relative_velocity_min=tuple(self.cfg.relative_velocity_min),
            relative_velocity_max=tuple(self.cfg.relative_velocity_max),
            max_horizontal_relative_speed=self.cfg.max_horizontal_relative_speed,
            max_relative_reference_acceleration=self.cfg.max_relative_reference_acceleration,
        )
        self._reference_adapter_config.validate()
        self._px4_like_controller = VectorizedPx4LikeController(
            VectorizedPx4LikeControllerConfig(
                velocity_gain=tuple(self.cfg.controller_velocity_gain),
                velocity_integral_gain=tuple(self.cfg.controller_velocity_integral_gain),
                velocity_derivative_gain=tuple(self.cfg.controller_velocity_derivative_gain),
                max_acceleration=self.cfg.controller_max_acceleration,
                max_tilt_rad=math.radians(self.cfg.controller_max_tilt_deg),
                min_thrust=self.cfg.controller_min_thrust,
                max_thrust=self.cfg.controller_max_thrust,
                attitude_gain=tuple(self.cfg.controller_attitude_gain),
                max_body_rate=tuple(self.cfg.controller_max_body_rate),
                rate_gain=tuple(self.cfg.controller_rate_gain),
                max_moment=tuple(self.cfg.controller_max_moment),
                yaw_ref_enu=self.cfg.controller_yaw_ref_enu,
            )
        )

        self._previous_relative_velocity_ref_d = torch.zeros(self.num_envs, 3, device=self.device)
        self._relative_velocity_ref_d = torch.zeros(self.num_envs, 3, device=self.device)
        self._deck_contact_velocity_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._velocity_reference_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._velocity_reference_ned = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_controller_diagnostics: dict[str, torch.Tensor] = {}

        # Episode-level M2 diagnostics are accumulated inside the environment and latched before reset.
        # This avoids evaluator races with DirectRLEnv's automatic reset and keeps M0/M1 untouched.
        self._episode_reference_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reference_norm_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_reference_norm_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_reference_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_action_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_action_square_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_action_abs_max = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference_norm_samples = torch.zeros(
            self.num_envs, self.max_episode_length, device=self.device
        )

        self._episode_controller_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_velocity_tracking_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_velocity_tracking_error_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_acceleration_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_tilt_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_thrust_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_body_rate_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_moment_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_max_desired_tilt = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_body_rate = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_moment = torch.zeros(self.num_envs, device=self.device)
        self._episode_controller_runtime_ms_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_controller_runtime_ms_max = torch.zeros(self.num_envs, device=self.device)
        self._controller_runtime_sample_capacity = self.max_episode_length * self.cfg.decimation
        self._controller_runtime_ms_samples = torch.zeros(
            self.num_envs, self._controller_runtime_sample_capacity, device=self.device
        )

        for name in (
            "_last_relative_velocity_reference_norm_mean",
            "_last_relative_velocity_reference_norm_p95",
            "_last_relative_velocity_reference_norm_max",
            "_last_reference_saturation_ratio",
            "_last_controller_velocity_tracking_error_mean",
            "_last_controller_velocity_tracking_error_max",
            "_last_controller_acceleration_saturation_ratio",
            "_last_controller_tilt_saturation_ratio",
            "_last_controller_thrust_saturation_ratio",
            "_last_controller_body_rate_saturation_ratio",
            "_last_controller_moment_saturation_ratio",
            "_last_max_desired_tilt",
            "_last_max_body_rate",
            "_last_max_moment",
            "_last_controller_runtime_ms_mean",
            "_last_controller_runtime_ms_p95",
            "_last_controller_runtime_ms_max",
        ):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self._last_action_mean = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_action_std = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_action_abs_max = torch.zeros(self.num_envs, 3, device=self.device)

        self._target_contact_point_d = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_contact_point_d[:, 2] = 0.5 * self.cfg.pad_thickness

        # PhysX exposes one flattened 3x3 inertia matrix per rigid body. The Crazyflie task applies
        # wrench to the single body selected by ``self._body_id``; use that same body's current inertia.
        body_inertias = self._robot.root_physx_view.get_inertias()[:, self._body_id, :]
        self._robot_inertia_b = body_inertias.reshape(self.num_envs, len(self._body_id), 3, 3)[:, 0].to(
            device=self.device, dtype=self._robot.data.root_quat_w.dtype
        )

    @staticmethod
    def _episode_percentile(
        samples: torch.Tensor,
        counts: torch.Tensor,
        env_ids: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
        """Compute an exact linearly interpolated percentile over each selected episode row."""
        selected_counts = counts[env_ids]
        selected_samples = samples[env_ids]
        positions = torch.arange(samples.shape[1], device=samples.device).unsqueeze(0)
        valid = positions < selected_counts.unsqueeze(1)
        masked = torch.where(valid, selected_samples, torch.full_like(selected_samples, float("inf")))
        ordered = torch.sort(masked, dim=1).values
        nonempty_counts = selected_counts.clamp_min(1)
        quantile_position = (nonempty_counts.float() - 1.0) * percentile
        lower = torch.floor(quantile_position).long()
        upper = torch.ceil(quantile_position).long()
        weight = quantile_position - lower.float()
        lower_values = ordered.gather(1, lower.unsqueeze(1)).squeeze(1)
        upper_values = ordered.gather(1, upper.unsqueeze(1)).squeeze(1)
        values = lower_values * (1.0 - weight) + upper_values * weight
        return torch.where(selected_counts > 0, values, torch.zeros_like(values))

    def _record_reference_diagnostics(self, actions: torch.Tensor) -> None:
        reference_norm = torch.linalg.norm(self._relative_velocity_ref_d, dim=-1)
        sample_index = self._episode_reference_step_count
        valid = sample_index < self._reference_norm_samples.shape[1]
        valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_ids.numel() > 0:
            self._reference_norm_samples[valid_ids, sample_index[valid_ids]] = reference_norm[valid_ids]
        self._episode_reference_step_count += 1
        self._episode_reference_norm_sum += reference_norm
        self._episode_reference_norm_max = torch.maximum(self._episode_reference_norm_max, reference_norm)
        self._episode_reference_saturated_count += self._reference_saturated.long()
        self._episode_action_sum += actions
        self._episode_action_square_sum += actions.square()
        self._episode_action_abs_max = torch.maximum(self._episode_action_abs_max, torch.abs(actions))

    def _record_controller_diagnostics(
        self,
        diagnostics: dict[str, torch.Tensor],
        moment_b: torch.Tensor,
        runtime_ms: float,
    ) -> None:
        sample_index = self._episode_controller_step_count
        valid = sample_index < self._controller_runtime_ms_samples.shape[1]
        valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_ids.numel() > 0:
            self._controller_runtime_ms_samples[valid_ids, sample_index[valid_ids]] = runtime_ms

        velocity_tracking_error = torch.linalg.norm(diagnostics["velocity_error_w"], dim=-1)
        body_rate = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1)
        moment = torch.linalg.norm(moment_b, dim=-1)
        self._episode_controller_step_count += 1
        self._episode_velocity_tracking_error_sum += velocity_tracking_error
        self._episode_velocity_tracking_error_max = torch.maximum(
            self._episode_velocity_tracking_error_max, velocity_tracking_error
        )
        self._episode_acceleration_saturated_count += diagnostics["acceleration_saturated"].long()
        self._episode_tilt_saturated_count += diagnostics["tilt_saturated"].long()
        self._episode_thrust_saturated_count += diagnostics["thrust_saturated"].long()
        self._episode_body_rate_saturated_count += diagnostics["body_rate_saturated"].long()
        self._episode_moment_saturated_count += diagnostics["moment_saturated"].long()
        self._episode_max_desired_tilt = torch.maximum(
            self._episode_max_desired_tilt, diagnostics["desired_tilt_rad"]
        )
        self._episode_max_body_rate = torch.maximum(self._episode_max_body_rate, body_rate)
        self._episode_max_moment = torch.maximum(self._episode_max_moment, moment)
        self._episode_controller_runtime_ms_sum += runtime_ms
        self._episode_controller_runtime_ms_max = torch.maximum(
            self._episode_controller_runtime_ms_max,
            torch.full_like(self._episode_controller_runtime_ms_max, runtime_ms),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # The high-level policy updates only once per DirectRLEnv step (25 Hz). The held velocity
        # reference is consumed by _apply_action at every 100 Hz physics substep.
        self._update_pad_motion()
        self._actions = actions.clone()

        # Reset the policy-relative slew state on completed episodes while keeping exact deck rigid-body
        # feedforward available on the first command of the new episode.
        new_episode = self.episode_length_buf == 0
        self._previous_relative_velocity_ref_d[new_episode] = 0.0

        deck_pose_w = self._deck_pose_command_w
        deck_velocity_w = self._deck_velocity_command_w
        target_contact_point_w = local_to_world_position(
            deck_pose_w[:, :3], deck_pose_w[:, 3:7], self._target_contact_point_d
        )
        (
            self._relative_velocity_ref_d,
            self._deck_contact_velocity_ref_w,
            self._velocity_reference_w,
            self._velocity_reference_ned,
        ) = build_velocity_reference(
            normalized_action=actions,
            deck_position_w=deck_pose_w[:, :3],
            deck_linear_velocity_w=deck_velocity_w[:, :3],
            deck_angular_velocity_w=deck_velocity_w[:, 3:],
            contact_point_w=target_contact_point_w,
            deck_quat_wxyz=deck_pose_w[:, 3:7],
            config=self._reference_adapter_config,
            previous_relative_velocity=self._previous_relative_velocity_ref_d,
            policy_dt=self.step_dt,
        )
        self._previous_relative_velocity_ref_d.copy_(self._relative_velocity_ref_d)
        self._reference_saturated = (
            torch.any(torch.abs(actions) > 1.0, dim=-1)
            | (
                torch.linalg.norm(self._relative_velocity_ref_d[:, :2], dim=-1)
                >= self.cfg.max_horizontal_relative_speed - 1.0e-6
            )
            | (self._relative_velocity_ref_d[:, 2] <= self.cfg.relative_velocity_min[2] + 1.0e-6)
            | (self._relative_velocity_ref_d[:, 2] >= self.cfg.relative_velocity_max[2] - 1.0e-6)
        )
        self._record_reference_diagnostics(actions)

    def _apply_action(self) -> None:
        sync_runtime = self.cfg.controller_runtime_sync and str(self.device).startswith("cuda")
        if sync_runtime:
            torch.cuda.synchronize()
        start = time.perf_counter()
        thrust_b, moment_b, diagnostics = self._px4_like_controller.compute(
            velocity_reference_w=self._velocity_reference_w,
            current_velocity_w=self._robot.data.root_lin_vel_w,
            current_quat_wxyz=self._robot.data.root_quat_w,
            current_angular_velocity_b=self._robot.data.root_ang_vel_b,
            mass=self._robot_mass,
            inertia_b=self._robot_inertia_b,
            gravity_magnitude=self._gravity_magnitude,
        )
        if sync_runtime:
            torch.cuda.synchronize()
        runtime_ms = 1000.0 * (time.perf_counter() - start)
        self._thrust[:, 0, :] = thrust_b
        self._moment[:, 0, :] = moment_b
        self._last_controller_diagnostics = diagnostics
        self._record_controller_diagnostics(diagnostics, moment_b, runtime_ms)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _latch_terminal_state(self, env_ids: torch.Tensor) -> None:
        super()._latch_terminal_state(env_ids)
        # Parent construction can call reset before the M2-only buffers are allocated.
        if not hasattr(self, "_episode_reference_step_count"):
            return

        reference_steps = self._episode_reference_step_count[env_ids].clamp_min(1)
        controller_steps = self._episode_controller_step_count[env_ids].clamp_min(1)
        self._last_relative_velocity_reference_norm_mean[env_ids] = (
            self._episode_reference_norm_sum[env_ids] / reference_steps
        )
        self._last_relative_velocity_reference_norm_p95[env_ids] = self._episode_percentile(
            self._reference_norm_samples,
            self._episode_reference_step_count,
            env_ids,
            0.95,
        )
        self._last_relative_velocity_reference_norm_max[env_ids] = self._episode_reference_norm_max[env_ids]
        self._last_reference_saturation_ratio[env_ids] = (
            self._episode_reference_saturated_count[env_ids].float() / reference_steps
        )

        action_mean = self._episode_action_sum[env_ids] / reference_steps.unsqueeze(1)
        action_second_moment = self._episode_action_square_sum[env_ids] / reference_steps.unsqueeze(1)
        self._last_action_mean[env_ids] = action_mean
        self._last_action_std[env_ids] = torch.sqrt(torch.clamp(action_second_moment - action_mean.square(), min=0.0))
        self._last_action_abs_max[env_ids] = self._episode_action_abs_max[env_ids]

        self._last_controller_velocity_tracking_error_mean[env_ids] = (
            self._episode_velocity_tracking_error_sum[env_ids] / controller_steps
        )
        self._last_controller_velocity_tracking_error_max[env_ids] = self._episode_velocity_tracking_error_max[env_ids]
        for target, count in (
            (self._last_controller_acceleration_saturation_ratio, self._episode_acceleration_saturated_count),
            (self._last_controller_tilt_saturation_ratio, self._episode_tilt_saturated_count),
            (self._last_controller_thrust_saturation_ratio, self._episode_thrust_saturated_count),
            (self._last_controller_body_rate_saturation_ratio, self._episode_body_rate_saturated_count),
            (self._last_controller_moment_saturation_ratio, self._episode_moment_saturated_count),
        ):
            target[env_ids] = count[env_ids].float() / controller_steps
        self._last_max_desired_tilt[env_ids] = self._episode_max_desired_tilt[env_ids]
        self._last_max_body_rate[env_ids] = self._episode_max_body_rate[env_ids]
        self._last_max_moment[env_ids] = self._episode_max_moment[env_ids]
        self._last_controller_runtime_ms_mean[env_ids] = (
            self._episode_controller_runtime_ms_sum[env_ids] / controller_steps
        )
        self._last_controller_runtime_ms_p95[env_ids] = self._episode_percentile(
            self._controller_runtime_ms_samples,
            self._episode_controller_step_count,
            env_ids,
            0.95,
        )
        self._last_controller_runtime_ms_max[env_ids] = self._episode_controller_runtime_ms_max[env_ids]

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        # During DirectRLEnv construction this override can be reached before the hierarchical buffers
        # exist, so always let the frozen parent reset first and only then reset new action state.
        super()._reset_idx(env_ids)
        if not hasattr(self, "_previous_relative_velocity_ref_d"):
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        completed = self._episode_reference_step_count[env_ids] > 0
        if torch.any(completed):
            completed_ids = env_ids[completed]
            log = self.extras.setdefault("log", {})
            log["Metrics/m2_relative_velocity_reference_norm_mean"] = self._last_relative_velocity_reference_norm_mean[
                completed_ids
            ].mean().item()
            log["Metrics/m2_reference_saturation_ratio"] = self._last_reference_saturation_ratio[completed_ids].mean().item()
            log["Metrics/m2_controller_velocity_tracking_error_mean"] = self._last_controller_velocity_tracking_error_mean[
                completed_ids
            ].mean().item()
            log["Metrics/m2_controller_acceleration_saturation_ratio"] = self._last_controller_acceleration_saturation_ratio[
                completed_ids
            ].mean().item()
            log["Metrics/m2_controller_tilt_saturation_ratio"] = self._last_controller_tilt_saturation_ratio[
                completed_ids
            ].mean().item()
            log["Metrics/m2_controller_thrust_saturation_ratio"] = self._last_controller_thrust_saturation_ratio[
                completed_ids
            ].mean().item()
            log["Metrics/m2_controller_body_rate_saturation_ratio"] = self._last_controller_body_rate_saturation_ratio[
                completed_ids
            ].mean().item()
            log["Metrics/m2_controller_moment_saturation_ratio"] = self._last_controller_moment_saturation_ratio[
                completed_ids
            ].mean().item()
            log["Metrics/m2_settled_landing_rate"] = self._last_successful_settle[completed_ids].float().mean().item()
            log["Metrics/m2_hard_contact_rate"] = self._last_hard_contact[completed_ids].float().mean().item()
            log["Metrics/m2_ground_crash_rate"] = self._last_ground_crash[completed_ids].float().mean().item()
            log["Metrics/m2_deck_miss_rate"] = self._last_deck_miss[completed_ids].float().mean().item()
            for axis, label in enumerate(("t1", "t2", "normal")):
                log[f"Metrics/m2_action_{label}_mean"] = self._last_action_mean[completed_ids, axis].mean().item()
                log[f"Metrics/m2_action_{label}_std"] = self._last_action_std[completed_ids, axis].mean().item()

        self._previous_relative_velocity_ref_d[env_ids] = 0.0
        self._relative_velocity_ref_d[env_ids] = 0.0
        self._deck_contact_velocity_ref_w[env_ids] = 0.0
        self._velocity_reference_w[env_ids] = 0.0
        self._velocity_reference_ned[env_ids] = 0.0
        self._reference_saturated[env_ids] = False

        for buffer in (
            self._episode_reference_step_count,
            self._episode_reference_norm_sum,
            self._episode_reference_norm_max,
            self._episode_reference_saturated_count,
            self._episode_controller_step_count,
            self._episode_velocity_tracking_error_sum,
            self._episode_velocity_tracking_error_max,
            self._episode_acceleration_saturated_count,
            self._episode_tilt_saturated_count,
            self._episode_thrust_saturated_count,
            self._episode_body_rate_saturated_count,
            self._episode_moment_saturated_count,
            self._episode_max_desired_tilt,
            self._episode_max_body_rate,
            self._episode_max_moment,
            self._episode_controller_runtime_ms_sum,
            self._episode_controller_runtime_ms_max,
        ):
            buffer[env_ids] = 0
        self._episode_action_sum[env_ids] = 0.0
        self._episode_action_square_sum[env_ids] = 0.0
        self._episode_action_abs_max[env_ids] = 0.0

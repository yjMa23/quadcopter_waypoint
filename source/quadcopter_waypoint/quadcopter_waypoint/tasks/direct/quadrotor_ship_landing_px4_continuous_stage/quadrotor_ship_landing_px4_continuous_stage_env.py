# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import time

import torch

from isaaclab.utils import configclass

from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_px4_hierarchical.quadrotor_ship_landing_px4_hierarchical_env import (
    QuadcopterShipLandingPx4HierarchicalEnv,
    QuadcopterShipLandingPx4HierarchicalEnvCfg,
)
from quadcopter_waypoint.utils.continuous_landing_stage import (
    ContinuousLandingGuidanceConfig,
    deck_heading_world,
    filter_landing_stage,
    limit_attitude_reference_rate,
    limit_attitude_tilt,
    limit_stage_conditioned_reference_slew,
    map_stage_conditioned_relative_velocity,
    normalized_stage_action,
    relative_angular_velocity,
    shortest_quaternion_slerp,
    smoothstep01,
    stage_conditioned_velocity_limits,
    terminal_alignment_weight,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    axis_angle_from_quat,
    local_to_world_position,
    quat_apply_inverse,
    quat_conjugate,
    quat_multiply,
)
from quadcopter_waypoint.utils.px4_reference_adapter import (
    deck_contact_point_velocity,
    deck_relative_to_world_velocity,
    world_to_ned_velocity,
)


@configclass
class QuadcopterShipLandingPx4ContinuousStageEnvCfg(QuadcopterShipLandingPx4HierarchicalEnvCfg):
    """Independent 22-D -> 4-D continuous-stage PX4-compatible landing-reference task."""

    decimation = 4
    action_space = 4
    observation_space = 22

    # S2 Theory-Gate values. They are high-level reference-planner parameters, not PX4 defaults.
    stage_filter_time_constant = 0.20
    stage_rate_limit = 2.0
    tangential_speed_approach = 0.80
    tangential_speed_terminal = 0.25
    max_descent_speed = 0.30
    ascent_speed_approach = 0.30
    ascent_speed_terminal = 0.15
    reference_accel_approach = (2.0, 2.0, 1.5)
    reference_accel_terminal = (0.8, 0.8, 0.5)
    attitude_blend_start_clearance = 0.50
    attitude_blend_full_clearance = 0.12
    terminal_attitude_max_tilt_deg = 35.0
    terminal_reference_max_rate = (2.0, 2.0, 1.5)

    # New S3-only coefficients without preregistered experimental values stay disabled. Their raw metrics
    # are nevertheless computed/logged so S7 can preregister coefficients without changing structure.
    stage_delta_reward_scale = 0.0
    terminal_attitude_alignment_reward_scale = 0.0
    relative_angular_alignment_reward_scale = 0.0
    relative_reference_delta_reward_scale = 0.0


class QuadcopterShipLandingPx4ContinuousStageEnv(QuadcopterShipLandingPx4HierarchicalEnv):
    """Continuous-stage high-level landing-reference planner with deterministic terminal attitude guidance."""

    cfg: QuadcopterShipLandingPx4ContinuousStageEnvCfg

    def __init__(
        self,
        cfg: QuadcopterShipLandingPx4ContinuousStageEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._continuous_guidance_config = ContinuousLandingGuidanceConfig(
            stage_filter_time_constant=self.cfg.stage_filter_time_constant,
            stage_rate_limit=self.cfg.stage_rate_limit,
            tangential_speed_approach=self.cfg.tangential_speed_approach,
            tangential_speed_terminal=self.cfg.tangential_speed_terminal,
            max_descent_speed=self.cfg.max_descent_speed,
            ascent_speed_approach=self.cfg.ascent_speed_approach,
            ascent_speed_terminal=self.cfg.ascent_speed_terminal,
            reference_accel_approach=tuple(self.cfg.reference_accel_approach),
            reference_accel_terminal=tuple(self.cfg.reference_accel_terminal),
            attitude_blend_start_clearance=self.cfg.attitude_blend_start_clearance,
            attitude_blend_full_clearance=self.cfg.attitude_blend_full_clearance,
            max_terminal_attitude_tilt_rad=math.radians(self.cfg.terminal_attitude_max_tilt_deg),
            max_terminal_reference_rate=tuple(self.cfg.terminal_reference_max_rate),
        )
        self._continuous_guidance_config.validate()

        # The M2 parent keeps action diagnostics in fixed 3-D buffers. Replace only this child's buffers
        # after parent construction so inherited M2 source and runtime semantics remain unchanged.
        self._episode_action_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self._episode_action_square_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self._episode_action_abs_max = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_action_mean = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_action_std = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_action_abs_max = torch.zeros(self.num_envs, 4, device=self.device)

        self._landing_stage = torch.zeros(self.num_envs, device=self.device)
        self._stage_raw = torch.zeros(self.num_envs, device=self.device)
        self._delta_stage = torch.zeros(self.num_envs, device=self.device)
        self._relative_velocity_target_d = torch.zeros(self.num_envs, 3, device=self.device)
        self._relative_reference_delta = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_attitude_reference_wxyz = self._robot.data.root_quat_w.clone()
        self._attitude_reference_wxyz = self._robot.data.root_quat_w.clone()
        self._previous_deck_heading_w = deck_heading_world(self._deck.data.root_quat_w)
        self._deck_heading_w = self._previous_deck_heading_w.clone()
        self._velocity_attitude_reference_wxyz = self._robot.data.root_quat_w.clone()
        self._terminal_alpha = torch.zeros(self.num_envs, device=self.device)
        self._terminal_attitude_tilt_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terminal_attitude_conflict_angle = torch.zeros(self.num_envs, device=self.device)
        self._attitude_reference_rate = torch.zeros(self.num_envs, 3, device=self.device)
        self._relative_angular_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._relative_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self._stage_tangential_limit = torch.zeros(self.num_envs, device=self.device)
        self._stage_descent_limit = torch.zeros(self.num_envs, device=self.device)
        self._stage_ascent_limit = torch.zeros(self.num_envs, device=self.device)

        self._episode_continuous_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_stage_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_stage_square_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_stage_min = torch.ones(self.num_envs, device=self.device)
        self._episode_stage_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_stage_variation_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_stage_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reference_variation_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_alpha_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_alpha_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_terminal_tilt_saturated_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_attitude_conflict_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_attitude_conflict_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_attitude_reference_rate_max = torch.zeros(self.num_envs, device=self.device)
        self._episode_contact_relative_angular_speed_max = torch.zeros(self.num_envs, device=self.device)

        for name in (
            "_last_stage_mean",
            "_last_stage_std",
            "_last_stage_min",
            "_last_stage_max",
            "_last_stage_variation",
            "_last_stage_saturation_ratio",
            "_last_reference_variation",
            "_last_terminal_alpha_mean",
            "_last_terminal_alpha_max",
            "_last_terminal_tilt_saturation_ratio",
            "_last_terminal_attitude_conflict_mean",
            "_last_terminal_attitude_conflict_max",
            "_last_attitude_reference_rate_max",
            "_last_contact_relative_angular_speed_max",
            "_last_terminal_relative_angular_speed",
        ):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))

        for key in (
            "continuous_height_tracking",
            "continuous_terminal_tangential",
            "continuous_contact_precision",
            "continuous_terminal_attitude_alignment",
            "continuous_relative_angular_alignment",
            "continuous_stage_delta",
            "continuous_reference_delta",
        ):
            self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _reset_continuous_reference_state(self, env_ids: torch.Tensor) -> None:
        """Reset caller-owned filter/reference memory deterministically for new episodes."""
        self._landing_stage[env_ids] = 0.0
        self._stage_raw[env_ids] = 0.0
        self._delta_stage[env_ids] = 0.0
        self._previous_relative_velocity_ref_d[env_ids] = 0.0
        self._relative_velocity_ref_d[env_ids] = 0.0
        self._relative_velocity_target_d[env_ids] = 0.0
        self._relative_reference_delta[env_ids] = 0.0
        current_attitude = self._robot.data.root_quat_w[env_ids]
        self._previous_attitude_reference_wxyz[env_ids] = current_attitude
        self._attitude_reference_wxyz[env_ids] = current_attitude
        current_heading = deck_heading_world(self._deck.data.root_quat_w[env_ids])
        self._previous_deck_heading_w[env_ids] = current_heading
        self._deck_heading_w[env_ids] = current_heading

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Update the high-level continuous reference exactly once per 25 Hz policy step."""
        self._update_pad_motion()
        self._actions = actions.clone()
        if actions.shape[-1] != 4:
            raise ValueError(f"continuous-stage action must have shape [..., 4], got {tuple(actions.shape)}")

        new_episode = self.episode_length_buf == 0
        new_episode_ids = torch.nonzero(new_episode, as_tuple=False).squeeze(-1)
        if new_episode_ids.numel() > 0:
            self._reset_continuous_reference_state(new_episode_ids)

        actions_xyz = actions[..., :3]
        action_stage = actions[..., 3]
        previous_stage = self._landing_stage.clone()
        self._stage_raw = normalized_stage_action(action_stage)
        self._landing_stage = filter_landing_stage(
            raw_stage=self._stage_raw,
            previous_stage=previous_stage,
            dt=self.step_dt,
            config=self._continuous_guidance_config,
        )
        self._delta_stage = self._landing_stage - previous_stage
        self._stage_tangential_limit, self._stage_descent_limit, self._stage_ascent_limit = (
            stage_conditioned_velocity_limits(self._landing_stage, self._continuous_guidance_config)
        )

        self._relative_velocity_target_d = map_stage_conditioned_relative_velocity(
            actions_xyz,
            self._landing_stage,
            self._continuous_guidance_config,
        )
        previous_relative_reference = self._previous_relative_velocity_ref_d.clone()
        self._relative_velocity_ref_d = limit_stage_conditioned_reference_slew(
            previous_relative_velocity=previous_relative_reference,
            target_relative_velocity=self._relative_velocity_target_d,
            stage=self._landing_stage,
            dt=self.step_dt,
            config=self._continuous_guidance_config,
        )
        self._relative_reference_delta = self._relative_velocity_ref_d - previous_relative_reference
        self._previous_relative_velocity_ref_d.copy_(self._relative_velocity_ref_d)

        deck_pose_w = self._deck_pose_command_w
        deck_velocity_w = self._deck_velocity_command_w
        target_contact_point_w = local_to_world_position(
            deck_pose_w[:, :3], deck_pose_w[:, 3:7], self._target_contact_point_d
        )
        self._deck_contact_velocity_ref_w = deck_contact_point_velocity(
            deck_position_w=deck_pose_w[:, :3],
            deck_linear_velocity_w=deck_velocity_w[:, :3],
            deck_angular_velocity_w=deck_velocity_w[:, 3:],
            contact_point_w=target_contact_point_w,
        )
        relative_velocity_w = deck_relative_to_world_velocity(
            deck_quat_wxyz=deck_pose_w[:, 3:7],
            relative_velocity_deck=self._relative_velocity_ref_d,
        )
        self._velocity_reference_w = self._deck_contact_velocity_ref_w + relative_velocity_w
        self._velocity_reference_ned = world_to_ned_velocity(self._velocity_reference_w)

        self._deck_heading_w = deck_heading_world(deck_pose_w[:, 3:7], self._previous_deck_heading_w)
        self._previous_deck_heading_w.copy_(self._deck_heading_w)
        self._velocity_attitude_reference_wxyz, _ = self._px4_like_controller.compute_velocity_attitude_reference(
            velocity_reference_w=self._velocity_reference_w,
            current_velocity_w=self._robot.data.root_lin_vel_w,
            gravity_magnitude=self._gravity_magnitude,
            heading_world=self._deck_heading_w,
        )

        kinematics = self._contact_kinematics()
        self._terminal_alpha = terminal_alignment_weight(
            self._landing_stage,
            kinematics["surface_clearance"],
            self._continuous_guidance_config,
        )
        q_deck = deck_pose_w[:, 3:7]
        q_blend = shortest_quaternion_slerp(
            self._velocity_attitude_reference_wxyz,
            q_deck,
            self._terminal_alpha,
        )
        q_feasible, self._terminal_attitude_tilt_saturated = limit_attitude_tilt(
            q_blend,
            self._deck_heading_w,
            self._continuous_guidance_config.max_terminal_attitude_tilt_rad,
        )
        previous_attitude_reference = self._previous_attitude_reference_wxyz.clone()
        self._attitude_reference_wxyz = limit_attitude_reference_rate(
            previous_attitude_reference,
            q_feasible,
            dt=self.step_dt,
            max_rate_world=self._continuous_guidance_config.max_terminal_reference_rate,
        )
        q_delta = quat_multiply(self._attitude_reference_wxyz, quat_conjugate(previous_attitude_reference))
        self._attitude_reference_rate = axis_angle_from_quat(q_delta) / self.step_dt
        self._previous_attitude_reference_wxyz.copy_(self._attitude_reference_wxyz)

        conflict_dot = torch.abs(torch.sum(self._velocity_attitude_reference_wxyz * q_deck, dim=-1)).clamp(0.0, 1.0)
        self._terminal_attitude_conflict_angle = 2.0 * torch.acos(conflict_dot)
        self._relative_angular_velocity_w = relative_angular_velocity(
            self._robot.data.root_ang_vel_w,
            kinematics["deck_ang_vel_w"],
        )
        self._relative_angular_speed = torch.linalg.norm(self._relative_angular_velocity_w, dim=-1)

        self._reference_saturated = torch.any(torch.abs(actions) > 1.0, dim=-1) | torch.any(
            torch.abs(self._relative_velocity_target_d - self._relative_velocity_ref_d) > 1.0e-6,
            dim=-1,
        )
        self._record_reference_diagnostics(actions)
        self._record_continuous_diagnostics(kinematics)

    def _apply_action(self) -> None:
        """Run the existing 100 Hz attitude/rate/moment loops using the held terminal attitude reference."""
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
            attitude_reference_wxyz=self._attitude_reference_wxyz,
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

    def _get_observations(self) -> dict:
        """Preserve the 22-D physical observation while replacing historical align state at index 15."""
        kinematics = self._contact_kinematics()
        deck_rel_pos_b = quat_apply_inverse(
            self._robot.data.root_quat_w,
            kinematics["deck_pos_w"] - self._robot.data.root_pos_w,
        )
        deck_surface_rel_linear_velocity_w = kinematics["surface_velocity_w"] - self._robot.data.root_lin_vel_w
        deck_normal_b = quat_apply_inverse(self._robot.data.root_quat_w, kinematics["deck_normal_w"])
        deck_relative_angular_velocity_b = quat_apply_inverse(
            self._robot.data.root_quat_w,
            kinematics["deck_ang_vel_w"] - self._robot.data.root_ang_vel_w,
        )
        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                deck_rel_pos_b,
                deck_surface_rel_linear_velocity_w,
                self._landing_stage.unsqueeze(-1),
                deck_normal_b,
                deck_relative_angular_velocity_b,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _compute_landing_terms(self) -> dict[str, torch.Tensor]:
        """Migrate only the new task's angular safe-contact predicate to deck-relative angular speed."""
        terms = super()._compute_landing_terms()
        kinematics = self._contact_kinematics()
        omega_rel_w = relative_angular_velocity(
            self._robot.data.root_ang_vel_w,
            kinematics["deck_ang_vel_w"],
        )
        relative_ang_vel_norm = torch.linalg.norm(omega_rel_w, dim=-1)
        safe_contact = (
            terms["deck_contact"]
            & (~terms["ground_contact"])
            & terms["inside_effective_deck"]
            & (terms["horizontal_error"] < self.cfg.landing_success_radius)
            & (~terms["hard_contact"])
            & (torch.abs(terms["normal_rel_speed"]) < self.cfg.safe_contact_normal_speed)
            & (terms["tangential_rel_speed"] < self.cfg.safe_contact_tangential_speed)
            & (terms["body_deck_normal_angle"] < self.cfg.safe_contact_body_deck_angle)
            & (relative_ang_vel_norm < self.cfg.safe_contact_ang_vel)
            & (terms["upright"] > self.cfg.safe_world_upright)
            & (terms["penetration"] <= self.cfg.success_max_penetration)
        )
        terms["relative_angular_velocity_w"] = omega_rel_w
        terms["relative_ang_vel_norm"] = relative_ang_vel_norm
        terms["safe_contact"] = safe_contact
        terms["landing_candidate"] = safe_contact
        return terms

    def _get_rewards(self) -> torch.Tensor:
        """Continuous S3 reward structure with no binary landing-window decision gate."""
        terms = self._compute_landing_terms()
        horizontal_error = terms["horizontal_error"]
        stage_weight = smoothstep01(self._landing_stage)
        progress_to_pad = self._previous_horizontal_error - horizontal_error
        descent_progress = self._previous_height_error - terms["height_error"]
        desired_height = self.cfg.approach_target_height - (
            self.cfg.approach_target_height - self.cfg.landing_target_height
        ) * stage_weight
        height_tracking_error = torch.abs(terms["robot_height_above_pad"] - desired_height)
        descent_vel = torch.square(terms["excess_descent_speed"])
        terminal_tangential = stage_weight * terms["horizontal_speed"]
        contact_precision = self._terminal_alpha * horizontal_error
        terminal_attitude_alignment = self._terminal_alpha * self._terminal_attitude_conflict_angle
        relative_angular_alignment = self._terminal_alpha * terms["relative_ang_vel_norm"]
        delta_stage = torch.abs(self._delta_stage)
        relative_reference_delta = torch.linalg.norm(self._relative_reference_delta, dim=-1)

        rewards = {
            "progress_to_pad": progress_to_pad * self.cfg.progress_reward_scale,
            "horizontal_error": horizontal_error * self.cfg.horizontal_error_reward_scale * self.step_dt,
            "continuous_height_tracking": height_tracking_error * self.cfg.height_tracking_reward_scale * self.step_dt,
            "rel_vel": terms["rel_vel"] * self.cfg.rel_vel_reward_scale * self.step_dt,
            "tilt": (1.0 - terms["upright"]) * self.cfg.tilt_reward_scale * self.step_dt,
            "descent_vel": descent_vel * self.cfg.descent_vel_reward_scale * self.step_dt,
            "post_align_descent": stage_weight * descent_progress * self.cfg.post_align_descent_reward_scale,
            "continuous_terminal_tangential": terminal_tangential
            * self.cfg.near_pad_horizontal_rel_vel_reward_scale
            * self.step_dt,
            "continuous_contact_precision": contact_precision * self.cfg.center_precision_reward_scale * self.step_dt,
            "continuous_terminal_attitude_alignment": terminal_attitude_alignment
            * self.cfg.terminal_attitude_alignment_reward_scale
            * self.step_dt,
            "continuous_relative_angular_alignment": relative_angular_alignment
            * self.cfg.relative_angular_alignment_reward_scale
            * self.step_dt,
            "continuous_stage_delta": delta_stage * self.cfg.stage_delta_reward_scale,
            "continuous_reference_delta": relative_reference_delta * self.cfg.relative_reference_delta_reward_scale,
            "landing_bonus": self._landing_success.float() * self.cfg.landing_bonus,
            "crash_penalty": self._crash.float() * self.cfg.crash_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value

        self._landing_touchdown_distance.zero_()
        self._landing_touchdown_rel_vel.zero_()
        self._landing_touchdown_distance[self._landing_success] = horizontal_error[self._landing_success]
        self._landing_touchdown_rel_vel[self._landing_success] = terms["rel_vel"][self._landing_success]
        self._landing_count += self._landing_success.long()
        self._crash_count += self._crash.long()
        self._previous_horizontal_error.copy_(horizontal_error)
        self._previous_height_error.copy_(terms["height_error"])
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the frozen physical failure taxonomy and the new relative-angular safe-contact contract."""
        self._ensure_attitude_metric_buffers()
        terms = self._compute_landing_terms()

        new_contact = terms["deck_contact"] & (~self._first_contact_seen)
        self._first_contact_xy_error[new_contact] = terms["horizontal_error"][new_contact]
        self._first_contact_normal_rel_speed[new_contact] = terms["normal_rel_speed"][new_contact]
        self._first_contact_tangential_rel_speed[new_contact] = terms["tangential_rel_speed"][new_contact]
        self._first_contact_force[new_contact] = terms["deck_force"][new_contact]
        self._first_contact_precision_ok[new_contact] = (
            terms["horizontal_error"][new_contact] < self.cfg.landing_success_radius
        )
        self._first_contact_deck_roll[new_contact] = self._deck_roll[new_contact]
        self._first_contact_deck_pitch[new_contact] = self._deck_pitch[new_contact]
        self._first_contact_deck_tilt[new_contact] = terms["deck_tilt"][new_contact]
        self._first_contact_deck_angular_speed[new_contact] = terms["deck_angular_speed"][new_contact]
        self._first_contact_body_deck_normal_angle[new_contact] = terms["body_deck_normal_angle"][new_contact]

        off_center_first_contact = new_contact & (~self._first_contact_precision_ok)
        self._first_contact_seen |= terms["deck_contact"]
        self._contact_age_steps[self._first_contact_seen] += 1
        self._max_contact_force = torch.maximum(self._max_contact_force, terms["deck_force"])
        self._max_contact_impulse = torch.maximum(self._max_contact_impulse, terms["contact_impulse"])
        self._minimum_surface_clearance = torch.minimum(
            self._minimum_surface_clearance, terms["landing_surface_clearance"]
        )
        self._maximum_penetration = torch.maximum(self._maximum_penetration, terms["penetration"])

        self._settle_hold_steps[terms["safe_contact"]] += 1
        self._settle_hold_steps[~terms["safe_contact"]] = 0
        self._successful_settle.copy_(
            (self._settle_hold_steps >= self.cfg.settle_hold_steps) & self._first_contact_precision_ok
        )
        self._settle_time[self._successful_settle] = (
            self._contact_age_steps[self._successful_settle].float() * self.step_dt
        )

        self._deck_contact.copy_(terms["deck_contact"])
        self._ground_contact.copy_(terms["ground_contact"])
        self._safe_contact.copy_(terms["safe_contact"])
        self._hard_contact.copy_(terms["hard_contact"] & (~self._successful_settle))
        self._deck_miss.copy_((terms["deck_miss"] | off_center_first_contact) & (~self._successful_settle))
        self._ground_crash.copy_(terms["ground_crash"] & (~self._successful_settle))
        self._landing_success.copy_(self._successful_settle)
        self._crash.copy_(
            (self._hard_contact | self._deck_miss | self._ground_crash | terms["workspace_crash"])
            & (~self._successful_settle)
        )
        contact_relative_speed = torch.where(
            terms["deck_contact"], terms["relative_ang_vel_norm"], torch.zeros_like(terms["relative_ang_vel_norm"])
        )
        self._episode_contact_relative_angular_speed_max = torch.maximum(
            self._episode_contact_relative_angular_speed_max, contact_relative_speed
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._crash | self._landing_success
        terminal = terminated | time_out
        self._terminal_body_deck_normal_angle_metric[terminal] = terms["body_deck_normal_angle"][terminal]
        self._terminal_normal_relative_speed_metric[terminal] = terms["normal_rel_speed"][terminal]
        self._terminal_tangential_relative_speed_metric[terminal] = terms["tangential_rel_speed"][terminal]
        return terminated, time_out

    def _record_continuous_diagnostics(self, kinematics: dict[str, torch.Tensor]) -> None:
        self._episode_continuous_step_count += 1
        self._episode_stage_sum += self._landing_stage
        self._episode_stage_square_sum += self._landing_stage.square()
        self._episode_stage_min = torch.minimum(self._episode_stage_min, self._landing_stage)
        self._episode_stage_max = torch.maximum(self._episode_stage_max, self._landing_stage)
        self._episode_stage_variation_sum += torch.abs(self._delta_stage)
        self._episode_stage_saturated_count += ((self._landing_stage <= 1.0e-6) | (self._landing_stage >= 1.0 - 1.0e-6)).long()
        self._episode_reference_variation_sum += torch.linalg.norm(self._relative_reference_delta, dim=-1)
        self._episode_alpha_sum += self._terminal_alpha
        self._episode_alpha_max = torch.maximum(self._episode_alpha_max, self._terminal_alpha)
        self._episode_terminal_tilt_saturated_count += self._terminal_attitude_tilt_saturated.long()
        self._episode_attitude_conflict_sum += self._terminal_attitude_conflict_angle
        self._episode_attitude_conflict_max = torch.maximum(
            self._episode_attitude_conflict_max, self._terminal_attitude_conflict_angle
        )
        attitude_rate_norm = torch.linalg.norm(self._attitude_reference_rate, dim=-1)
        self._episode_attitude_reference_rate_max = torch.maximum(
            self._episode_attitude_reference_rate_max, attitude_rate_norm
        )
        relative_speed = torch.linalg.norm(
            relative_angular_velocity(self._robot.data.root_ang_vel_w, kinematics["deck_ang_vel_w"]), dim=-1
        )
        contact_relative_speed = torch.where(self._deck_contact, relative_speed, torch.zeros_like(relative_speed))
        self._episode_contact_relative_angular_speed_max = torch.maximum(
            self._episode_contact_relative_angular_speed_max, contact_relative_speed
        )

    def _latch_terminal_state(self, env_ids: torch.Tensor) -> None:
        super()._latch_terminal_state(env_ids)
        if not hasattr(self, "_episode_continuous_step_count"):
            return
        terminal_terms = self._compute_landing_terms()
        self._last_terminal_relative_angular_speed[env_ids] = terminal_terms["relative_ang_vel_norm"][env_ids]
        counts = self._episode_continuous_step_count[env_ids].clamp_min(1)
        mean = self._episode_stage_sum[env_ids] / counts
        second = self._episode_stage_square_sum[env_ids] / counts
        self._last_stage_mean[env_ids] = mean
        self._last_stage_std[env_ids] = torch.sqrt(torch.clamp(second - mean.square(), min=0.0))
        self._last_stage_min[env_ids] = torch.where(
            self._episode_continuous_step_count[env_ids] > 0,
            self._episode_stage_min[env_ids],
            torch.zeros_like(mean),
        )
        self._last_stage_max[env_ids] = self._episode_stage_max[env_ids]
        self._last_stage_variation[env_ids] = self._episode_stage_variation_sum[env_ids]
        self._last_stage_saturation_ratio[env_ids] = self._episode_stage_saturated_count[env_ids].float() / counts
        self._last_reference_variation[env_ids] = self._episode_reference_variation_sum[env_ids]
        self._last_terminal_alpha_mean[env_ids] = self._episode_alpha_sum[env_ids] / counts
        self._last_terminal_alpha_max[env_ids] = self._episode_alpha_max[env_ids]
        self._last_terminal_tilt_saturation_ratio[env_ids] = (
            self._episode_terminal_tilt_saturated_count[env_ids].float() / counts
        )
        self._last_terminal_attitude_conflict_mean[env_ids] = self._episode_attitude_conflict_sum[env_ids] / counts
        self._last_terminal_attitude_conflict_max[env_ids] = self._episode_attitude_conflict_max[env_ids]
        self._last_attitude_reference_rate_max[env_ids] = self._episode_attitude_reference_rate_max[env_ids]
        self._last_contact_relative_angular_speed_max[env_ids] = self._episode_contact_relative_angular_speed_max[env_ids]

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            resolved_env_ids = self._robot._ALL_INDICES
        else:
            resolved_env_ids = env_ids
        completed_ids = None
        if hasattr(self, "_episode_continuous_step_count"):
            completed = self._episode_continuous_step_count[resolved_env_ids] > 0
            completed_ids = resolved_env_ids[completed]

        super()._reset_idx(env_ids)
        if not hasattr(self, "_landing_stage"):
            return

        if completed_ids is not None and completed_ids.numel() > 0:
            log = self.extras.setdefault("log", {})
            log["Metrics/continuous_stage_mean"] = self._last_stage_mean[completed_ids].mean().item()
            log["Metrics/continuous_stage_std"] = self._last_stage_std[completed_ids].mean().item()
            log["Metrics/continuous_stage_variation"] = self._last_stage_variation[completed_ids].mean().item()
            log["Metrics/continuous_stage_saturation_ratio"] = self._last_stage_saturation_ratio[completed_ids].mean().item()
            log["Metrics/continuous_reference_variation"] = self._last_reference_variation[completed_ids].mean().item()
            log["Metrics/continuous_terminal_alpha_mean"] = self._last_terminal_alpha_mean[completed_ids].mean().item()
            log["Metrics/continuous_terminal_alpha_max"] = self._last_terminal_alpha_max[completed_ids].mean().item()
            log["Metrics/continuous_terminal_tilt_saturation_ratio"] = self._last_terminal_tilt_saturation_ratio[
                completed_ids
            ].mean().item()
            log["Metrics/continuous_terminal_attitude_conflict_mean"] = self._last_terminal_attitude_conflict_mean[
                completed_ids
            ].mean().item()
            log["Metrics/continuous_attitude_reference_rate_max"] = self._last_attitude_reference_rate_max[
                completed_ids
            ].mean().item()
            log["Metrics/continuous_contact_relative_angular_speed_max"] = self._last_contact_relative_angular_speed_max[
                completed_ids
            ].mean().item()
            log["Metrics/continuous_terminal_relative_angular_speed"] = self._last_terminal_relative_angular_speed[
                completed_ids
            ].mean().item()

        self._reset_continuous_reference_state(resolved_env_ids)
        for buffer in (
            self._episode_continuous_step_count,
            self._episode_stage_sum,
            self._episode_stage_square_sum,
            self._episode_stage_max,
            self._episode_stage_variation_sum,
            self._episode_stage_saturated_count,
            self._episode_reference_variation_sum,
            self._episode_alpha_sum,
            self._episode_alpha_max,
            self._episode_terminal_tilt_saturated_count,
            self._episode_attitude_conflict_sum,
            self._episode_attitude_conflict_max,
            self._episode_attitude_reference_rate_max,
            self._episode_contact_relative_angular_speed_max,
        ):
            buffer[resolved_env_ids] = 0
        self._episode_stage_min[resolved_env_ids] = 1.0

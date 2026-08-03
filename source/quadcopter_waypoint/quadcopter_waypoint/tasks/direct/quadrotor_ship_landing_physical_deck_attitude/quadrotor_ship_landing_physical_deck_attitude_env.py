# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_physical_deck.quadrotor_ship_landing_physical_deck_env import (
    QuadcopterShipLandingPhysicalDeckEnv,
    QuadcopterShipLandingPhysicalDeckEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    body_deck_normal_angle,
    conservative_minimum_deck_bottom_height,
    deck_normal_world,
    deck_xy_error,
    decompose_relative_velocity,
    local_to_world_position,
    quat_apply_inverse,
    quat_from_euler_xyz,
    rigid_surface_point_velocity,
    signed_deck_surface_clearance,
    world_angular_velocity_from_xyz_rates,
    world_to_local_position,
)


@configclass
class QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg(QuadcopterShipLandingPhysicalDeckEnvCfg):
    """Physical translating/heaving deck with independent sinusoidal roll and pitch motion."""

    observation_space = 22

    # The higher center height guarantees every bottom corner remains above GroundSlab across the full
    # supported ±8 degree curriculum. The runtime safety check uses the configured deck dimensions.
    pad_base_height = 0.30
    ground_slab_top_height = 0.010
    deck_ground_safety_margin = 0.040
    deck_safety_max_roll_deg = 8.0
    deck_safety_max_pitch_deg = 8.0

    # Default task is Stage D. Stage A/B/C are selected through Hydra overrides without making more task IDs.
    deck_roll_amplitude_min_deg = 0.0
    deck_roll_amplitude_max_deg = 5.0
    deck_pitch_amplitude_min_deg = 0.0
    deck_pitch_amplitude_max_deg = 5.0
    deck_roll_frequency_min = 0.08
    deck_roll_frequency_max = 0.15
    deck_pitch_frequency_min = 0.08
    deck_pitch_frequency_max = 0.15

    # Pose is authored from absolute episode time. Velocity is written as an independently computed
    # derivative, and consistency errors are exposed for the dedicated motion diagnostic.
    strict_deck_motion_consistency = False
    deck_position_consistency_tolerance = 0.010
    deck_orientation_consistency_tolerance_rad = math.radians(0.75)
    deck_linear_velocity_consistency_tolerance = 0.08
    deck_angular_velocity_consistency_tolerance = 0.08

    safe_contact_body_deck_angle = math.radians(12.0)
    safe_world_upright = 0.90
    align_body_deck_angle = math.radians(20.0)
    hard_contact_impulse_threshold = 0.025
    success_max_penetration = 0.025
    max_physical_penetration = 0.030


class QuadcopterShipLandingPhysicalDeckAttitudeEnv(QuadcopterShipLandingPhysicalDeckEnv):
    """Phase 6C environment with deck-frame contact kinematics and attitude-aware landing success."""

    cfg: QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg

    def __init__(
        self,
        cfg: QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        self._validate_deck_ground_clearance(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_attitude_buffers()
        self._ensure_attitude_metric_buffers()

    @staticmethod
    def _validate_deck_ground_clearance(cfg: QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg) -> None:
        minimum_height = conservative_minimum_deck_bottom_height(
            base_height=cfg.pad_base_height,
            maximum_heave_amplitude=cfg.pad_heave_amplitude_max,
            deck_half_length=0.5 * cfg.deck_size_x,
            deck_half_width=0.5 * cfg.deck_size_y,
            deck_half_thickness=0.5 * cfg.pad_thickness,
            maximum_roll_rad=math.radians(cfg.deck_safety_max_roll_deg),
            maximum_pitch_rad=math.radians(cfg.deck_safety_max_pitch_deg),
        )
        required_height = cfg.ground_slab_top_height + cfg.deck_ground_safety_margin
        if minimum_height <= required_height:
            raise ValueError(
                "Unsafe physical-deck attitude configuration: conservative minimum deck bottom corner "
                f"height {minimum_height:.4f} m must exceed ground top + margin {required_height:.4f} m."
            )
        for axis in ("roll", "pitch"):
            minimum = getattr(cfg, f"deck_{axis}_amplitude_min_deg")
            maximum = getattr(cfg, f"deck_{axis}_amplitude_max_deg")
            safety_maximum = getattr(cfg, f"deck_safety_max_{axis}_deg")
            if minimum < 0.0 or maximum < minimum:
                raise ValueError(f"Invalid {axis} amplitude range: {minimum}..{maximum} deg")
            if maximum > safety_maximum:
                raise ValueError(
                    f"{axis} amplitude {maximum} deg exceeds the validated safety envelope {safety_maximum} deg."
                )

    def _ensure_attitude_buffers(self) -> None:
        if hasattr(self, "_deck_motion_time"):
            return
        self._deck_motion_time = torch.zeros(self.num_envs, device=self.device)
        self._deck_origin_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._deck_xy_velocity_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._deck_heave_phase0 = torch.zeros(self.num_envs, device=self.device)
        self._deck_roll_amp = torch.zeros(self.num_envs, device=self.device)
        self._deck_roll_omega = torch.zeros(self.num_envs, device=self.device)
        self._deck_roll_phase0 = torch.zeros(self.num_envs, device=self.device)
        self._deck_pitch_amp = torch.zeros(self.num_envs, device=self.device)
        self._deck_pitch_omega = torch.zeros(self.num_envs, device=self.device)
        self._deck_pitch_phase0 = torch.zeros(self.num_envs, device=self.device)
        self._deck_roll = torch.zeros(self.num_envs, device=self.device)
        self._deck_pitch = torch.zeros(self.num_envs, device=self.device)
        self._deck_pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._deck_pose_command_w[:, 3] = 1.0
        self._deck_velocity_command_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._deck_command_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_deck_position_consistency_error = torch.zeros(self.num_envs, device=self.device)
        self._max_deck_orientation_consistency_error = torch.zeros(self.num_envs, device=self.device)
        self._max_deck_linear_velocity_consistency_error = torch.zeros(self.num_envs, device=self.device)
        self._max_deck_angular_velocity_consistency_error = torch.zeros(self.num_envs, device=self.device)

    def _ensure_attitude_metric_buffers(self) -> None:
        if hasattr(self, "_first_contact_deck_roll"):
            return
        float_buffers = (
            "_first_contact_deck_roll",
            "_first_contact_deck_pitch",
            "_first_contact_deck_tilt",
            "_first_contact_deck_angular_speed",
            "_first_contact_body_deck_normal_angle",
            "_terminal_body_deck_normal_angle_metric",
            "_terminal_normal_relative_speed_metric",
            "_terminal_tangential_relative_speed_metric",
            "_max_contact_impulse",
        )
        last_float_buffers = tuple(name.replace("_first_contact", "_last_first_contact", 1) for name in float_buffers[:5]) + (
            "_last_terminal_body_deck_normal_angle",
            "_last_terminal_normal_relative_speed",
            "_last_terminal_tangential_relative_speed",
            "_last_max_contact_impulse",
            "_last_terminal_deck_roll",
            "_last_terminal_deck_pitch",
            "_last_terminal_deck_tilt",
            "_last_terminal_deck_angular_speed",
            "_last_max_deck_position_consistency_error",
            "_last_max_deck_orientation_consistency_error",
            "_last_max_deck_linear_velocity_consistency_error",
            "_last_max_deck_angular_velocity_consistency_error",
        )
        for name in float_buffers + last_float_buffers:
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))

    def _record_previous_command_consistency(self) -> None:
        valid = self._deck_command_valid
        if not torch.any(valid):
            return
        pose = self._deck.data.root_pose_w
        velocity = torch.cat((self._deck.data.root_lin_vel_w, self._deck.data.root_ang_vel_w), dim=-1)
        position_error = torch.linalg.norm(pose[:, :3] - self._deck_pose_command_w[:, :3], dim=-1)
        quat_dot = torch.abs(torch.sum(pose[:, 3:7] * self._deck_pose_command_w[:, 3:7], dim=-1)).clamp(0.0, 1.0)
        orientation_error = 2.0 * torch.acos(quat_dot)
        linear_velocity_error = torch.linalg.norm(velocity[:, :3] - self._deck_velocity_command_w[:, :3], dim=-1)
        angular_velocity_error = torch.linalg.norm(velocity[:, 3:] - self._deck_velocity_command_w[:, 3:], dim=-1)
        self._max_deck_position_consistency_error[valid] = torch.maximum(
            self._max_deck_position_consistency_error[valid], position_error[valid]
        )
        self._max_deck_orientation_consistency_error[valid] = torch.maximum(
            self._max_deck_orientation_consistency_error[valid], orientation_error[valid]
        )
        self._max_deck_linear_velocity_consistency_error[valid] = torch.maximum(
            self._max_deck_linear_velocity_consistency_error[valid], linear_velocity_error[valid]
        )
        self._max_deck_angular_velocity_consistency_error[valid] = torch.maximum(
            self._max_deck_angular_velocity_consistency_error[valid], angular_velocity_error[valid]
        )
        if self.cfg.strict_deck_motion_consistency:
            bad = valid & (
                (position_error > self.cfg.deck_position_consistency_tolerance)
                | (orientation_error > self.cfg.deck_orientation_consistency_tolerance_rad)
                | (linear_velocity_error > self.cfg.deck_linear_velocity_consistency_tolerance)
                | (angular_velocity_error > self.cfg.deck_angular_velocity_consistency_tolerance)
            )
            if torch.any(bad):
                env_id = int(torch.nonzero(bad, as_tuple=False)[0, 0])
                raise RuntimeError(
                    "Deck pose/velocity consistency failed in env "
                    f"{env_id}: pos={float(position_error[env_id]):.6f} m, "
                    f"orientation={float(orientation_error[env_id]):.6f} rad, "
                    f"linear_velocity={float(linear_velocity_error[env_id]):.6f} m/s, "
                    f"angular_velocity={float(angular_velocity_error[env_id]):.6f} rad/s."
                )

    def _compute_absolute_deck_state(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time = self._deck_motion_time[env_ids]
        heave_phase = self._deck_heave_phase0[env_ids] + self._pad_heave_omega[env_ids] * time
        roll_phase = self._deck_roll_phase0[env_ids] + self._deck_roll_omega[env_ids] * time
        pitch_phase = self._deck_pitch_phase0[env_ids] + self._deck_pitch_omega[env_ids] * time

        roll = self._deck_roll_amp[env_ids] * torch.sin(roll_phase)
        pitch = self._deck_pitch_amp[env_ids] * torch.sin(pitch_phase)
        yaw = torch.zeros_like(roll)
        roll_rate = self._deck_roll_amp[env_ids] * self._deck_roll_omega[env_ids] * torch.cos(roll_phase)
        pitch_rate = self._deck_pitch_amp[env_ids] * self._deck_pitch_omega[env_ids] * torch.cos(pitch_phase)
        yaw_rate = torch.zeros_like(roll)

        pose = torch.zeros(len(env_ids), 7, device=self.device)
        pose[:, :2] = self._deck_origin_xy_w[env_ids] + self._deck_xy_velocity_w[env_ids] * time.unsqueeze(-1)
        pose[:, 2] = self.cfg.pad_base_height + self._pad_heave_amp[env_ids] * torch.sin(heave_phase)
        pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        velocity = torch.zeros(len(env_ids), 6, device=self.device)
        velocity[:, :2] = self._deck_xy_velocity_w[env_ids]
        velocity[:, 2] = (
            self._pad_heave_amp[env_ids] * self._pad_heave_omega[env_ids] * torch.cos(heave_phase)
        )
        # The simulator expects angular velocity in world coordinates. Directly assigning roll_dot and
        # pitch_dot would be wrong once pitch is non-zero, so use the exact XYZ Euler-rate transform.
        velocity[:, 3:] = world_angular_velocity_from_xyz_rates(
            roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate
        )
        return pose, velocity, roll, pitch

    def _write_absolute_deck_state(self, env_ids: torch.Tensor) -> None:
        pose, velocity, roll, pitch = self._compute_absolute_deck_state(env_ids)
        self._deck.write_root_pose_to_sim(pose, env_ids)
        self._deck.write_root_velocity_to_sim(velocity, env_ids)
        self._deck_pose_command_w[env_ids] = pose
        self._deck_velocity_command_w[env_ids] = velocity
        self._deck_command_valid[env_ids] = True
        self._deck_roll[env_ids] = roll
        self._deck_pitch[env_ids] = pitch
        self._pad_heave_phase[env_ids] = (
            self._deck_heave_phase0[env_ids] + self._pad_heave_omega[env_ids] * self._deck_motion_time[env_ids]
        ).remainder(2.0 * math.pi)
        self._sync_pad_state_from_deck()

    def _update_pad_motion(self) -> None:
        self._ensure_heave_buffers()
        self._ensure_attitude_buffers()
        self._record_previous_command_consistency()
        self._deck_motion_time += self.step_dt
        self._write_absolute_deck_state(self._robot._ALL_INDICES)

    def _contact_kinematics(self) -> dict[str, torch.Tensor]:
        deck_pos_w = self._deck.data.root_pos_w
        deck_quat_w = self._deck.data.root_quat_w
        deck_lin_vel_w = self._deck.data.root_lin_vel_w
        deck_ang_vel_w = self._deck.data.root_ang_vel_w
        robot_pos_w = self._robot.data.root_pos_w
        robot_quat_w = self._robot.data.root_quat_w
        robot_body_z_w = deck_normal_world(robot_quat_w)
        robot_bottom_point_w = robot_pos_w - self.cfg.robot_landing_surface_offset * robot_body_z_w
        bottom_point_deck, horizontal_error = deck_xy_error(deck_pos_w, deck_quat_w, robot_bottom_point_w)
        surface_point_deck = bottom_point_deck.clone()
        surface_point_deck[:, 2] = 0.5 * self.cfg.pad_thickness
        surface_point_w = local_to_world_position(deck_pos_w, deck_quat_w, surface_point_deck)
        deck_normal_w = deck_normal_world(deck_quat_w)
        deck_surface_velocity_w = rigid_surface_point_velocity(
            deck_pos_w, deck_lin_vel_w, deck_ang_vel_w, surface_point_w
        )
        robot_bottom_velocity_w = rigid_surface_point_velocity(
            robot_pos_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            robot_bottom_point_w,
        )
        relative_velocity_w, normal_relative_speed, tangential_relative_speed = decompose_relative_velocity(
            robot_bottom_velocity_w, deck_surface_velocity_w, deck_normal_w
        )
        clearance = signed_deck_surface_clearance(
            deck_pos_w,
            deck_quat_w,
            0.5 * self.cfg.pad_thickness,
            robot_bottom_point_w,
        )
        robot_root_deck = world_to_local_position(deck_pos_w, deck_quat_w, robot_pos_w)
        return {
            "deck_pos_w": deck_pos_w,
            "deck_quat_w": deck_quat_w,
            "deck_lin_vel_w": deck_lin_vel_w,
            "deck_ang_vel_w": deck_ang_vel_w,
            "deck_normal_w": deck_normal_w,
            "robot_body_z_w": robot_body_z_w,
            "robot_bottom_point_w": robot_bottom_point_w,
            "robot_bottom_velocity_w": robot_bottom_velocity_w,
            "bottom_point_deck": bottom_point_deck,
            "robot_root_deck": robot_root_deck,
            "surface_point_w": surface_point_w,
            "surface_velocity_w": deck_surface_velocity_w,
            "relative_velocity_w": relative_velocity_w,
            "normal_relative_speed": normal_relative_speed,
            "tangential_relative_speed": tangential_relative_speed,
            "horizontal_error": horizontal_error,
            "surface_clearance": clearance,
        }

    def _get_observations(self) -> dict:
        kinematics = self._contact_kinematics()
        deck_rel_pos_b = quat_apply_inverse(
            self._robot.data.root_quat_w,
            kinematics["deck_pos_w"] - self._robot.data.root_pos_w,
        )
        # Preserve the first 16 columns' P6B semantics at zero attitude: world-frame deck/surface velocity
        # minus robot root velocity. New point-rotation effects appear only when deck angular velocity is non-zero.
        deck_surface_rel_linear_velocity_w = (
            kinematics["surface_velocity_w"] - self._robot.data.root_lin_vel_w
        )
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
                self._align_success.float().unsqueeze(-1),
                deck_normal_b,
                deck_relative_angular_velocity_b,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _compute_landing_terms(self) -> dict[str, torch.Tensor]:
        self._sync_pad_state_from_deck()
        kinematics = self._contact_kinematics()
        robot_pos_w = self._robot.data.root_pos_w
        robot_root_deck = kinematics["robot_root_deck"]
        horizontal_error = kinematics["horizontal_error"]
        robot_height_above_pad = robot_root_deck[:, 2]
        height_error = torch.abs(robot_height_above_pad)
        landing_surface_clearance = kinematics["surface_clearance"]
        relative_velocity_w = kinematics["relative_velocity_w"]
        normal_relative_speed = kinematics["normal_relative_speed"]
        tangential_relative_speed = kinematics["tangential_relative_speed"]
        rel_vel = torch.linalg.norm(relative_velocity_w, dim=-1)
        horizontal_speed = tangential_relative_speed
        descent_speed = torch.clamp(-normal_relative_speed, min=0.0)
        ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1)
        world_upright = torch.clamp(kinematics["robot_body_z_w"][:, 2], min=0.0, max=1.0)
        body_deck_angle = body_deck_normal_angle(
            self._robot.data.root_quat_w, kinematics["deck_normal_w"]
        )
        can_land = self._align_success | (self._align_hold_steps >= self.cfg.align_hold_steps)
        align_candidate = (
            (horizontal_error < self.cfg.align_radius)
            & (robot_height_above_pad > self.cfg.align_height_min)
            & (robot_height_above_pad < self.cfg.align_height_max)
            & (horizontal_speed < self.cfg.align_max_horizontal_speed)
            & (body_deck_angle < self.cfg.align_body_deck_angle)
            & (world_upright > self.cfg.align_upright)
        )

        desired_height_above_pad = torch.where(
            can_land,
            torch.full_like(horizontal_error, self.cfg.landing_target_height),
            torch.full_like(horizontal_error, self.cfg.approach_target_height),
        )
        height_tracking_error = torch.abs(robot_height_above_pad - desired_height_above_pad)
        excess_descent_speed = torch.clamp(descent_speed - self.cfg.descent_speed_limit, min=0.0)
        contact_clearance_error = torch.abs(
            landing_surface_clearance - self.cfg.landing_contact_target_clearance
        )
        near_pad_track_weight = torch.where(
            can_land,
            torch.clamp(
                (self.cfg.near_pad_track_height - robot_height_above_pad) / self.cfg.near_pad_track_height,
                min=0.0,
                max=1.0,
            ),
            torch.zeros_like(horizontal_error),
        )
        time_to_go = torch.clamp(
            robot_height_above_pad / self.cfg.expected_descent_speed,
            min=0.0,
            max=self.cfg.max_prediction_time,
        )
        predicted_bottom_deck_xy = (
            kinematics["bottom_point_deck"][:, :2]
            + quat_apply_inverse(self._deck.data.root_quat_w, relative_velocity_w)[:, :2]
            * time_to_go.unsqueeze(-1)
        )
        predicted_horizontal_error = torch.linalg.norm(predicted_bottom_deck_xy, dim=-1)

        deck_force = self._filtered_contact_force(self._deck_contact_sensor)
        ground_force = self._filtered_contact_force(self._ground_contact_sensor)
        deck_contact = deck_force > self.cfg.contact_force_threshold
        ground_contact = ground_force > self.cfg.contact_force_threshold
        contact_impulse = deck_force * self.cfg.sim.dt
        inside_effective_deck = (
            (torch.abs(kinematics["bottom_point_deck"][:, 0]) < 0.5 * self.cfg.deck_size_x - self.cfg.deck_edge_margin)
            & (torch.abs(kinematics["bottom_point_deck"][:, 1]) < 0.5 * self.cfg.deck_size_y - self.cfg.deck_edge_margin)
        )
        penetration = torch.clamp(-landing_surface_clearance, min=0.0)
        hard_contact = deck_contact & (
            (deck_force > self.cfg.hard_contact_force_threshold)
            | (contact_impulse > self.cfg.hard_contact_impulse_threshold)
            | (torch.abs(normal_relative_speed) > self.cfg.hard_contact_normal_speed)
            | (penetration > self.cfg.max_physical_penetration)
        )
        safe_contact = (
            deck_contact
            & (~ground_contact)
            & inside_effective_deck
            & (horizontal_error < self.cfg.landing_success_radius)
            & (~hard_contact)
            & (torch.abs(normal_relative_speed) < self.cfg.safe_contact_normal_speed)
            & (tangential_relative_speed < self.cfg.safe_contact_tangential_speed)
            & (ang_vel_norm < self.cfg.safe_contact_ang_vel)
            & (body_deck_angle < self.cfg.safe_contact_body_deck_angle)
            & (world_upright > self.cfg.safe_world_upright)
            & (penetration <= self.cfg.success_max_penetration)
        )
        deck_miss = (
            (~inside_effective_deck)
            & (landing_surface_clearance < 0.0)
            & (~ground_contact)
        )
        xy_distance_from_origin = torch.linalg.norm(
            robot_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        workspace_crash = (
            (robot_pos_w[:, 2] > self.cfg.max_flight_height)
            | (xy_distance_from_origin > self.cfg.max_xy_distance)
        ) & (~ground_contact)
        ground_crash = ground_contact | (robot_pos_w[:, 2] < self.cfg.min_crash_height)

        return {
            "horizontal_error": horizontal_error,
            "height_error": height_error,
            "robot_height_above_pad": robot_height_above_pad,
            "pad_surface_height_w": (
                kinematics["deck_pos_w"] + 0.5 * self.cfg.pad_thickness * kinematics["deck_normal_w"]
            )[:, 2],
            "robot_bottom_height_w": kinematics["robot_bottom_point_w"][:, 2],
            "landing_surface_clearance": landing_surface_clearance,
            "contact_clearance_error": contact_clearance_error,
            "distance_to_pad": torch.linalg.norm(robot_root_deck, dim=-1),
            "rel_vel": rel_vel,
            "horizontal_speed": horizontal_speed,
            "near_pad_track_weight": near_pad_track_weight,
            "predicted_horizontal_error": predicted_horizontal_error,
            "vertical_speed": normal_relative_speed,
            "descent_speed": descent_speed,
            "ang_vel_norm": ang_vel_norm,
            "upright": world_upright,
            "body_deck_normal_angle": body_deck_angle,
            "align_candidate": align_candidate,
            "can_land": can_land,
            "height_tracking_error": height_tracking_error,
            "excess_descent_speed": excess_descent_speed,
            "deck_force": deck_force,
            "ground_force": ground_force,
            "contact_impulse": contact_impulse,
            "deck_contact": deck_contact,
            "ground_contact": ground_contact,
            "safe_contact": safe_contact,
            "hard_contact": hard_contact,
            "deck_miss": deck_miss,
            "ground_crash": ground_crash,
            "workspace_crash": workspace_crash,
            "inside_effective_deck": inside_effective_deck,
            "normal_rel_speed": normal_relative_speed,
            "tangential_rel_speed": tangential_relative_speed,
            "penetration": penetration,
            "landing_candidate": safe_contact,
            "crash": ground_crash | hard_contact | deck_miss | workspace_crash,
            "deck_tilt": torch.acos(kinematics["deck_normal_w"][:, 2].clamp(-1.0, 1.0)),
            "deck_angular_speed": torch.linalg.norm(kinematics["deck_ang_vel_w"], dim=-1),
            "surface_velocity_w": kinematics["surface_velocity_w"],
            "relative_velocity_w": relative_velocity_w,
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_attitude_metric_buffers()
        terms = self._compute_landing_terms()
        new_contact = terms["deck_contact"] & (~self._first_contact_seen)
        self._first_contact_deck_roll[new_contact] = self._deck_roll[new_contact]
        self._first_contact_deck_pitch[new_contact] = self._deck_pitch[new_contact]
        self._first_contact_deck_tilt[new_contact] = terms["deck_tilt"][new_contact]
        self._first_contact_deck_angular_speed[new_contact] = terms["deck_angular_speed"][new_contact]
        self._first_contact_body_deck_normal_angle[new_contact] = terms["body_deck_normal_angle"][new_contact]
        self._max_contact_impulse = torch.maximum(self._max_contact_impulse, terms["contact_impulse"])
        terminated, time_out = super()._get_dones()
        terminal = terminated | time_out
        self._terminal_body_deck_normal_angle_metric[terminal] = terms["body_deck_normal_angle"][terminal]
        self._terminal_normal_relative_speed_metric[terminal] = terms["normal_rel_speed"][terminal]
        self._terminal_tangential_relative_speed_metric[terminal] = terms["tangential_rel_speed"][terminal]
        return terminated, time_out

    def _latch_terminal_state(self, env_ids: torch.Tensor) -> None:
        self._ensure_attitude_buffers()
        self._ensure_attitude_metric_buffers()
        super()._latch_terminal_state(env_ids)
        terms = self._compute_landing_terms()
        kinematics = self._contact_kinematics()
        self._terminal_relative_vel_w[env_ids] = kinematics["relative_velocity_w"][env_ids]
        self._terminal_horizontal_error[env_ids] = terms["horizontal_error"][env_ids]
        self._terminal_surface_clearance[env_ids] = terms["landing_surface_clearance"][env_ids]
        self._last_first_contact_deck_roll[env_ids] = self._first_contact_deck_roll[env_ids]
        self._last_first_contact_deck_pitch[env_ids] = self._first_contact_deck_pitch[env_ids]
        self._last_first_contact_deck_tilt[env_ids] = self._first_contact_deck_tilt[env_ids]
        self._last_first_contact_deck_angular_speed[env_ids] = self._first_contact_deck_angular_speed[env_ids]
        self._last_first_contact_body_deck_normal_angle[env_ids] = self._first_contact_body_deck_normal_angle[env_ids]
        self._last_terminal_body_deck_normal_angle[env_ids] = self._terminal_body_deck_normal_angle_metric[env_ids]
        self._last_terminal_normal_relative_speed[env_ids] = self._terminal_normal_relative_speed_metric[env_ids]
        self._last_terminal_tangential_relative_speed[env_ids] = self._terminal_tangential_relative_speed_metric[env_ids]
        self._last_max_contact_impulse[env_ids] = self._max_contact_impulse[env_ids]
        self._last_terminal_deck_roll[env_ids] = self._deck_roll[env_ids]
        self._last_terminal_deck_pitch[env_ids] = self._deck_pitch[env_ids]
        self._last_terminal_deck_tilt[env_ids] = terms["deck_tilt"][env_ids]
        self._last_terminal_deck_angular_speed[env_ids] = terms["deck_angular_speed"][env_ids]
        self._last_max_deck_position_consistency_error[env_ids] = self._max_deck_position_consistency_error[env_ids]
        self._last_max_deck_orientation_consistency_error[env_ids] = self._max_deck_orientation_consistency_error[env_ids]
        self._last_max_deck_linear_velocity_consistency_error[env_ids] = self._max_deck_linear_velocity_consistency_error[env_ids]
        self._last_max_deck_angular_velocity_consistency_error[env_ids] = self._max_deck_angular_velocity_consistency_error[env_ids]

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        self._ensure_attitude_buffers()
        self._ensure_attitude_metric_buffers()
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._deck_motion_time[env_ids] = 0.0
        self._deck_origin_xy_w[env_ids] = self._pad_pos_w[env_ids, :2]
        self._deck_xy_velocity_w[env_ids] = self._pad_vel_w[env_ids, :2]
        self._deck_heave_phase0[env_ids] = self._pad_heave_phase[env_ids]
        self._deck_roll_amp[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            math.radians(self.cfg.deck_roll_amplitude_min_deg),
            math.radians(self.cfg.deck_roll_amplitude_max_deg),
        )
        self._deck_pitch_amp[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            math.radians(self.cfg.deck_pitch_amplitude_min_deg),
            math.radians(self.cfg.deck_pitch_amplitude_max_deg),
        )
        self._deck_roll_omega[env_ids] = 2.0 * math.pi * torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.deck_roll_frequency_min, self.cfg.deck_roll_frequency_max
        )
        self._deck_pitch_omega[env_ids] = 2.0 * math.pi * torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.deck_pitch_frequency_min, self.cfg.deck_pitch_frequency_max
        )
        self._deck_roll_phase0[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 2.0 * math.pi)
        self._deck_pitch_phase0[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 2.0 * math.pi)
        self._deck_command_valid[env_ids] = False
        self._max_deck_position_consistency_error[env_ids] = 0.0
        self._max_deck_orientation_consistency_error[env_ids] = 0.0
        self._max_deck_linear_velocity_consistency_error[env_ids] = 0.0
        self._max_deck_angular_velocity_consistency_error[env_ids] = 0.0
        self._write_absolute_deck_state(env_ids)
        self._previous_horizontal_error[env_ids] = self._compute_landing_terms()["horizontal_error"][env_ids]
        self._previous_height_error[env_ids] = self._compute_landing_terms()["height_error"][env_ids]

        for name in (
            "_first_contact_deck_roll",
            "_first_contact_deck_pitch",
            "_first_contact_deck_tilt",
            "_first_contact_deck_angular_speed",
            "_first_contact_body_deck_normal_angle",
            "_terminal_body_deck_normal_angle_metric",
            "_terminal_normal_relative_speed_metric",
            "_terminal_tangential_relative_speed_metric",
            "_max_contact_impulse",
        ):
            getattr(self, name)[env_ids] = 0.0

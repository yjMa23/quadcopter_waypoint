# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

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

        self._target_contact_point_d = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_contact_point_d[:, 2] = 0.5 * self.cfg.pad_thickness

        # PhysX exposes one flattened 3x3 inertia matrix per rigid body. The Crazyflie task applies
        # wrench to the single body selected by ``self._body_id``; use that same body's current inertia.
        body_inertias = self._robot.root_physx_view.get_inertias()[:, self._body_id, :]
        self._robot_inertia_b = body_inertias.reshape(self.num_envs, len(self._body_id), 3, 3)[:, 0].to(
            device=self.device, dtype=self._robot.data.root_quat_w.dtype
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

    def _apply_action(self) -> None:
        thrust_b, moment_b, diagnostics = self._px4_like_controller.compute(
            velocity_reference_w=self._velocity_reference_w,
            current_velocity_w=self._robot.data.root_lin_vel_w,
            current_quat_wxyz=self._robot.data.root_quat_w,
            current_angular_velocity_b=self._robot.data.root_ang_vel_b,
            mass=self._robot_mass,
            inertia_b=self._robot_inertia_b,
            gravity_magnitude=self._gravity_magnitude,
        )
        self._thrust[:, 0, :] = thrust_b
        self._moment[:, 0, :] = moment_b
        self._last_controller_diagnostics = diagnostics
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        # During DirectRLEnv construction this override can be reached before the hierarchical buffers
        # exist, so always let the frozen parent reset first and only then reset new action state.
        super()._reset_idx(env_ids)
        if not hasattr(self, "_previous_relative_velocity_ref_d"):
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._previous_relative_velocity_ref_d[env_ids] = 0.0
        self._relative_velocity_ref_d[env_ids] = 0.0
        self._deck_contact_velocity_ref_w[env_ids] = 0.0
        self._velocity_reference_w[env_ids] = 0.0
        self._velocity_reference_ned[env_ids] = 0.0
        self._reference_saturated[env_ids] = False

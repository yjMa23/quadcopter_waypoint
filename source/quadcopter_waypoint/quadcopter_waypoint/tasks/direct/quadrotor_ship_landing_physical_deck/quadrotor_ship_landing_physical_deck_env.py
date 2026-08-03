# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_heave.quadrotor_ship_landing_heave_env import (
    QuadcopterShipLandingHeaveEnv,
    QuadcopterShipLandingHeaveEnvCfg,
)


@configclass
class QuadcopterShipLandingPhysicalDeckEnvCfg(QuadcopterShipLandingHeaveEnvCfg):
    """Horizontal translating/heaving deck with real collision and filtered contact reporting."""

    # ContactSensor rigid-body views need authored USD clones in Isaac Lab 5.1; Fabric-only clones
    # expose a single source prim and fail the sensor's per-environment body-count validation.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    deck_size_x = 0.50
    deck_size_y = 0.50
    pad_thickness = 0.04
    pad_base_height = 0.18
    pad_heave_amplitude_min = 0.08
    pad_heave_amplitude_max = 0.12
    pad_heave_frequency_min = 0.18
    pad_heave_frequency_max = 0.30

    deck: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Deck",
        spawn=sim_utils.CuboidCfg(
            size=(deck_size_x, deck_size_y, pad_thickness),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.42, 0.72), metallic=0.1),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, pad_base_height)),
    )
    ground_slab: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GroundSlab",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.18, 0.20)),
            activate_contact_sensors=True,
        ),
        # The top surface is 1 cm above the global plane so this filtered rigid body is the first ground contact.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005)),
    )
    deck_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Deck",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/body"],
    )
    ground_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/GroundSlab",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/body"],
    )

    # Stage-A physical-contact thresholds. They are deliberately wider than the final target so the
    # Phase-5D policy can be evaluated and fine-tuned without changing its 16-D observation contract.
    contact_force_threshold = 0.02
    hard_contact_force_threshold = 2.50
    hard_contact_normal_speed = 0.80
    safe_contact_normal_speed = 0.55
    safe_contact_tangential_speed = 0.30
    safe_contact_ang_vel = 1.50
    safe_contact_upright = 0.90
    deck_edge_margin = 0.025
    max_physical_penetration = 0.025
    settle_hold_steps = 3

    # Final Stage-D precision. Earlier curriculum runs override this to 0.18 and 0.14 m.
    landing_success_radius = 0.12
    near_center_height = 0.50
    horizontal_error_reward_scale = -2.5
    predicted_pad_error_reward_scale = -8.0
    center_precision_reward_scale = -30.0
    center_precision_square_reward_scale = -80.0
    off_center_contact_reward_scale = -25.0
    descent_speed_limit = 0.18
    descent_vel_reward_scale = -6.0
    rel_vel_reward_scale = -1.0
    near_pad_horizontal_rel_vel_reward_scale = -7.0
    landing_bonus = 80.0
    crash_penalty = -30.0


class QuadcopterShipLandingPhysicalDeckEnv(QuadcopterShipLandingHeaveEnv):
    """Phase 6B environment using the entity state and contact reports of a physical deck."""

    cfg: QuadcopterShipLandingPhysicalDeckEnvCfg

    def __init__(self, cfg: QuadcopterShipLandingPhysicalDeckEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._deck_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ground_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._safe_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hard_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._deck_miss = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ground_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._successful_settle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._settle_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._first_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._first_contact_precision_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._first_contact_xy_error = torch.zeros(self.num_envs, device=self.device)
        self._first_contact_normal_rel_speed = torch.zeros(self.num_envs, device=self.device)
        self._first_contact_tangential_rel_speed = torch.zeros(self.num_envs, device=self.device)
        self._first_contact_force = torch.zeros(self.num_envs, device=self.device)
        self._max_contact_force = torch.zeros(self.num_envs, device=self.device)
        self._contact_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._settle_time = torch.zeros(self.num_envs, device=self.device)
        self._minimum_surface_clearance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._maximum_penetration = torch.zeros(self.num_envs, device=self.device)

        self._last_deck_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_safe_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_hard_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_deck_miss = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ground_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_successful_settle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_first_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_first_contact_precision_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_first_contact_xy_error = torch.zeros(self.num_envs, device=self.device)
        self._last_first_contact_normal_rel_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_first_contact_tangential_rel_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_first_contact_force = torch.zeros(self.num_envs, device=self.device)
        self._last_max_contact_force = torch.zeros(self.num_envs, device=self.device)
        self._last_settle_time = torch.zeros(self.num_envs, device=self.device)
        self._last_minimum_surface_clearance = torch.zeros(self.num_envs, device=self.device)
        self._last_maximum_penetration = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["off_center_contact"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._deck = RigidObject(self.cfg.deck)
        self._ground_slab = RigidObject(self.cfg.ground_slab)
        self._deck_contact_sensor = ContactSensor(self.cfg.deck_contact_sensor)
        self._ground_contact_sensor = ContactSensor(self.cfg.ground_contact_sensor)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["deck"] = self._deck
        self.scene.rigid_objects["ground_slab"] = self._ground_slab
        self.scene.sensors["deck_contact"] = self._deck_contact_sensor
        self.scene.sensors["ground_contact"] = self._ground_contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _sync_pad_state_from_deck(self):
        """Mirror the physical entity state into the legacy policy interface; never integrate marker state."""
        self._pad_pos_w.copy_(self._deck.data.root_pos_w)
        self._pad_vel_w.copy_(self._deck.data.root_lin_vel_w)

    @staticmethod
    def _filtered_contact_force(sensor: ContactSensor) -> torch.Tensor:
        force_matrix_w = sensor.data.force_matrix_w
        if force_matrix_w is None:
            raise RuntimeError("Filtered contact sensor did not publish force_matrix_w.")
        return torch.linalg.norm(force_matrix_w.reshape(force_matrix_w.shape[0], -1, 3), dim=-1).amax(dim=1)

    def _update_pad_motion(self):
        self._ensure_heave_buffers()
        self._sync_pad_state_from_deck()
        self._pad_heave_phase += self._pad_heave_omega * self.step_dt
        self._pad_heave_phase.remainder_(2.0 * math.pi)

        target_pose_w = self._deck.data.root_pose_w.clone()
        target_pose_w[:, :2] += self._deck.data.root_lin_vel_w[:, :2] * self.step_dt
        target_pose_w[:, 2] = self.cfg.pad_base_height + self._pad_heave_amp * torch.sin(self._pad_heave_phase)
        target_pose_w[:, 3:7] = 0.0
        target_pose_w[:, 3] = 1.0

        target_velocity_w = torch.zeros(self.num_envs, 6, device=self.device)
        target_velocity_w[:, :2] = self._deck.data.root_lin_vel_w[:, :2]
        target_velocity_w[:, 2] = self._pad_heave_amp * self._pad_heave_omega * torch.cos(self._pad_heave_phase)
        self._deck.write_root_pose_to_sim(target_pose_w)
        self._deck.write_root_velocity_to_sim(target_velocity_w)
        self._sync_pad_state_from_deck()

    def _compute_landing_terms(self):
        self._sync_pad_state_from_deck()
        terms = super()._compute_landing_terms()

        deck_force = self._filtered_contact_force(self._deck_contact_sensor)
        ground_force = self._filtered_contact_force(self._ground_contact_sensor)
        deck_contact = deck_force > self.cfg.contact_force_threshold
        ground_contact = ground_force > self.cfg.contact_force_threshold

        relative_vel_w = self._robot.data.root_lin_vel_w - self._pad_vel_w
        normal_rel_speed = relative_vel_w[:, 2]
        tangential_rel_speed = torch.linalg.norm(relative_vel_w[:, :2], dim=1)
        deck_delta_xy = self._robot.data.root_pos_w[:, :2] - self._pad_pos_w[:, :2]
        inside_effective_deck = (
            (torch.abs(deck_delta_xy[:, 0]) < 0.5 * self.cfg.deck_size_x - self.cfg.deck_edge_margin)
            & (torch.abs(deck_delta_xy[:, 1]) < 0.5 * self.cfg.deck_size_y - self.cfg.deck_edge_margin)
        )
        penetration = torch.clamp(-terms["landing_surface_clearance"], min=0.0)
        hard_contact = deck_contact & (
            (deck_force > self.cfg.hard_contact_force_threshold)
            | (torch.abs(normal_rel_speed) > self.cfg.hard_contact_normal_speed)
            | (penetration > self.cfg.max_physical_penetration)
        )
        safe_contact = (
            deck_contact
            & inside_effective_deck
            & (terms["horizontal_error"] < self.cfg.landing_success_radius)
            & (~hard_contact)
            & (torch.abs(normal_rel_speed) < self.cfg.safe_contact_normal_speed)
            & (tangential_rel_speed < self.cfg.safe_contact_tangential_speed)
            & (terms["ang_vel_norm"] < self.cfg.safe_contact_ang_vel)
            & (terms["upright"] > self.cfg.safe_contact_upright)
            & (penetration <= self.cfg.max_physical_penetration)
        )
        deck_miss = (
            (~inside_effective_deck)
            & (terms["landing_surface_clearance"] < 0.0)
            & (~ground_contact)
        )
        workspace_crash = terms["crash"] & (~ground_contact)
        ground_crash = ground_contact | (self._robot.data.root_pos_w[:, 2] < self.cfg.min_crash_height)

        terms.update(
            {
                "deck_force": deck_force,
                "ground_force": ground_force,
                "deck_contact": deck_contact,
                "ground_contact": ground_contact,
                "safe_contact": safe_contact,
                "hard_contact": hard_contact,
                "deck_miss": deck_miss,
                "ground_crash": ground_crash,
                "workspace_crash": workspace_crash,
                "inside_effective_deck": inside_effective_deck,
                "normal_rel_speed": normal_rel_speed,
                "tangential_rel_speed": tangential_rel_speed,
                "penetration": penetration,
                # The physical contact report is the final success gate. Clearance remains an auxiliary
                # penetration metric and reward term, not a substitute for contact.
                "landing_candidate": safe_contact,
                "crash": ground_crash | hard_contact | deck_miss | workspace_crash,
            }
        )
        return terms

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        terms = self._compute_landing_terms()
        off_center_distance = torch.clamp(
            terms["horizontal_error"] - self.cfg.landing_success_radius, min=0.0
        )
        off_center_contact = terms["deck_contact"].float() * off_center_distance
        off_center_reward = off_center_contact * self.cfg.off_center_contact_reward_scale * self.step_dt
        self._episode_sums["off_center_contact"] += off_center_reward
        return reward + off_center_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terms = self._compute_landing_terms()
        self._align_hold_steps[terms["align_candidate"]] += 1
        self._align_hold_steps[~terms["align_candidate"]] = 0
        self._align_success |= self._align_hold_steps >= self.cfg.align_hold_steps

        new_contact = terms["deck_contact"] & (~self._first_contact_seen)
        self._first_contact_xy_error[new_contact] = terms["horizontal_error"][new_contact]
        self._first_contact_normal_rel_speed[new_contact] = terms["normal_rel_speed"][new_contact]
        self._first_contact_tangential_rel_speed[new_contact] = terms["tangential_rel_speed"][new_contact]
        self._first_contact_force[new_contact] = terms["deck_force"][new_contact]
        self._first_contact_precision_ok[new_contact] = (
            terms["horizontal_error"][new_contact] < self.cfg.landing_success_radius
        )
        off_center_first_contact = new_contact & (~self._first_contact_precision_ok)
        self._first_contact_seen |= terms["deck_contact"]
        self._contact_age_steps[self._first_contact_seen] += 1
        self._max_contact_force = torch.maximum(self._max_contact_force, terms["deck_force"])
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
        # A first contact outside the precision region is a terminal deck miss for this curriculum.
        # It cannot later become a valid "first-contact precise" landing by sliding across the deck.
        self._deck_miss.copy_((terms["deck_miss"] | off_center_first_contact) & (~self._successful_settle))
        self._ground_crash.copy_(terms["ground_crash"] & (~self._successful_settle))
        self._landing_success.copy_(self._successful_settle)
        self._crash.copy_(
            (self._hard_contact | self._deck_miss | self._ground_crash | terms["workspace_crash"])
            & (~self._successful_settle)
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._landing_success | self._crash
        return terminated, time_out

    def _latch_terminal_state(self, env_ids: torch.Tensor):
        super()._latch_terminal_state(env_ids)
        self._last_deck_contact[env_ids] = self._first_contact_seen[env_ids]
        self._last_safe_contact[env_ids] = self._safe_contact[env_ids]
        self._last_hard_contact[env_ids] = self._hard_contact[env_ids]
        self._last_deck_miss[env_ids] = self._deck_miss[env_ids]
        self._last_ground_crash[env_ids] = self._ground_crash[env_ids]
        self._last_successful_settle[env_ids] = self._successful_settle[env_ids]
        self._last_first_contact_seen[env_ids] = self._first_contact_seen[env_ids]
        self._last_first_contact_precision_ok[env_ids] = self._first_contact_precision_ok[env_ids]
        self._last_first_contact_xy_error[env_ids] = self._first_contact_xy_error[env_ids]
        self._last_first_contact_normal_rel_speed[env_ids] = self._first_contact_normal_rel_speed[env_ids]
        self._last_first_contact_tangential_rel_speed[env_ids] = self._first_contact_tangential_rel_speed[env_ids]
        self._last_first_contact_force[env_ids] = self._first_contact_force[env_ids]
        self._last_max_contact_force[env_ids] = self._max_contact_force[env_ids]
        self._last_settle_time[env_ids] = self._settle_time[env_ids]
        self._last_minimum_surface_clearance[env_ids] = self._minimum_surface_clearance[env_ids]
        self._last_maximum_penetration[env_ids] = self._maximum_penetration[env_ids]

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        deck_pose_w = torch.zeros(len(env_ids), 7, device=self.device)
        deck_pose_w[:, :3] = self._pad_pos_w[env_ids]
        deck_pose_w[:, 3] = 1.0
        deck_velocity_w = torch.zeros(len(env_ids), 6, device=self.device)
        deck_velocity_w[:, :3] = self._pad_vel_w[env_ids]
        self._deck.write_root_pose_to_sim(deck_pose_w, env_ids)
        self._deck.write_root_velocity_to_sim(deck_velocity_w, env_ids)
        self._sync_pad_state_from_deck()
        self._previous_horizontal_error[env_ids] = torch.linalg.norm(
            self._pad_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=1
        )
        self._previous_height_error[env_ids] = torch.abs(
            self._robot.data.root_pos_w[env_ids, 2] - self._pad_pos_w[env_ids, 2]
        )

        for buffer in (
            self._deck_contact,
            self._ground_contact,
            self._safe_contact,
            self._hard_contact,
            self._deck_miss,
            self._ground_crash,
            self._successful_settle,
            self._first_contact_seen,
            self._first_contact_precision_ok,
        ):
            buffer[env_ids] = False
        for buffer in (
            self._settle_hold_steps,
            self._contact_age_steps,
        ):
            buffer[env_ids] = 0
        for buffer in (
            self._first_contact_xy_error,
            self._first_contact_normal_rel_speed,
            self._first_contact_tangential_rel_speed,
            self._first_contact_force,
            self._max_contact_force,
            self._settle_time,
            self._maximum_penetration,
        ):
            buffer[env_ids] = 0.0
        self._minimum_surface_clearance[env_ids] = float("inf")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterShipLandingEnvWindow(BaseEnvWindow):
    """Window manager for the ship landing environment."""

    def __init__(self, env: QuadcopterShipLandingEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterShipLandingEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 16
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterShipLandingEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # align stage
    align_radius = 0.20
    align_height_min = 0.55
    align_height_max = 0.95
    align_max_horizontal_speed = 0.25
    align_upright = 0.92
    align_hold_steps = 15

    # landing success thresholds
    landing_success_radius = 0.16
    landing_success_height = 0.10
    landing_success_rel_vel = 0.30
    landing_success_ang_vel = 0.9
    landing_success_upright = 0.93
    landing_success_hold_steps = 4

    # height targets
    approach_target_height = 0.75
    landing_target_height = 0.08

    # descent control
    descent_speed_limit = 0.22

    # crash / workspace thresholds
    min_crash_height = 0.02
    max_flight_height = 2.5
    max_xy_distance = 2.5

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.03
    progress_reward_scale = 4.0
    height_progress_reward_scale = 0.0
    horizontal_error_reward_scale = -1.0
    height_tracking_reward_scale = -2.0
    rel_vel_reward_scale = -0.5
    tilt_reward_scale = -1.0
    descent_vel_reward_scale = -3.0
    align_bonus = 1.0
    align_hold_reward_scale = 0.5
    landing_bonus = 35.0
    post_align_descent_reward_scale = 6.0
    crash_penalty = -20.0


class QuadcopterShipLandingEnv(DirectRLEnv):
    cfg: QuadcopterShipLandingEnvCfg

    def __init__(self, cfg: QuadcopterShipLandingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Static landing pad state.
        self._pad_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._pad_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_horizontal_error = torch.zeros(self.num_envs, device=self.device)
        self._previous_height_error = torch.zeros(self.num_envs, device=self.device)
        self._align_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._align_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._landing_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._landing_touchdown_distance = torch.zeros(self.num_envs, device=self.device)
        self._landing_touchdown_rel_vel = torch.zeros(self.num_envs, device=self.device)
        self._landing_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._landing_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._crash_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_align_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_landing_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_landing_touchdown_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_landing_touchdown_rel_vel = torch.zeros(self.num_envs, device=self.device)
        self._last_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "progress_to_pad",
                "post_align_descent",
                "horizontal_error",
                "height_tracking",
                "rel_vel",
                "tilt",
                "descent_vel",
                "align_bonus",
                "align_hold",
                "landing_bonus",
                "crash_penalty",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        pad_rel_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._pad_pos_w,
        )
        pad_rel_vel_w = self._pad_vel_w - self._robot.data.root_lin_vel_w
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                pad_rel_pos_b,
                pad_rel_vel_w,
                self._align_success.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _compute_landing_terms(self):
        robot_pos_w = self._robot.data.root_pos_w
        robot_lin_vel_w = self._robot.data.root_lin_vel_w
        robot_ang_vel_b = self._robot.data.root_ang_vel_b

        horizontal_error = torch.linalg.norm(self._pad_pos_w[:, :2] - robot_pos_w[:, :2], dim=1)
        height_error = torch.abs(robot_pos_w[:, 2] - self._pad_pos_w[:, 2])
        robot_height_above_pad = robot_pos_w[:, 2] - self._pad_pos_w[:, 2]
        distance_to_pad = torch.linalg.norm(self._pad_pos_w - robot_pos_w, dim=1)
        rel_vel = torch.linalg.norm(robot_lin_vel_w - self._pad_vel_w, dim=1)
        horizontal_speed = torch.linalg.norm(robot_lin_vel_w[:, :2] - self._pad_vel_w[:, :2], dim=1)
        vertical_speed = robot_lin_vel_w[:, 2]
        descent_speed = torch.clamp(-vertical_speed, min=0.0)
        ang_vel_norm = torch.linalg.norm(robot_ang_vel_b, dim=1)
        upright = torch.clamp(-self._robot.data.projected_gravity_b[:, 2], min=0.0, max=1.0)
        can_land = self._align_success | (self._align_hold_steps >= self.cfg.align_hold_steps)

        align_candidate = (
            (horizontal_error < self.cfg.align_radius)
            & (robot_height_above_pad > self.cfg.align_height_min)
            & (robot_height_above_pad < self.cfg.align_height_max)
            & (horizontal_speed < self.cfg.align_max_horizontal_speed)
            & (upright > self.cfg.align_upright)
        )

        landing_candidate = (
            can_land
            & (horizontal_error < self.cfg.landing_success_radius)
            & (height_error < self.cfg.landing_success_height)
            & (rel_vel < self.cfg.landing_success_rel_vel)
            & (ang_vel_norm < self.cfg.landing_success_ang_vel)
            & (upright > self.cfg.landing_success_upright)
        )
        desired_height_above_pad = torch.where(
            can_land,
            torch.full_like(horizontal_error, self.cfg.landing_target_height),
            torch.full_like(horizontal_error, self.cfg.approach_target_height),
        )
        height_tracking_error = torch.abs(robot_height_above_pad - desired_height_above_pad)
        excess_descent_speed = torch.clamp(descent_speed - self.cfg.descent_speed_limit, min=0.0)

        xy_distance_from_origin = torch.linalg.norm(
            robot_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        crash = (
            (robot_pos_w[:, 2] < self.cfg.min_crash_height)
            | (robot_pos_w[:, 2] > self.cfg.max_flight_height)
            | (xy_distance_from_origin > self.cfg.max_xy_distance)
        )
        return {
            "horizontal_error": horizontal_error,
            "height_error": height_error,
            "robot_height_above_pad": robot_height_above_pad,
            "distance_to_pad": distance_to_pad,
            "rel_vel": rel_vel,
            "horizontal_speed": horizontal_speed,
            "vertical_speed": vertical_speed,
            "descent_speed": descent_speed,
            "ang_vel_norm": ang_vel_norm,
            "upright": upright,
            "align_candidate": align_candidate,
            "can_land": can_land,
            "landing_candidate": landing_candidate,
            "height_tracking_error": height_tracking_error,
            "excess_descent_speed": excess_descent_speed,
            "crash": crash,
        }

    def _get_rewards(self) -> torch.Tensor:
        terms = self._compute_landing_terms()
        lin_vel_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        horizontal_error = terms["horizontal_error"]
        progress_to_pad = self._previous_horizontal_error - horizontal_error
        post_align_descent = torch.where(
            terms["can_land"],
            self._previous_height_error - terms["height_error"],
            torch.zeros_like(horizontal_error),
        )
        descent_vel = torch.square(terms["excess_descent_speed"])

        rewards = {
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * self.step_dt,
            "progress_to_pad": progress_to_pad * self.cfg.progress_reward_scale,
            "post_align_descent": post_align_descent * self.cfg.post_align_descent_reward_scale,
            "horizontal_error": horizontal_error * self.cfg.horizontal_error_reward_scale * self.step_dt,
            "height_tracking": terms["height_tracking_error"] * self.cfg.height_tracking_reward_scale * self.step_dt,
            "rel_vel": terms["rel_vel"] * self.cfg.rel_vel_reward_scale * self.step_dt,
            "tilt": (1.0 - terms["upright"]) * self.cfg.tilt_reward_scale * self.step_dt,
            "descent_vel": descent_vel * self.cfg.descent_vel_reward_scale * self.step_dt,
            "align_bonus": terms["align_candidate"].float() * self.cfg.align_bonus * self.step_dt,
            "align_hold": self._align_success.float() * self.cfg.align_hold_reward_scale * self.step_dt,
            "landing_bonus": self._landing_success.float() * self.cfg.landing_bonus,
            "crash_penalty": self._crash.float() * self.cfg.crash_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
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
        terms = self._compute_landing_terms()
        self._align_hold_steps[terms["align_candidate"]] += 1
        self._align_hold_steps[~terms["align_candidate"]] = 0
        self._align_success |= self._align_hold_steps >= self.cfg.align_hold_steps
        terms = self._compute_landing_terms()
        self._landing_hold_steps[terms["landing_candidate"]] += 1
        self._landing_hold_steps[~terms["landing_candidate"]] = 0
        self._landing_success.copy_(self._landing_hold_steps >= self.cfg.landing_success_hold_steps)
        self._crash.copy_(terms["crash"] & (~self._landing_success))
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._crash | self._landing_success
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_pad = torch.linalg.norm(
            self._pad_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        align_success_rate = self._align_success[env_ids].float().mean()
        landing_success_rate = self._landing_success[env_ids].float().mean()
        success_ids = self._landing_success[env_ids]
        if torch.any(success_ids):
            mean_touchdown_distance = self._landing_touchdown_distance[env_ids][success_ids].mean()
            mean_touchdown_rel_vel = self._landing_touchdown_rel_vel[env_ids][success_ids].mean()
        else:
            mean_touchdown_distance = torch.tensor(0.0, device=self.device)
            mean_touchdown_rel_vel = torch.tensor(0.0, device=self.device)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/landing_success"] = torch.count_nonzero(self._landing_success[env_ids]).item()
        extras["Episode_Termination/crash"] = torch.count_nonzero(self._crash[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_pad"] = final_distance_to_pad.item()
        extras["Metrics/align_success_rate"] = align_success_rate.item()
        extras["Metrics/landing_success_rate"] = landing_success_rate.item()
        extras["Metrics/mean_touchdown_distance"] = mean_touchdown_distance.item()
        extras["Metrics/mean_touchdown_rel_vel"] = mean_touchdown_rel_vel.item()
        self.extras["log"].update(extras)
        self._last_align_success[env_ids] = self._align_success[env_ids]
        self._last_landing_success[env_ids] = self._landing_success[env_ids]
        self._last_landing_touchdown_distance[env_ids] = self._landing_touchdown_distance[env_ids]
        self._last_landing_touchdown_rel_vel[env_ids] = self._landing_touchdown_rel_vel[env_ids]
        self._last_crash[env_ids] = self._crash[env_ids]
        self._align_success[env_ids] = False
        self._align_hold_steps[env_ids] = 0
        self._landing_success[env_ids] = False
        self._landing_touchdown_distance[env_ids] = 0.0
        self._landing_touchdown_rel_vel[env_ids] = 0.0
        self._landing_hold_steps[env_ids] = 0
        self._landing_count[env_ids] = 0
        self._crash[env_ids] = False
        self._crash_count[env_ids] = 0

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        pad_xy = torch.zeros_like(self._pad_pos_w[env_ids, :2]).uniform_(-1.0, 1.0)
        pad_xy += self._terrain.env_origins[env_ids, :2]
        self._pad_pos_w[env_ids, :2] = pad_xy
        self._pad_pos_w[env_ids, 2] = 0.05
        self._pad_vel_w[env_ids] = 0.0
        default_root_state[:, 0:2] = (
            self._pad_pos_w[env_ids, 0:2] + torch.zeros_like(pad_xy).uniform_(-0.6, 0.6)
        )
        default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(0.8, 1.3)
        self._previous_horizontal_error[env_ids] = torch.linalg.norm(
            self._pad_pos_w[env_ids, :2] - default_root_state[:, :2], dim=1
        )
        self._previous_height_error[env_ids] = torch.abs(default_root_state[:, 2] - self._pad_pos_w[env_ids, 2])

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "pad_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.25, 0.25, 0.03)
                marker_cfg.prim_path = "/Visuals/Command/landing_pad"
                self.pad_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.pad_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "pad_pos_visualizer"):
                self.pad_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.pad_pos_visualizer.visualize(self._pad_pos_w)

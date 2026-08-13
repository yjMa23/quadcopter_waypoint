# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing.quadrotor_ship_landing_env import (
    QuadcopterShipLandingEnv,
    QuadcopterShipLandingEnvCfg,
)


@configclass
class QuadcopterShipLandingHeaveEnvCfg(QuadcopterShipLandingEnvCfg):
    """Ship landing with vertical deck heave.

    This task freezes the Deck-Contact Proxy Baseline DeckContact baseline and only adds a sinusoidal z-axis deck motion.
    Roll and pitch are intentionally not included in this stage.
    """

    # heave curriculum: z(t) = base_z + A sin(phase), vz(t) = A omega cos(phase)
    pad_base_height = 0.16
    pad_heave_amplitude_min = 0.08
    pad_heave_amplitude_max = 0.12
    pad_heave_frequency_min = 0.18
    pad_heave_frequency_max = 0.30

    # Make align less sensitive to the exact instantaneous deck height during heave.
    align_height_min = 0.50
    align_height_max = 1.00
    align_hold_steps = 8

    # Keep the same contact-proxy landing condition from Deck-Contact Proxy Baseline, but tighten xy precision for heave.
    landing_success_radius = 0.10
    landing_contact_clearance = 0.060
    max_landing_surface_penetration = 0.010
    landing_contact_target_clearance = 0.005

    # Center-precision reward for fine-tuning.
    near_center_height = 0.35
    center_precision_reward_scale = -8.0
    center_precision_square_reward_scale = -20.0


class QuadcopterShipLandingHeaveEnv(QuadcopterShipLandingEnv):
    """DeckContact ship landing task with z-axis sinusoidal deck heave."""

    cfg: QuadcopterShipLandingHeaveEnvCfg

    def __init__(self, cfg: QuadcopterShipLandingHeaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._episode_sums["center_precision"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_sums["center_precision_square"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
        descent_horizontal_rel_vel = torch.where(
            terms["can_land"], terms["horizontal_speed"], torch.zeros_like(horizontal_error)
        )
        near_pad_horizontal_rel_vel = terms["near_pad_track_weight"] * terms["horizontal_speed"]
        predicted_pad_error = torch.where(
            terms["can_land"], terms["predicted_horizontal_error"], torch.zeros_like(horizontal_error)
        )
        contact_clearance = torch.where(
            terms["can_land"], terms["contact_clearance_error"], torch.zeros_like(horizontal_error)
        )
        near_center_weight = torch.where(
            terms["can_land"],
            torch.clamp(
                (self.cfg.near_center_height - terms["robot_height_above_pad"]) / self.cfg.near_center_height,
                min=0.0,
                max=1.0,
            ),
            torch.zeros_like(horizontal_error),
        )
        center_precision = near_center_weight * horizontal_error
        center_precision_square = near_center_weight * torch.square(horizontal_error)

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
            "descent_horizontal_rel_vel": descent_horizontal_rel_vel
            * self.cfg.descent_horizontal_rel_vel_reward_scale
            * self.step_dt,
            "near_pad_horizontal_rel_vel": near_pad_horizontal_rel_vel
            * self.cfg.near_pad_horizontal_rel_vel_reward_scale
            * self.step_dt,
            "predicted_pad_error": predicted_pad_error * self.cfg.predicted_pad_error_reward_scale * self.step_dt,
            "contact_clearance": contact_clearance * self.cfg.contact_clearance_reward_scale * self.step_dt,
            "center_precision": center_precision * self.cfg.center_precision_reward_scale * self.step_dt,
            "center_precision_square": center_precision_square
            * self.cfg.center_precision_square_reward_scale
            * self.step_dt,
            "align_bonus": terms["align_candidate"].float() * self.cfg.align_bonus * self.step_dt,
            "align_hold": self._align_success.float() * self.cfg.align_hold_reward_scale * self.step_dt,
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

    def _ensure_heave_buffers(self):
        if hasattr(self, "_pad_heave_phase"):
            return
        self._pad_heave_phase = torch.zeros(self.num_envs, device=self.device)
        self._pad_heave_amp = torch.zeros(self.num_envs, device=self.device)
        self._pad_heave_omega = torch.zeros(self.num_envs, device=self.device)

    def _update_pad_motion(self):
        self._ensure_heave_buffers()
        self._pad_pos_w[:, :2] += self._pad_vel_w[:, :2] * self.step_dt
        self._pad_heave_phase += self._pad_heave_omega * self.step_dt
        self._pad_heave_phase.remainder_(2.0 * math.pi)
        self._pad_pos_w[:, 2] = self.cfg.pad_base_height + self._pad_heave_amp * torch.sin(self._pad_heave_phase)
        self._pad_vel_w[:, 2] = self._pad_heave_amp * self._pad_heave_omega * torch.cos(self._pad_heave_phase)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self._ensure_heave_buffers()
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._pad_heave_amp[env_ids] = torch.zeros_like(self._pad_heave_amp[env_ids]).uniform_(
            self.cfg.pad_heave_amplitude_min,
            self.cfg.pad_heave_amplitude_max,
        )
        heave_frequency = torch.zeros_like(self._pad_heave_omega[env_ids]).uniform_(
            self.cfg.pad_heave_frequency_min,
            self.cfg.pad_heave_frequency_max,
        )
        self._pad_heave_omega[env_ids] = 2.0 * math.pi * heave_frequency
        self._pad_heave_phase[env_ids] = torch.zeros_like(self._pad_heave_phase[env_ids]).uniform_(
            0.0,
            2.0 * math.pi,
        )
        self._pad_pos_w[env_ids, 2] = self.cfg.pad_base_height + self._pad_heave_amp[env_ids] * torch.sin(
            self._pad_heave_phase[env_ids]
        )
        self._pad_vel_w[env_ids, 2] = self._pad_heave_amp[env_ids] * self._pad_heave_omega[env_ids] * torch.cos(
            self._pad_heave_phase[env_ids]
        )

        # Recompute the initial height error after changing the deck height.
        self._previous_height_error[env_ids] = torch.abs(
            self._robot.data.root_pos_w[env_ids, 2] - self._pad_pos_w[env_ids, 2]
        )

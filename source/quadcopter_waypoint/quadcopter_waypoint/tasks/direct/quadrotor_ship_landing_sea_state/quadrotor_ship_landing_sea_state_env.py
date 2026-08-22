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
from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    conservative_minimum_deck_bottom_height,
    quat_from_euler_xyz,
    world_angular_velocity_from_xyz_rates,
)
from quadcopter_waypoint.utils.sea_state_motion import (
    SurrogateResponseConfig,
    angular_frequency_grid,
    sample_jonswap_components,
    scale_components_to_bound,
    surrogate_vessel_response_components,
    synthesize_components,
)


@configclass
class QuadcopterShipLandingSeaStateEnvCfg(QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg):
    """Stochastic JONSWAP benchmark preserving the PhysicalDeckAttitude task contract."""

    observation_space = 22

    # ``compatibility`` delegates deck motion to PhysicalDeckAttitude exactly. ``stochastic`` replaces
    # only the deck-motion generator while preserving reward, observation, contact, and success semantics.
    sea_state_mode = "stochastic"
    sea_state_benchmark_profile = "nominal"
    sea_state_num_components = 24
    sea_state_frequency_min_hz = 0.05
    sea_state_frequency_max_hz = 0.80

    # Engineering benchmark ranges. They are not labelled as validated WMO/real-vessel sea states.
    sea_state_hs_min_m = 0.18
    sea_state_hs_max_m = 0.30
    sea_state_tp_min_s = 3.8
    sea_state_tp_max_s = 5.8
    sea_state_gamma_min = 2.5
    sea_state_gamma_max = 4.0
    sea_state_heading_min_deg = -180.0
    sea_state_heading_max_deg = 180.0

    # Surrogate vessel response. These are benchmark transfer-function parameters, not identified RAOs.
    sea_state_heave_gain = 1.0
    sea_state_heave_natural_frequency_hz = 0.24
    sea_state_heave_damping_ratio = 0.85
    sea_state_roll_gain_deg_per_m = 50.0
    sea_state_roll_natural_frequency_hz = 0.13
    sea_state_roll_damping_ratio = 0.55
    sea_state_pitch_gain_deg_per_m = 50.0
    sea_state_pitch_natural_frequency_hz = 0.13
    sea_state_pitch_damping_ratio = 0.55

    # Phase-independent conservative response envelopes. Coefficients are scaled once per episode;
    # runtime clipping is deliberately forbidden because it would break spectrum/derivative consistency.
    sea_state_max_heave_m = 0.12
    sea_state_max_roll_deg = 8.0
    sea_state_max_pitch_deg = 8.0


class QuadcopterShipLandingSeaStateEnv(QuadcopterShipLandingPhysicalDeckAttitudeEnv):
    """Physical-deck landing task with a stochastic JONSWAP + surrogate-vessel deck motion."""

    cfg: QuadcopterShipLandingSeaStateEnvCfg

    def __init__(self, cfg: QuadcopterShipLandingSeaStateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_sea_state_buffers()

    @staticmethod
    def _validate_deck_ground_clearance(cfg: QuadcopterShipLandingSeaStateEnvCfg) -> None:
        QuadcopterShipLandingPhysicalDeckAttitudeEnv._validate_deck_ground_clearance(cfg)
        if cfg.sea_state_mode not in ("compatibility", "stochastic"):
            raise ValueError(f"Unsupported sea_state_mode={cfg.sea_state_mode!r}")
        if cfg.sea_state_num_components <= 0:
            raise ValueError("sea_state_num_components must be positive")
        if not 0.0 < cfg.sea_state_frequency_min_hz < cfg.sea_state_frequency_max_hz:
            raise ValueError("Invalid stochastic sea-state frequency range")
        if not 0.0 <= cfg.sea_state_hs_min_m <= cfg.sea_state_hs_max_m:
            raise ValueError("Invalid Hs range")
        if not 0.0 < cfg.sea_state_tp_min_s <= cfg.sea_state_tp_max_s:
            raise ValueError("Invalid Tp range")
        if not 1.0 <= cfg.sea_state_gamma_min <= cfg.sea_state_gamma_max:
            raise ValueError("Invalid JONSWAP gamma range")
        if cfg.sea_state_heading_max_deg < cfg.sea_state_heading_min_deg:
            raise ValueError("Invalid wave-heading range")
        if cfg.sea_state_max_roll_deg > cfg.deck_safety_max_roll_deg:
            raise ValueError("Sea-state roll envelope exceeds the validated PhysicalDeckAttitude roll envelope")
        if cfg.sea_state_max_pitch_deg > cfg.deck_safety_max_pitch_deg:
            raise ValueError("Sea-state pitch envelope exceeds the validated PhysicalDeckAttitude pitch envelope")

        minimum_height = conservative_minimum_deck_bottom_height(
            base_height=cfg.pad_base_height,
            maximum_heave_amplitude=cfg.sea_state_max_heave_m,
            deck_half_length=0.5 * cfg.deck_size_x,
            deck_half_width=0.5 * cfg.deck_size_y,
            deck_half_thickness=0.5 * cfg.pad_thickness,
            maximum_roll_rad=math.radians(cfg.sea_state_max_roll_deg),
            maximum_pitch_rad=math.radians(cfg.sea_state_max_pitch_deg),
        )
        required_height = cfg.ground_slab_top_height + cfg.deck_ground_safety_margin
        if minimum_height <= required_height:
            raise ValueError(
                "Unsafe sea-state response envelope: conservative minimum deck bottom corner "
                f"height {minimum_height:.4f} m must exceed ground top + margin {required_height:.4f} m."
            )

    def _ensure_sea_state_buffers(self) -> None:
        if hasattr(self, "_sea_state_ready"):
            return
        omega, delta_omega = angular_frequency_grid(
            self.cfg.sea_state_num_components,
            self.cfg.sea_state_frequency_min_hz,
            self.cfg.sea_state_frequency_max_hz,
            device=self.device,
        )
        self._sea_omega = omega
        self._sea_delta_omega = delta_omega
        self._sea_state_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sea_hs = torch.zeros(self.num_envs, device=self.device)
        self._sea_tp = torch.zeros(self.num_envs, device=self.device)
        self._sea_gamma = torch.zeros(self.num_envs, device=self.device)
        self._sea_heading = torch.zeros(self.num_envs, device=self.device)
        component_shape = (self.num_envs, self.cfg.sea_state_num_components)
        self._sea_wave_spectrum = torch.zeros(component_shape, device=self.device)
        self._sea_wave_amplitudes = torch.zeros(component_shape, device=self.device)
        self._sea_wave_phases = torch.zeros(component_shape, device=self.device)
        self._sea_heave_amplitudes = torch.zeros(component_shape, device=self.device)
        self._sea_heave_phases = torch.zeros(component_shape, device=self.device)
        self._sea_roll_amplitudes = torch.zeros(component_shape, device=self.device)
        self._sea_roll_phases = torch.zeros(component_shape, device=self.device)
        self._sea_pitch_amplitudes = torch.zeros(component_shape, device=self.device)
        self._sea_pitch_phases = torch.zeros(component_shape, device=self.device)
        self._sea_heave_scale = torch.ones(self.num_envs, device=self.device)
        self._sea_roll_scale = torch.ones(self.num_envs, device=self.device)
        self._sea_pitch_scale = torch.ones(self.num_envs, device=self.device)

        metric_buffers = (
            "_sea_motion_sample_count",
            "_sea_heave_sq_sum",
            "_sea_roll_sq_sum",
            "_sea_pitch_sq_sum",
            "_sea_heave_max_abs",
            "_sea_roll_max_abs",
            "_sea_pitch_max_abs",
            "_sea_heave_velocity_max_abs",
            "_sea_roll_rate_max_abs",
            "_sea_pitch_rate_max_abs",
            "_sea_deck_angular_speed_max",
        )
        for name in metric_buffers:
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))

        last_buffers = (
            "_last_sea_hs",
            "_last_sea_tp",
            "_last_sea_gamma",
            "_last_sea_heading",
            "_last_sea_heave_scale",
            "_last_sea_roll_scale",
            "_last_sea_pitch_scale",
            "_last_sea_heave_rms",
            "_last_sea_roll_rms",
            "_last_sea_pitch_rms",
            "_last_sea_heave_max_abs",
            "_last_sea_roll_max_abs",
            "_last_sea_pitch_max_abs",
            "_last_sea_heave_velocity_max_abs",
            "_last_sea_roll_rate_max_abs",
            "_last_sea_pitch_rate_max_abs",
            "_last_sea_deck_angular_speed_max",
        )
        for name in last_buffers:
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))

    def _response_config(self) -> SurrogateResponseConfig:
        return SurrogateResponseConfig(
            heave_gain=self.cfg.sea_state_heave_gain,
            heave_natural_frequency_hz=self.cfg.sea_state_heave_natural_frequency_hz,
            heave_damping_ratio=self.cfg.sea_state_heave_damping_ratio,
            roll_gain_rad_per_m=math.radians(self.cfg.sea_state_roll_gain_deg_per_m),
            roll_natural_frequency_hz=self.cfg.sea_state_roll_natural_frequency_hz,
            roll_damping_ratio=self.cfg.sea_state_roll_damping_ratio,
            pitch_gain_rad_per_m=math.radians(self.cfg.sea_state_pitch_gain_deg_per_m),
            pitch_natural_frequency_hz=self.cfg.sea_state_pitch_natural_frequency_hz,
            pitch_damping_ratio=self.cfg.sea_state_pitch_damping_ratio,
        )

    def _sample_stochastic_motion(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        self._sea_hs[env_ids] = torch.empty(count, device=self.device).uniform_(
            self.cfg.sea_state_hs_min_m, self.cfg.sea_state_hs_max_m
        )
        self._sea_tp[env_ids] = torch.empty(count, device=self.device).uniform_(
            self.cfg.sea_state_tp_min_s, self.cfg.sea_state_tp_max_s
        )
        self._sea_gamma[env_ids] = torch.empty(count, device=self.device).uniform_(
            self.cfg.sea_state_gamma_min, self.cfg.sea_state_gamma_max
        )
        self._sea_heading[env_ids] = torch.empty(count, device=self.device).uniform_(
            math.radians(self.cfg.sea_state_heading_min_deg),
            math.radians(self.cfg.sea_state_heading_max_deg),
        )
        spectrum, wave_amplitudes, wave_phases = sample_jonswap_components(
            self._sea_hs[env_ids],
            self._sea_tp[env_ids],
            self._sea_gamma[env_ids],
            self._sea_omega,
            self._sea_delta_omega,
        )
        response = surrogate_vessel_response_components(
            wave_amplitudes,
            wave_phases,
            self._sea_omega,
            self._sea_heading[env_ids],
            self._response_config(),
        )
        heave_amplitudes, heave_phases = response["heave"]
        roll_amplitudes, roll_phases = response["roll"]
        pitch_amplitudes, pitch_phases = response["pitch"]
        heave_amplitudes, heave_scale = scale_components_to_bound(
            heave_amplitudes, self.cfg.sea_state_max_heave_m
        )
        roll_amplitudes, roll_scale = scale_components_to_bound(
            roll_amplitudes, math.radians(self.cfg.sea_state_max_roll_deg)
        )
        pitch_amplitudes, pitch_scale = scale_components_to_bound(
            pitch_amplitudes, math.radians(self.cfg.sea_state_max_pitch_deg)
        )

        self._sea_wave_spectrum[env_ids] = spectrum
        self._sea_wave_amplitudes[env_ids] = wave_amplitudes
        self._sea_wave_phases[env_ids] = wave_phases
        self._sea_heave_amplitudes[env_ids] = heave_amplitudes
        self._sea_heave_phases[env_ids] = heave_phases
        self._sea_roll_amplitudes[env_ids] = roll_amplitudes
        self._sea_roll_phases[env_ids] = roll_phases
        self._sea_pitch_amplitudes[env_ids] = pitch_amplitudes
        self._sea_pitch_phases[env_ids] = pitch_phases
        self._sea_heave_scale[env_ids] = heave_scale
        self._sea_roll_scale[env_ids] = roll_scale
        self._sea_pitch_scale[env_ids] = pitch_scale
        self._sea_state_ready[env_ids] = True

    def _compute_stochastic_dofs(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time = self._deck_motion_time[env_ids]
        heave, heave_rate = synthesize_components(
            time,
            self._sea_omega,
            self._sea_heave_amplitudes[env_ids],
            self._sea_heave_phases[env_ids],
        )
        roll, roll_rate = synthesize_components(
            time,
            self._sea_omega,
            self._sea_roll_amplitudes[env_ids],
            self._sea_roll_phases[env_ids],
        )
        pitch, pitch_rate = synthesize_components(
            time,
            self._sea_omega,
            self._sea_pitch_amplitudes[env_ids],
            self._sea_pitch_phases[env_ids],
        )
        return heave, heave_rate, roll, roll_rate, pitch, pitch_rate

    def _compute_absolute_deck_state(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose, velocity, roll, pitch = super()._compute_absolute_deck_state(env_ids)
        if self.cfg.sea_state_mode == "compatibility" or not hasattr(self, "_sea_state_ready"):
            return pose, velocity, roll, pitch

        ready = self._sea_state_ready[env_ids]
        if not torch.any(ready):
            return pose, velocity, roll, pitch
        ready_env_ids = env_ids[ready]
        heave, heave_rate, stochastic_roll, roll_rate, stochastic_pitch, pitch_rate = self._compute_stochastic_dofs(
            ready_env_ids
        )
        local_indices = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        time = self._deck_motion_time[ready_env_ids]
        yaw = torch.zeros_like(stochastic_roll)
        yaw_rate = torch.zeros_like(stochastic_roll)

        pose[local_indices, :2] = self._deck_origin_xy_w[ready_env_ids] + self._deck_xy_velocity_w[
            ready_env_ids
        ] * time.unsqueeze(-1)
        pose[local_indices, 2] = self.cfg.pad_base_height + heave
        pose[local_indices, 3:7] = quat_from_euler_xyz(stochastic_roll, stochastic_pitch, yaw)
        velocity[local_indices, :2] = self._deck_xy_velocity_w[ready_env_ids]
        velocity[local_indices, 2] = heave_rate
        velocity[local_indices, 3:] = world_angular_velocity_from_xyz_rates(
            stochastic_roll,
            stochastic_pitch,
            yaw,
            roll_rate,
            pitch_rate,
            yaw_rate,
        )
        roll[local_indices] = stochastic_roll
        pitch[local_indices] = stochastic_pitch
        return pose, velocity, roll, pitch

    def _reset_sea_motion_metrics(self, env_ids: torch.Tensor) -> None:
        for name in (
            "_sea_motion_sample_count",
            "_sea_heave_sq_sum",
            "_sea_roll_sq_sum",
            "_sea_pitch_sq_sum",
            "_sea_heave_max_abs",
            "_sea_roll_max_abs",
            "_sea_pitch_max_abs",
            "_sea_heave_velocity_max_abs",
            "_sea_roll_rate_max_abs",
            "_sea_pitch_rate_max_abs",
            "_sea_deck_angular_speed_max",
        ):
            getattr(self, name)[env_ids] = 0.0

    def _record_sea_motion_metrics(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sea_state_mode != "stochastic":
            return
        ready = self._sea_state_ready[env_ids]
        if not torch.any(ready):
            return
        env_ids = env_ids[ready]
        heave, heave_rate, roll, roll_rate, pitch, pitch_rate = self._compute_stochastic_dofs(env_ids)
        yaw = torch.zeros_like(roll)
        angular_velocity = world_angular_velocity_from_xyz_rates(
            roll,
            pitch,
            yaw,
            roll_rate,
            pitch_rate,
            yaw,
        )
        self._sea_motion_sample_count[env_ids] += 1.0
        self._sea_heave_sq_sum[env_ids] += heave.square()
        self._sea_roll_sq_sum[env_ids] += roll.square()
        self._sea_pitch_sq_sum[env_ids] += pitch.square()
        self._sea_heave_max_abs[env_ids] = torch.maximum(self._sea_heave_max_abs[env_ids], torch.abs(heave))
        self._sea_roll_max_abs[env_ids] = torch.maximum(self._sea_roll_max_abs[env_ids], torch.abs(roll))
        self._sea_pitch_max_abs[env_ids] = torch.maximum(self._sea_pitch_max_abs[env_ids], torch.abs(pitch))
        self._sea_heave_velocity_max_abs[env_ids] = torch.maximum(
            self._sea_heave_velocity_max_abs[env_ids], torch.abs(heave_rate)
        )
        self._sea_roll_rate_max_abs[env_ids] = torch.maximum(self._sea_roll_rate_max_abs[env_ids], torch.abs(roll_rate))
        self._sea_pitch_rate_max_abs[env_ids] = torch.maximum(
            self._sea_pitch_rate_max_abs[env_ids], torch.abs(pitch_rate)
        )
        self._sea_deck_angular_speed_max[env_ids] = torch.maximum(
            self._sea_deck_angular_speed_max[env_ids], torch.linalg.norm(angular_velocity, dim=-1)
        )

    def _update_pad_motion(self) -> None:
        if self.cfg.sea_state_mode == "compatibility":
            super()._update_pad_motion()
            return
        self._ensure_heave_buffers()
        self._ensure_attitude_buffers()
        self._ensure_sea_state_buffers()
        self._record_previous_command_consistency()
        self._deck_motion_time += self.step_dt
        self._write_absolute_deck_state(self._robot._ALL_INDICES)
        self._record_sea_motion_metrics(self._robot._ALL_INDICES)

    def _latch_terminal_state(self, env_ids: torch.Tensor) -> None:
        self._ensure_sea_state_buffers()
        super()._latch_terminal_state(env_ids)
        if self.cfg.sea_state_mode != "stochastic":
            return
        count = self._sea_motion_sample_count[env_ids].clamp_min(1.0)
        self._last_sea_hs[env_ids] = self._sea_hs[env_ids]
        self._last_sea_tp[env_ids] = self._sea_tp[env_ids]
        self._last_sea_gamma[env_ids] = self._sea_gamma[env_ids]
        self._last_sea_heading[env_ids] = self._sea_heading[env_ids]
        self._last_sea_heave_scale[env_ids] = self._sea_heave_scale[env_ids]
        self._last_sea_roll_scale[env_ids] = self._sea_roll_scale[env_ids]
        self._last_sea_pitch_scale[env_ids] = self._sea_pitch_scale[env_ids]
        self._last_sea_heave_rms[env_ids] = torch.sqrt(self._sea_heave_sq_sum[env_ids] / count)
        self._last_sea_roll_rms[env_ids] = torch.sqrt(self._sea_roll_sq_sum[env_ids] / count)
        self._last_sea_pitch_rms[env_ids] = torch.sqrt(self._sea_pitch_sq_sum[env_ids] / count)
        self._last_sea_heave_max_abs[env_ids] = self._sea_heave_max_abs[env_ids]
        self._last_sea_roll_max_abs[env_ids] = self._sea_roll_max_abs[env_ids]
        self._last_sea_pitch_max_abs[env_ids] = self._sea_pitch_max_abs[env_ids]
        self._last_sea_heave_velocity_max_abs[env_ids] = self._sea_heave_velocity_max_abs[env_ids]
        self._last_sea_roll_rate_max_abs[env_ids] = self._sea_roll_rate_max_abs[env_ids]
        self._last_sea_pitch_rate_max_abs[env_ids] = self._sea_pitch_rate_max_abs[env_ids]
        self._last_sea_deck_angular_speed_max[env_ids] = self._sea_deck_angular_speed_max[env_ids]

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        self._ensure_sea_state_buffers()
        if env_ids is None or len(env_ids) == self.num_envs:
            reset_env_ids = self._robot._ALL_INDICES
        else:
            reset_env_ids = env_ids
        self._sea_state_ready[reset_env_ids] = False
        super()._reset_idx(env_ids)
        self._reset_sea_motion_metrics(reset_env_ids)
        if self.cfg.sea_state_mode == "compatibility":
            return

        self._sample_stochastic_motion(reset_env_ids)
        self._write_absolute_deck_state(reset_env_ids)
        self._record_sea_motion_metrics(reset_env_ids)
        self._previous_horizontal_error[reset_env_ids] = self._compute_landing_terms()["horizontal_error"][reset_env_ids]
        self._previous_height_error[reset_env_ids] = self._compute_landing_terms()["height_error"][reset_env_ids]

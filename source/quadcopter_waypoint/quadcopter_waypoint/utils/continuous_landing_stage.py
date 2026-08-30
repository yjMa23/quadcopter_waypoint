"""Pure Continuous-Stage landing and terminal-attitude guidance mathematics.

This module intentionally depends only on PyTorch and the project's existing quaternion helpers.
It contains no simulator, reward, ROS 2, PX4 runtime, or policy-framework dependency. Quaternions
use the project-wide ``(w, x, y, z)`` convention and rotate local/body vectors into world.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_rotation_matrix,
    quat_multiply,
    quat_normalize,
)


@dataclass(frozen=True)
class ContinuousLandingGuidanceConfig:
    """Frozen first-version engineering values for continuous landing guidance."""

    stage_filter_time_constant: float = 0.20
    stage_rate_limit: float = 2.0
    tangential_speed_approach: float = 0.80
    tangential_speed_terminal: float = 0.25
    max_descent_speed: float = 0.30
    ascent_speed_approach: float = 0.30
    ascent_speed_terminal: float = 0.15
    reference_accel_approach: tuple[float, float, float] = (2.0, 2.0, 1.5)
    reference_accel_terminal: tuple[float, float, float] = (0.8, 0.8, 0.5)
    attitude_blend_start_clearance: float = 0.50
    attitude_blend_full_clearance: float = 0.12
    max_terminal_attitude_tilt_rad: float = math.radians(35.0)
    max_terminal_reference_rate: tuple[float, float, float] = (2.0, 2.0, 1.5)

    def validate(self) -> None:
        scalar_values = {
            "stage_filter_time_constant": self.stage_filter_time_constant,
            "stage_rate_limit": self.stage_rate_limit,
            "tangential_speed_approach": self.tangential_speed_approach,
            "tangential_speed_terminal": self.tangential_speed_terminal,
            "max_descent_speed": self.max_descent_speed,
            "ascent_speed_approach": self.ascent_speed_approach,
            "ascent_speed_terminal": self.ascent_speed_terminal,
            "attitude_blend_start_clearance": self.attitude_blend_start_clearance,
            "attitude_blend_full_clearance": self.attitude_blend_full_clearance,
            "max_terminal_attitude_tilt_rad": self.max_terminal_attitude_tilt_rad,
        }
        if any(not math.isfinite(value) for value in scalar_values.values()):
            raise ValueError("continuous landing guidance config must contain only finite values")
        if self.stage_filter_time_constant <= 0.0:
            raise ValueError("stage_filter_time_constant must be positive")
        if self.stage_rate_limit <= 0.0:
            raise ValueError("stage_rate_limit must be positive")
        if self.tangential_speed_approach <= 0.0 or self.tangential_speed_terminal <= 0.0:
            raise ValueError("tangential speed limits must be positive")
        if self.tangential_speed_terminal > self.tangential_speed_approach:
            raise ValueError("terminal tangential speed must not exceed approach speed")
        if self.max_descent_speed <= 0.0:
            raise ValueError("max_descent_speed must be positive")
        if self.ascent_speed_approach <= 0.0 or self.ascent_speed_terminal <= 0.0:
            raise ValueError("ascent speed limits must be positive")
        if self.ascent_speed_terminal > self.ascent_speed_approach:
            raise ValueError("terminal ascent speed must not exceed approach speed")
        if not self.attitude_blend_start_clearance > self.attitude_blend_full_clearance >= 0.0:
            raise ValueError("attitude blend clearances must satisfy start > full >= 0")
        if not 0.0 < self.max_terminal_attitude_tilt_rad < 0.5 * math.pi:
            raise ValueError("max_terminal_attitude_tilt_rad must be within (0, pi/2)")
        for name in ("reference_accel_approach", "reference_accel_terminal", "max_terminal_reference_rate"):
            values = getattr(self, name)
            if len(values) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} must contain three positive finite values")
        if any(
            terminal > approach
            for approach, terminal in zip(self.reference_accel_approach, self.reference_accel_terminal, strict=True)
        ):
            raise ValueError("terminal reference acceleration must not exceed approach acceleration")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} contains NaN or Inf")


def _require_vector3(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim < 1 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3], got {tuple(tensor.shape)}")
    _require_finite(name, tensor)


def _require_quaternion(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim < 1 or tensor.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [..., 4], got {tuple(tensor.shape)}")
    _require_finite(name, tensor)
    if bool(torch.any(torch.linalg.norm(tensor, dim=-1) <= 1.0e-9)):
        raise ValueError(f"{name} contains a zero-length quaternion")


def _require_positive_dt(dt: float) -> None:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")


def _field_for_batch(name: str, field: torch.Tensor, batch_shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
    _require_finite(name, field)
    if field.device != like.device or field.dtype != like.dtype:
        raise ValueError(f"{name} must match reference tensor device and dtype")
    if field.shape == batch_shape:
        return field
    if field.shape == batch_shape + (1,):
        return field.squeeze(-1)
    if field.ndim == 0:
        return field.expand(batch_shape)
    raise ValueError(f"{name} must have shape {tuple(batch_shape)} or {tuple(batch_shape + (1,))}, got {tuple(field.shape)}")


def smoothstep01(stage: torch.Tensor) -> torch.Tensor:
    """Return the shared cubic stage shaping ``C(s)=3s²-2s³`` on ``[0,1]``."""
    _require_finite("stage", stage)
    stage = stage.clamp(0.0, 1.0)
    return stage.square() * (3.0 - 2.0 * stage)


def normalized_stage_action(action_stage: torch.Tensor) -> torch.Tensor:
    """Map normalized stage action ``[-1,1]`` to raw stage ``[0,1]``."""
    _require_finite("action_stage", action_stage)
    return ((action_stage + 1.0) * 0.5).clamp(0.0, 1.0)


def filter_landing_stage(
    raw_stage: torch.Tensor,
    previous_stage: torch.Tensor,
    dt: float,
    config: ContinuousLandingGuidanceConfig = ContinuousLandingGuidanceConfig(),
) -> torch.Tensor:
    """Apply the frozen first-order low-pass and explicit stage-rate limiter.

    State is caller-owned. Episode reset semantics are therefore explicit: pass ``previous_stage=0``.
    """
    config.validate()
    _require_positive_dt(dt)
    _require_finite("raw_stage", raw_stage)
    _require_finite("previous_stage", previous_stage)
    if raw_stage.shape != previous_stage.shape:
        raise ValueError("raw_stage and previous_stage must have identical shapes")
    if raw_stage.device != previous_stage.device or raw_stage.dtype != previous_stage.dtype:
        raise ValueError("raw_stage and previous_stage must match device and dtype")

    raw = raw_stage.clamp(0.0, 1.0)
    previous = previous_stage.clamp(0.0, 1.0)
    beta = 1.0 - torch.exp(raw.new_tensor(-dt / config.stage_filter_time_constant))
    lowpass = previous + beta * (raw - previous)
    delta_max = config.stage_rate_limit * dt
    delta = (lowpass - previous).clamp(-delta_max, delta_max)
    return (previous + delta).clamp(0.0, 1.0)


def stage_conditioned_velocity_limits(
    stage: torch.Tensor,
    config: ContinuousLandingGuidanceConfig = ContinuousLandingGuidanceConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(tangential, descent, ascent)`` speed envelopes in m/s."""
    config.validate()
    shaped = smoothstep01(stage)
    tangential = config.tangential_speed_approach - (
        config.tangential_speed_approach - config.tangential_speed_terminal
    ) * shaped
    descent = config.max_descent_speed * shaped
    ascent = config.ascent_speed_approach - (
        config.ascent_speed_approach - config.ascent_speed_terminal
    ) * shaped
    return tangential, descent, ascent


def map_stage_conditioned_relative_velocity(
    normalized_action_xyz: torch.Tensor,
    stage: torch.Tensor,
    config: ContinuousLandingGuidanceConfig = ContinuousLandingGuidanceConfig(),
) -> torch.Tensor:
    """Map normalized XYZ action to deck-relative velocity under the current stage envelope."""
    config.validate()
    _require_vector3("normalized_action_xyz", normalized_action_xyz)
    stage = _field_for_batch("stage", stage, normalized_action_xyz.shape[:-1], normalized_action_xyz)
    action = normalized_action_xyz.clamp(-1.0, 1.0)
    tangential_limit, descent_limit, ascent_limit = stage_conditioned_velocity_limits(stage, config)

    tangent = action[..., :2] * tangential_limit.unsqueeze(-1)
    tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    scale = torch.clamp(tangential_limit.unsqueeze(-1) / tangent_norm.clamp_min(1.0e-9), max=1.0)
    tangent = tangent * scale
    normal_action = action[..., 2]
    normal = torch.where(normal_action >= 0.0, normal_action * ascent_limit, normal_action * descent_limit)
    return torch.cat((tangent, normal.unsqueeze(-1)), dim=-1)


def limit_stage_conditioned_reference_slew(
    previous_relative_velocity: torch.Tensor,
    target_relative_velocity: torch.Tensor,
    stage: torch.Tensor,
    dt: float,
    config: ContinuousLandingGuidanceConfig = ContinuousLandingGuidanceConfig(),
) -> torch.Tensor:
    """Limit only the policy-relative reference using stage-conditioned per-axis acceleration."""
    config.validate()
    _require_positive_dt(dt)
    _require_vector3("previous_relative_velocity", previous_relative_velocity)
    _require_vector3("target_relative_velocity", target_relative_velocity)
    if previous_relative_velocity.shape != target_relative_velocity.shape:
        raise ValueError("previous_relative_velocity and target_relative_velocity must have identical shapes")
    if previous_relative_velocity.device != target_relative_velocity.device or previous_relative_velocity.dtype != target_relative_velocity.dtype:
        raise ValueError("previous_relative_velocity and target_relative_velocity must match device and dtype")
    stage = _field_for_batch("stage", stage, previous_relative_velocity.shape[:-1], previous_relative_velocity)

    shaped = smoothstep01(stage).unsqueeze(-1)
    approach = previous_relative_velocity.new_tensor(config.reference_accel_approach)
    terminal = previous_relative_velocity.new_tensor(config.reference_accel_terminal)
    acceleration_limit = approach - (approach - terminal) * shaped
    delta_limit = acceleration_limit * dt
    delta = torch.clamp(target_relative_velocity - previous_relative_velocity, min=-delta_limit, max=delta_limit)
    return previous_relative_velocity + delta


def terminal_alignment_weight(
    stage: torch.Tensor,
    signed_clearance: torch.Tensor,
    config: ContinuousLandingGuidanceConfig = ContinuousLandingGuidanceConfig(),
) -> torch.Tensor:
    """Return ``alpha=C(stage)*C(clearance proximity)`` for terminal deck alignment."""
    config.validate()
    _require_finite("stage", stage)
    _require_finite("signed_clearance", signed_clearance)
    if stage.device != signed_clearance.device or stage.dtype != signed_clearance.dtype:
        raise ValueError("stage and signed_clearance must match device and dtype")
    try:
        stage, signed_clearance = torch.broadcast_tensors(stage, signed_clearance)
    except RuntimeError as exc:
        raise ValueError("stage and signed_clearance must be broadcast-compatible") from exc
    clearance = signed_clearance.clamp_min(0.0)
    proximity = (
        (config.attitude_blend_start_clearance - clearance)
        / (config.attitude_blend_start_clearance - config.attitude_blend_full_clearance)
    ).clamp(0.0, 1.0)
    return smoothstep01(stage) * smoothstep01(proximity)


def deck_heading_world(deck_quat_wxyz: torch.Tensor, previous_heading: torch.Tensor | None = None) -> torch.Tensor:
    """Project deck local +x into the world horizontal plane with deterministic fallback."""
    _require_quaternion("deck_quat_wxyz", deck_quat_wxyz)
    deck_quat_wxyz = quat_normalize(deck_quat_wxyz)
    local_x = deck_quat_wxyz.new_tensor([1.0, 0.0, 0.0]).expand(*deck_quat_wxyz.shape[:-1], 3)
    deck_x_world = quat_apply(deck_quat_wxyz, local_x)
    horizontal = deck_x_world.clone()
    horizontal[..., 2] = 0.0
    horizontal_norm = torch.linalg.norm(horizontal, dim=-1, keepdim=True)
    current = horizontal / horizontal_norm.clamp_min(1.0e-9)

    world_x = deck_x_world.new_tensor([1.0, 0.0, 0.0]).expand_as(deck_x_world)
    fallback = world_x
    if previous_heading is not None:
        _require_vector3("previous_heading", previous_heading)
        if previous_heading.shape != deck_x_world.shape:
            raise ValueError("previous_heading must match deck quaternion batch shape")
        if previous_heading.device != deck_x_world.device or previous_heading.dtype != deck_x_world.dtype:
            raise ValueError("previous_heading must match deck quaternion device and dtype")
        previous_horizontal = previous_heading.clone()
        previous_horizontal[..., 2] = 0.0
        previous_norm = torch.linalg.norm(previous_horizontal, dim=-1, keepdim=True)
        previous_valid = previous_norm >= 1.0e-6
        previous_unit = previous_horizontal / previous_norm.clamp_min(1.0e-9)
        fallback = torch.where(previous_valid, previous_unit, world_x)
    return torch.where(horizontal_norm >= 1.0e-6, current, fallback)


def shortest_quaternion_slerp(q_velocity: torch.Tensor, q_deck: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Shortest-path quaternion SLERP with stable normalized-lerp near zero angular separation."""
    _require_quaternion("q_velocity", q_velocity)
    _require_quaternion("q_deck", q_deck)
    if q_velocity.shape != q_deck.shape:
        raise ValueError("q_velocity and q_deck must have identical shapes")
    if q_velocity.device != q_deck.device or q_velocity.dtype != q_deck.dtype:
        raise ValueError("q_velocity and q_deck must match device and dtype")
    alpha = _field_for_batch("alpha", alpha, q_velocity.shape[:-1], q_velocity).clamp(0.0, 1.0)

    q_velocity = quat_normalize(q_velocity)
    q_deck = quat_normalize(q_deck)
    dot = torch.sum(q_velocity * q_deck, dim=-1, keepdim=True)
    q_deck = torch.where(dot < 0.0, -q_deck, q_deck)
    dot = torch.sum(q_velocity * q_deck, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    alpha_expanded = alpha.unsqueeze(-1)
    sin_theta = torch.sin(theta)
    denominator = sin_theta.clamp_min(1.0e-9)
    slerp = (
        torch.sin((1.0 - alpha_expanded) * theta) / denominator * q_velocity
        + torch.sin(alpha_expanded * theta) / denominator * q_deck
    )
    lerp = (1.0 - alpha_expanded) * q_velocity + alpha_expanded * q_deck
    result = torch.where(theta <= 1.0e-5, lerp, slerp)
    return quat_normalize(result)


def _rotation_from_body_z_and_heading(body_z_world: torch.Tensor, heading_world: torch.Tensor) -> torch.Tensor:
    heading = heading_world.clone()
    heading[..., 2] = 0.0
    heading_norm = torch.linalg.norm(heading, dim=-1, keepdim=True)
    world_x = heading.new_tensor([1.0, 0.0, 0.0]).expand_as(heading)
    heading = torch.where(heading_norm >= 1.0e-6, heading / heading_norm.clamp_min(1.0e-9), world_x)

    body_y = torch.cross(body_z_world, heading, dim=-1)
    body_y_norm = torch.linalg.norm(body_y, dim=-1, keepdim=True)
    world_y = heading.new_tensor([0.0, 1.0, 0.0]).expand_as(heading)
    fallback_body_y = torch.cross(body_z_world, world_y, dim=-1)
    fallback_norm = torch.linalg.norm(fallback_body_y, dim=-1, keepdim=True)
    world_z = heading.new_tensor([0.0, 0.0, 1.0]).expand_as(heading)
    second_fallback = torch.cross(body_z_world, world_z, dim=-1)
    fallback_body_y = torch.where(fallback_norm >= 1.0e-6, fallback_body_y, second_fallback)
    body_y = torch.where(body_y_norm >= 1.0e-6, body_y, fallback_body_y)
    body_y = body_y / torch.linalg.norm(body_y, dim=-1, keepdim=True).clamp_min(1.0e-9)
    body_x = torch.cross(body_y, body_z_world, dim=-1)
    body_x = body_x / torch.linalg.norm(body_x, dim=-1, keepdim=True).clamp_min(1.0e-9)
    return torch.stack((body_x, body_y, body_z_world), dim=-1)


def limit_attitude_tilt(
    q_reference: torch.Tensor,
    deterministic_heading: torch.Tensor,
    max_tilt_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project infeasible body-z directions onto the world-up tilt cone and report saturation."""
    if not math.isfinite(max_tilt_rad) or not 0.0 < max_tilt_rad < 0.5 * math.pi:
        raise ValueError("max_tilt_rad must be finite and within (0, pi/2)")
    _require_quaternion("q_reference", q_reference)
    _require_vector3("deterministic_heading", deterministic_heading)
    if q_reference.shape[:-1] != deterministic_heading.shape[:-1]:
        raise ValueError("q_reference and deterministic_heading batch shapes must match")
    if q_reference.device != deterministic_heading.device or q_reference.dtype != deterministic_heading.dtype:
        raise ValueError("q_reference and deterministic_heading must match device and dtype")

    q_reference = quat_normalize(q_reference)
    local_z = q_reference.new_tensor([0.0, 0.0, 1.0]).expand(*q_reference.shape[:-1], 3)
    body_z = quat_apply(q_reference, local_z)
    tilt = torch.acos(body_z[..., 2].clamp(-1.0, 1.0))
    saturated = tilt > max_tilt_rad

    horizontal = body_z.clone()
    horizontal[..., 2] = 0.0
    horizontal_norm = torch.linalg.norm(horizontal, dim=-1, keepdim=True)
    heading_horizontal = deterministic_heading.clone()
    heading_horizontal[..., 2] = 0.0
    heading_norm = torch.linalg.norm(heading_horizontal, dim=-1, keepdim=True)
    world_x = body_z.new_tensor([1.0, 0.0, 0.0]).expand_as(body_z)
    fallback_horizontal = torch.where(
        heading_norm >= 1.0e-6,
        heading_horizontal / heading_norm.clamp_min(1.0e-9),
        world_x,
    )
    horizontal_direction = torch.where(
        horizontal_norm >= 1.0e-6,
        horizontal / horizontal_norm.clamp_min(1.0e-9),
        fallback_horizontal,
    )
    limited_body_z = math.sin(max_tilt_rad) * horizontal_direction
    limited_body_z[..., 2] = math.cos(max_tilt_rad)
    rotation = _rotation_from_body_z_and_heading(limited_body_z, deterministic_heading)
    projected = quat_from_rotation_matrix(rotation)
    return torch.where(saturated.unsqueeze(-1), projected, q_reference), saturated


def limit_attitude_reference_rate(
    previous_q_reference: torch.Tensor,
    target_q_reference: torch.Tensor,
    dt: float,
    max_rate_world: tuple[float, float, float],
) -> torch.Tensor:
    """Clamp shortest world-frame quaternion increment components by per-axis angular-rate limits."""
    _require_positive_dt(dt)
    _require_quaternion("previous_q_reference", previous_q_reference)
    _require_quaternion("target_q_reference", target_q_reference)
    if previous_q_reference.shape != target_q_reference.shape:
        raise ValueError("previous_q_reference and target_q_reference must have identical shapes")
    if previous_q_reference.device != target_q_reference.device or previous_q_reference.dtype != target_q_reference.dtype:
        raise ValueError("previous_q_reference and target_q_reference must match device and dtype")
    if len(max_rate_world) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in max_rate_world):
        raise ValueError("max_rate_world must contain three positive finite values")

    previous = quat_normalize(previous_q_reference)
    target = quat_normalize(target_q_reference)
    delta = quat_multiply(target, quat_conjugate(previous))
    phi_world = axis_angle_from_quat(delta)
    max_rate = previous.new_tensor(max_rate_world)
    omega_world = torch.clamp(phi_world / dt, min=-max_rate, max=max_rate)
    limited_delta = quat_from_axis_angle(omega_world * dt)
    return quat_normalize(quat_multiply(limited_delta, previous))


def relative_angular_velocity(omega_uav_world: torch.Tensor, omega_deck_world: torch.Tensor) -> torch.Tensor:
    """Return world-frame relative angular velocity ``omega_uav - omega_deck``."""
    _require_vector3("omega_uav_world", omega_uav_world)
    _require_vector3("omega_deck_world", omega_deck_world)
    if omega_uav_world.shape != omega_deck_world.shape:
        raise ValueError("omega_uav_world and omega_deck_world must have identical shapes")
    if omega_uav_world.device != omega_deck_world.device or omega_uav_world.dtype != omega_deck_world.dtype:
        raise ValueError("omega_uav_world and omega_deck_world must match device and dtype")
    return omega_uav_world - omega_deck_world

"""PX4-compatible deck-relative velocity reference math.

The module is intentionally independent of Isaac Sim, ROS 2, and PX4 runtime packages. Simulation
world vectors follow the project deployment contract ``ENU = [East, North, Up]``. PX4 local velocity
references follow ``NED = [North, East, Down]``. Project quaternions use Isaac Lab's ``(w, x, y, z)``
order and rotate local-frame vectors into the parent/world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import quat_apply, rigid_surface_point_velocity


@dataclass(frozen=True)
class Px4ReferenceAdapterConfig:
    """Physical bounds for the deployable 3-D deck-relative velocity action."""

    relative_velocity_min: tuple[float, float, float] = (-0.80, -0.80, -0.40)
    relative_velocity_max: tuple[float, float, float] = (0.80, 0.80, 0.30)
    max_horizontal_relative_speed: float = 0.80
    max_relative_reference_acceleration: float = 2.0

    def validate(self) -> None:
        if len(self.relative_velocity_min) != 3 or len(self.relative_velocity_max) != 3:
            raise ValueError("relative velocity bounds must contain exactly three values")
        for lower, upper in zip(self.relative_velocity_min, self.relative_velocity_max, strict=True):
            if not lower < upper:
                raise ValueError(f"relative velocity lower bound {lower} must be below upper bound {upper}")
        if self.max_horizontal_relative_speed <= 0.0:
            raise ValueError("max_horizontal_relative_speed must be positive")
        if self.max_relative_reference_acceleration <= 0.0:
            raise ValueError("max_relative_reference_acceleration must be positive")


def _require_vector3(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim < 1 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3], got {tuple(tensor.shape)}")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} contains NaN or Inf")


def _bounds_like(action: torch.Tensor, config: Px4ReferenceAdapterConfig) -> tuple[torch.Tensor, torch.Tensor]:
    lower = action.new_tensor(config.relative_velocity_min)
    upper = action.new_tensor(config.relative_velocity_max)
    return lower, upper


def normalized_action_to_relative_velocity(
    normalized_action: torch.Tensor,
    config: Px4ReferenceAdapterConfig = Px4ReferenceAdapterConfig(),
) -> torch.Tensor:
    """Map normalized ``[t1, t2, normal]`` actions to physical deck-frame velocity in m/s.

    The policy action is first clamped to ``[-1, 1]`` and then mapped with a zero-preserving
    piecewise-linear scale into configured per-axis physical bounds. The tangential vector norm is additionally limited so a diagonal action
    cannot exceed the intended horizontal speed envelope.
    """
    config.validate()
    _require_vector3("normalized_action", normalized_action)
    _require_finite("normalized_action", normalized_action)

    action = normalized_action.clamp(-1.0, 1.0)
    lower, upper = _bounds_like(action, config)
    # Preserve the physically important neutral command: zero normalized action must mean zero
    # deck-relative velocity even when ascent/descent ranges are intentionally asymmetric.
    relative_velocity = torch.where(action >= 0.0, action * upper, action * (-lower))

    tangent = relative_velocity[..., :2]
    tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    scale = torch.clamp(config.max_horizontal_relative_speed / tangent_norm.clamp_min(1.0e-9), max=1.0)
    relative_velocity = torch.cat((tangent * scale, relative_velocity[..., 2:3]), dim=-1)
    return relative_velocity


def limit_relative_velocity_slew(
    previous_relative_velocity: torch.Tensor,
    target_relative_velocity: torch.Tensor,
    dt: float,
    max_acceleration: float,
) -> torch.Tensor:
    """Limit per-axis change of the policy-relative velocity reference.

    Deck rigid-body feedforward is intentionally not included here. Limiting only the policy-relative
    component preserves exact ``v_contact`` compensation while bounding abrupt policy command changes.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if max_acceleration <= 0.0:
        raise ValueError(f"max_acceleration must be positive, got {max_acceleration}")
    _require_vector3("previous_relative_velocity", previous_relative_velocity)
    _require_vector3("target_relative_velocity", target_relative_velocity)
    if previous_relative_velocity.shape != target_relative_velocity.shape:
        raise ValueError(
            "previous_relative_velocity and target_relative_velocity must have identical shapes, got "
            f"{tuple(previous_relative_velocity.shape)} and {tuple(target_relative_velocity.shape)}"
        )
    _require_finite("previous_relative_velocity", previous_relative_velocity)
    _require_finite("target_relative_velocity", target_relative_velocity)

    max_delta = max_acceleration * dt
    delta = (target_relative_velocity - previous_relative_velocity).clamp(-max_delta, max_delta)
    return previous_relative_velocity + delta


def deck_contact_point_velocity(
    deck_position_w: torch.Tensor,
    deck_linear_velocity_w: torch.Tensor,
    deck_angular_velocity_w: torch.Tensor,
    contact_point_w: torch.Tensor,
) -> torch.Tensor:
    """Return ``v_deck + omega_deck x r_contact`` in world/ENU coordinates."""
    for name, tensor in (
        ("deck_position_w", deck_position_w),
        ("deck_linear_velocity_w", deck_linear_velocity_w),
        ("deck_angular_velocity_w", deck_angular_velocity_w),
        ("contact_point_w", contact_point_w),
    ):
        _require_vector3(name, tensor)
        _require_finite(name, tensor)
    return rigid_surface_point_velocity(
        deck_position_w,
        deck_linear_velocity_w,
        deck_angular_velocity_w,
        contact_point_w,
    )


def deck_relative_to_world_velocity(
    deck_quat_wxyz: torch.Tensor,
    relative_velocity_deck: torch.Tensor,
) -> torch.Tensor:
    """Rotate deck-frame ``[t1, t2, normal]`` velocity into world/ENU coordinates."""
    _require_vector3("relative_velocity_deck", relative_velocity_deck)
    if deck_quat_wxyz.ndim < 1 or deck_quat_wxyz.shape[-1] != 4:
        raise ValueError(f"deck_quat_wxyz must have shape [..., 4], got {tuple(deck_quat_wxyz.shape)}")
    if deck_quat_wxyz.shape[:-1] != relative_velocity_deck.shape[:-1]:
        raise ValueError(
            "deck quaternion and relative velocity batch shapes must match, got "
            f"{tuple(deck_quat_wxyz.shape)} and {tuple(relative_velocity_deck.shape)}"
        )
    _require_finite("deck_quat_wxyz", deck_quat_wxyz)
    _require_finite("relative_velocity_deck", relative_velocity_deck)
    return quat_apply(deck_quat_wxyz, relative_velocity_deck)


def world_to_ned_velocity(world_velocity_enu: torch.Tensor) -> torch.Tensor:
    """Convert ENU velocity ``[East, North, Up]`` to PX4 local NED ``[North, East, Down]``."""
    _require_vector3("world_velocity_enu", world_velocity_enu)
    _require_finite("world_velocity_enu", world_velocity_enu)
    return torch.stack(
        (world_velocity_enu[..., 1], world_velocity_enu[..., 0], -world_velocity_enu[..., 2]),
        dim=-1,
    )


def ned_to_world_velocity(ned_velocity: torch.Tensor) -> torch.Tensor:
    """Convert PX4 local NED velocity ``[North, East, Down]`` to ENU ``[East, North, Up]``."""
    _require_vector3("ned_velocity", ned_velocity)
    _require_finite("ned_velocity", ned_velocity)
    return torch.stack((ned_velocity[..., 1], ned_velocity[..., 0], -ned_velocity[..., 2]), dim=-1)


def build_velocity_reference(
    normalized_action: torch.Tensor,
    deck_position_w: torch.Tensor,
    deck_linear_velocity_w: torch.Tensor,
    deck_angular_velocity_w: torch.Tensor,
    contact_point_w: torch.Tensor,
    deck_quat_wxyz: torch.Tensor,
    config: Px4ReferenceAdapterConfig = Px4ReferenceAdapterConfig(),
    previous_relative_velocity: torch.Tensor | None = None,
    policy_dt: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deployable velocity references from a normalized deck-relative policy action.

    Returns ``(relative_velocity_deck, contact_velocity_world, velocity_ref_world, velocity_ref_ned)``.
    If a previous relative reference is supplied, the policy component is slew limited before the
    exact rigid-body contact-point feedforward is added.
    """
    relative_velocity_deck = normalized_action_to_relative_velocity(normalized_action, config)
    if previous_relative_velocity is not None:
        if policy_dt is None:
            raise ValueError("policy_dt is required when previous_relative_velocity is provided")
        relative_velocity_deck = limit_relative_velocity_slew(
            previous_relative_velocity,
            relative_velocity_deck,
            policy_dt,
            config.max_relative_reference_acceleration,
        )

    contact_velocity_world = deck_contact_point_velocity(
        deck_position_w,
        deck_linear_velocity_w,
        deck_angular_velocity_w,
        contact_point_w,
    )
    relative_velocity_world = deck_relative_to_world_velocity(deck_quat_wxyz, relative_velocity_deck)
    if contact_velocity_world.shape != relative_velocity_world.shape:
        raise ValueError(
            "contact velocity and relative velocity shapes must match, got "
            f"{tuple(contact_velocity_world.shape)} and {tuple(relative_velocity_world.shape)}"
        )
    velocity_ref_world = contact_velocity_world + relative_velocity_world
    velocity_ref_ned = world_to_ned_velocity(velocity_ref_world)
    return relative_velocity_deck, contact_velocity_world, velocity_ref_world, velocity_ref_ned

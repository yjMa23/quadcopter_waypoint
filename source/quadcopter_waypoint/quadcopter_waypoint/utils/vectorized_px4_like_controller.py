"""Minimal GPU-vectorized PX4-like controller for hierarchical RL training.

This is a control-structure surrogate, not a source-level PX4 reimplementation. It preserves the
velocity -> acceleration -> attitude -> body-rate -> torque hierarchy while remaining pure PyTorch
for large Isaac Lab batches. World coordinates follow the project's ENU contract and body coordinates
follow the Isaac/FLU convention used by the existing quadrotor tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import quat_apply


@dataclass(frozen=True)
class VectorizedPx4LikeControllerConfig:
    """Configuration for the minimal training-only PX4-like control surrogate."""

    velocity_gain: tuple[float, float, float] = (2.0, 2.0, 2.5)
    velocity_integral_gain: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_derivative_gain: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_acceleration: float = 5.0
    max_tilt_rad: float = math.radians(35.0)
    # min/max thrust are expressed as multiples of vehicle weight. max=1.9 matches the frozen task's
    # existing thrust-to-weight authority without hard-coding a Crazyflie mass.
    min_thrust: float = 0.0
    max_thrust: float = 1.9
    attitude_gain: tuple[float, float, float] = (6.0, 6.0, 4.0)
    max_body_rate: tuple[float, float, float] = (6.0, 6.0, 4.0)
    rate_gain: tuple[float, float, float] = (12.0, 12.0, 8.0)
    max_moment: tuple[float, float, float] = (0.01, 0.01, 0.01)
    yaw_ref_enu: float = 0.0

    def validate(self) -> None:
        for name in (
            "velocity_gain",
            "velocity_integral_gain",
            "velocity_derivative_gain",
            "attitude_gain",
            "max_body_rate",
            "rate_gain",
            "max_moment",
        ):
            values = getattr(self, name)
            if len(values) != 3:
                raise ValueError(f"{name} must contain exactly three values")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
        if any(value < 0.0 for value in self.velocity_gain):
            raise ValueError("velocity_gain must be non-negative")
        if any(value != 0.0 for value in self.velocity_integral_gain):
            raise ValueError("first-version controller requires velocity_integral_gain == 0 (no anti-windup model yet)")
        if any(value < 0.0 for value in self.velocity_derivative_gain):
            raise ValueError("velocity_derivative_gain must be non-negative")
        if self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be positive")
        if not 0.0 < self.max_tilt_rad < 0.5 * math.pi:
            raise ValueError("max_tilt_rad must be within (0, pi/2)")
        if self.min_thrust < 0.0 or self.max_thrust <= self.min_thrust:
            raise ValueError("thrust-to-weight limits must satisfy 0 <= min_thrust < max_thrust")
        if any(value <= 0.0 for value in self.max_body_rate):
            raise ValueError("max_body_rate must be positive")
        if any(value < 0.0 for value in self.rate_gain):
            raise ValueError("rate_gain must be non-negative")
        if any(value <= 0.0 for value in self.max_moment):
            raise ValueError("max_moment must be positive")
        if not math.isfinite(self.yaw_ref_enu):
            raise ValueError("yaw_ref_enu must be finite")


def _require_vector3(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim < 1 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3], got {tuple(tensor.shape)}")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} contains NaN or Inf")


def _require_quaternion(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim < 1 or tensor.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [..., 4], got {tuple(tensor.shape)}")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} contains NaN or Inf")


def _as_gain(values: tuple[float, float, float], like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor(values)


def _clamp_vector_norm(vector: torch.Tensor, maximum_norm: float) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    scale = torch.clamp(maximum_norm / norm.clamp_min(1.0e-9), max=1.0)
    return vector * scale, norm.squeeze(-1) > maximum_norm


def _body_axes_world(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    basis = torch.eye(3, device=quat_wxyz.device, dtype=quat_wxyz.dtype)
    batch_shape = quat_wxyz.shape[:-1]
    body_x = quat_apply(quat_wxyz, basis[0].expand(*batch_shape, 3))
    body_y = quat_apply(quat_wxyz, basis[1].expand(*batch_shape, 3))
    body_z = quat_apply(quat_wxyz, basis[2].expand(*batch_shape, 3))
    return body_x, body_y, body_z


def _desired_rotation_from_thrust_and_yaw(
    desired_specific_force_w: torch.Tensor,
    yaw_ref_enu: float,
    max_tilt_rad: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return desired world-from-body rotation, limited specific force, and tilt saturation mask."""
    force = desired_specific_force_w.clone()
    horizontal = force[..., :2]
    horizontal_norm = torch.linalg.norm(horizontal, dim=-1, keepdim=True)
    vertical = force[..., 2:3].clamp_min(1.0e-6)
    max_horizontal = vertical * math.tan(max_tilt_rad)
    horizontal_scale = torch.clamp(max_horizontal / horizontal_norm.clamp_min(1.0e-9), max=1.0)
    tilt_saturated = (horizontal_norm > max_horizontal).squeeze(-1)
    force = torch.cat((horizontal * horizontal_scale, vertical), dim=-1)

    body_z_des = force / torch.linalg.norm(force, dim=-1, keepdim=True).clamp_min(1.0e-9)
    heading = force.new_tensor([math.cos(yaw_ref_enu), math.sin(yaw_ref_enu), 0.0]).expand_as(force)
    body_y_des = torch.cross(body_z_des, heading, dim=-1)
    body_y_norm = torch.linalg.norm(body_y_des, dim=-1, keepdim=True)
    fallback_heading = force.new_tensor([0.0, 1.0, 0.0]).expand_as(force)
    fallback_body_y = torch.cross(body_z_des, fallback_heading, dim=-1)
    body_y_des = torch.where(body_y_norm > 1.0e-6, body_y_des, fallback_body_y)
    body_y_des = body_y_des / torch.linalg.norm(body_y_des, dim=-1, keepdim=True).clamp_min(1.0e-9)
    body_x_des = torch.cross(body_y_des, body_z_des, dim=-1)
    rotation_wb_des = torch.stack((body_x_des, body_y_des, body_z_des), dim=-1)
    return rotation_wb_des, force, tilt_saturated


def _rotation_matrix_from_quaternion(quat_wxyz: torch.Tensor) -> torch.Tensor:
    body_x, body_y, body_z = _body_axes_world(quat_wxyz)
    return torch.stack((body_x, body_y, body_z), dim=-1)


def _so3_attitude_error(current_rotation_wb: torch.Tensor, desired_rotation_wb: torch.Tensor) -> torch.Tensor:
    error_matrix = (
        desired_rotation_wb.transpose(-1, -2) @ current_rotation_wb
        - current_rotation_wb.transpose(-1, -2) @ desired_rotation_wb
    )
    return 0.5 * torch.stack(
        (error_matrix[..., 2, 1], error_matrix[..., 0, 2], error_matrix[..., 1, 0]),
        dim=-1,
    )


def _apply_inertia(inertia_b: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    if inertia_b.shape == vector_b.shape:
        return inertia_b * vector_b
    if inertia_b.ndim == vector_b.ndim + 1 and inertia_b.shape[-2:] == (3, 3):
        if inertia_b.shape[:-2] != vector_b.shape[:-1]:
            raise ValueError(
                f"inertia batch shape {tuple(inertia_b.shape)} does not match vector shape {tuple(vector_b.shape)}"
            )
        return (inertia_b @ vector_b.unsqueeze(-1)).squeeze(-1)
    raise ValueError(
        "inertia_b must be diagonal [..., 3] or full [..., 3, 3], got "
        f"{tuple(inertia_b.shape)} for vector {tuple(vector_b.shape)}"
    )


class VectorizedPx4LikeController:
    """Training-only vectorized velocity/attitude/rate controller surrogate."""

    def __init__(self, config: VectorizedPx4LikeControllerConfig = VectorizedPx4LikeControllerConfig()):
        config.validate()
        self.config = config

    def compute(
        self,
        velocity_reference_w: torch.Tensor,
        current_velocity_w: torch.Tensor,
        current_quat_wxyz: torch.Tensor,
        current_angular_velocity_b: torch.Tensor,
        mass: torch.Tensor | float,
        inertia_b: torch.Tensor,
        gravity_magnitude: torch.Tensor | float,
        current_acceleration_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute body-FLU thrust and moment for a batch of vehicles.

        ``min_thrust`` and ``max_thrust`` in the config are weight-normalized limits. The returned
        thrust is a 3-D body vector with only +z populated so it can be passed to the existing
        rigid-body wrench path without changing the frozen Direct task.
        """
        for name, tensor in (
            ("velocity_reference_w", velocity_reference_w),
            ("current_velocity_w", current_velocity_w),
            ("current_angular_velocity_b", current_angular_velocity_b),
        ):
            _require_vector3(name, tensor)
        _require_quaternion("current_quat_wxyz", current_quat_wxyz)
        if velocity_reference_w.shape != current_velocity_w.shape:
            raise ValueError("velocity reference and current velocity must have identical shapes")
        if current_angular_velocity_b.shape != current_velocity_w.shape:
            raise ValueError("angular velocity and linear velocity batch shapes must match")
        if current_quat_wxyz.shape[:-1] != current_velocity_w.shape[:-1]:
            raise ValueError("quaternion and velocity batch shapes must match")

        velocity_error = velocity_reference_w - current_velocity_w
        velocity_gain = _as_gain(self.config.velocity_gain, current_velocity_w)
        acceleration_command = velocity_gain * velocity_error

        derivative_gain = _as_gain(self.config.velocity_derivative_gain, current_velocity_w)
        if bool(torch.any(derivative_gain != 0.0)):
            if current_acceleration_w is None:
                raise ValueError("current_acceleration_w is required when velocity_derivative_gain is non-zero")
            _require_vector3("current_acceleration_w", current_acceleration_w)
            if current_acceleration_w.shape != current_velocity_w.shape:
                raise ValueError("current_acceleration_w must match velocity batch shape")
            acceleration_command = acceleration_command - derivative_gain * current_acceleration_w

        acceleration_command, acceleration_saturated = _clamp_vector_norm(
            acceleration_command, self.config.max_acceleration
        )

        if isinstance(gravity_magnitude, torch.Tensor):
            if not bool(torch.all(torch.isfinite(gravity_magnitude))) or bool(torch.any(gravity_magnitude <= 0.0)):
                raise ValueError("gravity_magnitude tensor must contain positive finite values")
            gravity = gravity_magnitude.to(device=current_velocity_w.device, dtype=current_velocity_w.dtype)
        else:
            if not math.isfinite(gravity_magnitude) or gravity_magnitude <= 0.0:
                raise ValueError("gravity_magnitude must be positive and finite")
            gravity = current_velocity_w.new_tensor(gravity_magnitude)
        while gravity.ndim < current_velocity_w.ndim - 1:
            gravity = gravity.unsqueeze(-1)
        desired_specific_force = acceleration_command.clone()
        desired_specific_force[..., 2] += gravity

        desired_rotation_wb, limited_specific_force, tilt_saturated = _desired_rotation_from_thrust_and_yaw(
            desired_specific_force,
            self.config.yaw_ref_enu,
            self.config.max_tilt_rad,
        )

        if isinstance(mass, torch.Tensor):
            if not bool(torch.all(torch.isfinite(mass))) or bool(torch.any(mass <= 0.0)):
                raise ValueError("mass tensor must contain positive finite values")
            mass_tensor = mass.to(device=current_velocity_w.device, dtype=current_velocity_w.dtype)
        else:
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError("mass must be positive and finite")
            mass_tensor = current_velocity_w.new_tensor(mass)
        while mass_tensor.ndim < current_velocity_w.ndim - 1:
            mass_tensor = mass_tensor.unsqueeze(-1)

        requested_thrust = mass_tensor * torch.linalg.norm(limited_specific_force, dim=-1)
        weight = mass_tensor * gravity
        min_thrust_force = self.config.min_thrust * weight
        max_thrust_force = self.config.max_thrust * weight
        thrust_force = torch.clamp(requested_thrust, min=min_thrust_force, max=max_thrust_force)
        thrust_saturated = (requested_thrust < min_thrust_force) | (requested_thrust > max_thrust_force)
        thrust_b = torch.zeros_like(current_velocity_w)
        thrust_b[..., 2] = thrust_force

        current_rotation_wb = _rotation_matrix_from_quaternion(current_quat_wxyz)
        attitude_error = _so3_attitude_error(current_rotation_wb, desired_rotation_wb)
        attitude_gain = _as_gain(self.config.attitude_gain, current_velocity_w)
        body_rate_reference = -attitude_gain * attitude_error
        max_body_rate = _as_gain(self.config.max_body_rate, current_velocity_w)
        body_rate_saturated = torch.any(torch.abs(body_rate_reference) > max_body_rate, dim=-1)
        body_rate_reference = torch.clamp(body_rate_reference, min=-max_body_rate, max=max_body_rate)

        rate_gain = _as_gain(self.config.rate_gain, current_velocity_w)
        angular_acceleration_command = rate_gain * (body_rate_reference - current_angular_velocity_b)
        inertia_times_alpha = _apply_inertia(inertia_b, angular_acceleration_command)
        inertia_times_omega = _apply_inertia(inertia_b, current_angular_velocity_b)
        gyroscopic_moment = torch.cross(current_angular_velocity_b, inertia_times_omega, dim=-1)
        requested_moment = inertia_times_alpha + gyroscopic_moment
        max_moment = _as_gain(self.config.max_moment, current_velocity_w)
        moment_saturated = torch.any(torch.abs(requested_moment) > max_moment, dim=-1)
        moment_b = torch.clamp(requested_moment, min=-max_moment, max=max_moment)

        if not bool(torch.all(torch.isfinite(thrust_b))) or not bool(torch.all(torch.isfinite(moment_b))):
            raise ValueError("controller produced NaN or Inf")

        body_z_des = desired_rotation_wb[..., :, 2]
        tilt_rad = torch.acos(body_z_des[..., 2].clamp(-1.0, 1.0))
        diagnostics = {
            "velocity_error_w": velocity_error,
            "acceleration_command_w": acceleration_command,
            "desired_specific_force_w": limited_specific_force,
            "desired_tilt_rad": tilt_rad,
            "attitude_error_b": attitude_error,
            "body_rate_reference_b": body_rate_reference,
            "requested_thrust": requested_thrust,
            "requested_moment_b": requested_moment,
            "acceleration_saturated": acceleration_saturated,
            "tilt_saturated": tilt_saturated,
            "thrust_saturated": thrust_saturated,
            "body_rate_saturated": body_rate_saturated,
            "moment_saturated": moment_saturated,
        }
        return thrust_b, moment_b, diagnostics

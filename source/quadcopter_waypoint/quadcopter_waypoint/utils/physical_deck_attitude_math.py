"""Geometry and motion helpers for the Phase-6C physical attitude deck.

The module intentionally depends only on PyTorch so quaternion, frame, contact-point velocity, and
height-safety logic can be validated without launching Isaac Sim. Quaternions use Isaac Lab's
``(w, x, y, z)`` convention throughout.
"""

from __future__ import annotations

import math

import torch


def quat_normalize(quat_wxyz: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Return unit quaternions while rejecting zero-length inputs through clamping."""
    return quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the quaternion conjugate in ``(w, x, y, z)`` order."""
    return torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1)


def quat_multiply(lhs_wxyz: torch.Tensor, rhs_wxyz: torch.Tensor) -> torch.Tensor:
    """Hamilton product for matching batches of ``(w, x, y, z)`` quaternions."""
    if lhs_wxyz.shape != rhs_wxyz.shape:
        raise ValueError(f"Quaternion shape mismatch: {lhs_wxyz.shape} != {rhs_wxyz.shape}")
    w1, x1, y1, z1 = lhs_wxyz.unbind(dim=-1)
    w2, x2, y2, z2 = rhs_wxyz.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from the local frame into the parent frame."""
    quat_wxyz = quat_normalize(quat_wxyz)
    xyz = quat_wxyz[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat_wxyz[..., :1] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def quat_apply_inverse(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from the parent frame into the quaternion's local frame."""
    return quat_apply(quat_conjugate(quat_normalize(quat_wxyz)), vector)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert XYZ roll/pitch/yaw angles to Isaac Lab ``(w, x, y, z)`` quaternions."""
    cy = torch.cos(0.5 * yaw)
    sy = torch.sin(0.5 * yaw)
    cr = torch.cos(0.5 * roll)
    sr = torch.sin(0.5 * roll)
    cp = torch.cos(0.5 * pitch)
    sp = torch.sin(0.5 * pitch)
    return quat_normalize(
        torch.stack(
            (
                cy * cr * cp + sy * sr * sp,
                cy * sr * cp - sy * cr * sp,
                cy * cr * sp + sy * sr * cp,
                sy * cr * cp - cy * sr * sp,
            ),
            dim=-1,
        )
    )


def world_angular_velocity_from_xyz_rates(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    roll_rate: torch.Tensor,
    pitch_rate: torch.Tensor,
    yaw_rate: torch.Tensor,
) -> torch.Tensor:
    """Map XYZ Euler rates to angular velocity expressed in the world frame.

    For ``R = Rz(yaw) Ry(pitch) Rx(roll)``, the world-frame angular velocity is the sum of each
    instantaneous rotation axis expressed in world coordinates. In Phase 6C ``yaw == yaw_rate == 0``,
    which reduces to ``[roll_rate*cos(pitch), pitch_rate, -roll_rate*sin(pitch)]``. Keeping the full
    transform prevents the common but incorrect direct assignment ``omega=[roll_rate,pitch_rate,yaw_rate]``.
    """
    del roll  # The world-frame mapping for this XYZ convention does not depend on roll itself.
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    omega_x = roll_rate * cos_yaw * cos_pitch - pitch_rate * sin_yaw
    omega_y = roll_rate * sin_yaw * cos_pitch + pitch_rate * cos_yaw
    omega_z = yaw_rate - roll_rate * sin_pitch
    return torch.stack((omega_x, omega_y, omega_z), dim=-1)


def axis_angle_from_quat(quat_wxyz: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Convert shortest-path unit quaternions to axis-angle vectors."""
    quat_wxyz = quat_normalize(quat_wxyz)
    # q and -q encode the same rotation. Positive scalar part selects the shortest log-map branch.
    quat_wxyz = torch.where(quat_wxyz[..., :1] < 0.0, -quat_wxyz, quat_wxyz)
    vector = quat_wxyz[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quat_wxyz[..., :1].clamp_min(eps))
    scale = torch.where(sin_half > eps, angle / sin_half, torch.full_like(sin_half, 2.0))
    return vector * scale


def world_angular_velocity_from_quat_delta(
    quat_now_wxyz: torch.Tensor, quat_next_wxyz: torch.Tensor, dt: float
) -> torch.Tensor:
    """Estimate average world angular velocity using ``q_next * inverse(q_now)`` and a log map."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    quat_now_wxyz = quat_normalize(quat_now_wxyz)
    quat_next_wxyz = quat_normalize(quat_next_wxyz)
    quat_delta = quat_multiply(quat_next_wxyz, quat_conjugate(quat_now_wxyz))
    return axis_angle_from_quat(quat_delta) / dt


def world_to_local_position(
    frame_pos_w: torch.Tensor, frame_quat_wxyz: torch.Tensor, point_pos_w: torch.Tensor
) -> torch.Tensor:
    """Express world points in a moving local frame."""
    return quat_apply_inverse(frame_quat_wxyz, point_pos_w - frame_pos_w)


def local_to_world_position(
    frame_pos_w: torch.Tensor, frame_quat_wxyz: torch.Tensor, point_pos_local: torch.Tensor
) -> torch.Tensor:
    """Express local-frame points in world coordinates."""
    return frame_pos_w + quat_apply(frame_quat_wxyz, point_pos_local)


def deck_normal_world(deck_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the deck's local +z surface normal in world coordinates."""
    local_z = torch.zeros((*deck_quat_wxyz.shape[:-1], 3), device=deck_quat_wxyz.device, dtype=deck_quat_wxyz.dtype)
    local_z[..., 2] = 1.0
    return quat_apply(deck_quat_wxyz, local_z)


def rigid_surface_point_velocity(
    center_pos_w: torch.Tensor,
    center_lin_vel_w: torch.Tensor,
    angular_vel_w: torch.Tensor,
    surface_point_w: torch.Tensor,
) -> torch.Tensor:
    """Compute rigid-body point velocity ``v_center + omega x (point-center)`` in world coordinates."""
    lever_arm_w = surface_point_w - center_pos_w
    return center_lin_vel_w + torch.cross(angular_vel_w, lever_arm_w, dim=-1)


def decompose_relative_velocity(
    robot_velocity_w: torch.Tensor,
    surface_velocity_w: torch.Tensor,
    surface_normal_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relative velocity, signed normal component, and tangential speed.

    Negative normal velocity means the robot is moving into the deck surface.
    """
    normal = surface_normal_w / torch.linalg.norm(surface_normal_w, dim=-1, keepdim=True).clamp_min(1.0e-9)
    relative = robot_velocity_w - surface_velocity_w
    normal_speed = torch.sum(relative * normal, dim=-1)
    tangent = relative - normal_speed.unsqueeze(-1) * normal
    tangent_speed = torch.linalg.norm(tangent, dim=-1)
    return relative, normal_speed, tangent_speed


def signed_deck_surface_clearance(
    deck_pos_w: torch.Tensor,
    deck_quat_wxyz: torch.Tensor,
    deck_half_thickness: float,
    robot_bottom_point_w: torch.Tensor,
) -> torch.Tensor:
    """Signed distance from a robot bottom point to the infinite deck top plane.

    Positive values are above the top surface; negative values indicate penetration.
    """
    point_deck = world_to_local_position(deck_pos_w, deck_quat_wxyz, robot_bottom_point_w)
    return point_deck[..., 2] - deck_half_thickness


def deck_xy_error(
    deck_pos_w: torch.Tensor, deck_quat_wxyz: torch.Tensor, point_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the point in deck coordinates and its planar distance from deck center."""
    point_deck = world_to_local_position(deck_pos_w, deck_quat_wxyz, point_w)
    return point_deck, torch.linalg.norm(point_deck[..., :2], dim=-1)


def body_deck_normal_angle(robot_quat_wxyz: torch.Tensor, deck_normal_w: torch.Tensor) -> torch.Tensor:
    """Return the angle between robot body +z and the deck surface normal."""
    body_z_w = deck_normal_world(robot_quat_wxyz)
    cosine = torch.sum(body_z_w * deck_normal_w, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def conservative_minimum_deck_bottom_height(
    base_height: float,
    maximum_heave_amplitude: float,
    deck_half_length: float,
    deck_half_width: float,
    deck_half_thickness: float,
    maximum_roll_rad: float,
    maximum_pitch_rad: float,
) -> float:
    """Conservatively bound the lowest deck bottom-corner world-z height.

    With yaw fixed at zero and roll/pitch below 90 degrees, the vertical contribution of a bottom
    corner is bounded by ``hx*sin(|pitch|) + hy*sin(|roll|) + hz``. Omitting cosine factors is slightly
    conservative and guarantees the returned height is no greater than the true minimum.
    """
    for name, value in (
        ("maximum_heave_amplitude", maximum_heave_amplitude),
        ("deck_half_length", deck_half_length),
        ("deck_half_width", deck_half_width),
        ("deck_half_thickness", deck_half_thickness),
        ("maximum_roll_rad", maximum_roll_rad),
        ("maximum_pitch_rad", maximum_pitch_rad),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if maximum_roll_rad >= 0.5 * math.pi or maximum_pitch_rad >= 0.5 * math.pi:
        raise ValueError("The conservative Phase-6C height bound requires roll/pitch below 90 degrees.")
    vertical_extent = (
        deck_half_length * math.sin(maximum_pitch_rad)
        + deck_half_width * math.sin(maximum_roll_rad)
        + deck_half_thickness
    )
    return base_height - maximum_heave_amplitude - vertical_extent

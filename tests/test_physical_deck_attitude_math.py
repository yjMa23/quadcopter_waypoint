import math

import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    body_deck_normal_angle,
    conservative_minimum_deck_bottom_height,
    deck_normal_world,
    deck_xy_error,
    decompose_relative_velocity,
    local_to_world_position,
    quat_apply,
    quat_from_euler_xyz,
    rigid_surface_point_velocity,
    signed_deck_surface_clearance,
    world_angular_velocity_from_quat_delta,
    world_angular_velocity_from_xyz_rates,
    world_to_local_position,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def test_world_deck_frame_position_round_trip():
    deck_pos = _tensor([[1.2, -0.4, 0.8]])
    quat = quat_from_euler_xyz(_tensor([0.17]), _tensor([-0.11]), _tensor([0.0]))
    local = _tensor([[0.13, -0.08, 0.04]])
    world = local_to_world_position(deck_pos, quat, local)
    recovered = world_to_local_position(deck_pos, quat, world)
    torch.testing.assert_close(recovered, local, atol=1.0e-10, rtol=1.0e-10)


def test_deck_normal_and_body_alignment_angle():
    roll = _tensor([math.radians(5.0)])
    pitch = _tensor([math.radians(-3.0)])
    quat = quat_from_euler_xyz(roll, pitch, _tensor([0.0]))
    normal = deck_normal_world(quat)
    torch.testing.assert_close(torch.linalg.norm(normal, dim=-1), _tensor([1.0]), atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(body_deck_normal_angle(quat, normal), _tensor([0.0]), atol=1.0e-8, rtol=0.0)


def test_xyz_rate_mapping_matches_quaternion_log_map():
    roll = _tensor([0.08, -0.10])
    pitch = _tensor([-0.06, 0.11])
    yaw = _tensor([0.0, 0.0])
    roll_rate = _tensor([0.21, -0.17])
    pitch_rate = _tensor([-0.12, 0.09])
    yaw_rate = _tensor([0.0, 0.0])
    dt = 1.0e-6
    quat_now = quat_from_euler_xyz(roll, pitch, yaw)
    quat_next = quat_from_euler_xyz(
        roll + roll_rate * dt,
        pitch + pitch_rate * dt,
        yaw + yaw_rate * dt,
    )
    analytic = world_angular_velocity_from_xyz_rates(
        roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate
    )
    finite_delta = world_angular_velocity_from_quat_delta(quat_now, quat_next, dt)
    torch.testing.assert_close(finite_delta, analytic, atol=3.0e-7, rtol=3.0e-6)


def test_contact_point_velocity_uses_omega_cross_r():
    center = _tensor([[1.0, 2.0, 3.0]])
    center_velocity = _tensor([[0.2, -0.1, 0.05]])
    omega = _tensor([[0.0, 0.0, 2.0]])
    point = _tensor([[1.5, 2.0, 3.0]])
    velocity = rigid_surface_point_velocity(center, center_velocity, omega, point)
    torch.testing.assert_close(velocity, _tensor([[0.2, 0.9, 0.05]]))


def test_normal_and_tangential_relative_velocity_decomposition():
    robot_velocity = _tensor([[0.3, -0.2, -0.4]])
    surface_velocity = _tensor([[0.1, -0.1, 0.05]])
    normal = _tensor([[0.0, 0.0, 1.0]])
    relative, normal_speed, tangential_speed = decompose_relative_velocity(
        robot_velocity, surface_velocity, normal
    )
    torch.testing.assert_close(relative, _tensor([[0.2, -0.1, -0.45]]))
    torch.testing.assert_close(normal_speed, _tensor([-0.45]))
    torch.testing.assert_close(tangential_speed, _tensor([math.sqrt(0.05)]))


def test_inclined_plane_signed_clearance_and_deck_xy():
    deck_pos = _tensor([[0.4, -0.2, 0.6]])
    quat = quat_from_euler_xyz(_tensor([0.12]), _tensor([-0.09]), _tensor([0.0]))
    local_bottom = _tensor([[0.11, -0.07, 0.02 + 0.031]])
    bottom_world = local_to_world_position(deck_pos, quat, local_bottom)
    clearance = signed_deck_surface_clearance(deck_pos, quat, 0.02, bottom_world)
    point_deck, xy_error = deck_xy_error(deck_pos, quat, bottom_world)
    torch.testing.assert_close(clearance, _tensor([0.031]), atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(point_deck, local_bottom, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(xy_error, _tensor([math.hypot(0.11, -0.07)]), atol=1.0e-10, rtol=1.0e-10)


def test_minimum_height_bound_is_safe_for_all_sampled_corners_and_angles():
    base_height = 0.30
    heave = 0.12
    half_length = 0.25
    half_width = 0.25
    half_thickness = 0.02
    max_roll = math.radians(8.0)
    max_pitch = math.radians(8.0)
    bound = conservative_minimum_deck_bottom_height(
        base_height,
        heave,
        half_length,
        half_width,
        half_thickness,
        max_roll,
        max_pitch,
    )
    minimum_sampled = float("inf")
    for roll in torch.linspace(-max_roll, max_roll, 41, dtype=torch.float64):
        for pitch in torch.linspace(-max_pitch, max_pitch, 41, dtype=torch.float64):
            quat = quat_from_euler_xyz(roll.reshape(1), pitch.reshape(1), _tensor([0.0]))
            for x in (-half_length, half_length):
                for y in (-half_width, half_width):
                    corner_w = quat_apply(quat, _tensor([[x, y, -half_thickness]]))
                    minimum_sampled = min(minimum_sampled, base_height - heave + float(corner_w[0, 2]))
    assert bound <= minimum_sampled + 1.0e-12
    assert bound > 0.05


def test_minimum_height_validation_rejects_unsupported_tilt():
    try:
        conservative_minimum_deck_bottom_height(0.3, 0.1, 0.25, 0.25, 0.02, math.pi / 2, 0.0)
    except ValueError as exc:
        assert "below 90 degrees" in str(exc)
    else:
        raise AssertionError("Expected unsupported 90-degree roll to fail")

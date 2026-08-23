import math

import pytest
import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    quat_from_euler_xyz,
    rigid_surface_point_velocity,
)
from quadcopter_waypoint.utils.px4_reference_adapter import (
    Px4ReferenceAdapterConfig,
    build_velocity_reference,
    deck_contact_point_velocity,
    deck_relative_to_world_velocity,
    limit_relative_velocity_slew,
    ned_to_world_velocity,
    normalized_action_to_relative_velocity,
    world_to_ned_velocity,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def _identity_quat(batch_size=1):
    quat = torch.zeros(batch_size, 4, dtype=torch.float64)
    quat[:, 0] = 1.0
    return quat


def test_zero_action_is_exact_zero_relative_velocity_even_with_asymmetric_normal_bounds():
    velocity = normalized_action_to_relative_velocity(_tensor([[0.0, 0.0, 0.0]]))
    torch.testing.assert_close(velocity, _tensor([[0.0, 0.0, 0.0]]), atol=0.0, rtol=0.0)


def test_stationary_level_deck_zero_relative_velocity_produces_zero_world_and_ned_reference():
    relative, contact, world, ned = build_velocity_reference(
        normalized_action=_tensor([[0.0, 0.0, 0.0]]),
        deck_position_w=_tensor([[0.0, 0.0, 0.3]]),
        deck_linear_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        deck_angular_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        contact_point_w=_tensor([[0.1, -0.2, 0.32]]),
        deck_quat_wxyz=_identity_quat(),
    )
    zero = _tensor([[0.0, 0.0, 0.0]])
    torch.testing.assert_close(relative, zero)
    torch.testing.assert_close(contact, zero)
    torch.testing.assert_close(world, zero)
    torch.testing.assert_close(ned, zero)


def test_constant_translating_deck_is_exact_velocity_feedforward():
    deck_velocity = _tensor([[0.35, -0.18, 0.07]])
    _, contact, world, _ = build_velocity_reference(
        _tensor([[0.0, 0.0, 0.0]]),
        _tensor([[0.0, 0.0, 0.3]]),
        deck_velocity,
        _tensor([[0.0, 0.0, 0.0]]),
        _tensor([[0.0, 0.0, 0.32]]),
        _identity_quat(),
    )
    torch.testing.assert_close(contact, deck_velocity)
    torch.testing.assert_close(world, deck_velocity)


def test_heaving_deck_zero_relative_action_tracks_heave_velocity():
    _, _, world, ned = build_velocity_reference(
        _tensor([[0.0, 0.0, 0.0]]),
        _tensor([[0.0, 0.0, 0.3]]),
        _tensor([[0.0, 0.0, 0.22]]),
        _tensor([[0.0, 0.0, 0.0]]),
        _tensor([[0.0, 0.0, 0.32]]),
        _identity_quat(),
    )
    torch.testing.assert_close(world, _tensor([[0.0, 0.0, 0.22]]))
    torch.testing.assert_close(ned, _tensor([[0.0, 0.0, -0.22]]))


def test_roll_rate_contact_velocity_matches_omega_cross_r():
    center = _tensor([[0.0, 0.0, 0.3]])
    center_velocity = _tensor([[0.0, 0.0, 0.0]])
    omega = _tensor([[2.0, 0.0, 0.0]])
    point = _tensor([[0.0, 0.4, 0.3]])
    velocity = deck_contact_point_velocity(center, center_velocity, omega, point)
    torch.testing.assert_close(velocity, _tensor([[0.0, 0.0, 0.8]]))


def test_pitch_rate_contact_velocity_matches_omega_cross_r():
    center = _tensor([[0.0, 0.0, 0.3]])
    omega = _tensor([[0.0, 3.0, 0.0]])
    point = _tensor([[0.2, 0.0, 0.3]])
    velocity = deck_contact_point_velocity(center, _tensor([[0.0, 0.0, 0.0]]), omega, point)
    torch.testing.assert_close(velocity, _tensor([[0.0, 0.0, -0.6]]))


def test_combined_translation_and_angular_velocity_reuses_physical_deck_rigid_body_math():
    center = _tensor([[1.0, -2.0, 0.4], [0.2, 0.1, 0.3]])
    linear = _tensor([[0.2, 0.3, -0.1], [-0.4, 0.1, 0.2]])
    omega = _tensor([[0.1, -0.2, 0.3], [0.0, 0.4, -0.2]])
    point = _tensor([[1.3, -1.8, 0.45], [0.0, 0.2, 0.35]])
    adapter = deck_contact_point_velocity(center, linear, omega, point)
    frozen_math = rigid_surface_point_velocity(center, linear, omega, point)
    torch.testing.assert_close(adapter, frozen_math, atol=0.0, rtol=0.0)


def test_pure_normal_descent_uses_deck_normal_not_world_z():
    quat = quat_from_euler_xyz(
        _tensor([math.radians(15.0)]),
        _tensor([math.radians(-10.0)]),
        _tensor([0.0]),
    )
    relative_deck = normalized_action_to_relative_velocity(_tensor([[0.0, 0.0, -1.0]]))
    world = deck_relative_to_world_velocity(quat, relative_deck)
    expected = deck_relative_to_world_velocity(quat, _tensor([[0.0, 0.0, -0.40]]))
    torch.testing.assert_close(relative_deck, _tensor([[0.0, 0.0, -0.40]]))
    torch.testing.assert_close(world, expected)
    assert abs(float(world[0, 0])) > 1.0e-3 or abs(float(world[0, 1])) > 1.0e-3


def test_pure_tangential_correction_rotates_with_deck_frame():
    quat = quat_from_euler_xyz(_tensor([0.0]), _tensor([0.0]), _tensor([math.pi / 2.0]))
    world = deck_relative_to_world_velocity(quat, _tensor([[0.4, 0.0, 0.0]]))
    torch.testing.assert_close(world, _tensor([[0.0, 0.4, 0.0]]), atol=1.0e-12, rtol=1.0e-12)


def test_action_saturation_and_horizontal_vector_norm_limit():
    action = _tensor([[2.0, 2.0, -3.0], [-2.0, 0.0, 3.0]])
    velocity = normalized_action_to_relative_velocity(action)
    horizontal_norm = torch.linalg.norm(velocity[:, :2], dim=-1)
    assert torch.all(horizontal_norm <= 0.80 + 1.0e-12)
    torch.testing.assert_close(velocity[0, 2], _tensor(-0.40))
    torch.testing.assert_close(velocity[1, 2], _tensor(0.30))
    torch.testing.assert_close(velocity[1, 0], _tensor(-0.80))


def test_relative_velocity_slew_limit_is_vectorized_and_per_axis():
    previous = _tensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.2]])
    target = _tensor([[1.0, -1.0, -0.4], [-1.0, 1.0, -0.4]])
    limited = limit_relative_velocity_slew(previous, target, dt=0.04, max_acceleration=2.0)
    expected = _tensor([[0.08, -0.08, -0.08], [0.02, -0.02, 0.12]])
    torch.testing.assert_close(limited, expected)


def test_enu_ned_round_trip():
    enu = _tensor([[1.2, -0.3, 0.7], [-0.5, 2.0, -1.1]])
    ned = world_to_ned_velocity(enu)
    torch.testing.assert_close(ned, _tensor([[-0.3, 1.2, -0.7], [2.0, -0.5, 1.1]]))
    torch.testing.assert_close(ned_to_world_velocity(ned), enu)


def test_deck_frame_transform_batch_matches_known_yaw_rotation():
    yaw = _tensor([0.0, math.pi / 2.0])
    quat = quat_from_euler_xyz(_tensor([0.0, 0.0]), _tensor([0.0, 0.0]), yaw)
    relative = _tensor([[0.2, -0.1, -0.05], [0.2, 0.0, 0.0]])
    world = deck_relative_to_world_velocity(quat, relative)
    torch.testing.assert_close(world[0], relative[0])
    torch.testing.assert_close(world[1], _tensor([0.0, 0.2, 0.0]), atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize(
    "bad_action",
    [
        [[float("nan"), 0.0, 0.0]],
        [[float("inf"), 0.0, 0.0]],
        [[float("-inf"), 0.0, 0.0]],
    ],
)
def test_nan_inf_action_rejected(bad_action):
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalized_action_to_relative_velocity(_tensor(bad_action))


def test_build_reference_applies_policy_slew_before_exact_deck_feedforward():
    config = Px4ReferenceAdapterConfig(max_relative_reference_acceleration=2.0)
    relative, contact, world, _ = build_velocity_reference(
        normalized_action=_tensor([[1.0, 0.0, 0.0]]),
        deck_position_w=_tensor([[0.0, 0.0, 0.3]]),
        deck_linear_velocity_w=_tensor([[0.4, 0.0, 0.0]]),
        deck_angular_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        contact_point_w=_tensor([[0.0, 0.0, 0.32]]),
        deck_quat_wxyz=_identity_quat(),
        config=config,
        previous_relative_velocity=_tensor([[0.0, 0.0, 0.0]]),
        policy_dt=0.04,
    )
    torch.testing.assert_close(relative, _tensor([[0.08, 0.0, 0.0]]))
    torch.testing.assert_close(contact, _tensor([[0.4, 0.0, 0.0]]))
    torch.testing.assert_close(world, _tensor([[0.48, 0.0, 0.0]]))

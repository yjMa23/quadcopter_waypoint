import math

import pytest
import torch

from quadcopter_waypoint.utils.continuous_landing_stage import (
    ContinuousLandingGuidanceConfig,
    deck_heading_world,
    filter_landing_stage,
    limit_attitude_reference_rate,
    limit_attitude_tilt,
    limit_stage_conditioned_reference_slew,
    map_stage_conditioned_relative_velocity,
    normalized_stage_action,
    relative_angular_velocity,
    shortest_quaternion_slerp,
    smoothstep01,
    stage_conditioned_velocity_limits,
    terminal_alignment_weight,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import (
    quat_apply,
    quat_from_axis_angle,
    quat_from_euler_xyz,
)


def _tensor(values, dtype=torch.float64, device="cpu"):
    return torch.tensor(values, dtype=dtype, device=device)


def _identity_quat(batch_size=1, dtype=torch.float64, device="cpu"):
    quat = torch.zeros(batch_size, 4, dtype=dtype, device=device)
    quat[:, 0] = 1.0
    return quat


def _assert_same_rotation(actual, expected, atol=1.0e-10):
    actual = actual / torch.linalg.norm(actual, dim=-1, keepdim=True)
    expected = expected / torch.linalg.norm(expected, dim=-1, keepdim=True)
    dot = torch.abs(torch.sum(actual * expected, dim=-1))
    torch.testing.assert_close(dot, torch.ones_like(dot), atol=atol, rtol=0.0)


def test_default_config_matches_frozen_engineering_values():
    cfg = ContinuousLandingGuidanceConfig()
    cfg.validate()
    assert cfg.stage_filter_time_constant == 0.20
    assert cfg.stage_rate_limit == 2.0
    assert cfg.tangential_speed_approach == 0.80
    assert cfg.tangential_speed_terminal == 0.25
    assert cfg.max_descent_speed == 0.30
    assert cfg.ascent_speed_approach == 0.30
    assert cfg.ascent_speed_terminal == 0.15
    assert cfg.reference_accel_approach == (2.0, 2.0, 1.5)
    assert cfg.reference_accel_terminal == (0.8, 0.8, 0.5)
    assert cfg.attitude_blend_start_clearance == 0.50
    assert cfg.attitude_blend_full_clearance == 0.12
    assert cfg.max_terminal_attitude_tilt_rad == pytest.approx(math.radians(35.0))
    assert cfg.max_terminal_reference_rate == (2.0, 2.0, 1.5)


def test_config_validation_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        ContinuousLandingGuidanceConfig(stage_filter_time_constant=0.0).validate()
    with pytest.raises(ValueError):
        ContinuousLandingGuidanceConfig(attitude_blend_start_clearance=0.1, attitude_blend_full_clearance=0.2).validate()
    with pytest.raises(ValueError):
        ContinuousLandingGuidanceConfig(max_terminal_reference_rate=(2.0, 0.0, 1.5)).validate()


def test_stage_mapping_clamp_and_smoothstep_endpoints():
    action = _tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    stage = normalized_stage_action(action)
    torch.testing.assert_close(stage, _tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    shaped = smoothstep01(_tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(shaped, _tensor([0.0, 0.5, 1.0]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_stage_mapping_rejects_nonfinite(bad):
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalized_stage_action(_tensor([bad]))


def test_stage_filter_exact_rate_limit_monotonic_and_reset_semantics():
    cfg = ContinuousLandingGuidanceConfig()
    previous = _tensor([0.0])
    raw = _tensor([1.0])
    first = filter_landing_stage(raw, previous, dt=0.04, config=cfg)
    torch.testing.assert_close(first, _tensor([0.08]), atol=1.0e-12, rtol=0.0)
    second = filter_landing_stage(raw, first, dt=0.04, config=cfg)
    assert float(second[0]) > float(first[0])
    assert float(second[0]) <= 0.16 + 1.0e-12
    reset = filter_landing_stage(_tensor([0.0]), _tensor([0.0]), dt=0.04, config=cfg)
    torch.testing.assert_close(reset, _tensor([0.0]), atol=0.0, rtol=0.0)


def test_stage_filter_rejects_invalid_dt_and_nonfinite_state():
    with pytest.raises(ValueError, match="dt"):
        filter_landing_stage(_tensor([1.0]), _tensor([0.0]), dt=0.0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        filter_landing_stage(_tensor([1.0]), _tensor([float("nan")]), dt=0.04)


def test_velocity_limits_match_approach_and_terminal_contract():
    stage = _tensor([0.0, 1.0])
    tangential, down, up = stage_conditioned_velocity_limits(stage)
    torch.testing.assert_close(tangential, _tensor([0.80, 0.25]))
    torch.testing.assert_close(down, _tensor([0.0, 0.30]))
    torch.testing.assert_close(up, _tensor([0.30, 0.15]))


def test_velocity_mapping_zero_sign_diagonal_norm_and_clamp():
    stage = _tensor([0.0, 1.0, 1.0, 0.5])
    action = _tensor([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0],
        [2.0, -2.0, 1.0],
    ])
    velocity = map_stage_conditioned_relative_velocity(action, stage)
    torch.testing.assert_close(velocity[0], _tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(velocity[1, 2], _tensor(-0.30))
    assert float(velocity[3, 2]) > 0.0
    limits, _, _ = stage_conditioned_velocity_limits(stage)
    assert torch.all(torch.linalg.norm(velocity[:, :2], dim=-1) <= limits + 1.0e-12)


def test_velocity_envelope_is_continuous_at_intermediate_stage():
    stages = _tensor([0.49, 0.50, 0.51])
    velocity = map_stage_conditioned_relative_velocity(_tensor([[1.0, 0.0, -1.0]]).expand(3, -1), stages)
    assert torch.max(torch.abs(velocity[1:] - velocity[:-1])) < 0.02


def test_stage_conditioned_slew_exact_approach_and_terminal_limits():
    previous = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = _tensor([[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]])
    stage = _tensor([0.0, 1.0])
    limited = limit_stage_conditioned_reference_slew(previous, target, stage, dt=0.04)
    torch.testing.assert_close(limited[0], _tensor([0.08, -0.08, 0.06]))
    torch.testing.assert_close(limited[1], _tensor([0.032, -0.032, 0.02]))
    assert torch.all(torch.abs(limited[1]) < torch.abs(limited[0]))


def test_terminal_alignment_weight_contract_and_penetration():
    stage = _tensor([1.0, 0.0, 1.0, 1.0, 0.5])
    clearance = _tensor([0.60, 0.0, 0.12, -0.05, 0.31])
    alpha = terminal_alignment_weight(stage, clearance)
    torch.testing.assert_close(alpha[:4], _tensor([0.0, 0.0, 1.0, 1.0]), atol=1.0e-12, rtol=0.0)
    assert 0.0 < float(alpha[4]) < 1.0
    assert torch.all((alpha >= 0.0) & (alpha <= 1.0))


def test_terminal_alignment_weight_is_continuous_at_clearance_boundaries():
    eps = 1.0e-6
    clearances = _tensor([0.50 + eps, 0.50, 0.50 - eps, 0.12 + eps, 0.12, 0.12 - eps])
    alpha = terminal_alignment_weight(torch.ones_like(clearances), clearances)
    assert torch.max(torch.abs(alpha[1:3] - alpha[:2])) < 1.0e-4
    assert torch.max(torch.abs(alpha[4:6] - alpha[3:5])) < 1.0e-4


def test_deck_heading_world_known_yaw_and_degenerate_fallback():
    quat = quat_from_euler_xyz(_tensor([0.0, 0.0]), _tensor([0.0, math.pi / 2.0]), _tensor([math.pi / 2.0, 0.0]))
    previous = _tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    heading = deck_heading_world(quat, previous)
    torch.testing.assert_close(heading[0], _tensor([0.0, 1.0, 0.0]), atol=1.0e-10, rtol=0.0)
    torch.testing.assert_close(heading[1], _tensor([0.0, 1.0, 0.0]), atol=1.0e-10, rtol=0.0)
    fallback = deck_heading_world(quat[1:2], None)
    torch.testing.assert_close(fallback[0], _tensor([1.0, 0.0, 0.0]), atol=1.0e-10, rtol=0.0)


def test_shortest_quaternion_slerp_endpoints_sign_parity_and_near_equal():
    q0 = _identity_quat(2)
    q1 = quat_from_euler_xyz(_tensor([0.1, 0.0]), _tensor([-0.2, 0.0]), _tensor([0.3, 1.0e-8]))
    alpha0 = _tensor([0.0, 0.0])
    alpha1 = _tensor([1.0, 1.0])
    _assert_same_rotation(shortest_quaternion_slerp(q0, q1, alpha0), q0)
    _assert_same_rotation(shortest_quaternion_slerp(q0, q1, alpha1), q1)
    half_a = shortest_quaternion_slerp(q0, q1, _tensor([0.5, 0.5]))
    half_b = shortest_quaternion_slerp(q0, -q1, _tensor([0.5, 0.5]))
    _assert_same_rotation(half_a, half_b)
    torch.testing.assert_close(torch.linalg.norm(half_a, dim=-1), _tensor([1.0, 1.0]), atol=1.0e-12, rtol=0.0)


def test_tilt_limit_preserves_feasible_attitude_and_clamps_excess():
    heading = _tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q = quat_from_euler_xyz(_tensor([math.radians(20.0), math.radians(50.0)]), _tensor([0.0, 0.0]), _tensor([0.0, 0.0]))
    limited, saturated = limit_attitude_tilt(q, heading, math.radians(35.0))
    _assert_same_rotation(limited[0:1], q[0:1])
    assert not bool(saturated[0])
    assert bool(saturated[1])
    body_z = quat_apply(limited, _tensor([[0.0, 0.0, 1.0]]).expand(2, -1))
    tilt = torch.acos(body_z[:, 2].clamp(-1.0, 1.0))
    assert float(tilt[1]) <= math.radians(35.0) + 1.0e-10
    torch.testing.assert_close(torch.linalg.norm(limited, dim=-1), _tensor([1.0, 1.0]), atol=1.0e-12, rtol=0.0)


def test_tilt_limit_retains_heading_and_is_finite_for_degenerate_heading():
    q = quat_from_euler_xyz(_tensor([0.0, 0.0]), _tensor([math.radians(55.0), math.radians(55.0)]), _tensor([0.0, 0.0]))
    headings = _tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    limited, saturated = limit_attitude_tilt(q, headings, math.radians(35.0))
    assert torch.all(saturated)
    assert torch.all(torch.isfinite(limited))
    body_x = quat_apply(limited[0:1], _tensor([[1.0, 0.0, 0.0]]))[0]
    body_x_horizontal = body_x.clone()
    body_x_horizontal[2] = 0.0
    body_x_horizontal = body_x_horizontal / torch.linalg.norm(body_x_horizontal)
    assert float(torch.dot(body_x_horizontal, headings[0])) > 0.99


def test_attitude_reference_rate_limit_keeps_below_limit_and_clamps_world_axes_exactly():
    previous = _identity_quat(4)
    dt = 0.04
    phi = _tensor([
        [0.04, 0.00, 0.00],
        [0.20, 0.00, 0.00],
        [0.00, -0.20, 0.00],
        [0.00, 0.00, 0.20],
    ])
    target = quat_from_axis_angle(phi)
    limited = limit_attitude_reference_rate(previous, target, dt, (2.0, 2.0, 1.5))
    _assert_same_rotation(limited[0:1], target[0:1])
    expected_phi = _tensor([
        [0.04, 0.00, 0.00],
        [0.08, 0.00, 0.00],
        [0.00, -0.08, 0.00],
        [0.00, 0.00, 0.06],
    ])
    _assert_same_rotation(limited, quat_from_axis_angle(expected_phi))


def test_attitude_reference_rate_limit_handles_quaternion_sign_and_invalid_dt():
    previous = _identity_quat(1)
    target = quat_from_axis_angle(_tensor([[0.0, 0.0, 0.03]]))
    a = limit_attitude_reference_rate(previous, target, 0.04, (2.0, 2.0, 1.5))
    b = limit_attitude_reference_rate(previous, -target, 0.04, (2.0, 2.0, 1.5))
    _assert_same_rotation(a, b)
    with pytest.raises(ValueError, match="dt"):
        limit_attitude_reference_rate(previous, target, 0.0, (2.0, 2.0, 1.5))


def test_relative_angular_velocity_subtraction_sign_and_rotated_norm():
    uav = _tensor([[1.0, -2.0, 0.5], [0.2, 0.3, -0.4]])
    deck = _tensor([[1.0, -2.0, 0.5], [-0.1, 0.5, 0.2]])
    relative = relative_angular_velocity(uav, deck)
    torch.testing.assert_close(relative[0], _tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(relative[1], _tensor([0.3, -0.2, -0.6]))
    q = quat_from_euler_xyz(_tensor([0.2]), _tensor([-0.3]), _tensor([0.4]))
    rotated = quat_apply(q, relative[1:2])
    torch.testing.assert_close(torch.linalg.norm(rotated, dim=-1), torch.linalg.norm(relative[1:2], dim=-1), atol=1.0e-12, rtol=0.0)


def test_single_vector_contract():
    action = _tensor([1.0, -1.0, -1.0])
    stage = _tensor(0.5)
    velocity = map_stage_conditioned_relative_velocity(action, stage)
    q = shortest_quaternion_slerp(_tensor([1.0, 0.0, 0.0, 0.0]), quat_from_axis_angle(_tensor([0.0, 0.1, 0.0])), stage)
    limited, saturated = limit_attitude_tilt(q, _tensor([1.0, 0.0, 0.0]), math.radians(35.0))
    assert velocity.shape == (3,)
    assert q.shape == (4,)
    assert limited.shape == (4,)
    assert saturated.ndim == 0
    assert torch.all(torch.isfinite(velocity))
    assert torch.all(torch.isfinite(limited))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_dtype_contract(dtype):
    action = _tensor([[1.0, -0.5, -1.0], [0.0, 0.2, 0.5]], dtype=dtype)
    stage = _tensor([0.25, 0.75], dtype=dtype)
    velocity = map_stage_conditioned_relative_velocity(action, stage)
    alpha = terminal_alignment_weight(stage, _tensor([0.3, 0.1], dtype=dtype))
    q = shortest_quaternion_slerp(_identity_quat(2, dtype=dtype), quat_from_axis_angle(_tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=dtype)), stage)
    assert velocity.dtype == dtype
    assert alpha.dtype == dtype
    assert q.dtype == dtype
    assert velocity.shape == (2, 3)
    assert alpha.shape == (2,)
    assert q.shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_dtype_and_cpu_numerical_parity():
    action_cpu = _tensor([[0.8, -0.6, -1.0], [0.2, 0.4, 0.5]], dtype=torch.float32)
    stage_cpu = _tensor([0.2, 0.9], dtype=torch.float32)
    velocity_cpu = map_stage_conditioned_relative_velocity(action_cpu, stage_cpu)
    velocity_cuda = map_stage_conditioned_relative_velocity(action_cpu.cuda(), stage_cpu.cuda())
    assert velocity_cuda.is_cuda
    assert velocity_cuda.dtype == torch.float32
    torch.testing.assert_close(velocity_cuda.cpu(), velocity_cpu, atol=1.0e-6, rtol=1.0e-6)

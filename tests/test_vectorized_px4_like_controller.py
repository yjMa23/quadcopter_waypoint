import math

import pytest
import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import quat_apply, quat_from_euler_xyz
from quadcopter_waypoint.utils.vectorized_px4_like_controller import (
    VectorizedPx4LikeController,
    VectorizedPx4LikeControllerConfig,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def _identity_quat(batch_size):
    quat = torch.zeros(batch_size, 4, dtype=torch.float64)
    quat[:, 0] = 1.0
    return quat


def _inertia(batch_size):
    return _tensor([[1.4e-5, 1.4e-5, 2.2e-5]]).expand(batch_size, -1).clone()


def test_hover_equilibrium_outputs_weight_and_zero_moment():
    controller = VectorizedPx4LikeController()
    thrust, moment, diagnostics = controller.compute(
        velocity_reference_w=_tensor([[0.0, 0.0, 0.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        current_quat_wxyz=_identity_quat(1),
        current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    torch.testing.assert_close(thrust, _tensor([[0.0, 0.0, 0.03 * 9.81]]), atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(moment, _tensor([[0.0, 0.0, 0.0]]), atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(diagnostics["desired_tilt_rad"], _tensor([0.0]), atol=1.0e-12, rtol=0.0)


def test_horizontal_velocity_error_commands_tilt_toward_correction():
    controller = VectorizedPx4LikeController()
    _, moment, diagnostics = controller.compute(
        velocity_reference_w=_tensor([[1.0, 0.0, 0.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        current_quat_wxyz=_identity_quat(1),
        current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    assert float(diagnostics["desired_tilt_rad"][0]) > 0.0
    assert float(diagnostics["body_rate_reference_b"][0, 1]) > 0.0
    assert float(moment[0, 1]) > 0.0


def test_vertical_climb_and_descent_change_collective_thrust_about_hover():
    controller = VectorizedPx4LikeController()
    reference = _tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    thrust, _, _ = controller.compute(
        velocity_reference_w=reference,
        current_velocity_w=torch.zeros_like(reference),
        current_quat_wxyz=_identity_quat(3),
        current_angular_velocity_b=torch.zeros_like(reference),
        mass=0.03,
        inertia_b=_inertia(3),
        gravity_magnitude=9.81,
    )
    assert float(thrust[0, 2]) > float(thrust[2, 2]) > float(thrust[1, 2])


def test_max_acceleration_is_vector_norm_clamped():
    config = VectorizedPx4LikeControllerConfig(max_acceleration=1.5)
    controller = VectorizedPx4LikeController(config)
    _, _, diagnostics = controller.compute(
        velocity_reference_w=_tensor([[20.0, -20.0, 10.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        current_quat_wxyz=_identity_quat(1),
        current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    torch.testing.assert_close(
        torch.linalg.norm(diagnostics["acceleration_command_w"], dim=-1),
        _tensor([1.5]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert bool(diagnostics["acceleration_saturated"][0])


def test_max_tilt_clamps_desired_body_z():
    max_tilt = math.radians(10.0)
    config = VectorizedPx4LikeControllerConfig(max_acceleration=20.0, max_tilt_rad=max_tilt)
    controller = VectorizedPx4LikeController(config)
    _, _, diagnostics = controller.compute(
        velocity_reference_w=_tensor([[20.0, 0.0, 0.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        current_quat_wxyz=_identity_quat(1),
        current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    torch.testing.assert_close(diagnostics["desired_tilt_rad"], _tensor([max_tilt]), atol=1.0e-12, rtol=1.0e-12)
    assert bool(diagnostics["tilt_saturated"][0])


def test_thrust_is_clamped_as_weight_multiple():
    config = VectorizedPx4LikeControllerConfig(max_thrust=1.2)
    controller = VectorizedPx4LikeController(config)
    thrust, _, diagnostics = controller.compute(
        velocity_reference_w=_tensor([[0.0, 0.0, 20.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        current_quat_wxyz=_identity_quat(1),
        current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    torch.testing.assert_close(thrust[0, 2], _tensor(1.2 * 0.03 * 9.81), atol=1.0e-12, rtol=1.0e-12)
    assert bool(diagnostics["thrust_saturated"][0])


def test_body_rate_and_moment_limits_are_applied():
    config = VectorizedPx4LikeControllerConfig(
        attitude_gain=(100.0, 100.0, 100.0),
        max_body_rate=(0.2, 0.2, 0.2),
        rate_gain=(1000.0, 1000.0, 1000.0),
        max_moment=(1.0e-5, 1.0e-5, 1.0e-5),
    )
    controller = VectorizedPx4LikeController(config)
    quat = quat_from_euler_xyz(_tensor([0.0]), _tensor([0.0]), _tensor([math.pi / 2.0]))
    _, moment, diagnostics = controller.compute(
        velocity_reference_w=_tensor([[0.0, 0.0, 0.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        current_quat_wxyz=quat,
        current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    assert torch.all(torch.abs(diagnostics["body_rate_reference_b"]) <= 0.2 + 1.0e-12)
    assert bool(diagnostics["body_rate_saturated"][0])
    assert torch.all(torch.abs(moment) <= 1.0e-5 + 1.0e-12)
    assert bool(diagnostics["moment_saturated"][0])


def test_full_inertia_matrix_matches_diagonal_representation():
    controller = VectorizedPx4LikeController()
    diagonal = _inertia(2)
    full = torch.diag_embed(diagonal)
    inputs = dict(
        velocity_reference_w=_tensor([[0.3, 0.1, 0.0], [-0.2, 0.4, 0.1]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]]),
        current_quat_wxyz=_identity_quat(2),
        current_angular_velocity_b=_tensor([[0.1, -0.2, 0.05], [-0.1, 0.15, -0.08]]),
        mass=_tensor([0.03, 0.032]),
        gravity_magnitude=9.81,
    )
    thrust_diag, moment_diag, _ = controller.compute(inertia_b=diagonal, **inputs)
    thrust_full, moment_full, _ = controller.compute(inertia_b=full, **inputs)
    torch.testing.assert_close(thrust_full, thrust_diag)
    torch.testing.assert_close(moment_full, moment_diag)


def test_large_batch_is_vectorized_and_finite():
    batch = 4096
    controller = VectorizedPx4LikeController()
    reference = torch.zeros(batch, 3, dtype=torch.float64)
    reference[:, 0] = torch.linspace(-0.8, 0.8, batch, dtype=torch.float64)
    thrust, moment, diagnostics = controller.compute(
        velocity_reference_w=reference,
        current_velocity_w=torch.zeros_like(reference),
        current_quat_wxyz=_identity_quat(batch),
        current_angular_velocity_b=torch.zeros_like(reference),
        mass=0.03,
        inertia_b=_inertia(batch),
        gravity_magnitude=9.81,
    )
    assert thrust.shape == (batch, 3)
    assert moment.shape == (batch, 3)
    assert diagnostics["body_rate_reference_b"].shape == (batch, 3)
    assert torch.all(torch.isfinite(thrust))
    assert torch.all(torch.isfinite(moment))


def test_exposed_velocity_attitude_reference_matches_default_compute_path():
    controller = VectorizedPx4LikeController()
    reference = _tensor([[0.35, -0.20, 0.15], [-0.25, 0.10, -0.05]])
    current_velocity = _tensor([[0.05, -0.02, 0.01], [0.02, 0.03, -0.01]])
    current_quat = _identity_quat(2)
    current_rate = _tensor([[0.1, -0.05, 0.02], [-0.08, 0.04, -0.01]])
    inputs = dict(
        velocity_reference_w=reference,
        current_velocity_w=current_velocity,
        current_quat_wxyz=current_quat,
        current_angular_velocity_b=current_rate,
        mass=_tensor([0.03, 0.032]),
        inertia_b=_inertia(2),
        gravity_magnitude=9.81,
    )
    thrust_default, moment_default, diagnostics_default = controller.compute(**inputs)
    q_velocity, attitude_diagnostics = controller.compute_velocity_attitude_reference(
        velocity_reference_w=reference,
        current_velocity_w=current_velocity,
        gravity_magnitude=9.81,
    )
    thrust_explicit, moment_explicit, diagnostics_explicit = controller.compute(
        **inputs, attitude_reference_wxyz=q_velocity
    )
    torch.testing.assert_close(thrust_explicit, thrust_default, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(moment_explicit, moment_default, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        diagnostics_explicit["body_rate_reference_b"],
        diagnostics_default["body_rate_reference_b"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        attitude_diagnostics["velocity_error_w"], diagnostics_default["velocity_error_w"], atol=0.0, rtol=0.0
    )


def test_velocity_attitude_reference_accepts_deterministic_heading():
    controller = VectorizedPx4LikeController()
    heading = _tensor([[0.0, 1.0, 0.0]])
    q_velocity, diagnostics = controller.compute_velocity_attitude_reference(
        velocity_reference_w=_tensor([[0.0, 0.0, 0.0]]),
        current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
        gravity_magnitude=9.81,
        heading_world=heading,
    )
    body_x = quat_apply(q_velocity, _tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(body_x, heading, atol=1.0e-12, rtol=1.0e-12)
    assert q_velocity.dtype == torch.float64
    assert q_velocity.device.type == "cpu"
    assert not bool(diagnostics["tilt_saturated"][0])


def test_external_attitude_reference_changes_moment_but_not_velocity_thrust():
    controller = VectorizedPx4LikeController()
    reference = _tensor([[0.0, 0.0, 0.0]])
    common = dict(
        velocity_reference_w=reference,
        current_velocity_w=torch.zeros_like(reference),
        current_quat_wxyz=_identity_quat(1),
        current_angular_velocity_b=torch.zeros_like(reference),
        mass=0.03,
        inertia_b=_inertia(1),
        gravity_magnitude=9.81,
    )
    thrust_default, moment_default, _ = controller.compute(**common)
    external = quat_from_euler_xyz(_tensor([math.radians(5.0)]), _tensor([0.0]), _tensor([0.0]))
    thrust_external, moment_external, _ = controller.compute(**common, attitude_reference_wxyz=external)
    torch.testing.assert_close(thrust_external, thrust_default, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(moment_default, torch.zeros_like(moment_default), atol=1.0e-12, rtol=0.0)
    assert float(torch.linalg.norm(moment_external)) > 0.0


def test_nonzero_integral_gain_is_rejected_until_anti_windup_is_defined():
    with pytest.raises(ValueError, match="velocity_integral_gain == 0"):
        VectorizedPx4LikeControllerConfig(velocity_integral_gain=(0.1, 0.0, 0.0)).validate()


def test_derivative_gain_requires_acceleration_measurement():
    controller = VectorizedPx4LikeController(
        VectorizedPx4LikeControllerConfig(velocity_derivative_gain=(0.1, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="current_acceleration_w"):
        controller.compute(
            velocity_reference_w=_tensor([[0.0, 0.0, 0.0]]),
            current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
            current_quat_wxyz=_identity_quat(1),
            current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
            mass=0.03,
            inertia_b=_inertia(1),
            gravity_magnitude=9.81,
        )


@pytest.mark.parametrize(
    "reference",
    [
        [[float("nan"), 0.0, 0.0]],
        [[float("inf"), 0.0, 0.0]],
    ],
)
def test_nan_inf_reference_is_rejected(reference):
    controller = VectorizedPx4LikeController()
    with pytest.raises(ValueError, match="NaN or Inf"):
        controller.compute(
            velocity_reference_w=_tensor(reference),
            current_velocity_w=_tensor([[0.0, 0.0, 0.0]]),
            current_quat_wxyz=_identity_quat(1),
            current_angular_velocity_b=_tensor([[0.0, 0.0, 0.0]]),
            mass=0.03,
            inertia_b=_inertia(1),
            gravity_magnitude=9.81,
        )

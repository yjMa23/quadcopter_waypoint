import math

import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import conservative_minimum_deck_bottom_height
from quadcopter_waypoint.utils.sea_state_motion import (
    SurrogateResponseConfig,
    angular_frequency_grid,
    conservative_component_bound,
    jonswap_spectrum_omega,
    sample_jonswap_components,
    scale_components_to_bound,
    surrogate_vessel_response_components,
    synthesize_components,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def test_jonswap_spectrum_is_finite_non_negative_and_hs_normalized():
    omega, delta_omega = angular_frequency_grid(256, 0.04, 1.20, dtype=torch.float64)
    hs = _tensor([0.20, 0.32])
    tp = _tensor([4.0, 5.5])
    gamma = _tensor([3.3, 2.5])
    spectrum = jonswap_spectrum_omega(omega, hs, tp, gamma, delta_omega=delta_omega)

    assert torch.all(torch.isfinite(spectrum))
    assert torch.all(spectrum >= 0.0)
    m0 = torch.sum(spectrum, dim=-1) * delta_omega
    torch.testing.assert_close(m0, (hs / 4.0) ** 2, atol=1.0e-12, rtol=1.0e-10)

    peak_frequency_hz = omega[torch.argmax(spectrum[0])] / (2.0 * math.pi)
    assert abs(float(peak_frequency_hz) - 1.0 / float(tp[0])) < 0.02


def test_same_seed_reproduces_components_and_different_seed_changes_phase():
    omega, delta_omega = angular_frequency_grid(32, 0.05, 0.90, dtype=torch.float64)
    hs = _tensor([0.24, 0.24])
    tp = _tensor([4.5, 4.5])
    gamma = _tensor([3.3, 3.3])

    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)
    generator_c = torch.Generator().manual_seed(1235)
    spectrum_a, amplitudes_a, phases_a = sample_jonswap_components(
        hs, tp, gamma, omega, delta_omega, generator=generator_a
    )
    spectrum_b, amplitudes_b, phases_b = sample_jonswap_components(
        hs, tp, gamma, omega, delta_omega, generator=generator_b
    )
    _, _, phases_c = sample_jonswap_components(hs, tp, gamma, omega, delta_omega, generator=generator_c)

    torch.testing.assert_close(spectrum_a, spectrum_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(amplitudes_a, amplitudes_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(phases_a, phases_b, atol=0.0, rtol=0.0)
    assert not torch.equal(phases_a, phases_c)


def test_surrogate_response_uses_heading_projection_and_is_not_wave_elevation_copy():
    omega, _ = angular_frequency_grid(8, 0.06, 0.50, dtype=torch.float64)
    amplitudes = torch.full((2, 8), 0.01, dtype=torch.float64)
    phases = torch.zeros_like(amplitudes)
    heading = _tensor([0.0, math.pi / 2.0])
    response = surrogate_vessel_response_components(
        amplitudes,
        phases,
        omega,
        heading,
        SurrogateResponseConfig(),
    )

    roll_amplitudes, _ = response["roll"]
    pitch_amplitudes, _ = response["pitch"]
    assert torch.allclose(roll_amplitudes[0], torch.zeros_like(roll_amplitudes[0]), atol=1.0e-12, rtol=0.0)
    assert torch.allclose(pitch_amplitudes[1], torch.zeros_like(pitch_amplitudes[1]), atol=1.0e-12, rtol=0.0)
    assert torch.any(torch.abs(roll_amplitudes[1]) > 0.0)
    assert torch.any(torch.abs(pitch_amplitudes[0]) > 0.0)
    assert not torch.allclose(response["heave"][0], amplitudes)


def test_analytical_component_derivative_matches_central_difference():
    omega, _ = angular_frequency_grid(24, 0.05, 0.80, dtype=torch.float64)
    generator = torch.Generator().manual_seed(9)
    amplitudes = 0.03 * torch.rand((3, 24), dtype=torch.float64, generator=generator)
    phases = 2.0 * math.pi * torch.rand((3, 24), dtype=torch.float64, generator=generator)
    time = _tensor([0.2, 1.3, 3.1])
    _, analytic_rate = synthesize_components(time, omega, amplitudes, phases)

    dt = 1.0e-6
    plus, _ = synthesize_components(time + dt, omega, amplitudes, phases)
    minus, _ = synthesize_components(time - dt, omega, amplitudes, phases)
    numerical_rate = (plus - minus) / (2.0 * dt)
    torch.testing.assert_close(analytic_rate, numerical_rate, atol=2.0e-9, rtol=2.0e-8)


def test_surrogate_heave_roll_pitch_analytical_derivatives_match_finite_difference():
    omega, delta_omega = angular_frequency_grid(24, 0.05, 0.80, dtype=torch.float64)
    generator = torch.Generator().manual_seed(77)
    hs = _tensor([0.22, 0.28])
    tp = _tensor([4.2, 5.1])
    gamma = _tensor([3.3, 3.8])
    _, wave_amplitudes, wave_phases = sample_jonswap_components(
        hs, tp, gamma, omega, delta_omega, generator=generator
    )
    response = surrogate_vessel_response_components(
        wave_amplitudes,
        wave_phases,
        omega,
        _tensor([0.4, -1.1]),
        SurrogateResponseConfig(
            roll_gain_rad_per_m=math.radians(50.0),
            pitch_gain_rad_per_m=math.radians(50.0),
        ),
    )
    time = _tensor([0.7, 2.3])
    dt = 1.0e-6
    for amplitudes, phases in response.values():
        _, analytic_rate = synthesize_components(time, omega, amplitudes, phases)
        plus, _ = synthesize_components(time + dt, omega, amplitudes, phases)
        minus, _ = synthesize_components(time - dt, omega, amplitudes, phases)
        torch.testing.assert_close(analytic_rate, (plus - minus) / (2.0 * dt), atol=3.0e-9, rtol=3.0e-8)


def test_sea_state_envelope_preserves_conservative_ground_clearance():
    minimum_height = conservative_minimum_deck_bottom_height(
        base_height=0.30,
        maximum_heave_amplitude=0.12,
        deck_half_length=0.25,
        deck_half_width=0.25,
        deck_half_thickness=0.02,
        maximum_roll_rad=math.radians(8.0),
        maximum_pitch_rad=math.radians(8.0),
    )
    assert minimum_height > 0.010 + 0.040


def test_conservative_bound_and_uniform_scaling_are_safe():
    amplitudes = _tensor([[0.04, -0.03, 0.02], [0.01, 0.01, 0.01]])
    bound = conservative_component_bound(amplitudes)
    torch.testing.assert_close(bound, _tensor([0.09, 0.03]))

    scaled, scale = scale_components_to_bound(amplitudes, maximum_abs_value=0.05)
    torch.testing.assert_close(scale, _tensor([5.0 / 9.0, 1.0]))
    assert torch.all(conservative_component_bound(scaled) <= 0.05 + 1.0e-12)

    omega = _tensor([0.4, 0.9, 1.6])
    phases = _tensor([[0.0, 0.3, 1.1], [0.2, 0.4, 0.6]])
    for time_value in torch.linspace(0.0, 10.0, 101, dtype=torch.float64):
        values, _ = synthesize_components(torch.full((2,), time_value), omega, scaled, phases)
        assert torch.all(torch.abs(values) <= 0.05 + 1.0e-12)


def test_component_scaling_preserves_pose_rate_consistency_that_runtime_clamp_breaks():
    omega = _tensor([1.0])
    amplitudes = _tensor([[0.20]])
    phases = _tensor([[0.0]])
    time = _tensor([0.50])
    dt = 1.0e-6

    _, raw_rate = synthesize_components(time, omega, amplitudes, phases)
    raw_plus, _ = synthesize_components(time + dt, omega, amplitudes, phases)
    raw_minus, _ = synthesize_components(time - dt, omega, amplitudes, phases)
    clamped_numerical_rate = (raw_plus.clamp(-0.10, 0.10) - raw_minus.clamp(-0.10, 0.10)) / (2.0 * dt)
    assert float(torch.abs(raw_rate - clamped_numerical_rate)) > 0.09

    scaled, _ = scale_components_to_bound(amplitudes, maximum_abs_value=0.10)
    _, scaled_rate = synthesize_components(time, omega, scaled, phases)
    scaled_plus, _ = synthesize_components(time + dt, omega, scaled, phases)
    scaled_minus, _ = synthesize_components(time - dt, omega, scaled, phases)
    scaled_numerical_rate = (scaled_plus - scaled_minus) / (2.0 * dt)
    torch.testing.assert_close(scaled_rate, scaled_numerical_rate, atol=1.0e-10, rtol=1.0e-9)

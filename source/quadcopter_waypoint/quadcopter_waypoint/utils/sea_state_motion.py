"""Pure-PyTorch stochastic sea-state motion helpers.

The functions in this module deliberately avoid Isaac Sim dependencies so the JONSWAP synthesis,
surrogate vessel response, analytic derivatives, reproducibility, and conservative response bounds
can be unit-tested in a normal Python process.

This is a benchmark motion model. The second-order vessel-response layer is intentionally a simple,
configurable surrogate and must not be interpreted as an identified vessel RAO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SurrogateResponseConfig:
    """Frequency-response parameters for heave/roll/pitch benchmark motion.

    ``*_gain`` maps incident wave elevation to the requested output DOF. Heave gain is dimensionless;
    roll/pitch gains are radians per metre. Each DOF uses a standard second-order transfer function.
    """

    heave_gain: float = 1.0
    heave_natural_frequency_hz: float = 0.24
    heave_damping_ratio: float = 0.85
    roll_gain_rad_per_m: float = math.radians(50.0)
    roll_natural_frequency_hz: float = 0.13
    roll_damping_ratio: float = 0.55
    pitch_gain_rad_per_m: float = math.radians(50.0)
    pitch_natural_frequency_hz: float = 0.13
    pitch_damping_ratio: float = 0.55


def angular_frequency_grid(
    num_components: int,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, float]:
    """Return midpoint angular-frequency bins and their constant ``delta_omega``."""
    if num_components <= 0:
        raise ValueError(f"num_components must be positive, got {num_components}")
    if minimum_frequency_hz <= 0.0 or maximum_frequency_hz <= minimum_frequency_hz:
        raise ValueError(
            "frequency range must satisfy 0 < minimum_frequency_hz < maximum_frequency_hz, "
            f"got {minimum_frequency_hz}..{maximum_frequency_hz}"
        )
    delta_frequency = (maximum_frequency_hz - minimum_frequency_hz) / num_components
    frequencies_hz = minimum_frequency_hz + (torch.arange(num_components, device=device, dtype=dtype) + 0.5) * delta_frequency
    return 2.0 * math.pi * frequencies_hz, 2.0 * math.pi * delta_frequency


def jonswap_spectrum_omega(
    omega: torch.Tensor,
    hs: torch.Tensor,
    tp: torch.Tensor,
    gamma: torch.Tensor,
    *,
    delta_omega: float,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Return a discretely normalized JONSWAP angular-frequency spectrum.

    The spectral shape is the standard Pierson-Moskowitz tail with the JONSWAP peak-enhancement
    factor. Instead of relying on a closed-form Phillips constant, the finite discretization is
    normalized so ``sum(S_k * delta_omega) == (Hs/4)^2``. This makes the finite benchmark spectrum
    explicit and reproducible for the exact bins used by the simulator.
    """
    if delta_omega <= 0.0:
        raise ValueError(f"delta_omega must be positive, got {delta_omega}")
    if torch.any(omega <= 0.0):
        raise ValueError("omega must be strictly positive")
    if torch.any(hs < 0.0) or torch.any(tp <= 0.0) or torch.any(gamma < 1.0):
        raise ValueError("JONSWAP parameters require Hs >= 0, Tp > 0, and gamma >= 1")

    omega = omega.reshape(1, -1)
    hs = hs.reshape(-1, 1).to(device=omega.device, dtype=omega.dtype)
    tp = tp.reshape(-1, 1).to(device=omega.device, dtype=omega.dtype)
    gamma = gamma.reshape(-1, 1).to(device=omega.device, dtype=omega.dtype)
    omega_peak = 2.0 * math.pi / tp
    sigma = torch.where(omega <= omega_peak, torch.full_like(omega, 0.07), torch.full_like(omega, 0.09))
    peak_exponent = torch.exp(-0.5 * ((omega - omega_peak) / (sigma * omega_peak)) ** 2)

    # The scale is normalized below, but retaining g^2 makes the unnormalized shape conventional.
    shape = (
        gravity**2
        * omega.pow(-5)
        * torch.exp(-1.25 * (omega_peak / omega) ** 4)
        * gamma.pow(peak_exponent)
    )
    raw_m0 = torch.sum(shape, dim=-1, keepdim=True) * delta_omega
    target_m0 = (hs / 4.0) ** 2
    scale = torch.where(target_m0 > 0.0, target_m0 / raw_m0.clamp_min(torch.finfo(shape.dtype).tiny), torch.zeros_like(target_m0))
    return shape * scale


def component_amplitudes_from_spectrum(spectrum: torch.Tensor, delta_omega: float) -> torch.Tensor:
    """Convert one-sided spectral density bins to cosine-component amplitudes."""
    if delta_omega <= 0.0:
        raise ValueError(f"delta_omega must be positive, got {delta_omega}")
    if torch.any(spectrum < 0.0):
        raise ValueError("spectrum must be non-negative")
    return torch.sqrt(2.0 * spectrum * delta_omega)


def second_order_frequency_response(
    omega: torch.Tensor,
    *,
    natural_frequency_hz: float,
    damping_ratio: float,
    gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return magnitude and phase of a stable second-order low-pass response."""
    if natural_frequency_hz <= 0.0:
        raise ValueError(f"natural_frequency_hz must be positive, got {natural_frequency_hz}")
    if damping_ratio <= 0.0:
        raise ValueError(f"damping_ratio must be positive, got {damping_ratio}")
    omega_n = 2.0 * math.pi * natural_frequency_hz
    real = omega_n**2 - omega**2
    imag = 2.0 * damping_ratio * omega_n * omega
    denominator = torch.sqrt(real**2 + imag**2)
    magnitude = abs(gain) * omega_n**2 / denominator.clamp_min(torch.finfo(omega.dtype).tiny)
    phase = -torch.atan2(imag, real)
    if gain < 0.0:
        phase = phase + math.pi
    return magnitude, phase


def surrogate_vessel_response_components(
    wave_amplitudes: torch.Tensor,
    wave_phases: torch.Tensor,
    omega: torch.Tensor,
    heading_rad: torch.Tensor,
    config: SurrogateResponseConfig,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map wave components to heave/roll/pitch through a configurable surrogate response.

    Roll and pitch use simple heading projection: beam-sea excitation scales with ``sin(heading)``
    for roll and head-sea excitation with ``cos(heading)`` for pitch. The model is intentionally
    interpretable and replaceable by measured or tabulated RAOs later.
    """
    if wave_amplitudes.shape != wave_phases.shape:
        raise ValueError(f"wave amplitude/phase shape mismatch: {wave_amplitudes.shape} != {wave_phases.shape}")
    if wave_amplitudes.shape[-1] != omega.numel():
        raise ValueError("last wave-component dimension must match omega")
    heading_rad = heading_rad.reshape(-1, 1).to(device=wave_amplitudes.device, dtype=wave_amplitudes.dtype)
    omega = omega.to(device=wave_amplitudes.device, dtype=wave_amplitudes.dtype)

    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    definitions = (
        (
            "heave",
            config.heave_natural_frequency_hz,
            config.heave_damping_ratio,
            config.heave_gain,
            torch.ones_like(heading_rad),
        ),
        (
            "roll",
            config.roll_natural_frequency_hz,
            config.roll_damping_ratio,
            config.roll_gain_rad_per_m,
            torch.sin(heading_rad),
        ),
        (
            "pitch",
            config.pitch_natural_frequency_hz,
            config.pitch_damping_ratio,
            config.pitch_gain_rad_per_m,
            torch.cos(heading_rad),
        ),
    )
    for name, natural_frequency_hz, damping_ratio, gain, directional_scale in definitions:
        magnitude, response_phase = second_order_frequency_response(
            omega,
            natural_frequency_hz=natural_frequency_hz,
            damping_ratio=damping_ratio,
            gain=gain,
        )
        amplitudes = wave_amplitudes * magnitude.reshape(1, -1) * directional_scale
        phases = wave_phases + response_phase.reshape(1, -1)
        outputs[name] = (amplitudes, phases)
    return outputs


def conservative_component_bound(amplitudes: torch.Tensor) -> torch.Tensor:
    """Return ``sum(abs(A_k))``, a phase-independent bound for each realization."""
    return torch.sum(torch.abs(amplitudes), dim=-1)


def scale_components_to_bound(
    amplitudes: torch.Tensor,
    maximum_abs_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly scale each realization so its conservative component bound fits an envelope."""
    if maximum_abs_value <= 0.0:
        raise ValueError(f"maximum_abs_value must be positive, got {maximum_abs_value}")
    bound = conservative_component_bound(amplitudes)
    scale = torch.minimum(torch.ones_like(bound), maximum_abs_value / bound.clamp_min(torch.finfo(amplitudes.dtype).tiny))
    return amplitudes * scale.unsqueeze(-1), scale


def synthesize_components(
    time_s: torch.Tensor,
    omega: torch.Tensor,
    amplitudes: torch.Tensor,
    phases: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize cosine components and their exact analytical time derivative."""
    if amplitudes.shape != phases.shape:
        raise ValueError(f"component amplitude/phase shape mismatch: {amplitudes.shape} != {phases.shape}")
    if amplitudes.shape[-1] != omega.numel():
        raise ValueError("last component dimension must match omega")
    time_s = time_s.reshape(-1, 1).to(device=amplitudes.device, dtype=amplitudes.dtype)
    omega = omega.reshape(1, -1).to(device=amplitudes.device, dtype=amplitudes.dtype)
    angle = omega * time_s + phases
    value = torch.sum(amplitudes * torch.cos(angle), dim=-1)
    rate = -torch.sum(amplitudes * omega * torch.sin(angle), dim=-1)
    return value, rate


def sample_jonswap_components(
    hs: torch.Tensor,
    tp: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    delta_omega: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random phases and return ``(spectrum, amplitudes, phases)`` for a JONSWAP realization."""
    spectrum = jonswap_spectrum_omega(omega, hs, tp, gamma, delta_omega=delta_omega)
    amplitudes = component_amplitudes_from_spectrum(spectrum, delta_omega)
    phases = 2.0 * math.pi * torch.rand(
        amplitudes.shape,
        device=amplitudes.device,
        dtype=amplitudes.dtype,
        generator=generator,
    )
    return spectrum, amplitudes, phases

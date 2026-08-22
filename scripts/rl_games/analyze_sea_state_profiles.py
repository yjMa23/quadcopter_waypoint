#!/usr/bin/env python3
"""Offline realization diagnostics for factor-isolated Sea-State benchmark profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from quadcopter_waypoint.utils.physical_deck_attitude_math import world_angular_velocity_from_xyz_rates
from quadcopter_waypoint.utils.sea_state_motion import (
    SurrogateResponseConfig,
    angular_frequency_grid,
    sample_jonswap_components,
    scale_components_to_bound,
    surrogate_vessel_response_components,
    synthesize_components,
)
from quadcopter_waypoint.utils.sea_state_profiles import load_sea_state_profiles


DEFAULTS = {
    "sea_state_heave_gain": 1.0,
    "sea_state_heave_natural_frequency_hz": 0.24,
    "sea_state_heave_damping_ratio": 0.85,
    "sea_state_roll_gain_deg_per_m": 50.0,
    "sea_state_roll_natural_frequency_hz": 0.13,
    "sea_state_roll_damping_ratio": 0.55,
    "sea_state_pitch_gain_deg_per_m": 50.0,
    "sea_state_pitch_natural_frequency_hz": 0.13,
    "sea_state_pitch_damping_ratio": 0.55,
    "sea_state_hs_min_m": 0.18,
    "sea_state_hs_max_m": 0.30,
    "sea_state_tp_min_s": 3.8,
    "sea_state_tp_max_s": 5.8,
    "sea_state_gamma_min": 2.5,
    "sea_state_gamma_max": 4.0,
    "sea_state_heading_min_deg": -180.0,
    "sea_state_heading_max_deg": 180.0,
}


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q / 100.0))


def uniform(count: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    return low + (high - low) * torch.rand(count, generator=generator)


def response_config(env: dict[str, float]) -> SurrogateResponseConfig:
    get = lambda key: env.get(key, DEFAULTS[key])
    return SurrogateResponseConfig(
        heave_gain=float(get("sea_state_heave_gain")),
        heave_natural_frequency_hz=float(get("sea_state_heave_natural_frequency_hz")),
        heave_damping_ratio=float(get("sea_state_heave_damping_ratio")),
        roll_gain_rad_per_m=math.radians(float(get("sea_state_roll_gain_deg_per_m"))),
        roll_natural_frequency_hz=float(get("sea_state_roll_natural_frequency_hz")),
        roll_damping_ratio=float(get("sea_state_roll_damping_ratio")),
        pitch_gain_rad_per_m=math.radians(float(get("sea_state_pitch_gain_deg_per_m"))),
        pitch_natural_frequency_hz=float(get("sea_state_pitch_natural_frequency_hz")),
        pitch_damping_ratio=float(get("sea_state_pitch_damping_ratio")),
    )


def analyze_profile(name: str, profile: dict, count: int, seed: int, sample_dt: float) -> dict[str, float | int | str]:
    env = {**DEFAULTS, **profile["env"]}
    if env.get("sea_state_mode") == "compatibility":
        return {
            "profile": name,
            "family": profile["family"],
            "severity_rank": profile["severity_rank"],
            "samples": 0,
            "status": "compatibility_not_stochastic",
        }

    generator = torch.Generator().manual_seed(seed + 1009 * int(profile["severity_rank"]))
    hs = uniform(count, float(env["sea_state_hs_min_m"]), float(env["sea_state_hs_max_m"]), generator)
    tp = uniform(count, float(env["sea_state_tp_min_s"]), float(env["sea_state_tp_max_s"]), generator)
    gamma = uniform(count, float(env["sea_state_gamma_min"]), float(env["sea_state_gamma_max"]), generator)
    heading = uniform(
        count,
        math.radians(float(env["sea_state_heading_min_deg"])),
        math.radians(float(env["sea_state_heading_max_deg"])),
        generator,
    )
    omega, delta_omega = angular_frequency_grid(24, 0.05, 0.80)
    _, wave_amplitudes, wave_phases = sample_jonswap_components(
        hs, tp, gamma, omega, delta_omega, generator=generator
    )
    response = surrogate_vessel_response_components(
        wave_amplitudes, wave_phases, omega, heading, response_config(env)
    )
    heave_amp, heave_phase = response["heave"]
    roll_amp, roll_phase = response["roll"]
    pitch_amp, pitch_phase = response["pitch"]
    heave_amp, heave_scale = scale_components_to_bound(heave_amp, 0.12)
    roll_amp, roll_scale = scale_components_to_bound(roll_amp, math.radians(8.0))
    pitch_amp, pitch_scale = scale_components_to_bound(pitch_amp, math.radians(8.0))

    heave_max = torch.zeros(count)
    roll_max = torch.zeros(count)
    pitch_max = torch.zeros(count)
    tilt_max = torch.zeros(count)
    heave_rate_max = torch.zeros(count)
    roll_rate_max = torch.zeros(count)
    pitch_rate_max = torch.zeros(count)
    angular_speed_max = torch.zeros(count)
    heave_sq = torch.zeros(count)
    roll_sq = torch.zeros(count)
    pitch_sq = torch.zeros(count)
    steps = 0
    for time_value in torch.arange(0.0, 10.0 + 0.5 * sample_dt, sample_dt):
        time = torch.full((count,), float(time_value))
        heave, heave_rate = synthesize_components(time, omega, heave_amp, heave_phase)
        roll, roll_rate = synthesize_components(time, omega, roll_amp, roll_phase)
        pitch, pitch_rate = synthesize_components(time, omega, pitch_amp, pitch_phase)
        angular_velocity = world_angular_velocity_from_xyz_rates(
            roll, pitch, torch.zeros_like(roll), roll_rate, pitch_rate, torch.zeros_like(roll)
        )
        heave_max = torch.maximum(heave_max, heave.abs())
        roll_max = torch.maximum(roll_max, roll.abs())
        pitch_max = torch.maximum(pitch_max, pitch.abs())
        tilt_max = torch.maximum(tilt_max, torch.sqrt(roll.square() + pitch.square()))
        heave_rate_max = torch.maximum(heave_rate_max, heave_rate.abs())
        roll_rate_max = torch.maximum(roll_rate_max, roll_rate.abs())
        pitch_rate_max = torch.maximum(pitch_rate_max, pitch_rate.abs())
        angular_speed_max = torch.maximum(angular_speed_max, torch.linalg.norm(angular_velocity, dim=-1))
        heave_sq += heave.square()
        roll_sq += roll.square()
        pitch_sq += pitch.square()
        steps += 1

    minimum_scale = torch.minimum(torch.minimum(heave_scale, roll_scale), pitch_scale)
    result: dict[str, float | int | str] = {
        "profile": name,
        "family": profile["family"],
        "severity_rank": profile["severity_rank"],
        "samples": count,
        "status": "PASS",
        "hs_mean_m": float(hs.mean()),
        "tp_mean_s": float(tp.mean()),
        "gamma_mean": float(gamma.mean()),
        "heave_rms_p50_m": percentile(torch.sqrt(heave_sq / steps), 50),
        "heave_max_p50_m": percentile(heave_max, 50),
        "heave_max_p95_m": percentile(heave_max, 95),
        "tilt_max_p50_deg": math.degrees(percentile(tilt_max, 50)),
        "tilt_max_p95_deg": math.degrees(percentile(tilt_max, 95)),
        "roll_max_p95_deg": math.degrees(percentile(roll_max, 95)),
        "pitch_max_p95_deg": math.degrees(percentile(pitch_max, 95)),
        "heave_velocity_max_p50_mps": percentile(heave_rate_max, 50),
        "heave_velocity_max_p95_mps": percentile(heave_rate_max, 95),
        "roll_rate_max_p95_radps": percentile(roll_rate_max, 95),
        "pitch_rate_max_p95_radps": percentile(pitch_rate_max, 95),
        "deck_angular_speed_max_p50_radps": percentile(angular_speed_max, 50),
        "deck_angular_speed_max_p95_radps": percentile(angular_speed_max, 95),
        "scaling_fraction": float((minimum_scale < 0.999).float().mean()),
        "min_scale_p05": percentile(minimum_scale, 5),
        "min_scale_p50": percentile(minimum_scale, 50),
        "heave_scale_p05": percentile(heave_scale, 5),
        "roll_scale_p05": percentile(roll_scale, 5),
        "pitch_scale_p05": percentile(pitch_scale, 5),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("benchmarks/sea_state/profiles.yaml"))
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--sample_dt", type=float, default=0.05)
    parser.add_argument("--output_csv", type=Path, default=Path("benchmarks/sea_state/profile_realization_summary.csv"))
    parser.add_argument("--output_json", type=Path, default=Path("benchmarks/sea_state/profile_realization_summary.json"))
    args = parser.parse_args()

    profiles = load_sea_state_profiles(args.profiles)
    rows = [analyze_profile(name, profile, args.samples, args.seed, args.sample_dt) for name, profile in profiles.items()]
    stochastic_rows = [row for row in rows if row["status"] == "PASS"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(stochastic_rows[0].keys())
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

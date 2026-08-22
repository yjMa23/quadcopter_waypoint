"""Benchmark-profile helpers for the frozen Sea-State v1 motion model.

Profiles live outside the environment implementation so robustness studies can change controlled
input distributions without changing JONSWAP synthesis, surrogate response math, analytic
pose/velocity generation, or the PhysicalDeckAttitude landing contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ALLOWED_ENV_FIELDS = {
    "sea_state_mode",
    "sea_state_hs_min_m",
    "sea_state_hs_max_m",
    "sea_state_tp_min_s",
    "sea_state_tp_max_s",
    "sea_state_gamma_min",
    "sea_state_gamma_max",
    "sea_state_heading_min_deg",
    "sea_state_heading_max_deg",
    "sea_state_heave_gain",
    "sea_state_heave_natural_frequency_hz",
    "sea_state_heave_damping_ratio",
    "sea_state_roll_gain_deg_per_m",
    "sea_state_roll_natural_frequency_hz",
    "sea_state_roll_damping_ratio",
    "sea_state_pitch_gain_deg_per_m",
    "sea_state_pitch_natural_frequency_hz",
    "sea_state_pitch_damping_ratio",
}

REQUIRED_PROFILE_FIELDS = {"family", "severity_rank", "env"}


def load_sea_state_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate named engineering benchmark profiles from YAML."""
    profile_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(profile_path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise ValueError(f"Expected top-level 'profiles' mapping in {profile_path}")

    profiles = data["profiles"]
    for name, profile in profiles.items():
        if not isinstance(name, str) or not isinstance(profile, dict):
            raise ValueError("Profile names and definitions must be mappings")
        missing = REQUIRED_PROFILE_FIELDS - profile.keys()
        if missing:
            raise ValueError(f"Profile {name!r} missing fields: {sorted(missing)}")
        if not isinstance(profile["severity_rank"], int) or profile["severity_rank"] < 0:
            raise ValueError(f"Profile {name!r} severity_rank must be a non-negative integer")
        env = profile["env"]
        if not isinstance(env, dict):
            raise ValueError(f"Profile {name!r} env must be a mapping")
        unknown = set(env) - ALLOWED_ENV_FIELDS
        if unknown:
            raise ValueError(f"Profile {name!r} has unsupported env fields: {sorted(unknown)}")
    return profiles


def apply_sea_state_profile(cfg: Any, profile_name: str, profile: dict[str, Any]) -> None:
    """Apply one validated profile to an environment config object."""
    for key, value in profile["env"].items():
        if key not in ALLOWED_ENV_FIELDS:
            raise ValueError(f"Unsupported profile field {key!r}")
        if not hasattr(cfg, key):
            raise AttributeError(f"Environment config does not expose {key!r}")
        setattr(cfg, key, value)
    cfg.sea_state_benchmark_profile = profile_name


def hydra_env_overrides(profile_name: str, profile: dict[str, Any]) -> list[str]:
    """Return deterministic Hydra ``env.key=value`` overrides for eval_metrics.py."""
    overrides = [f"env.sea_state_benchmark_profile={profile_name}"]
    for key, value in profile["env"].items():
        if key not in ALLOWED_ENV_FIELDS:
            raise ValueError(f"Unsupported profile field {key!r}")
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        overrides.append(f"env.{key}={text}")
    return overrides

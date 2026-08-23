# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python helpers shared by quadcopter evaluation and regression tests."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PAD_SPEED_BUCKETS = ("0.00-0.05", "0.05-0.10", "0.10-0.15", ">=0.15")
DECK_TILT_BUCKETS = ("0-2deg", "2-4deg", "4-6deg", ">=6deg")
DECK_ANGULAR_SPEED_BUCKETS = ("0.00-0.04", "0.04-0.08", "0.08-0.12", ">=0.12")

PX4_HIERARCHICAL_SCALAR_LATCHES = {
    "relative_velocity_reference_norm_mean": "_last_relative_velocity_reference_norm_mean",
    "relative_velocity_reference_norm_p95": "_last_relative_velocity_reference_norm_p95",
    "relative_velocity_reference_norm_max": "_last_relative_velocity_reference_norm_max",
    "reference_saturation_ratio": "_last_reference_saturation_ratio",
    "controller_velocity_tracking_error_mean": "_last_controller_velocity_tracking_error_mean",
    "controller_velocity_tracking_error_max": "_last_controller_velocity_tracking_error_max",
    "controller_acceleration_saturation_ratio": "_last_controller_acceleration_saturation_ratio",
    "controller_tilt_saturation_ratio": "_last_controller_tilt_saturation_ratio",
    "controller_thrust_saturation_ratio": "_last_controller_thrust_saturation_ratio",
    "controller_body_rate_saturation_ratio": "_last_controller_body_rate_saturation_ratio",
    "controller_moment_saturation_ratio": "_last_controller_moment_saturation_ratio",
    "max_desired_tilt": "_last_max_desired_tilt",
    "max_body_rate": "_last_max_body_rate",
    "max_moment": "_last_max_moment",
    "controller_runtime_ms_mean": "_last_controller_runtime_ms_mean",
    "controller_runtime_ms_p95": "_last_controller_runtime_ms_p95",
    "controller_runtime_ms_max": "_last_controller_runtime_ms_max",
    "reward_descent_phase_active_ratio": "_last_reward_descent_phase_active_ratio",
    "can_land_but_reward_gate_inactive_ratio": "_last_can_land_but_reward_gate_inactive_ratio",
    "reward_gate_transition_count": "_last_reward_gate_transition_count",
    "reward_gate_horizontal_error_violation_ratio": "_last_reward_gate_horizontal_error_violation_ratio",
    "reward_gate_horizontal_speed_violation_ratio": "_last_reward_gate_horizontal_speed_violation_ratio",
    "reward_gate_attitude_violation_ratio": "_last_reward_gate_attitude_violation_ratio",
    "reward_gate_too_high_violation_ratio": "_last_reward_gate_too_high_violation_ratio",
}
PX4_HIERARCHICAL_VECTOR_LATCHES = {
    "action_mean": "_last_action_mean",
    "action_std": "_last_action_std",
    "action_abs_max": "_last_action_abs_max",
}


def has_px4_hierarchical_diagnostics(task: Any) -> bool:
    """Return whether ``task`` exposes the complete optional M2 terminal diagnostic contract."""
    latch_names = tuple(PX4_HIERARCHICAL_SCALAR_LATCHES.values()) + tuple(PX4_HIERARCHICAL_VECTOR_LATCHES.values())
    return all(hasattr(task, name) for name in latch_names)


def pad_speed_bucket(pad_speed: float) -> str:
    """Return the half-open horizontal pad-speed bucket containing ``pad_speed``."""
    if pad_speed < 0.05:
        return PAD_SPEED_BUCKETS[0]
    if pad_speed < 0.10:
        return PAD_SPEED_BUCKETS[1]
    if pad_speed < 0.15:
        return PAD_SPEED_BUCKETS[2]
    return PAD_SPEED_BUCKETS[3]


def deck_tilt_bucket(tilt_radians: float) -> str:
    """Return the requested first-contact deck-tilt bucket."""
    tilt_degrees = math.degrees(abs(float(tilt_radians)))
    if tilt_degrees < 2.0:
        return DECK_TILT_BUCKETS[0]
    if tilt_degrees < 4.0:
        return DECK_TILT_BUCKETS[1]
    if tilt_degrees < 6.0:
        return DECK_TILT_BUCKETS[2]
    return DECK_TILT_BUCKETS[3]


def deck_angular_speed_bucket(angular_speed: float) -> str:
    """Return a half-open deck angular-speed bucket in rad/s."""
    angular_speed = abs(float(angular_speed))
    if angular_speed < 0.04:
        return DECK_ANGULAR_SPEED_BUCKETS[0]
    if angular_speed < 0.08:
        return DECK_ANGULAR_SPEED_BUCKETS[1]
    if angular_speed < 0.12:
        return DECK_ANGULAR_SPEED_BUCKETS[2]
    return DECK_ANGULAR_SPEED_BUCKETS[3]


def select_terminal_value(latched: Any, fallback: Any, valid: bool) -> Any:
    """Prefer an exact terminal latch and use a pre-step fallback only for legacy tasks."""
    return latched if valid else fallback


def mean_or_nan(values: Iterable[float]) -> float:
    """Compute a mean without dividing by zero or replacing an empty set with a misleading zero."""
    collected = [float(value) for value in values]
    return sum(collected) / len(collected) if collected else float("nan")


def percentile_or_nan(values: Iterable[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile, returning NaN for an empty collection."""
    collected = sorted(float(value) for value in values)
    if not collected:
        return float("nan")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if len(collected) == 1:
        return collected[0]
    position = (len(collected) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return collected[lower]
    weight = position - lower
    return collected[lower] * (1.0 - weight) + collected[upper] * weight


def successful_values(episodes: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """Collect one touchdown metric only from episodes explicitly marked successful."""
    return [float(episode[key]) for episode in episodes if bool(episode.get("success", False))]


def summarize_ship_landing(episodes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Build the core ship-landing summary from per-episode records."""
    episode_count = len(episodes)
    if episode_count == 0:
        return {
            "episodes": 0.0,
            "landing_success_rate": float("nan"),
            "align_success_rate": float("nan"),
            "crash_rate": float("nan"),
            "timeout_rate": float("nan"),
            "touchdown_distance_mean": float("nan"),
            "touchdown_distance_p50": float("nan"),
            "touchdown_distance_p90": float("nan"),
            "touchdown_distance_p95": float("nan"),
        }

    touchdown_distances = successful_values(episodes, "touchdown_distance")
    return {
        "episodes": float(episode_count),
        "landing_success_rate": sum(bool(ep.get("success", False)) for ep in episodes) / episode_count,
        "align_success_rate": sum(bool(ep.get("align_success", False)) for ep in episodes) / episode_count,
        "crash_rate": sum(bool(ep.get("crash", False)) for ep in episodes) / episode_count,
        "timeout_rate": sum(bool(ep.get("time_out", False)) for ep in episodes) / episode_count,
        "touchdown_distance_mean": mean_or_nan(touchdown_distances),
        "touchdown_distance_p50": percentile_or_nan(touchdown_distances, 50.0),
        "touchdown_distance_p90": percentile_or_nan(touchdown_distances, 90.0),
        "touchdown_distance_p95": percentile_or_nan(touchdown_distances, 95.0),
    }

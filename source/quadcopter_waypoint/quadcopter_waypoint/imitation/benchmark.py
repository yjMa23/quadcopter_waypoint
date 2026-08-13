"""Pure-Python aggregation helpers for imitation-learning benchmark closed-loop evaluations and learning curves."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

import numpy as np

RATE_FIELDS = {
    "contact_success_rate": "contact_success",
    "settled_landing_rate": "settled_landing",
    "hard_contact_rate": "hard_contact",
    "ground_crash_rate": "ground_crash",
    "deck_miss_rate": "deck_miss",
    "timeout_rate": "time_out",
}

FLOAT_FIELDS = {
    "first_contact_xy_error_mean_m": "first_contact_xy_error_deck_frame",
    "first_contact_normal_relative_speed_mean_mps": "first_contact_normal_rel_speed",
    "first_contact_tangential_relative_speed_mean_mps": "first_contact_tangential_rel_speed",
    "first_contact_body_deck_normal_angle_mean_rad": "first_contact_body_deck_normal_angle",
    "touchdown_distance_mean_m": "touchdown_distance",
    "maximum_penetration_mean_m": "maximum_penetration",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_evaluation_csv(path: str | Path) -> list[dict[str, str]]:
    """Read one formal evaluation CSV and reject empty files."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"evaluation CSV is empty: {path}")
    return rows


def summarize_episode_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the physical-deck-attitude task/imitation-learning benchmark episode CSV fields used in the fair comparison."""
    records = list(rows)
    if not records:
        raise ValueError("at least one episode row is required")
    summary: dict[str, Any] = {"episodes": len(records)}
    for output_name, field in RATE_FIELDS.items():
        if field not in records[0]:
            raise KeyError(f"evaluation row is missing {field}")
        summary[output_name] = mean(_as_bool(row[field]) for row in records)
    successful = [row for row in records if _as_bool(row["settled_landing"])]
    summary["settled_episodes"] = len(successful)
    for output_name, field in FLOAT_FIELDS.items():
        source = successful if field.startswith("first_contact") or field == "touchdown_distance" else records
        values = [_finite_float(row.get(field)) for row in source]
        finite = [value for value in values if value is not None]
        summary[output_name] = float(np.mean(finite)) if finite else None
        p95_name = output_name.replace("_mean_", "_p95_")
        summary[p95_name] = float(np.percentile(finite, 95)) if finite else None
    failure_counts = {
        "hard_contact": sum(_as_bool(row["hard_contact"]) for row in records),
        "ground_crash": sum(_as_bool(row["ground_crash"]) for row in records),
        "deck_miss": sum(_as_bool(row["deck_miss"]) for row in records),
        "timeout": sum(_as_bool(row["time_out"]) for row in records),
    }
    unexplained = len(records) - len(successful) - sum(failure_counts.values())
    if unexplained > 0:
        failure_counts["other"] = unexplained
    summary["failure_counts"] = failure_counts
    return summary


def summarize_evaluation_csv(path: str | Path) -> dict[str, Any]:
    """Read and summarize one evaluation CSV."""
    summary = summarize_episode_rows(read_evaluation_csv(path))
    summary["csv"] = str(Path(path))
    return summary


def aggregate_seed_summaries(per_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute mean/std across equal-sized evaluation seeds plus pooled count rates."""
    if not per_seed:
        raise ValueError("per_seed summaries cannot be empty")
    episode_counts = {int(value["episodes"]) for value in per_seed.values()}
    if len(episode_counts) != 1:
        raise ValueError("fair seed aggregation requires equal episode counts")
    aggregate: dict[str, Any] = {
        "seeds": len(per_seed),
        "episodes_per_seed": next(iter(episode_counts)),
        "episodes": sum(int(value["episodes"]) for value in per_seed.values()),
    }
    numeric_keys = sorted(
        key
        for key in next(iter(per_seed.values()))
        if key.endswith("_rate") or key.endswith("_mean_m") or key.endswith("_mean_mps") or key.endswith("_mean_rad")
    )
    for key in numeric_keys:
        values = [float(value[key]) for value in per_seed.values() if value.get(key) is not None]
        if values:
            aggregate[key] = {"mean": mean(values), "std": pstdev(values)}
    failure_totals: dict[str, int] = {}
    for value in per_seed.values():
        for name, count in value.get("failure_counts", {}).items():
            failure_totals[name] = failure_totals.get(name, 0) + int(count)
    aggregate["failure_counts"] = failure_totals
    return aggregate


def threshold_crossing_steps(
    curve: Iterable[Mapping[str, Any]],
    thresholds: Iterable[float] = (0.8, 0.9, 0.92),
    metric: str = "settled_landing_rate",
    step_field: str = "environment_steps",
) -> dict[str, int | None]:
    """Return the first evaluated environment-step count reaching each threshold."""
    ordered = sorted(curve, key=lambda row: int(row[step_field]))
    result: dict[str, int | None] = {}
    for threshold in thresholds:
        crossing = next(
            (int(row[step_field]) for row in ordered if float(row[metric]) >= float(threshold)),
            None,
        )
        result[f"{int(round(threshold * 100))}%"] = crossing
    return result

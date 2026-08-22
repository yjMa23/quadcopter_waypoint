#!/usr/bin/env python3
"""Aggregate Sea-State evaluation CSVs into realized-motion robustness curves and boundary diagnostics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

from quadcopter_waypoint.utils.sea_state_profiles import load_sea_state_profiles

BUCKETS = {
    "sea_deck_angular_speed_max": [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, math.inf],
    "deck_tilt_max_deg": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, math.inf],
    "sea_heave_velocity_max_abs": [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, math.inf],
    "sea_tp": [0.0, 3.0, 4.0, 5.0, 6.0, math.inf],
    "sea_hs": [0.0, 0.16, 0.20, 0.24, 0.30, 0.36, math.inf],
}

DOMINANT_METRIC = {
    "frequency_shift": "sea_deck_angular_speed_max",
    "tilt_shift": "deck_tilt_max_deg",
    "heave_rate_shift": "sea_heave_velocity_max_abs",
    "combined_shift": "sea_deck_angular_speed_max",
    "nominal": "sea_deck_angular_speed_max",
    "compatibility": "deck_tilt_max_deg",
}


def bool_value(row: dict[str, str], key: str) -> bool:
    return row.get(key, "False").strip().lower() == "true"


def float_value(row: dict[str, str], key: str) -> float:
    text = row.get(key, "")
    if text in ("", None):
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def numeric_metric(row: dict[str, str], metric: str) -> float:
    if metric == "deck_tilt_max_deg":
        roll = float_value(row, "sea_roll_max_abs")
        pitch = float_value(row, "sea_pitch_max_abs")
        return math.degrees(math.hypot(roll, pitch))
    return float_value(row, metric)


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    values = finite(values)
    return sum(values) / len(values) if values else math.nan


def percentile(values: list[float], q: float) -> float:
    values = sorted(finite(values))
    if not values:
        return math.nan
    position = (len(values) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def bucket_label(value: float, edges: list[float]) -> tuple[int, str]:
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if lower <= value < upper:
            upper_text = "inf" if math.isinf(upper) else f"{upper:g}"
            return index, f"[{lower:g},{upper_text})"
    return len(edges) - 2, f"[{edges[-2]:g},inf)"


def metadata_from_path(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("__")
    if len(parts) >= 3 and parts[-1].startswith("seed"):
        return parts[0], "__".join(parts[1:-1]), parts[-1][4:]
    return "unknown", path.stem, ""


def load_rows(input_specs: list[str], input_globs: list[str]) -> list[dict[str, str]]:
    sources: list[tuple[str | None, Path]] = []
    for spec in input_specs:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=CSV, got {spec!r}")
        label, path_text = spec.split("=", 1)
        sources.append((label, Path(path_text).expanduser().resolve()))
    for pattern in input_globs:
        for path_text in sorted(glob.glob(pattern)):
            sources.append((None, Path(path_text).expanduser().resolve()))
    if not sources:
        raise ValueError("At least one --input or --input_glob is required")

    rows: list[dict[str, str]] = []
    for explicit_label, path in sources:
        policy, inferred_profile, seed = metadata_from_path(path)
        profile = explicit_label or inferred_profile
        if explicit_label is not None:
            policy = "explicit"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["policy_label"] = policy
                row["profile"] = profile
                row["eval_seed"] = seed
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def outcome(row: dict[str, str]) -> str:
    if bool_value(row, "settled_landing"):
        return "settled_landing"
    if bool_value(row, "ground_crash"):
        return "ground_crash"
    if bool_value(row, "hard_contact"):
        return "hard_contact"
    if bool_value(row, "deck_miss"):
        return "deck_miss"
    if bool_value(row, "time_out"):
        return "timeout"
    return "other_failure"


def summarize_profiles(rows: list[dict[str, str]], profiles: dict | None) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy_label"], row["profile"])].append(row)
    summary_rows: list[dict[str, object]] = []
    for (policy, profile_name), subset in sorted(groups.items()):
        count = len(subset)
        profile = profiles.get(profile_name, {}) if profiles else {}
        min_scales = [float_value(row, "sea_min_scale") for row in subset]
        tilt = [numeric_metric(row, "deck_tilt_max_deg") for row in subset]
        angular = [numeric_metric(row, "sea_deck_angular_speed_max") for row in subset]
        heave_velocity = [numeric_metric(row, "sea_heave_velocity_max_abs") for row in subset]
        summary_rows.append(
            {
                "policy_label": policy,
                "profile": profile_name,
                "family": profile.get("family", "unknown"),
                "severity_rank": profile.get("severity_rank", -1),
                "episodes": count,
                "eval_seeds": ",".join(sorted({row["eval_seed"] for row in subset if row["eval_seed"]})),
                "settled_landing_rate": sum(bool_value(row, "settled_landing") for row in subset) / count,
                "deck_miss_rate": sum(bool_value(row, "deck_miss") for row in subset) / count,
                "hard_contact_rate": sum(bool_value(row, "hard_contact") for row in subset) / count,
                "ground_crash_rate": sum(bool_value(row, "ground_crash") for row in subset) / count,
                "timeout_rate": sum(bool_value(row, "time_out") for row in subset) / count,
                "tp_mean_s": mean([float_value(row, "sea_tp") for row in subset]),
                "hs_mean_m": mean([float_value(row, "sea_hs") for row in subset]),
                "deck_tilt_max_mean_deg": mean(tilt),
                "deck_tilt_max_p95_deg": percentile(tilt, 95),
                "deck_angular_speed_max_mean_radps": mean(angular),
                "deck_angular_speed_max_p95_radps": percentile(angular, 95),
                "heave_velocity_max_mean_mps": mean(heave_velocity),
                "heave_velocity_max_p95_mps": percentile(heave_velocity, 95),
                "scaling_fraction": mean([1.0 if value < 0.999 else 0.0 for value in finite(min_scales)]),
                "min_scale_p05": percentile(min_scales, 5),
                "min_scale_p50": percentile(min_scales, 50),
            }
        )
    return summary_rows


def robustness_buckets(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric, edges in BUCKETS.items():
        groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            value = numeric_metric(row, metric)
            if not math.isfinite(value):
                continue
            index, label = bucket_label(value, edges)
            groups[(row["policy_label"], index, label)].append(row)
        for (policy, index, label), subset in sorted(groups.items()):
            count = len(subset)
            output.append(
                {
                    "policy_label": policy,
                    "metric": metric,
                    "bucket_index": index,
                    "bucket": label,
                    "episodes": count,
                    "metric_mean": mean([numeric_metric(row, metric) for row in subset]),
                    "settled_landing_rate": sum(bool_value(row, "settled_landing") for row in subset) / count,
                    "deck_miss_rate": sum(bool_value(row, "deck_miss") for row in subset) / count,
                    "hard_contact_rate": sum(bool_value(row, "hard_contact") for row in subset) / count,
                    "ground_crash_rate": sum(bool_value(row, "ground_crash") for row in subset) / count,
                    "timeout_rate": sum(bool_value(row, "time_out") for row in subset) / count,
                }
            )
    return output


def failure_analysis(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy_label"], row["profile"], outcome(row))].append(row)
    output: list[dict[str, object]] = []
    for (policy, profile, result), subset in sorted(groups.items()):
        count = len(subset)
        output.append(
            {
                "policy_label": policy,
                "profile": profile,
                "outcome": result,
                "episodes": count,
                "deck_tilt_max_mean_deg": mean([numeric_metric(row, "deck_tilt_max_deg") for row in subset]),
                "deck_angular_speed_max_mean_radps": mean(
                    [numeric_metric(row, "sea_deck_angular_speed_max") for row in subset]
                ),
                "heave_velocity_max_mean_mps": mean(
                    [numeric_metric(row, "sea_heave_velocity_max_abs") for row in subset]
                ),
                "first_contact_normal_rel_speed_mean_mps": mean(
                    [abs(float_value(row, "first_contact_normal_rel_speed")) for row in subset]
                ),
                "first_contact_tangential_rel_speed_mean_mps": mean(
                    [abs(float_value(row, "first_contact_tangential_rel_speed")) for row in subset]
                ),
                "body_deck_normal_angle_mean_deg": math.degrees(
                    mean([float_value(row, "first_contact_body_deck_normal_angle") for row in subset])
                ),
                "max_contact_impulse_mean_ns": mean([float_value(row, "max_contact_impulse") for row in subset]),
            }
        )
    return output


def boundary_candidates(profile_rows: list[dict[str, object]]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in profile_rows:
        family = str(row["family"])
        if family in {"unknown", "compatibility", "nominal"}:
            continue
        grouped[(str(row["policy_label"]), family)].append(row)
    for (policy, family), family_rows in sorted(grouped.items()):
        ordered = sorted(family_rows, key=lambda row: (int(row["severity_rank"]), str(row["profile"])))
        below = [row for row in ordered if float(row["settled_landing_rate"]) < 0.95]
        in_target = [row for row in ordered if 0.75 <= float(row["settled_landing_rate"]) <= 0.90]
        near_target = [row for row in ordered if 0.75 <= float(row["settled_landing_rate"]) < 0.95]
        chosen = in_target[0] if in_target else (near_target[0] if near_target else (below[0] if below else None))
        if chosen is None:
            statuses.append({"policy_label": policy, "family": family, "status": "no robustness boundary found"})
            continue
        success = float(chosen["settled_landing_rate"])
        candidates.append(
            {
                "policy_label": policy,
                "family": family,
                "candidate_profile": chosen["profile"],
                "severity_rank": chosen["severity_rank"],
                "dominant_motion_metric": DOMINANT_METRIC.get(family, "unknown"),
                "settled_landing_rate": success,
                "deck_miss_rate": chosen["deck_miss_rate"],
                "hard_contact_rate": chosen["hard_contact_rate"],
                "sample_count": chosen["episodes"],
                "candidate_quality": "adaptation_relevant" if 0.75 <= success <= 0.90 else "transition_signal",
            }
        )
    return {"candidates": candidates, "family_status": statuses}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], metavar="LABEL=CSV")
    parser.add_argument("--input_glob", action="append", default=[])
    parser.add_argument("--profiles", type=Path, default=Path("benchmarks/sea_state/profiles.yaml"))
    parser.add_argument("--output_csv", type=Path, default=Path("benchmarks/sea_state/robustness_curves.csv"))
    parser.add_argument("--output_summary", type=Path, default=Path("benchmarks/sea_state/pilot_summary.json"))
    parser.add_argument("--output_profiles", type=Path, default=Path("benchmarks/sea_state/pilot_results.csv"))
    parser.add_argument("--output_boundary", type=Path, default=Path("benchmarks/sea_state/boundary_candidates.json"))
    parser.add_argument("--output_failure", type=Path, default=Path("benchmarks/sea_state/failure_analysis.csv"))
    args = parser.parse_args()

    profiles = load_sea_state_profiles(args.profiles) if args.profiles.exists() else None
    rows = load_rows(args.input, args.input_glob)
    profile_rows = summarize_profiles(rows, profiles)
    curves = robustness_buckets(rows)
    failures = failure_analysis(rows)
    boundaries = boundary_candidates(profile_rows)

    write_csv(args.output_profiles, profile_rows)
    write_csv(args.output_csv, curves)
    write_csv(args.output_failure, failures)
    summary = {
        "episodes": len(rows),
        "policies": sorted({row["policy_label"] for row in rows}),
        "profiles": profile_rows,
        "boundary": boundaries,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")
    args.output_boundary.write_text(json.dumps(boundaries, indent=2) + "\n")
    print(json.dumps(boundaries, indent=2))
    print(f"[INFO] wrote {args.output_profiles}, {args.output_csv}, {args.output_failure}")


if __name__ == "__main__":
    main()

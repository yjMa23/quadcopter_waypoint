#!/usr/bin/env python3
"""Aggregate Phase-6C per-episode CSV files into the frozen benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


BOOL_FIELDS = {
    "align_success",
    "success",
    "crash",
    "terminated",
    "time_out",
    "contact_success",
    "settled_landing",
    "hard_contact",
    "ground_crash",
    "deck_miss",
    "first_contact_seen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--csv-dir", type=Path, default=Path("logs/rl_games/quadcopter_ship_landing_physical_deck_attitude"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_value(key: str, value: str) -> Any:
    if key in BOOL_FIELDS:
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv(path: Path, seed: int | None = None) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = [{key: parse_value(key, value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if seed is not None:
        for row in rows:
            row["seed"] = seed
    return rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successful = [row for row in rows if row["settled_landing"]]
    first_contact_successful = [row for row in successful if row["first_contact_seen"]]

    def rate(key: str) -> float:
        return sum(bool(row[key]) for row in rows) / count

    def mean(key: str, source: list[dict[str, Any]] = rows) -> float:
        return sum(float(row[key]) for row in source) / len(source) if source else float("nan")

    return {
        "episodes": count,
        "contact_success_rate": rate("contact_success"),
        "settled_landing_rate": rate("settled_landing"),
        "hard_contact_rate": rate("hard_contact"),
        "ground_crash_rate": rate("ground_crash"),
        "deck_miss_rate": rate("deck_miss"),
        "timeout_rate": rate("time_out"),
        "successful_first_contact_xy_error_deck_frame_p95_m": percentile(
            [float(row["first_contact_xy_error_deck_frame"]) for row in first_contact_successful], 95.0
        ),
        "successful_first_contact_normal_relative_speed_p95_mps": percentile(
            [abs(float(row["first_contact_normal_rel_speed"])) for row in first_contact_successful], 95.0
        ),
        "successful_first_contact_body_deck_normal_angle_p95_deg": percentile(
            [math.degrees(float(row["first_contact_body_deck_normal_angle"])) for row in first_contact_successful],
            95.0,
        ),
        "successful_touchdown_distance_p95_m": percentile(
            [float(row["touchdown_distance"]) for row in successful], 95.0
        ),
        "maximum_penetration_max_m": max(float(row["maximum_penetration"]) for row in rows),
        "max_contact_force_mean_success_n": mean("max_contact_force", successful),
        "max_contact_impulse_mean_success_ns": mean("max_contact_impulse", successful),
        "settle_time_mean_success_s": mean("settle_time", successful),
        "deck_tilt_max_deg": max(math.degrees(float(row["first_contact_deck_tilt"])) for row in rows),
        "deck_angular_speed_max_radps": max(float(row["first_contact_deck_angular_speed"]) for row in rows),
        "deck_position_consistency_error_max_m": max(
            float(row["max_deck_position_consistency_error"]) for row in rows
        ),
        "deck_orientation_consistency_error_max_deg": math.degrees(
            max(float(row["max_deck_orientation_consistency_error"]) for row in rows)
        ),
        "deck_linear_velocity_consistency_error_max_mps": max(
            float(row["max_deck_linear_velocity_consistency_error"]) for row in rows
        ),
        "deck_angular_velocity_consistency_error_max_radps": max(
            float(row["max_deck_angular_velocity_consistency_error"]) for row in rows
        ),
    }


def bucket_summaries(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {bucket: summarize(bucket_rows) for bucket, bucket_rows in sorted(grouped.items())}


def compact_stage(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    summary = summarize(rows)
    return {
        "csv": str(path),
        "episodes": summary["episodes"],
        "settled_landing_rate": summary["settled_landing_rate"],
        "hard_contact_rate": summary["hard_contact_rate"],
        "ground_crash_rate": summary["ground_crash_rate"],
        "timeout_rate": summary["timeout_rate"],
        "first_contact_xy_p95_m": summary["successful_first_contact_xy_error_deck_frame_p95_m"],
        "first_contact_normal_speed_p95_mps": summary[
            "successful_first_contact_normal_relative_speed_p95_mps"
        ],
    }


def main() -> None:
    args = parse_args()
    csv_dir = args.csv_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    final_paths = {seed: csv_dir / f"p6c_final_seed{seed}.csv" for seed in (42, 43, 44)}
    zero_paths = {seed: csv_dir / f"zero_tilt_final_seed{seed}.csv" for seed in (42, 43, 44)}
    final_by_seed = {seed: read_csv(path, seed) for seed, path in final_paths.items()}
    zero_by_seed = {seed: read_csv(path, seed) for seed, path in zero_paths.items()}
    final_rows = [row for seed in (42, 43, 44) for row in final_by_seed[seed]]
    zero_rows = [row for seed in (42, 43, 44) for row in zero_by_seed[seed]]

    training_dir = csv_dir / "2026-08-03_22-15-55" / "nn"
    candidate_paths = {
        "ep1000": training_dir / "last_quadcopter_ship_landing_physical_deck_attitude_ep_1000_rew_49.98572.pth",
        "ep1010": training_dir / "last_quadcopter_ship_landing_physical_deck_attitude_ep_1010_rew_46.976643.pth",
        "ep1020": training_dir / "last_quadcopter_ship_landing_physical_deck_attitude_ep_1020_rew_52.032288.pth",
    }
    candidate_csvs = {
        "ep1000": csv_dir / "candidate_ep1000_seed42.csv",
        "ep1010": csv_dir / "candidate_ep1010_seed42.csv",
        "ep1020": csv_dir / "candidate_ep1020_seed42.csv",
    }
    candidates = {}
    for name in candidate_paths:
        candidate_summary = compact_stage(candidate_csvs[name])
        candidate_summary.update(
            {
                "checkpoint": str(candidate_paths[name]),
                "checkpoint_sha256": sha256_file(candidate_paths[name]),
            }
        )
        candidates[name] = candidate_summary

    command_template = (
        "python scripts/rl_games/eval_metrics.py "
        "--task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 "
        f"--checkpoint {checkpoint.relative_to(Path.cwd())} --num_envs=64 --episodes=256 "
        "--seed=<42|43|44> --csv=<OUTPUT.csv> --headless "
        "env.deck_roll_amplitude_min_deg=0.0 env.deck_roll_amplitude_max_deg=5.0 "
        "env.deck_pitch_amplitude_min_deg=0.0 env.deck_pitch_amplitude_max_deg=5.0 "
        "env.deck_roll_frequency_min=0.08 env.deck_roll_frequency_max=0.15 "
        "env.deck_pitch_frequency_min=0.08 env.deck_pitch_frequency_max=0.15"
    )
    benchmark = {
        "phase": "P6C-PhysicalDeckAttitude",
        "task_id": "Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0",
        "rl_games_experiment": "quadcopter_ship_landing_physical_deck_attitude",
        "code_commit": args.code_commit,
        "checkpoint": str(checkpoint.relative_to(Path.cwd())),
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluation_command_template": command_template,
        "evaluation": {
            "seeds": [42, 43, 44],
            "episodes_per_seed": 256,
            "num_envs": 64,
            "headless": True,
            "curriculum": {
                "roll_amplitude_deg": [0.0, 5.0],
                "pitch_amplitude_deg": [0.0, 5.0],
                "roll_frequency_hz": [0.08, 0.15],
                "pitch_frequency_hz": [0.08, 0.15],
                "yaw_oscillation": False,
            },
            "aggregate": summarize(final_rows),
            "per_seed": {str(seed): summarize(final_by_seed[seed]) for seed in (42, 43, 44)},
            "deck_tilt_buckets": bucket_summaries(final_rows, "deck_tilt_bucket"),
            "deck_angular_speed_buckets": bucket_summaries(final_rows, "deck_angular_speed_bucket"),
            "csv_files": {str(seed): str(path) for seed, path in final_paths.items()},
        },
        "zero_tilt_compatibility": {
            "aggregate": summarize(zero_rows),
            "per_seed": {str(seed): summarize(zero_by_seed[seed]) for seed in (42, 43, 44)},
            "csv_files": {str(seed): str(path) for seed, path in zero_paths.items()},
        },
        "curriculum_evaluations": {
            "stage_a_zero_tilt": compact_stage(csv_dir / "zero_tilt_metrics_smoke_seed42.csv"),
            "stage_b_roll_2deg_zero_shot": compact_stage(csv_dir / "stage_b_zero_shot_2deg_seed42.csv"),
            "stage_c_roll_pitch_3deg_zero_shot": compact_stage(csv_dir / "stage_c_zero_shot_3deg_seed42.csv"),
            "stage_d_roll_pitch_5deg_zero_shot": compact_stage(csv_dir / "stage_d_zero_shot_5deg_seed42.csv"),
            "stage_d_training": {
                "run_directory": str(csv_dir / "2026-08-03_22-15-55"),
                "epochs": [991, 1020],
                "num_envs": 256,
                "learning_rate_config": 1.0e-4,
                "horizon_length": 24,
                "minibatch_size": 384,
                "candidate_comparison": candidates,
                "selection": "All fine-tuned candidates regressed; retain the expanded P6B ep990 checkpoint.",
            },
        },
        "physics_validation": {
            "one_env_report": "benchmarks/phase6c_physical_deck_attitude/physics_check_1env.json",
            "sixteen_env_report": "benchmarks/phase6c_physical_deck_attitude/physics_check_16env.json",
            "gui_report": "benchmarks/phase6c_physical_deck_attitude/gui_check.json",
            "human_visual_inspection_completed": False,
        },
        "metric_definitions": {
            "surface_point_velocity": "deck center linear velocity + deck world angular velocity cross (surface point - deck center)",
            "normal_relative_speed": "dot(robot bottom-point velocity - deck surface-point velocity, deck normal)",
            "tangential_relative_speed": "norm(relative velocity - normal_relative_speed * deck normal)",
            "xy_error": "norm of robot bottom point x/y coordinates in deck frame",
            "surface_clearance": "robot bottom-point deck-frame z minus deck half thickness",
            "body_deck_normal_angle": "acos(clamp(dot(robot body-z world, deck normal world), -1, 1))",
            "settled_landing": "attitude-aware safe physical deck contact held for the configured settle steps",
            "hard_contact": "deck contact exceeding normal-speed, force/impulse, or penetration threshold",
        },
        "acceptance": {
            "settled_landing_rate_min": 0.92,
            "ground_crash_rate_max": 0.01,
            "hard_contact_rate_max": 0.02,
            "timeout_rate_max": 0.03,
            "first_contact_xy_p95_m_max": 0.12,
            "first_contact_normal_speed_p95_mps_max": 0.45,
            "body_deck_normal_angle_p95_deg_max": 10.0,
            "maximum_penetration_m_max": 0.03,
        },
    }
    aggregate = benchmark["evaluation"]["aggregate"]
    acceptance = benchmark["acceptance"]
    benchmark["acceptance_result"] = {
        "passed": (
            aggregate["settled_landing_rate"] >= acceptance["settled_landing_rate_min"]
            and aggregate["ground_crash_rate"] <= acceptance["ground_crash_rate_max"]
            and aggregate["hard_contact_rate"] <= acceptance["hard_contact_rate_max"]
            and aggregate["timeout_rate"] <= acceptance["timeout_rate_max"]
            and aggregate["successful_first_contact_xy_error_deck_frame_p95_m"]
            <= acceptance["first_contact_xy_p95_m_max"]
            and aggregate["successful_first_contact_normal_relative_speed_p95_mps"]
            <= acceptance["first_contact_normal_speed_p95_mps_max"]
            and aggregate["successful_first_contact_body_deck_normal_angle_p95_deg"]
            <= acceptance["body_deck_normal_angle_p95_deg_max"]
            and aggregate["maximum_penetration_max_m"] <= acceptance["maximum_penetration_m_max"]
        )
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, indent=2, allow_nan=False) + "\n")
    print(json.dumps(benchmark["evaluation"]["aggregate"], indent=2))
    print(f"acceptance passed: {benchmark['acceptance_result']['passed']}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

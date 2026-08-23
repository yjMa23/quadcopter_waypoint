#!/usr/bin/env python3
"""Offline S0/S1/D1 comparison for the M2 D1 descent-reward gating sanity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_m2_reward_compatibility import (
    DIRECT_CAN_LAND_TERMS,
    PHASE_SENSITIVE_TERMS,
    SNAPSHOT_ITERATIONS,
    load_scalar_series,
    previous_or_exact,
    snapshot_contributions,
    summarize_deterministic_rows,
    write_reward_terms_csv,
)


DEFAULT_RUNS = {
    "S0": Path("logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-00-25/summaries"),
    "S1": Path("logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-17-58/summaries"),
    "D1": Path("logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_17-15-36/summaries"),
}

DEFAULT_EVALUATIONS = {
    "S0": {
        10: Path("benchmarks/px4_hierarchical_training/sanity_ep10_seed145.csv"),
        20: Path("benchmarks/px4_hierarchical_training/sanity_ep20_seed145.csv"),
        30: Path("benchmarks/px4_hierarchical_training/sanity_ep30_seed145.csv"),
    },
    "S1": {
        10: Path("benchmarks/px4_hierarchical_training/sanity_s1_ep10_seed145.csv"),
        20: Path("benchmarks/px4_hierarchical_training/sanity_s1_ep20_seed145.csv"),
        30: Path("benchmarks/px4_hierarchical_training/sanity_s1_ep30_seed145.csv"),
    },
    "D1": {
        10: Path("benchmarks/px4_hierarchical_training/sanity_d1_ep10_seed145.csv"),
        20: Path("benchmarks/px4_hierarchical_training/sanity_d1_ep20_seed145.csv"),
        30: Path("benchmarks/px4_hierarchical_training/sanity_d1_ep30_seed145.csv"),
    },
}

TRAINING_TAGS = {
    "reward": "rewards/iter",
    "episode_length_steps": "episode_lengths/iter",
}

DETERMINISTIC_MEANS = {
    "reference_saturation": "reference_saturation_ratio",
    "controller_tracking_mean": "controller_velocity_tracking_error_mean",
    "controller_acceleration_saturation": "controller_acceleration_saturation_ratio",
    "controller_tilt_saturation": "controller_tilt_saturation_ratio",
    "controller_thrust_saturation": "controller_thrust_saturation_ratio",
    "controller_body_rate_saturation": "controller_body_rate_saturation_ratio",
    "controller_moment_saturation": "controller_moment_saturation_ratio",
    "action_t1_mean": "action_t1_mean",
    "action_t2_mean": "action_t2_mean",
    "action_normal_mean": "action_normal_mean",
    "action_t1_std": "action_t1_std",
    "action_t2_std": "action_t2_std",
    "action_normal_std": "action_normal_std",
}

D1_TRAINING_GATE_TAGS = {
    "reward_descent_phase_active_ratio": "Episode/Metrics/m2_reward_descent_phase_active_ratio",
    "can_land_but_reward_gate_inactive_ratio": "Episode/Metrics/m2_can_land_but_reward_gate_inactive_ratio",
    "reward_gate_transition_count": "Episode/Metrics/m2_reward_gate_transition_count",
    "terminal_outside_align_after_latch_rate": "Episode/Metrics/m2_terminal_outside_align_after_latch_rate",
    "timeout_outside_align_after_latch_rate": "Episode/Metrics/m2_timeout_outside_align_after_latch_rate",
    "reward_gate_horizontal_error_violation_ratio": "Episode/Metrics/m2_reward_gate_horizontal_error_violation_ratio",
    "reward_gate_horizontal_speed_violation_ratio": "Episode/Metrics/m2_reward_gate_horizontal_speed_violation_ratio",
    "reward_gate_attitude_violation_ratio": "Episode/Metrics/m2_reward_gate_attitude_violation_ratio",
    "reward_gate_too_high_violation_ratio": "Episode/Metrics/m2_reward_gate_too_high_violation_ratio",
}

D1_GATE_FIELDS = (
    "reward_descent_phase_active_ratio",
    "can_land_but_reward_gate_inactive_ratio",
    "reward_gate_transition_count",
    "reward_gate_horizontal_error_violation_ratio",
    "reward_gate_horizontal_speed_violation_ratio",
    "reward_gate_attitude_violation_ratio",
    "reward_gate_too_high_violation_ratio",
)


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def mean_numeric(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") not in {"", "nan", "NaN"}]
    return float("nan") if not values else sum(values) / len(values)


def rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(as_bool(row[key]) for row in rows) / len(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No evaluator rows in {path}")
    return rows


def write_snapshot_comparison(
    output_path: Path,
    runs: dict[str, dict[str, dict[int, float]]],
    evaluations: dict[str, dict[int, list[dict[str, str]]]],
) -> None:
    metrics = (
        "reward",
        "episode_length_steps",
        "reward_per_step",
        "align",
        "settled",
        "crash",
        "deck_miss",
        "ground_crash",
        "hard_contact",
        "timeout",
        *DETERMINISTIC_MEANS.keys(),
    )
    columns = ["metric"] + [f"{run}_ep{iteration}" for run in runs for iteration in SNAPSHOT_ITERATIONS]
    table: dict[str, dict[str, float | str]] = {metric: {"metric": metric} for metric in metrics}

    for run_name, scalars in runs.items():
        for iteration in SNAPSHOT_ITERATIONS:
            column = f"{run_name}_ep{iteration}"
            _, reward = previous_or_exact(scalars.get(TRAINING_TAGS["reward"], {}), iteration)
            _, length = previous_or_exact(scalars.get(TRAINING_TAGS["episode_length_steps"], {}), iteration)
            table["reward"][column] = "" if reward is None else reward
            table["episode_length_steps"][column] = "" if length is None else length
            table["reward_per_step"][column] = "" if reward is None or not length else reward / length

            rows = evaluations[run_name][iteration]
            table["align"][column] = rate(rows, "align_success")
            table["settled"][column] = rate(rows, "settled_landing")
            table["crash"][column] = rate(rows, "crash")
            table["deck_miss"][column] = rate(rows, "deck_miss")
            table["ground_crash"][column] = rate(rows, "ground_crash")
            table["hard_contact"][column] = rate(rows, "hard_contact")
            table["timeout"][column] = rate(rows, "time_out")
            for metric, field in DETERMINISTIC_MEANS.items():
                table[metric][column] = mean_numeric(rows, field)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table[metric] for metric in metrics)


def write_training_gate_snapshot(output_path: Path, d1_scalars: dict[str, dict[int, float]]) -> None:
    fieldnames = ["metric"] + [f"D1_ep{iteration}" for iteration in SNAPSHOT_ITERATIONS]
    rows_out: list[dict[str, float | str]] = []
    for metric, tag in D1_TRAINING_GATE_TAGS.items():
        row: dict[str, float | str] = {"metric": metric}
        series = d1_scalars.get(tag, {})
        for iteration in SNAPSHOT_ITERATIONS:
            _, value = previous_or_exact(series, iteration)
            row[f"D1_ep{iteration}"] = "" if value is None else value
        rows_out.append(row)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


def write_reward_attribution(
    output_path: Path, runs: dict[str, dict[str, dict[int, float]]]
) -> None:
    fieldnames = [
        "run",
        "iteration",
        "reward",
        "episode_length_steps",
        "reward_per_step",
        "phase_sensitive_sum",
        "phase_sensitive_abs",
        "total_reward_term_abs",
        "phase_sensitive_magnitude_share_pct",
        "predicted_pad_error",
        "predicted_pad_error_share_pct",
        "contact_clearance",
        "contact_clearance_share_pct",
        "height_tracking",
        "height_tracking_share_pct",
        "post_align_descent",
        "descent_horizontal_rel_vel",
        "near_pad_horizontal_rel_vel",
        "center_precision",
        "center_precision_square",
    ]
    rows_out: list[dict[str, float | str | int]] = []
    for run_name, scalars in runs.items():
        for iteration in SNAPSHOT_ITERATIONS:
            contributions, _, length = snapshot_contributions(scalars, iteration)
            _, reward = previous_or_exact(scalars.get("rewards/iter", {}), iteration)
            total_abs = sum(abs(value) for value in contributions.values())
            phase_values = [contributions.get(term, 0.0) for term in PHASE_SENSITIVE_TERMS]
            row: dict[str, float | str | int] = {
                "run": run_name,
                "iteration": iteration,
                "reward": "" if reward is None else reward,
                "episode_length_steps": "" if length is None else length,
                "reward_per_step": "" if reward is None or not length else reward / length,
                "phase_sensitive_sum": sum(phase_values),
                "phase_sensitive_abs": sum(abs(value) for value in phase_values),
                "total_reward_term_abs": total_abs,
                "phase_sensitive_magnitude_share_pct": 0.0
                if total_abs == 0.0
                else 100.0 * sum(abs(value) for value in phase_values) / total_abs,
            }
            for term in (
                "predicted_pad_error",
                "contact_clearance",
                "height_tracking",
                "post_align_descent",
                "descent_horizontal_rel_vel",
                "near_pad_horizontal_rel_vel",
                "center_precision",
                "center_precision_square",
            ):
                value = contributions.get(term, 0.0)
                row[term] = value
                if term in {"predicted_pad_error", "contact_clearance", "height_tracking"}:
                    row[f"{term}_share_pct"] = 0.0 if total_abs == 0.0 else 100.0 * abs(value) / total_abs
            rows_out.append(row)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


def write_deterministic_gate_summary(
    output_path: Path, evaluations: dict[str, dict[int, list[dict[str, str]]]]
) -> None:
    fieldnames = [
        "run",
        "iteration",
        "episodes",
        "align_rate",
        "settled_rate",
        "crash_rate",
        "deck_miss_rate",
        "ground_crash_rate",
        "hard_contact_rate",
        "timeout_rate",
        "reference_saturation_ratio",
        "controller_tracking_mean",
        "aligned_episodes",
        "aligned_terminal_outside_rate",
        "aligned_timeout_episodes",
        "aligned_timeout_terminal_outside_rate",
        "reward_descent_phase_active_ratio_all",
        "reward_descent_phase_active_ratio_aligned",
        "can_land_but_reward_gate_inactive_ratio_all",
        "can_land_but_reward_gate_inactive_ratio_aligned",
        "aligned_gate_inactive_gt_0_5_rate",
        "aligned_gate_inactive_gt_0_8_rate",
        "reward_gate_transition_count_all",
        "reward_gate_transition_count_aligned",
        "reward_gate_horizontal_error_violation_ratio_aligned",
        "reward_gate_horizontal_speed_violation_ratio_aligned",
        "reward_gate_attitude_violation_ratio_aligned",
        "reward_gate_too_high_violation_ratio_aligned",
        "mean_descent_speed_all",
        "mean_descent_speed_aligned",
        "mean_descent_speed_aligned_timeouts",
    ]
    rows_out: list[dict[str, float | str | int]] = []
    for run_name, checkpoints in evaluations.items():
        for iteration, rows in checkpoints.items():
            aligned = [row for row in rows if as_bool(row["align_success"])]
            aligned_timeouts = [row for row in aligned if as_bool(row["time_out"])]
            post_latch = summarize_deterministic_rows(rows)
            row_out: dict[str, float | str | int] = {
                "run": run_name,
                "iteration": iteration,
                "episodes": len(rows),
                "align_rate": rate(rows, "align_success"),
                "settled_rate": rate(rows, "settled_landing"),
                "crash_rate": rate(rows, "crash"),
                "deck_miss_rate": rate(rows, "deck_miss"),
                "ground_crash_rate": rate(rows, "ground_crash"),
                "hard_contact_rate": rate(rows, "hard_contact"),
                "timeout_rate": rate(rows, "time_out"),
                "reference_saturation_ratio": mean_numeric(rows, "reference_saturation_ratio"),
                "controller_tracking_mean": mean_numeric(rows, "controller_velocity_tracking_error_mean"),
                "aligned_episodes": len(aligned),
                "aligned_terminal_outside_rate": post_latch["aligned_terminal_outside_radius_rate"],
                "aligned_timeout_episodes": len(aligned_timeouts),
                "aligned_timeout_terminal_outside_rate": post_latch[
                    "aligned_timeout_terminal_outside_radius_rate"
                ],
                "mean_descent_speed_all": mean_numeric(rows, "mean_descent_speed"),
                "mean_descent_speed_aligned": mean_numeric(aligned, "mean_descent_speed") if aligned else "",
                "mean_descent_speed_aligned_timeouts": mean_numeric(aligned_timeouts, "mean_descent_speed")
                if aligned_timeouts
                else "",
            }
            if run_name == "D1":
                for field in D1_GATE_FIELDS:
                    row_out[f"{field}_all"] = mean_numeric(rows, field)
                    row_out[f"{field}_aligned"] = mean_numeric(aligned, field) if aligned else ""
                inactive_aligned = [float(row["can_land_but_reward_gate_inactive_ratio"]) for row in aligned]
                row_out["aligned_gate_inactive_gt_0_5_rate"] = (
                    sum(value > 0.5 for value in inactive_aligned) / len(inactive_aligned) if inactive_aligned else ""
                )
                row_out["aligned_gate_inactive_gt_0_8_rate"] = (
                    sum(value > 0.8 for value in inactive_aligned) / len(inactive_aligned) if inactive_aligned else ""
                )
            rows_out.append(row_out)

    # Some D1-only source keys are deliberately omitted from fieldnames; map the four selected aligned fields.
    for row in rows_out:
        if row["run"] != "D1":
            continue
        row["reward_descent_phase_active_ratio_all"] = row.pop("reward_descent_phase_active_ratio_all")
        row["reward_descent_phase_active_ratio_aligned"] = row.pop("reward_descent_phase_active_ratio_aligned")
        row["can_land_but_reward_gate_inactive_ratio_all"] = row.pop("can_land_but_reward_gate_inactive_ratio_all")
        row["can_land_but_reward_gate_inactive_ratio_aligned"] = row.pop(
            "can_land_but_reward_gate_inactive_ratio_aligned"
        )
        row["reward_gate_transition_count_all"] = row.pop("reward_gate_transition_count_all")
        row["reward_gate_transition_count_aligned"] = row.pop("reward_gate_transition_count_aligned")
        row["reward_gate_horizontal_error_violation_ratio_aligned"] = row.pop(
            "reward_gate_horizontal_error_violation_ratio_aligned"
        )
        row["reward_gate_horizontal_speed_violation_ratio_aligned"] = row.pop(
            "reward_gate_horizontal_speed_violation_ratio_aligned"
        )
        row["reward_gate_attitude_violation_ratio_aligned"] = row.pop(
            "reward_gate_attitude_violation_ratio_aligned"
        )
        row["reward_gate_too_high_violation_ratio_aligned"] = row.pop(
            "reward_gate_too_high_violation_ratio_aligned"
        )
        # Remove D1-only all-population cause ratios not requested in the compact evidence table.
        for key in list(row):
            if key.endswith("_all") and key.startswith("reward_gate_") and key not in {
                "reward_gate_transition_count_all"
            }:
                row.pop(key)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)


def write_commands(output_path: Path, args: argparse.Namespace) -> None:
    output_path.write_text(
        "source /home/j/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && "
        "export PYTHONPATH=source/quadcopter_waypoint && "
        "python scripts/rl_games/analyze_m2_d1_reward_gating.py "
        f"--s0 {args.s0} --s1 {args.s1} --d1 {args.d1} --output-dir {args.output_dir}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0", type=Path, default=DEFAULT_RUNS["S0"])
    parser.add_argument("--s1", type=Path, default=DEFAULT_RUNS["S1"])
    parser.add_argument("--d1", type=Path, default=DEFAULT_RUNS["D1"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/px4_hierarchical_training/d1_reward_gating_sanity"),
    )
    args = parser.parse_args()

    run_paths = {"S0": args.s0, "S1": args.s1, "D1": args.d1}
    for name, path in run_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} TensorBoard summary directory not found: {path}")
    for run_name, checkpoints in DEFAULT_EVALUATIONS.items():
        for iteration, path in checkpoints.items():
            if not path.exists():
                raise FileNotFoundError(f"{run_name} ep{iteration} evaluator CSV not found: {path}")

    runs = {name: load_scalar_series(path) for name, path in run_paths.items()}
    evaluations = {
        run_name: {iteration: load_rows(path) for iteration, path in checkpoints.items()}
        for run_name, checkpoints in DEFAULT_EVALUATIONS.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_commands(args.output_dir / "commands.txt", args)
    write_snapshot_comparison(args.output_dir / "snapshot_comparison.csv", runs, evaluations)
    write_training_gate_snapshot(args.output_dir / "training_gate_snapshot.csv", runs["D1"])
    write_reward_terms_csv(args.output_dir / "reward_terms_s0_s1_d1.csv", runs)
    write_reward_attribution(args.output_dir / "reward_attribution.csv", runs)
    write_deterministic_gate_summary(args.output_dir / "deterministic_gate_summary.csv", evaluations)
    print(f"Wrote D1 reward-gating analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

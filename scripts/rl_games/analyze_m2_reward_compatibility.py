#!/usr/bin/env python3
"""Offline TensorBoard analysis for the M2 D0 reward compatibility audit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MAX_EPISODE_LENGTH_S = 10.0
M2_STEP_DT_S = 0.04
SNAPSHOT_ITERATIONS = (10, 20, 30)
REWARD_PREFIX = "Episode/Episode_Reward/"

DEFAULT_RUNS = {
    "S0": Path("logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-00-25/summaries"),
    "S1": Path("logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-17-58/summaries"),
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
}

ALIGN_RADIUS_M = 0.25

COMPARISON_TAGS = {
    "training_reward": "rewards/iter",
    "episode_length_steps": "episode_lengths/iter",
    "align_success_rate": "Episode/Metrics/align_success_rate",
    "landing_success_rate": "Episode/Metrics/landing_success_rate",
    "settled_landing_rate": "Episode/Metrics/m2_settled_landing_rate",
    "hard_contact_rate": "Episode/Metrics/m2_hard_contact_rate",
    "ground_crash_rate": "Episode/Metrics/m2_ground_crash_rate",
    "deck_miss_rate": "Episode/Metrics/m2_deck_miss_rate",
    "termination_crash_signal": "Episode/Episode_Termination/crash",
    "termination_timeout_signal": "Episode/Episode_Termination/time_out",
    "reference_norm_mean": "Episode/Metrics/m2_relative_velocity_reference_norm_mean",
    "reference_saturation_ratio": "Episode/Metrics/m2_reference_saturation_ratio",
    "controller_tracking_error_mean": "Episode/Metrics/m2_controller_velocity_tracking_error_mean",
    "controller_acceleration_saturation_ratio": "Episode/Metrics/m2_controller_acceleration_saturation_ratio",
    "controller_tilt_saturation_ratio": "Episode/Metrics/m2_controller_tilt_saturation_ratio",
    "controller_thrust_saturation_ratio": "Episode/Metrics/m2_controller_thrust_saturation_ratio",
    "controller_body_rate_saturation_ratio": "Episode/Metrics/m2_controller_body_rate_saturation_ratio",
    "controller_moment_saturation_ratio": "Episode/Metrics/m2_controller_moment_saturation_ratio",
    "action_t1_mean": "Episode/Metrics/m2_action_t1_mean",
    "action_t1_std": "Episode/Metrics/m2_action_t1_std",
    "action_t2_mean": "Episode/Metrics/m2_action_t2_mean",
    "action_t2_std": "Episode/Metrics/m2_action_t2_std",
    "action_normal_mean": "Episode/Metrics/m2_action_normal_mean",
    "action_normal_std": "Episode/Metrics/m2_action_normal_std",
}

TREND_TAGS = {
    "align": "Episode/Metrics/align_success_rate",
    "crash": "Episode/Episode_Termination/crash",
    "deck_miss": "Episode/Metrics/m2_deck_miss_rate",
    "timeout": "Episode/Episode_Termination/time_out",
    "episode_length": "episode_lengths/iter",
}

DIRECT_CAN_LAND_TERMS = {
    "post_align_descent",
    "descent_horizontal_rel_vel",
    "near_pad_horizontal_rel_vel",
    "predicted_pad_error",
    "contact_clearance",
    "center_precision",
    "center_precision_square",
}
PHASE_SENSITIVE_TERMS = DIRECT_CAN_LAND_TERMS | {"height_tracking"}


def restore_episode_contribution(logged_reward_rate: float, max_episode_length_s: float = MAX_EPISODE_LENGTH_S) -> float:
    """Undo the environment's Episode_Reward division by max_episode_length_s."""
    return logged_reward_rate * max_episode_length_s


def length_normalized(value: float, episode_length_steps: float | None) -> float | None:
    """Return an aggregate per-step diagnostic, or None when episode length is unavailable."""
    if episode_length_steps is None or episode_length_steps <= 0.0:
        return None
    return value / episode_length_steps


def previous_or_exact(series: dict[int, float], iteration: int) -> tuple[int | None, float | None]:
    """Return the exact value or the latest earlier value, preserving its source iteration."""
    eligible = [step for step in series if step <= iteration]
    if not eligible:
        return None, None
    source_step = max(eligible)
    return source_step, series[source_step]


def relative_magnitude_percent(values: dict[str, float]) -> dict[str, float]:
    """Return each term's absolute contribution as a percent of total absolute magnitude."""
    denominator = sum(abs(value) for value in values.values())
    if denominator == 0.0:
        return {key: 0.0 for key in values}
    return {key: 100.0 * abs(value) / denominator for key, value in values.items()}


def negative_ranks(values: dict[str, float]) -> dict[str, int | None]:
    """Rank negative terms by descending absolute magnitude; non-negative terms receive no rank."""
    negatives = sorted(
        ((key, abs(value)) for key, value in values.items() if value < 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    ranks: dict[str, int | None] = {key: None for key in values}
    for rank, (key, _) in enumerate(negatives, start=1):
        ranks[key] = rank
    return ranks


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def pearson(values_x: Iterable[float], values_y: Iterable[float]) -> float | None:
    x = list(values_x)
    y = list(values_y)
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y, strict=True)) / denominator


def spearman(values_x: Iterable[float], values_y: Iterable[float]) -> float | None:
    x = list(values_x)
    y = list(values_y)
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(_rankdata(x), _rankdata(y))


def load_scalar_series(summary_dir: Path) -> dict[str, dict[int, float]]:
    accumulator = EventAccumulator(str(summary_dir))
    accumulator.Reload()
    result: dict[str, dict[int, float]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        result[tag] = {int(event.step): float(event.value) for event in accumulator.Scalars(tag)}
    return result


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes"}


def summarize_deterministic_rows(
    rows: list[dict[str, str]], align_radius_m: float = ALIGN_RADIUS_M
) -> dict[str, float | int]:
    """Summarize existing deterministic evaluator rows for post-latch terminal drift evidence."""
    aligned = [row for row in rows if _as_bool(row["align_success"])]
    aligned_timeouts = [row for row in aligned if _as_bool(row["time_out"])]
    aligned_crashes = [row for row in aligned if _as_bool(row["crash"])]
    aligned_settled = [row for row in aligned if _as_bool(row["settled_landing"])]

    def terminal_xy(row: dict[str, str]) -> float:
        return float(row["terminal_horizontal_error"])

    def outside(row: dict[str, str]) -> bool:
        return terminal_xy(row) >= align_radius_m

    def safe_ratio(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else numerator / denominator

    aligned_outside = sum(outside(row) for row in aligned)
    timeout_outside = sum(outside(row) for row in aligned_timeouts)
    return {
        "episodes": len(rows),
        "aligned_episodes": len(aligned),
        "aligned_rate": safe_ratio(len(aligned), len(rows)),
        "aligned_terminal_outside_radius": aligned_outside,
        "aligned_terminal_outside_radius_rate": safe_ratio(aligned_outside, len(aligned)),
        "aligned_timeout_episodes": len(aligned_timeouts),
        "aligned_timeout_rate": safe_ratio(len(aligned_timeouts), len(aligned)),
        "aligned_timeout_terminal_outside_radius": timeout_outside,
        "aligned_timeout_terminal_outside_radius_rate": safe_ratio(timeout_outside, len(aligned_timeouts)),
        "aligned_crash_episodes": len(aligned_crashes),
        "aligned_crash_rate": safe_ratio(len(aligned_crashes), len(aligned)),
        "aligned_settled_episodes": len(aligned_settled),
        "aligned_terminal_xy_error_mean_m": 0.0
        if not aligned
        else sum(terminal_xy(row) for row in aligned) / len(aligned),
        "aligned_timeout_terminal_xy_error_mean_m": 0.0
        if not aligned_timeouts
        else sum(terminal_xy(row) for row in aligned_timeouts) / len(aligned_timeouts),
        "aligned_timeout_terminal_clearance_mean_m": 0.0
        if not aligned_timeouts
        else sum(float(row["terminal_surface_clearance"]) for row in aligned_timeouts) / len(aligned_timeouts),
    }


def common_series_pairs(
    left: dict[int, float], right: dict[int, float]
) -> tuple[list[int], list[float], list[float]]:
    steps = sorted(set(left).intersection(right))
    return steps, [left[step] for step in steps], [right[step] for step in steps]


def _format(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_reward_terms_csv(
    output_path: Path, runs: dict[str, dict[str, dict[int, float]]]
) -> None:
    fieldnames = [
        "run",
        "iteration",
        "term",
        "logged_episode_reward_rate",
        "approx_episode_contribution",
        "absolute_contribution",
        "relative_total_magnitude_pct",
        "negative_rank",
        "episode_length_source_iteration",
        "episode_length_steps",
        "episode_length_seconds",
        "approx_contribution_per_step",
        "approx_contribution_per_second",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run_name, scalars in runs.items():
            reward_tags = sorted(tag for tag in scalars if tag.startswith(REWARD_PREFIX))
            iterations = sorted({step for tag in reward_tags for step in scalars[tag]})
            episode_lengths = scalars.get("episode_lengths/iter", {})
            for iteration in iterations:
                contributions = {
                    tag.removeprefix(REWARD_PREFIX): restore_episode_contribution(scalars[tag][iteration])
                    for tag in reward_tags
                    if iteration in scalars[tag]
                }
                percentages = relative_magnitude_percent(contributions)
                ranks = negative_ranks(contributions)
                length_source, episode_length_steps = previous_or_exact(episode_lengths, iteration)
                episode_length_seconds = (
                    episode_length_steps * M2_STEP_DT_S if episode_length_steps is not None else None
                )
                for term, contribution in sorted(contributions.items()):
                    logged_rate = scalars[REWARD_PREFIX + term][iteration]
                    per_step = length_normalized(contribution, episode_length_steps)
                    per_second = (
                        contribution / episode_length_seconds
                        if episode_length_seconds is not None and episode_length_seconds > 0.0
                        else None
                    )
                    writer.writerow(
                        {
                            "run": run_name,
                            "iteration": iteration,
                            "term": term,
                            "logged_episode_reward_rate": logged_rate,
                            "approx_episode_contribution": contribution,
                            "absolute_contribution": abs(contribution),
                            "relative_total_magnitude_pct": percentages[term],
                            "negative_rank": ranks[term] if ranks[term] is not None else "",
                            "episode_length_source_iteration": length_source if length_source is not None else "",
                            "episode_length_steps": episode_length_steps if episode_length_steps is not None else "",
                            "episode_length_seconds": episode_length_seconds if episode_length_seconds is not None else "",
                            "approx_contribution_per_step": per_step if per_step is not None else "",
                            "approx_contribution_per_second": per_second if per_second is not None else "",
                        }
                    )


def write_snapshot_comparison_csv(
    output_path: Path, runs: dict[str, dict[str, dict[int, float]]]
) -> None:
    fieldnames = ["metric"]
    for run_name in runs:
        for iteration in SNAPSHOT_ITERATIONS:
            fieldnames.append(f"{run_name}_ep{iteration}")
            fieldnames.append(f"{run_name}_ep{iteration}_source_iteration")
    rows: list[dict[str, float | str | int]] = []
    for metric, tag in COMPARISON_TAGS.items():
        row: dict[str, float | str | int] = {"metric": metric}
        for run_name, scalars in runs.items():
            series = scalars.get(tag, {})
            for iteration in SNAPSHOT_ITERATIONS:
                source_step, value = previous_or_exact(series, iteration)
                row[f"{run_name}_ep{iteration}"] = "" if value is None else value
                row[f"{run_name}_ep{iteration}_source_iteration"] = "" if source_step is None else source_step
        rows.append(row)

    normalized_row: dict[str, float | str | int] = {"metric": "training_reward_per_episode_step_approx"}
    for run_name, scalars in runs.items():
        rewards = scalars.get("rewards/iter", {})
        lengths = scalars.get("episode_lengths/iter", {})
        for iteration in SNAPSHOT_ITERATIONS:
            reward_step, reward = previous_or_exact(rewards, iteration)
            length_step, length = previous_or_exact(lengths, iteration)
            value = length_normalized(reward, length) if reward is not None else None
            normalized_row[f"{run_name}_ep{iteration}"] = "" if value is None else value
            sources = [step for step in (reward_step, length_step) if step is not None]
            normalized_row[f"{run_name}_ep{iteration}_source_iteration"] = (
                "" if not sources else "/".join(str(step) for step in sources)
            )
    rows.append(normalized_row)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase_group_csv(
    output_path: Path, runs: dict[str, dict[str, dict[int, float]]]
) -> None:
    fieldnames = [
        "run",
        "iteration",
        "training_reward",
        "episode_length_steps",
        "training_reward_per_step_approx",
        "reward_term_sum_approx",
        "reward_term_absolute_magnitude",
        "direct_can_land_sum_approx",
        "direct_can_land_negative_magnitude",
        "phase_sensitive_sum_approx",
        "phase_sensitive_absolute_magnitude",
        "phase_sensitive_magnitude_share_pct",
        "align_success_rate",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run_name, scalars in runs.items():
            iterations = sorted(scalars.get("rewards/iter", {}))
            for iteration in iterations:
                contributions, _, episode_length = snapshot_contributions(scalars, iteration)
                total_sum = sum(contributions.values())
                total_abs = sum(abs(value) for value in contributions.values())
                direct_values = [contributions.get(term, 0.0) for term in DIRECT_CAN_LAND_TERMS]
                phase_values = [contributions.get(term, 0.0) for term in PHASE_SENSITIVE_TERMS]
                training_reward = scalars["rewards/iter"][iteration]
                _, align = previous_or_exact(scalars.get("Episode/Metrics/align_success_rate", {}), iteration)
                writer.writerow(
                    {
                        "run": run_name,
                        "iteration": iteration,
                        "training_reward": training_reward,
                        "episode_length_steps": "" if episode_length is None else episode_length,
                        "training_reward_per_step_approx": ""
                        if episode_length is None
                        else length_normalized(training_reward, episode_length),
                        "reward_term_sum_approx": total_sum,
                        "reward_term_absolute_magnitude": total_abs,
                        "direct_can_land_sum_approx": sum(direct_values),
                        "direct_can_land_negative_magnitude": sum(abs(value) for value in direct_values if value < 0.0),
                        "phase_sensitive_sum_approx": sum(phase_values),
                        "phase_sensitive_absolute_magnitude": sum(abs(value) for value in phase_values),
                        "phase_sensitive_magnitude_share_pct": 0.0
                        if total_abs == 0.0
                        else 100.0 * sum(abs(value) for value in phase_values) / total_abs,
                        "align_success_rate": "" if align is None else align,
                    }
                )


def write_correlations_csv(
    output_path: Path, runs: dict[str, dict[str, dict[int, float]]]
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for run_name, scalars in runs.items():
        reward_series = scalars.get("rewards/iter", {})
        for diagnostic, tag in TREND_TAGS.items():
            diagnostic_series = scalars.get(tag, {})
            steps, rewards, values = common_series_pairs(reward_series, diagnostic_series)
            rows.append(
                {
                    "run": run_name,
                    "diagnostic": diagnostic,
                    "samples": len(steps),
                    "first_iteration": steps[0] if steps else "",
                    "last_iteration": steps[-1] if steps else "",
                    "pearson": "" if (value := pearson(rewards, values)) is None else value,
                    "spearman": "" if (value := spearman(rewards, values)) is None else value,
                }
            )
    fieldnames = ["run", "diagnostic", "samples", "first_iteration", "last_iteration", "pearson", "spearman"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def snapshot_contributions(
    scalars: dict[str, dict[int, float]], iteration: int
) -> tuple[dict[str, float], int | None, float | None]:
    reward_tags = sorted(tag for tag in scalars if tag.startswith(REWARD_PREFIX))
    contributions: dict[str, float] = {}
    for tag in reward_tags:
        source_step, value = previous_or_exact(scalars[tag], iteration)
        if source_step == iteration and value is not None:
            contributions[tag.removeprefix(REWARD_PREFIX)] = restore_episode_contribution(value)
    length_step, episode_length = previous_or_exact(scalars.get("episode_lengths/iter", {}), iteration)
    return contributions, length_step, episode_length


def write_summary_markdown(
    output_path: Path,
    runs: dict[str, dict[str, dict[int, float]]],
    correlation_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# M2 Reward Term Summary",
        "",
        "This file is generated from the existing S0/S1 TensorBoard logs only. No simulator or training is invoked.",
        "",
        "`Episode/Episode_Reward/*` is logged by the environment as mean episodic sum divided by 10 s. The tables below multiply it by 10 s to recover the logger's approximate mean episodic contribution for that reset cohort.",
        "",
        "Episode-length normalization is an aggregate diagnostic approximation. `episode_lengths/iter` is in environment steps; M2 uses 0.04 s per environment step. When ep30 has no exact episode-length scalar, the latest earlier scalar is shown with its source iteration.",
        "",
    ]
    for run_name, scalars in runs.items():
        lines.extend([f"## {run_name}", ""])
        rewards = scalars.get("rewards/iter", {})
        lengths = scalars.get("episode_lengths/iter", {})
        lines.append("| iteration | training reward | episode length (steps) | source | reward/step approx | top negative contributors | top positive contributors |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for iteration in SNAPSHOT_ITERATIONS:
            reward_source, reward = previous_or_exact(rewards, iteration)
            length_source, episode_length = previous_or_exact(lengths, iteration)
            contributions, _, _ = snapshot_contributions(scalars, iteration)
            negatives = sorted(
                ((term, value) for term, value in contributions.items() if value < 0.0),
                key=lambda item: item[1],
            )[:5]
            positives = sorted(
                ((term, value) for term, value in contributions.items() if value > 0.0),
                key=lambda item: item[1],
                reverse=True,
            )[:4]
            normalized = length_normalized(reward, episode_length) if reward is not None else None
            negative_text = "; ".join(f"{term} {_format(value, 2)}" for term, value in negatives) or "none"
            positive_text = "; ".join(f"{term} +{_format(value, 2)}" for term, value in positives) or "none"
            source_text = f"r{reward_source if reward_source is not None else 'NA'}/l{length_source if length_source is not None else 'NA'}"
            lines.append(
                f"| {iteration} | {_format(reward, 3)} | {_format(episode_length, 2)} | {source_text} | {_format(normalized, 5)} | {negative_text} | {positive_text} |"
            )
        lines.append("")

    lines.extend(["## Phase-gating diagnostics", ""])
    lines.append("| run | diagnostic | n | Pearson | Spearman |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for run_name, scalars in runs.items():
        iterations = sorted(scalars.get("rewards/iter", {}))
        align_series = scalars.get("Episode/Metrics/align_success_rate", {})
        length_series = scalars.get("episode_lengths/iter", {})
        phase_negative: dict[int, float] = {}
        phase_share: dict[int, float] = {}
        normalized_reward: dict[int, float] = {}
        lengths: dict[int, float] = {}
        for iteration in iterations:
            contributions, length_source, episode_length = snapshot_contributions(scalars, iteration)
            direct_values = [contributions.get(term, 0.0) for term in DIRECT_CAN_LAND_TERMS]
            phase_values = [contributions.get(term, 0.0) for term in PHASE_SENSITIVE_TERMS]
            total_abs = sum(abs(value) for value in contributions.values())
            phase_negative[iteration] = sum(abs(value) for value in direct_values if value < 0.0)
            phase_share[iteration] = 0.0 if total_abs == 0.0 else 100.0 * sum(abs(value) for value in phase_values) / total_abs
            if episode_length is not None and length_source == iteration:
                normalized_reward[iteration] = scalars["rewards/iter"][iteration] / episode_length
                lengths[iteration] = episode_length
        diagnostics = [
            ("direct can_land negative magnitude vs align", phase_negative, align_series),
            ("phase-sensitive magnitude share vs align", phase_share, align_series),
            ("length-normalized reward vs episode length", normalized_reward, lengths),
        ]
        for label, left, right in diagnostics:
            steps, xs, ys = common_series_pairs(left, right)
            lines.append(
                f"| {run_name} | {label} | {len(steps)} | {_format(pearson(xs, ys), 3)} | {_format(spearman(xs, ys), 3)} |"
            )
    lines.extend(
        [
            "",
            "The first two diagnostics use reward-term aggregates and alignment metrics emitted from the same environment reset logging path. A strong positive association means post-latch reward magnitude appears precisely in iterations where more reset episodes have latched alignment; it does not by itself prove the policy later drifted within those episodes.",
            "",
            "## Reward-versus-behavior trend diagnostics",
            "",
        ]
    )
    lines.append("| run | diagnostic | n | Pearson | Spearman |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in correlation_rows:
        pearson_value = float(row["pearson"]) if row["pearson"] != "" else None
        spearman_value = float(row["spearman"]) if row["spearman"] != "" else None
        lines.append(
            f"| {row['run']} | {row['diagnostic']} | {row['samples']} | {_format(pearson_value, 3)} | {_format(spearman_value, 3)} |"
        )
    lines.extend(
        [
            "",
            "These correlations use approximately 30 optimizer iterations and are descriptive only. The reset-cohort logger signals for `Episode_Termination/crash` and `time_out` are not probabilities and can exceed 1; they must not be interpreted as percentage rates.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def write_deterministic_post_latch_csv(
    output_path: Path, evaluations: dict[str, dict[int, Path]]
) -> list[dict[str, float | int | str]]:
    rows_out: list[dict[str, float | int | str]] = []
    for run_name, checkpoints in evaluations.items():
        for iteration, path in sorted(checkpoints.items()):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            summary = summarize_deterministic_rows(rows)
            rows_out.append({"run": run_name, "iteration": iteration, **summary})
    fieldnames = list(rows_out[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def write_commands(output_path: Path, run_paths: dict[str, Path], output_dir: Path) -> None:
    command = (
        "source /home/j/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && "
        "export PYTHONPATH=source/quadcopter_waypoint && "
        "python scripts/rl_games/analyze_m2_reward_compatibility.py "
        f"--s0 {run_paths['S0']} --s1 {run_paths['S1']} --output-dir {output_dir}"
    )
    output_path.write_text(command + "\n")


def append_deterministic_summary(
    output_path: Path, deterministic_rows: list[dict[str, float | int | str]]
) -> None:
    with output_path.open("a") as handle:
        handle.write("## Existing deterministic evaluator: post-latch terminal state\n\n")
        handle.write(
            "`align_success` is latched, so an aligned episode whose terminal horizontal error is >= 0.25 m is direct evidence that the episode later ended outside the horizontal part of the instantaneous alignment envelope. This still does not reconstruct exactly when alignment was lost.\n\n"
        )
        handle.write(
            "| run | iter | aligned | aligned ending outside 0.25 m | aligned timeouts | timeout outside 0.25 m | timeout terminal xy mean | timeout clearance mean |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in deterministic_rows:
            handle.write(
                f"| {row['run']} | {row['iteration']} | {row['aligned_episodes']}/{row['episodes']} "
                f"({_format(float(row['aligned_rate']) * 100.0, 2)}%) | "
                f"{row['aligned_terminal_outside_radius']}/{row['aligned_episodes']} "
                f"({_format(float(row['aligned_terminal_outside_radius_rate']) * 100.0, 2)}%) | "
                f"{row['aligned_timeout_episodes']} | "
                f"{row['aligned_timeout_terminal_outside_radius']}/{row['aligned_timeout_episodes']} "
                f"({_format(float(row['aligned_timeout_terminal_outside_radius_rate']) * 100.0, 2)}%) | "
                f"{_format(float(row['aligned_timeout_terminal_xy_error_mean_m']), 3)} m | "
                f"{_format(float(row['aligned_timeout_terminal_clearance_mean_m']), 3)} m |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0", type=Path, default=DEFAULT_RUNS["S0"])
    parser.add_argument("--s1", type=Path, default=DEFAULT_RUNS["S1"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/px4_hierarchical_training/reward_compatibility_audit"),
    )
    args = parser.parse_args()

    run_paths = {"S0": args.s0, "S1": args.s1}
    for name, path in run_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} TensorBoard summary directory not found: {path}")
    for run_name, checkpoints in DEFAULT_EVALUATIONS.items():
        for iteration, path in checkpoints.items():
            if not path.exists():
                raise FileNotFoundError(f"{run_name} ep{iteration} deterministic evaluator CSV not found: {path}")

    runs = {name: load_scalar_series(path) for name, path in run_paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_commands(args.output_dir / "commands.txt", run_paths, args.output_dir)
    write_reward_terms_csv(args.output_dir / "reward_terms_s0_s1.csv", runs)
    write_snapshot_comparison_csv(args.output_dir / "snapshot_comparison.csv", runs)
    write_phase_group_csv(args.output_dir / "phase_group_summary.csv", runs)
    correlation_rows = write_correlations_csv(args.output_dir / "trend_correlations.csv", runs)
    deterministic_rows = write_deterministic_post_latch_csv(
        args.output_dir / "deterministic_post_latch_summary.csv", DEFAULT_EVALUATIONS
    )
    write_summary_markdown(args.output_dir / "reward_term_summary.md", runs, correlation_rows)
    append_deterministic_summary(args.output_dir / "reward_term_summary.md", deterministic_rows)
    print(f"Wrote D0 offline audit tables to {args.output_dir}")


if __name__ == "__main__":
    main()

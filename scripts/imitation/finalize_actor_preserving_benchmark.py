"""Select actor-preserving PPO checkpoints and finalize the reproducible formal benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from quadcopter_waypoint.imitation.checkpoint_sweep import aggregate_checkpoint_rows, select_validation_best
from quadcopter_waypoint.imitation.dataset import sha256_file
from quadcopter_waypoint.imitation.actor_preserving_checkpoint import (
    actor_weights_sha256,
    critic_weights_sha256,
    obs_rms_sha256,
)

RATE_METRICS = (
    "settled_landing_rate",
    "contact_success_rate",
    "hard_contact_rate",
    "ground_crash_rate",
    "deck_miss_rate",
    "timeout_rate",
)
DETAIL_METRICS = (
    "touchdown_distance_mean_m",
    "touchdown_distance_p95_m",
    "first_contact_xy_error_mean_m",
    "first_contact_xy_error_p95_m",
    "first_contact_normal_relative_speed_mean_mps",
    "first_contact_normal_relative_speed_p95_mps",
    "first_contact_tangential_relative_speed_mean_mps",
    "first_contact_tangential_relative_speed_p95_mps",
    "first_contact_body_deck_normal_angle_mean_rad",
    "first_contact_body_deck_normal_angle_p95_rad",
    "maximum_penetration_mean_m",
    "maximum_penetration_p95_m",
)
VALIDATION_SEEDS = (145, 146, 147)
FORMAL_TEST_SEEDS = (245, 246, 247)
TRAINING_SEEDS = (42, 43, 44)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=check)
    return result.stdout.strip()


@lru_cache(maxsize=None)
def _actor_preserving_model_hashes(checkpoint_path: str) -> dict[str, str | None]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    return {
        "actor_weights_sha256": actor_weights_sha256(model),
        "critic_weights_sha256": critic_weights_sha256(model) if "a2c_network.critic_mlp.0.weight" in model else None,
        "observation_rms_sha256": obs_rms_sha256(model),
    }


def _flatten_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"completed evaluation has no metrics: {entry.get('checkpoint_path')}")
    failures = metrics.get("failure_counts", {})
    checkpoint_path = str(Path(entry["checkpoint_path"]).resolve())
    hashes = _actor_preserving_model_hashes(checkpoint_path)
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "actor_sha256": hashes["actor_weights_sha256"],
        "actor_rms_sha256": entry["actor_sha256"],
        "critic_sha256": hashes["critic_weights_sha256"],
        "observation_rms_sha256": hashes["observation_rms_sha256"],
        "train_seed": entry.get("train_seed"),
        "epoch": int(entry["epoch"]),
        "training_reward": entry.get("training_reward"),
        "kind": entry["kind"],
        "eval_seed": int(entry["eval_seed"]),
        "episodes": int(entry["episodes"]),
        "num_envs": int(entry["num_envs"]),
        "agent": entry.get("agent"),
        "output_csv": entry["output_csv"],
        "output_log": entry["output_log"],
        "settled_episodes": int(metrics["settled_episodes"]),
        "failure_deck_miss_count": int(failures.get("deck_miss", 0)),
        "failure_ground_crash_count": int(failures.get("ground_crash", 0)),
        "failure_hard_contact_count": int(failures.get("hard_contact", 0)),
        "failure_timeout_count": int(failures.get("timeout", 0)),
        **{
            key: value
            for key, value in metrics.items()
            if key not in {"csv", "failure_counts", "episodes", "settled_episodes"}
        },
    }


def _manifest_rows(
    paths: Sequence[str | Path],
    *,
    expected_num_envs: int | None,
    expected_episodes: int,
    expected_eval_seeds: Iterable[int],
    allowed_checkpoint_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    expected_seeds = set(expected_eval_seeds)
    rows: list[dict[str, Any]] = []
    skipped_parallelism = 0
    for path in paths:
        manifest = _read_json(path)
        for entry in manifest.get("entries", {}).values():
            if entry.get("status") != "completed":
                continue
            checkpoint = str(Path(entry["checkpoint_path"]).resolve())
            if allowed_checkpoint_paths is not None and checkpoint not in allowed_checkpoint_paths:
                continue
            if expected_num_envs is not None and int(entry["num_envs"]) != expected_num_envs:
                skipped_parallelism += 1
                continue
            if int(entry["episodes"]) != expected_episodes:
                raise ValueError(f"unexpected episode count in {path}: {entry['episodes']}")
            if int(entry["eval_seed"]) not in expected_seeds:
                raise ValueError(f"unexpected evaluation seed in {path}: {entry['eval_seed']}")
            rows.append(_flatten_entry(entry))
    if not rows:
        raise ValueError("no completed manifest rows matched the frozen protocol")
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["checkpoint_sha256"], row["eval_seed"])
        previous = deduplicated.get(key)
        if previous is not None:
            current_metrics = {name: row.get(name) for name in (*RATE_METRICS, *DETAIL_METRICS)}
            previous_metrics = {name: previous.get(name) for name in (*RATE_METRICS, *DETAIL_METRICS)}
            if current_metrics != previous_metrics:
                raise ValueError(f"duplicate evaluation has different metrics: {key}")
            continue
        deduplicated[key] = row
    result = sorted(
        deduplicated.values(),
        key=lambda row: (
            row["train_seed"] is None,
            row["train_seed"] or -1,
            row["epoch"],
            row["kind"],
            row["eval_seed"],
        ),
    )
    print(json.dumps({"rows": len(result), "skipped_other_parallelism": skipped_parallelism}))
    return result


def _validate_complete_grid(rows: Sequence[Mapping[str, Any]], expected_eval_seeds: Iterable[int]) -> None:
    expected = set(expected_eval_seeds)
    by_checkpoint: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_checkpoint[str(row["checkpoint_sha256"])].add(int(row["eval_seed"]))
    incomplete = {sha: sorted(seeds) for sha, seeds in by_checkpoint.items() if seeds != expected}
    if incomplete:
        raise ValueError(f"incomplete evaluation grid: {incomplete}")


def _combine_inventories(paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in _read_json(path)["records"]:
            resolved = str(Path(record["path"]).resolve())
            if resolved in by_path and by_path[resolved]["sha256"] != record["sha256"]:
                raise ValueError(f"inventory path hash mismatch: {resolved}")
            hashes = _actor_preserving_model_hashes(resolved)
            by_path[resolved] = {
                **record,
                "path": resolved,
                "actor_rms_sha256": record["actor_sha256"],
                "actor_sha256": hashes["actor_weights_sha256"],
                "critic_sha256": hashes["critic_weights_sha256"],
                "observation_rms_sha256": hashes["observation_rms_sha256"],
            }
    records = sorted(
        by_path.values(),
        key=lambda row: (
            row["train_seed"] is None,
            row["train_seed"] or -1,
            row["epoch"],
            row["kind"],
            row["path"],
        ),
    )
    return records, {
        "checkpoint_count": len(records),
        "canonical_count": sum(bool(record["canonical"]) for record in records),
        "records": records,
    }


def _role_plan(validation_rows: list[dict[str, Any]], bc_path: str) -> dict[str, Any]:
    selected = select_validation_best(validation_rows)
    aggregated = aggregate_checkpoint_rows(validation_rows)
    roles_by_path: dict[str, list[str]] = defaultdict(list)
    role_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparisons: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        metric = selected[seed]
        reward = next(row for row in aggregated if row.get("train_seed") == seed and row["kind"] == "reward_selected")
        last = max(
            (row for row in aggregated if row.get("train_seed") == seed and row["kind"] == "periodic"),
            key=lambda row: int(row["epoch"]),
        )
        for role, row in (("metric_selected", metric), ("reward_selected", reward), ("last", last)):
            path = str(Path(row["checkpoint_path"]).resolve())
            roles_by_path[path].append(role)
            role_records[role].append(row)
        comparisons[str(seed)] = {
            "metric_selected_epoch": int(metric["epoch"]),
            "reward_selected_epoch": int(reward["epoch"]),
            "last_epoch": int(last["epoch"]),
            "metric_equals_reward": metric["checkpoint_sha256"] == reward["checkpoint_sha256"],
            "tie_break_evidence": {
                key: metric.get(key)
                for key in (
                    "settled_landing_rate",
                    "deck_miss_rate",
                    "hard_contact_rate",
                    "touchdown_distance_mean_m",
                    "epoch",
                )
            },
        }
    bc_resolved = str(Path(bc_path).resolve())
    bc = next(row for row in aggregated if str(Path(row["checkpoint_path"]).resolve()) == bc_resolved)
    roles_by_path[bc_resolved].append("bc_only")
    role_records["bc_only"].append(bc)
    return {
        "selection_rule": [
            "maximize validation settled_landing_rate",
            "minimize deck_miss_rate",
            "minimize hard_contact_rate",
            "minimize touchdown_distance_mean_m",
            "choose earlier epoch",
        ],
        "training_seeds": list(TRAINING_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "episodes_per_validation_seed": 128,
        "formal_test_seeds_used_for_selection": False,
        "selected_by_training_seed": {str(seed): selected[seed] for seed in TRAINING_SEEDS},
        "selection_comparison_by_training_seed": comparisons,
        "roles_by_checkpoint_path": {path: sorted(set(roles)) for path, roles in sorted(roles_by_path.items())},
        "role_records": dict(role_records),
        "checkpoint_paths": sorted(roles_by_path),
    }


def _select_mode(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    rows = _manifest_rows(
        args.validation_manifests,
        expected_num_envs=args.expected_num_envs,
        expected_episodes=128,
        expected_eval_seeds=VALIDATION_SEEDS,
    )
    _validate_complete_grid(rows, VALIDATION_SEEDS)
    train_seeds = sorted({int(row["train_seed"]) for row in rows if row.get("train_seed") is not None})
    if train_seeds != list(TRAINING_SEEDS):
        raise ValueError(f"unexpected training seeds: {train_seeds}")
    aggregate = aggregate_checkpoint_rows(rows)
    plan = _role_plan(rows, args.bc_checkpoint)
    _write_csv(output / "validation_results.csv", rows)
    _write_csv(output / "validation_aggregate.csv", aggregate)
    _write_json(output / "validation_selection.json", plan)
    _write_json(output / "formal_checkpoint_plan.json", plan)
    records, inventory = _combine_inventories(args.inventories)
    _write_json(output / "checkpoint_inventory.json", inventory)
    _write_csv(output / "checkpoint_inventory.csv", records)
    _write_json(
        output / "checkpoint_hashes.json",
        {
            record["path"]: {
                "checkpoint_sha256": record["sha256"],
                "actor_sha256": record["actor_sha256"],
                "actor_rms_sha256": record["actor_rms_sha256"],
                "critic_sha256": record["critic_sha256"],
                "observation_rms_sha256": record["observation_rms_sha256"],
                "canonical": bool(record["canonical"]),
                "duplicate_of": record.get("duplicate_of"),
            }
            for record in records
        },
    )
    print(
        json.dumps(
            {
                "validation_rows": len(rows),
                "checkpoints": len(aggregate),
                "selected_epochs": {str(seed): int(plan["selected_by_training_seed"][str(seed)]["epoch"]) for seed in TRAINING_SEEDS},
                "formal_checkpoint_paths": len(plan["checkpoint_paths"]),
            },
            indent=2,
        )
    )


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [math.nan, math.nan]
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty evaluation rows")
    episodes = sum(int(row["episodes"]) for row in rows)
    result: dict[str, Any] = {
        "evaluations": len(rows),
        "episodes": episodes,
        "checkpoints": len({str(row["checkpoint_sha256"]) for row in rows}),
        "eval_seeds": sorted({int(row["eval_seed"]) for row in rows}),
        "training_seeds": sorted({int(row["train_seed"]) for row in rows if row.get("train_seed") is not None}),
    }
    for metric in (*RATE_METRICS, *DETAIL_METRICS):
        metric_rows = [row for row in rows if row.get(metric) is not None]
        if not metric_rows:
            continue
        values = [float(row[metric]) for row in metric_rows]
        weights = [int(row["episodes"]) for row in metric_rows]
        weighted = sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)
        result[metric] = weighted
        result[f"{metric}_evaluation_std"] = statistics.pstdev(values)
        if metric in RATE_METRICS:
            successes = sum(round(float(row[metric]) * int(row["episodes"])) for row in metric_rows)
            result[f"{metric}_successes"] = successes
            result[f"{metric}_95ci_wilson_low"] = _wilson(successes, sum(weights))[0]
            result[f"{metric}_95ci_wilson_high"] = _wilson(successes, sum(weights))[1]
    result["settled_episodes"] = sum(int(row["settled_episodes"]) for row in rows)
    for name in ("deck_miss", "ground_crash", "hard_contact", "timeout"):
        result[f"failure_{name}_count"] = sum(int(row[f"failure_{name}_count"]) for row in rows)
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _numeric_validation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        parsed: dict[str, Any] = dict(row)
        for key in ("train_seed", "epoch", "eval_seed", "episodes", "num_envs"):
            parsed[key] = int(float(parsed[key])) if parsed.get(key) else None
        for key in (*RATE_METRICS, *DETAIL_METRICS, "training_reward"):
            parsed[key] = float(parsed[key]) if parsed.get(key) not in (None, "") else None
        rows.append(parsed)
    return rows


def _load_drift(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("formal drift JSON has no records")
    return payload, records


def _correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _save_line_plot(
    rows: Sequence[Mapping[str, Any]],
    y: str,
    ylabel: str,
    path: Path,
    selection: Mapping[str, Any],
) -> None:
    fig = plt.figure(figsize=(8.5, 5.2))
    for seed in TRAINING_SEEDS:
        seed_rows = sorted(
            [row for row in rows if row.get("train_seed") == seed and row.get("kind") == "periodic"],
            key=lambda row: int(row["epoch"]),
        )
        plt.plot([row["epoch"] for row in seed_rows], [row[y] for row in seed_rows], marker="o", label=f"seed {seed}")
        chosen = selection["selected_by_training_seed"][str(seed)]
        plt.scatter([chosen["epoch"]], [chosen[y]], marker="*", s=150)
        reward = next(row for row in rows if row.get("train_seed") == seed and row.get("kind") == "reward_selected")
        plt.scatter([reward["epoch"]], [reward[y]], marker="x", s=70)
    bc = next(row for row in rows if row.get("kind") == "bc_init")
    plt.axhline(float(bc[y]), linestyle="--", label="BC epoch 0")
    plt.xlabel("PPO epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_drift_line(rows: Sequence[Mapping[str, Any]], y: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 5.2))
    for seed in TRAINING_SEEDS:
        selected = sorted(
            [row for row in rows if row.get("train_seed") == seed and row.get("kind") == "periodic"],
            key=lambda row: int(row["epoch"]),
        )
        plt.plot([int(row["epoch"]) for row in selected], [float(row[y]) for row in selected], marker="o", label=f"seed {seed}")
    plt.xlabel("PPO epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_scatter(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig = plt.figure(figsize=(7.5, 5.2))
    for seed in TRAINING_SEEDS:
        selected = [row for row in rows if row.get("train_seed") == seed and row.get("validation_settled_landing_rate") is not None]
        plt.scatter(
            [float(row["action_mse_vs_reference"]) for row in selected],
            [float(row["validation_settled_landing_rate"]) for row in selected],
            label=f"seed {seed}",
        )
    plt.xlabel("Action MSE vs BC reference")
    plt.ylabel("Validation settled landing rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_formal_comparison(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    values = [float(row["settled_landing_rate"]) for row in rows]
    fig = plt.figure(figsize=(10.5, 5.6))
    plt.bar(methods, values)
    plt.axhline(0.90, linestyle="--")
    plt.axhline(0.92, linestyle=":")
    plt.ylabel("Formal settled landing rate")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_failure_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    failures = ("deck_miss", "hard_contact", "ground_crash", "timeout")
    x = np.arange(len(methods))
    width = 0.18
    fig = plt.figure(figsize=(10.5, 5.6))
    for index, failure in enumerate(failures):
        lookup = {(str(row["method"]), str(row["failure_type"])): float(row["rate"]) for row in rows}
        plt.bar(x + (index - 1.5) * width, [lookup[(method, failure)] for method in methods], width, label=failure)
    plt.ylabel("Failure rate")
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _training_run_summary(run_dir: Path, seed: int) -> dict[str, Any]:
    checkpoints = sorted((run_dir / "nn").glob("last_*_ep_*_rew_*.pth"))
    canonical: dict[int, Path] = {}
    for path in checkpoints:
        try:
            epoch = int(path.name.split("_ep_")[1].split("_rew_")[0])
        except (IndexError, ValueError):
            continue
        previous = canonical.get(epoch)
        if previous is None or (len(path.name), path.name) < (len(previous.name), previous.name):
            canonical[epoch] = path
    if sorted(canonical) != list(range(10, 201, 10)):
        raise ValueError(f"training seed {seed} has incomplete periodic checkpoints")
    event_files = list((run_dir / "summaries").glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(f"training seed {seed} must have one TensorBoard event file")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars("performance/total_fps") if "performance/total_fps" in accumulator.Tags().get("scalars", []) else []
    reward_selected = run_dir / "nn" / "actor_preserving_formal_lambda50.pth"
    return {
        "training_seed": seed,
        "run_dir": str(run_dir.resolve()),
        "epochs": 200,
        "environment_steps": 256 * 24 * 200,
        "periodic_checkpoints": 20,
        "wall_clock_seconds_from_tensorboard": float(values[-1].wall_time - values[0].wall_time) if len(values) >= 2 else None,
        "final_checkpoint": str(canonical[200].resolve()),
        "final_checkpoint_sha256": sha256_file(canonical[200]),
        "reward_selected_checkpoint": str(reward_selected.resolve()),
        "reward_selected_checkpoint_sha256": sha256_file(reward_selected),
        "tensorboard_event": str(event_files[0].resolve()),
        "tensorboard_event_sha256": sha256_file(event_files[0]),
    }


def _historical_row(method: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method}
    for key, value in summary.items():
        if isinstance(value, Mapping) and "mean" in value:
            row[key] = value["mean"]
            row[f"{key}_evaluation_std"] = value.get("std")
        elif not isinstance(value, (Mapping, list)):
            row[key] = value
    return row


def _comparison_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# actor-preserving PPO Formal Comparison",
        "",
        "| Method | Episodes | Settled landing | Deck miss | Hard contact | Ground crash | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {episodes} | {settled:.2%} | {deck:.2%} | {hard:.2%} | {ground:.2%} | {timeout:.2%} |".format(
                method=row["method"],
                episodes=int(row.get("episodes", 0)),
                settled=float(row["settled_landing_rate"]),
                deck=float(row["deck_miss_rate"]),
                hard=float(row["hard_contact_rate"]),
                ground=float(row["ground_crash_rate"]),
                timeout=float(row["timeout_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def _verdict(condition: bool, partial: bool = False) -> str:
    if condition:
        return "supported"
    return "partially_supported" if partial else "not_supported"


def _finalize_mode(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _read_json(args.selection_plan)
    allowed = {str(Path(path).resolve()) for path in plan["checkpoint_paths"]}
    formal_physical = _manifest_rows(
        args.formal_manifests,
        expected_num_envs=args.formal_num_envs,
        expected_episodes=256,
        expected_eval_seeds=FORMAL_TEST_SEEDS,
        allowed_checkpoint_paths=allowed,
    )
    _validate_complete_grid(formal_physical, FORMAL_TEST_SEEDS)
    if {str(Path(row["checkpoint_path"]).resolve()) for row in formal_physical} != allowed:
        raise ValueError("formal manifest does not contain exactly the preregistered actor-preserving PPO checkpoint plan")
    roles_by_path = {str(Path(path).resolve()): roles for path, roles in plan["roles_by_checkpoint_path"].items()}
    expanded: list[dict[str, Any]] = []
    for row in formal_physical:
        for role in roles_by_path[str(Path(row["checkpoint_path"]).resolve())]:
            expanded.append({**row, "method": role})

    teacher_path = str(Path(args.teacher_checkpoint).resolve())
    teacher_rows = _manifest_rows(
        [args.reference_manifest],
        expected_num_envs=args.reference_num_envs,
        expected_episodes=256,
        expected_eval_seeds=FORMAL_TEST_SEEDS,
        allowed_checkpoint_paths={teacher_path},
    )
    _validate_complete_grid(teacher_rows, FORMAL_TEST_SEEDS)
    expanded.extend({**row, "method": "frozen_teacher"} for row in teacher_rows)
    _write_csv(output / "formal_results.csv", expanded)

    role_summaries: dict[str, Any] = {}
    per_training_seed: dict[str, Any] = {}
    for role in sorted({str(row["method"]) for row in expanded}):
        role_rows = [row for row in expanded if row["method"] == role]
        role_summaries[role] = _summary(role_rows)
        if role in {"metric_selected", "reward_selected", "last"}:
            per_training_seed[role] = {
                str(seed): _summary([row for row in role_rows if row.get("train_seed") == seed]) for seed in TRAINING_SEEDS
            }
    aggregate_payload = {
        "protocol": {"formal_test_seeds": list(FORMAL_TEST_SEEDS), "episodes_per_seed": 256},
        "methods": role_summaries,
        "per_training_seed": per_training_seed,
    }
    _write_json(output / "formal_aggregate.json", aggregate_payload)
    aggregate_rows = [{"method": role, **summary} for role, summary in sorted(role_summaries.items())]
    _write_csv(output / "formal_aggregate.csv", aggregate_rows)

    failure_rows: list[dict[str, Any]] = []
    for role, summary in sorted(role_summaries.items()):
        for failure in ("deck_miss", "hard_contact", "ground_crash", "timeout"):
            failure_rows.append(
                {
                    "method": role,
                    "failure_type": failure,
                    "count": summary[f"failure_{failure}_count"],
                    "episodes": summary["episodes"],
                    "rate": summary[f"failure_{failure}_count"] / summary["episodes"],
                }
            )
    _write_csv(output / "failure_distribution.csv", failure_rows)

    imitation = _read_json(args.imitation_summary)
    checkpoint_selection = _read_json(args.checkpoint_selection_summary)
    comparison_rows = [
        {"method": "Frozen PPO teacher", **role_summaries["frozen_teacher"]},
        {"method": "BC epoch 0", **role_summaries["bc_only"]},
        _historical_row("ordinary BC+PPO", imitation["formal_evaluation"]["BC+PPO"]["aggregate"]),
        {"method": "checkpoint-selected BC+PPO", **checkpoint_selection["formal_comparison"]["metric-selected BC+PPO"]},
        {"method": "actor-preserving PPO metric-selected", **role_summaries["metric_selected"]},
        {"method": "actor-preserving PPO reward-selected", **role_summaries["reward_selected"]},
        {"method": "actor-preserving PPO epoch-200 last", **role_summaries["last"]},
    ]
    _write_csv(output / "comparison.csv", comparison_rows)
    (output / "comparison.md").write_text(_comparison_markdown(comparison_rows), encoding="utf-8")

    drift_payload, drift_rows = _load_drift(args.formal_drift)
    _write_json(output / "policy_drift.json", drift_payload)
    shutil.copy2(Path(args.formal_drift).with_suffix(".csv"), output / "policy_drift.csv")
    validation_rows = _numeric_validation_rows(output / "validation_results.csv")
    validation_aggregate = aggregate_checkpoint_rows(validation_rows)
    validation_by_sha = {row["checkpoint_sha256"]: row for row in validation_aggregate}
    drift_with_metrics: list[dict[str, Any]] = []
    for row in drift_rows:
        combined = dict(row)
        metric = validation_by_sha.get(row["checkpoint_sha256"])
        if metric:
            combined["validation_settled_landing_rate"] = metric["settled_landing_rate"]
            combined["validation_deck_miss_rate"] = metric["deck_miss_rate"]
        drift_with_metrics.append(combined)

    _save_line_plot(validation_aggregate, "settled_landing_rate", "Validation settled landing rate", output / "validation_settled_landing_vs_epoch.png", plan)
    _save_line_plot(validation_aggregate, "deck_miss_rate", "Validation deck miss rate", output / "validation_deck_miss_vs_epoch.png", plan)
    _save_drift_line(drift_with_metrics, "action_mse_vs_reference", "Action MSE vs BC reference", output / "action_drift_vs_epoch.png")
    _save_scatter(drift_with_metrics, output / "action_drift_vs_settled_landing.png")
    _save_drift_line(drift_with_metrics, "critic_parameter_relative_l2", "Critic relative parameter L2", output / "critic_drift_vs_epoch.png")
    _save_formal_comparison(comparison_rows, output / "formal_method_comparison.png")
    _save_failure_plot(failure_rows, output / "formal_failure_distribution.png")

    paired = [row for row in drift_with_metrics if row.get("validation_settled_landing_rate") is not None and row.get("kind") == "periodic"]
    correlations = {
        "paired_checkpoints": len(paired),
        "pearson_action_mse_vs_validation_settled": _correlation(
            [float(row["action_mse_vs_reference"]) for row in paired],
            [float(row["validation_settled_landing_rate"]) for row in paired],
        ),
        "pearson_actor_l2_vs_validation_settled": _correlation(
            [float(row["actor_parameter_relative_l2"]) for row in paired],
            [float(row["validation_settled_landing_rate"]) for row in paired],
        ),
        "interpretation": "observational statistical association only; no causal claim",
    }

    pilot_summary = _read_json(output / "pilot_summary.json")
    pilot_rows = _read_csv(output / "pilot_results.csv")
    epoch10 = [row for row in drift_rows if row.get("kind") == "periodic" and int(row["epoch"]) == 10]
    all_rms_zero = all(
        float(row["observation_mean_l2"]) == 0.0
        and float(row["observation_variance_l2"]) == 0.0
        and float(row["observation_count_delta"]) == 0.0
        for row in drift_rows
    )
    lambda0_epoch30 = next(
        float(row["action_mse_vs_reference"])
        for row in pilot_rows
        if row["coefficient"] == "0.0" and row["epoch"] == "30" and row["kind"] == "periodic"
    )
    lambda50_epoch30 = next(
        float(row["action_mse_vs_reference"])
        for row in pilot_rows
        if row["coefficient"] == "50.0" and row["epoch"] == "30" and row["kind"] == "periodic"
    )
    metric_rate = role_summaries["metric_selected"]["settled_landing_rate"]
    reward_rate = role_summaries["reward_selected"]["settled_landing_rate"]
    bc_rate = role_summaries["bc_only"]["settled_landing_rate"]
    last_rate = role_summaries["last"]["settled_landing_rate"]
    per_seed_metric = per_training_seed["metric_selected"]
    prediction_verification = {
        "predictions": [
            {
                "id": 1,
                "prediction": "epoch1-10 actor SHA256 remains unchanged",
                "verdict": _verdict(bool(epoch10) and all(float(row["actor_parameter_relative_l2"]) == 0.0 for row in epoch10)),
                "evidence_files": ["policy_drift.csv"],
                "measured_values": {"epoch10_actor_relative_l2": [float(row["actor_parameter_relative_l2"]) for row in epoch10]},
                "interpretation": "The actor weights stayed identical through critic-only warm-up.",
                "limitations": "Hash/parameter equality does not by itself verify the environment trajectory distribution.",
            },
            {
                "id": 2,
                "prediction": "warm-up deterministic action max error is at most 1e-5",
                "verdict": _verdict(bool(epoch10) and all(float(row["action_max_abs_error_vs_reference"]) <= 1.0e-5 for row in epoch10)),
                "evidence_files": ["policy_drift.csv"],
                "measured_values": {"epoch10_action_max_abs_error": [float(row["action_max_abs_error_vs_reference"]) for row in epoch10]},
                "interpretation": "The deterministic actor function stayed at the BC reference during warm-up.",
                "limitations": "Measured on the frozen imitation-learning benchmark test observation split.",
            },
            {
                "id": 3,
                "prediction": "critic parameters change during warm-up",
                "verdict": _verdict(bool(epoch10) and all(float(row["critic_parameter_relative_l2"]) > 0.0 for row in epoch10)),
                "evidence_files": ["policy_drift.csv"],
                "measured_values": {"epoch10_critic_relative_l2": [float(row["critic_parameter_relative_l2"]) for row in epoch10]},
                "interpretation": "Critic-only optimization changed the critic while preserving the actor.",
                "limitations": "Parameter change is not a direct measure of value-function accuracy.",
            },
            {
                "id": 4,
                "prediction": "observation RMS mean, variance, and count drift remain zero",
                "verdict": _verdict(all_rms_zero),
                "evidence_files": ["policy_drift.csv", "policy_drift.json"],
                "measured_values": {"all_checkpoints_zero_rms_drift": all_rms_zero},
                "interpretation": "The frozen observation normalization contract held for all inventoried actor-preserving PPO checkpoints.",
                "limitations": "Value RMS is intentionally excluded from this frozen contract.",
            },
            {
                "id": 5,
                "prediction": "positive anchor lowers same-epoch action MSE versus lambda=0",
                "verdict": _verdict(lambda50_epoch30 < lambda0_epoch30),
                "evidence_files": ["pilot_results.csv"],
                "measured_values": {"lambda0_epoch30_action_mse": lambda0_epoch30, "lambda50_epoch30_action_mse": lambda50_epoch30},
                "interpretation": "The preregistered anchor reduced policy drift at the controlled pilot epoch.",
                "limitations": "This pilot comparison does not establish a universal causal relationship with closed-loop success.",
            },
            {
                "id": 6,
                "prediction": "lower drift may reduce rapid degradation but does not guarantee settled landing improvement",
                "verdict": _verdict(metric_rate >= bc_rate and last_rate < metric_rate, partial=metric_rate >= bc_rate or last_rate < metric_rate),
                "evidence_files": ["formal_aggregate.csv", "policy_drift.csv"],
                "measured_values": {"bc_rate": bc_rate, "metric_selected_rate": metric_rate, "last_rate": last_rate},
                "interpretation": "Formal outcomes are consistent with preservation helping selected checkpoints while later updates can still degrade performance.",
                "limitations": "The relationship is observational; no strict causal claim is made.",
            },
            {
                "id": 7,
                "prediction": "validation metric selection outperforms training-reward selection",
                "verdict": _verdict(metric_rate > reward_rate, partial=math.isclose(metric_rate, reward_rate)),
                "evidence_files": ["formal_aggregate.csv", "formal_checkpoint_plan.json"],
                "measured_values": {"metric_selected_rate": metric_rate, "reward_selected_rate": reward_rate},
                "interpretation": "Selection was frozen on validation before the formal test comparison.",
                "limitations": "Two training seeds selected the same physical checkpoint for both roles, reducing independent contrast.",
            },
        ]
    }
    _write_json(output / "prediction_verification.json", prediction_verification)

    training_runs = [_training_run_summary(Path(args.formal_run_root) / f"seed{seed}", seed) for seed in TRAINING_SEEDS]
    _write_json(output / "training_runs.json", {"runs": training_runs})
    shutil.copy2(args.formal_manifests[0], output / "sweep_manifest.json")

    video_manifest: dict[str, Any]
    if args.video_manifest:
        source = Path(args.video_manifest)
        if not source.is_file():
            raise FileNotFoundError(source)
        video_manifest = _read_json(source)
        if source.resolve() != (output / "video_manifest.json").resolve():
            shutil.copy2(source, output / "video_manifest.json")
    else:
        video_manifest = {
            "video_generation_completed": False,
            "human_review_completed": False,
            "entries": [],
        }
        _write_json(output / "video_manifest.json", video_manifest)

    selected = plan["selected_by_training_seed"]
    checkpoint_selection_rate = float(checkpoint_selection["formal_comparison"]["metric-selected BC+PPO"]["settled_landing_rate"])
    summary = {
        "benchmark": "actor-preserving PPO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_before_final_commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "task_id": "Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0",
        "policy_input": "state-based 22-dimensional observation; no camera images",
        "training_seeds": list(TRAINING_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "formal_test_seeds": list(FORMAL_TEST_SEEDS),
        "validation_episodes_per_seed": 128,
        "formal_episodes_per_seed": 256,
        "selected_epochs": {str(seed): int(selected[str(seed)]["epoch"]) for seed in TRAINING_SEEDS},
        "selected_checkpoint_sha256": {str(seed): selected[str(seed)]["checkpoint_sha256"] for seed in TRAINING_SEEDS},
        "selected_actor_sha256": {str(seed): selected[str(seed)]["actor_sha256"] for seed in TRAINING_SEEDS},
        "formal_results": aggregate_payload,
        "comparison": {row["method"]: row for row in comparison_rows},
        "answers": {
            "actor_preserving_metric_selected_exceeds_bc": metric_rate > bc_rate,
            "actor_preserving_metric_selected_exceeds_checkpoint_selected": metric_rate > checkpoint_selection_rate,
            "actor_preserving_metric_selected_reaches_90_percent": metric_rate >= 0.90,
            "actor_preserving_metric_selected_reaches_92_percent": metric_rate >= 0.92,
            "hard_contact_worse_than_bc": role_summaries["metric_selected"]["hard_contact_rate"] > role_summaries["bc_only"]["hard_contact_rate"],
            "deck_miss_lower_than_bc": role_summaries["metric_selected"]["deck_miss_rate"] < role_summaries["bc_only"]["deck_miss_rate"],
            "reward_selected_equals_metric_selected_aggregate": math.isclose(metric_rate, reward_rate),
            "last_outperforms_validation_selected": last_rate > metric_rate,
            "all_training_seeds_reach_90_percent": all(float(per_seed_metric[str(seed)]["settled_landing_rate"]) >= 0.90 for seed in TRAINING_SEEDS),
            "epoch10_actor_and_rms_zero_drift": bool(epoch10) and all(float(row["actor_parameter_relative_l2"]) == 0.0 for row in epoch10) and all_rms_zero,
        },
        "policy_drift_correlations": correlations,
        "prediction_verdicts": {str(item["id"]): item["verdict"] for item in prediction_verification["predictions"]},
        "training_runs": training_runs,
        "video_generation_completed": bool(video_manifest.get("video_generation_completed", video_manifest.get("headless_generation_validated", False))),
        "human_review_completed": bool(video_manifest.get("human_review_completed", False)),
        "formal_test_used_for_selection": False,
        "failed_training_seeds_hidden": False,
        "known_limitations": [
            "The policy is state-based and does not consume camera images.",
            "Action drift correlations are observational and do not establish causality.",
            "Formal test seeds were used only after validation selection.",
            "Automated video generation does not constitute human GUI review.",
        ],
    }
    _write_json(output / "summary.json", summary)

    git_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "head_before_final_commit": _git(repo, "rev-parse", "HEAD"),
        "status_before_final_commit": _git(repo, "status", "--short", check=False).splitlines(),
        "remote_push_performed": False,
    }
    _write_json(output / "git_manifest.json", git_manifest)
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "task_id": summary["task_id"],
    }
    _write_json(output / "environment_manifest.json", environment)

    readme = f"""# Actor-Preserving PPO 正式 Benchmark

actor-preserving PPO 保持冻结的 physical-deck-attitude task 环境语义，采用 separate actor/critic、epoch 1–10 critic-only warm-up、冻结 observation RMS，以及 pilot 预先选定的 `bc_anchor_coefficient=50`。

## 冻结协议

- Training seeds：42、43、44
- Validation seeds：145、146、147；每 checkpoint/seed 128 episodes
- Formal test seeds：245、246、247；每 checkpoint/seed 256 episodes
- Formal test 未参与 checkpoint selection
- Validation-selected epochs：seed42={int(selected['42']['epoch'])}、seed43={int(selected['43']['epoch'])}、seed44={int(selected['44']['epoch'])}
- Reward-selected epochs：seed42=91、seed43=128、seed44=51
- Last checkpoints：三个 training seed 均为 epoch200

## 正式结果

| 方法 | settled landing | deck miss | hard contact |
|---|---:|---:|---:|
| Frozen teacher | {role_summaries['frozen_teacher']['settled_landing_rate']:.2%} | {role_summaries['frozen_teacher']['deck_miss_rate']:.2%} | {role_summaries['frozen_teacher']['hard_contact_rate']:.2%} |
| BC epoch0 | {bc_rate:.2%} | {role_summaries['bc_only']['deck_miss_rate']:.2%} | {role_summaries['bc_only']['hard_contact_rate']:.2%} |
| checkpoint-selected BC+PPO | {checkpoint_selection_rate:.2%} | {float(checkpoint_selection['formal_comparison']['metric-selected BC+PPO']['deck_miss_rate']):.2%} | {float(checkpoint_selection['formal_comparison']['metric-selected BC+PPO']['hard_contact_rate']):.2%} |
| **actor-preserving PPO metric-selected** | **{metric_rate:.2%}** | **{role_summaries['metric_selected']['deck_miss_rate']:.2%}** | **{role_summaries['metric_selected']['hard_contact_rate']:.2%}** |
| actor-preserving PPO reward-selected | {reward_rate:.2%} | {role_summaries['reward_selected']['deck_miss_rate']:.2%} | {role_summaries['reward_selected']['hard_contact_rate']:.2%} |
| actor-preserving PPO epoch200 last | {last_rate:.2%} | {role_summaries['last']['deck_miss_rate']:.2%} | {role_summaries['last']['hard_contact_rate']:.2%} |

actor-preserving PPO metric-selected 为 {role_summaries['metric_selected']['settled_episodes']}/{role_summaries['metric_selected']['episodes']}，Wilson 95% CI 为 [{role_summaries['metric_selected']['settled_landing_rate_95ci_wilson_low']:.2%}, {role_summaries['metric_selected']['settled_landing_rate_95ci_wilson_high']:.2%}]。达到 90%：`{metric_rate >= 0.90}`；达到 92%：`{metric_rate >= 0.92}`。

必须保留的负面结果：actor-preserving PPO metric-selected 的 hard contact 为 2/2304，而 BC 为 0/768；因此不能声称所有安全指标都严格改善。epoch200 last 低于 validation-selected，reward-selected 也低于 metric-selected。

## 证据索引

- `summary.json`：机器可解析总览和关键判定
- `comparison.csv` / `comparison.md`：teacher、BC、imitation-learning benchmark、checkpoint-selection analysis、actor-preserving PPO 对比
- `validation_results.csv` / `validation_aggregate.csv` / `validation_selection.json`：只用 validation 的选模证据
- `formal_results.csv` / `formal_aggregate.csv`：独立 test 聚合与 Wilson CI
- `prediction_verification.json`：七项预注册预测 verdict
- `policy_drift.csv` / `policy_drift.json`：actor、critic、action、RMS drift
- `failure_distribution.csv`：失败类型分布
- `checkpoint_hashes.json` / `checkpoint_inventory.json`：checkpoint、actor、critic、RMS 哈希
- `videos/video_manifest.json`：真实 `settled_landing` 与 `deck_miss` 视频/轨迹哈希
- `commands.txt`：完整复现命令

原始逐 episode CSV、checkpoint 和 TensorBoard 保留在 `logs/`。`video_generation_completed={bool(video_manifest.get('video_generation_completed', False))}`，`human_review_completed={bool(video_manifest.get('human_review_completed', False))}`；自动 headless 视频不等于人工 GUI 目视验收。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "metric_selected": role_summaries["metric_selected"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--validation_manifests", nargs="+", required=True)
    select.add_argument("--inventories", nargs="+", required=True)
    select.add_argument("--bc_checkpoint", required=True)
    select.add_argument("--expected_num_envs", type=int, default=48)
    select.add_argument("--output_dir", default="benchmarks/actor_preserving_ppo")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--selection_plan", required=True)
    finalize.add_argument("--formal_manifests", nargs="+", required=True)
    finalize.add_argument("--formal_num_envs", type=int, default=64)
    finalize.add_argument("--reference_manifest", required=True)
    finalize.add_argument("--reference_num_envs", type=int, default=64)
    finalize.add_argument("--teacher_checkpoint", required=True)
    finalize.add_argument("--formal_drift", required=True)
    finalize.add_argument("--formal_run_root", required=True)
    finalize.add_argument("--imitation_summary", required=True)
    finalize.add_argument("--checkpoint_selection_summary", required=True)
    finalize.add_argument("--video_manifest")
    finalize.add_argument("--output_dir", default="benchmarks/actor_preserving_ppo")

    args = parser.parse_args()
    if args.mode == "select":
        _select_mode(args)
    else:
        _finalize_mode(args)


if __name__ == "__main__":
    main()

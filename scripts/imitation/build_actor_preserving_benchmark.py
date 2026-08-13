"""Build reproducible actor-preserving PPO pilot/formal benchmark tables from raw manifests and drift JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "settled_landing_rate",
    "contact_success_rate",
    "hard_contact_rate",
    "ground_crash_rate",
    "deck_miss_rate",
    "timeout_rate",
    "touchdown_distance_mean_m",
)
SELECTION_RULE = [
    "maximize validation settled_landing_rate",
    "minimize deck_miss_rate",
    "minimize hard_contact_rate",
    "minimize touchdown_distance_mean_m",
    "use lower action drift only as a later diagnostic tie-break",
    "prefer the smaller coefficient if still tied",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _coefficient(path: str) -> float | None:
    match = re.search(r"actor_preserving_(?:pilot|formal)_lambda(?P<value>\d+(?:p\d+)?)", path)
    if match:
        return float(match.group("value").replace("p", "."))
    return None


def _aggregate_manifest(manifest_path: Path, drift_path: Path | None = None) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    completed = [entry for entry in manifest.get("entries", {}).values() if entry.get("status") == "completed"]
    if not completed or any(entry.get("status") == "failed" for entry in manifest.get("entries", {}).values()):
        raise ValueError(f"manifest is empty or contains failed entries: {manifest_path}")
    by_path: dict[str, list[dict[str, Any]]] = {}
    for entry in completed:
        by_path.setdefault(str(Path(entry["checkpoint_path"]).resolve()), []).append(entry)

    drift_by_path: dict[str, dict[str, Any]] = {}
    if drift_path is not None:
        payload = _load_json(drift_path)
        drift_by_path = {
            str(Path(row["checkpoint_path"]).resolve()): row for row in payload.get("records", [])
        }

    rows: list[dict[str, Any]] = []
    for checkpoint_path, entries in by_path.items():
        episodes = sum(int(entry["episodes"]) for entry in entries)
        first = entries[0]
        row: dict[str, Any] = {
            "coefficient": _coefficient(checkpoint_path),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": first["checkpoint_sha256"],
            "actor_sha256": first["actor_sha256"],
            "train_seed": first.get("train_seed"),
            "epoch": int(first["epoch"]),
            "kind": first["kind"],
            "training_reward": first.get("training_reward"),
            "eval_seeds": ",".join(str(value) for value in sorted(int(entry["eval_seed"]) for entry in entries)),
            "episodes": episodes,
        }
        for metric in METRICS:
            values = [float(entry["metrics"][metric]) for entry in entries]
            row[metric] = sum(
                float(entry["metrics"][metric]) * int(entry["episodes"]) for entry in entries
            ) / episodes
            row[f"{metric}_seed_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        drift = drift_by_path.get(checkpoint_path, {})
        for key in (
            "critic_sha256",
            "observation_rms_sha256",
            "action_mse_vs_reference",
            "action_max_abs_error_vs_reference",
            "actor_parameter_relative_l2",
            "critic_parameter_relative_l2",
            "observation_mean_l2",
            "observation_variance_l2",
            "observation_count_delta",
            "fixed_sigma_l2",
        ):
            row[key] = drift.get(key)
        rows.append(row)
    rows.sort(key=lambda row: (row["coefficient"] is None, row["coefficient"] or -1.0, row["epoch"], row["kind"]))
    return rows


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    drift = row.get("action_mse_vs_reference")
    return (
        -float(row["settled_landing_rate"]),
        float(row["deck_miss_rate"]),
        float(row["hard_contact_rate"]),
        float(row["touchdown_distance_mean_m"]),
        float(drift) if drift is not None else math.inf,
        float(row["coefficient"]) if row["coefficient"] is not None else math.inf,
        int(row["epoch"]),
    )


def _build_pilot(output: Path, manifest: Path, drift: Path) -> dict[str, Any]:
    rows = _aggregate_manifest(manifest, drift)
    _write_csv(output / "pilot_results.csv", rows)
    candidate_rows = [row for row in rows if row["coefficient"] is not None]
    best_by_coefficient: dict[str, dict[str, Any]] = {}
    for coefficient in sorted({float(row["coefficient"]) for row in candidate_rows}):
        selected = min(
            (row for row in candidate_rows if float(row["coefficient"]) == coefficient),
            key=_selection_key,
        )
        best_by_coefficient[str(coefficient)] = selected
    selected = min(best_by_coefficient.values(), key=_selection_key)
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_manifest": str(manifest.resolve()),
        "raw_policy_drift": str(drift.resolve()),
        "selection_rule": SELECTION_RULE,
        "candidates": [0.0, 10.0, 50.0],
        "best_by_coefficient": best_by_coefficient,
        "selected_coefficient": float(selected["coefficient"]),
        "selected_checkpoint": selected["checkpoint_path"],
        "selected_epoch": int(selected["epoch"]),
        "formal_test_seeds_used": False,
        "targeted_revision_after_pilot": False,
    }
    _write_json(output / "pilot_summary.json", summary)
    return summary


def _write_preregistered_config(output: Path, selected_coefficient: float) -> None:
    payload = {
        "actor_preserving_preregistered_config": {
            "task_id": "Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0",
            "observation_dim": 22,
            "action_dim": 4,
            "network": {"separate": True, "units": [64, 64], "activation": "elu", "fixed_sigma": True},
            "migration": {"schema_version": "actor-preserving-separate-v1", "critic_seed": 2026, "parity_max_abs_error": 1.0e-5},
            "warmup_epochs": 10,
            "freeze_lr_scheduler_during_warmup": True,
            "freeze_observation_rms": True,
            "bc_anchor": {
                "type": "mse_mean_action",
                "coefficient": selected_coefficient,
                "pilot_candidates": [0.0, 10.0, 50.0],
            },
            "training": {"num_envs": 256, "horizon_length": 24, "max_epochs": 200, "environment_steps_per_seed": 1228800},
            "seeds": {
                "training": [42, 43, 44],
                "validation": [145, 146, 147],
                "formal_test": [245, 246, 247],
            },
            "evaluation": {"validation_episodes_per_seed": 128, "formal_test_episodes_per_seed": 256},
        }
    }
    (output / "preregistered_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/actor_preserving_ppo")
    parser.add_argument("--pilot_manifest", required=True)
    parser.add_argument("--pilot_drift", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pilot = _build_pilot(output, Path(args.pilot_manifest).resolve(), Path(args.pilot_drift).resolve())
    _write_preregistered_config(output, float(pilot["selected_coefficient"]))
    _write_json(
        output / "environment_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "python": "/home/j/anaconda3/envs/env_isaaclab/bin/python",
            "display": None,
            "headless_training_and_evaluation": True,
        },
    )
    print(json.dumps({"output": str(output), "selected_coefficient": pilot["selected_coefficient"]}, indent=2))


if __name__ == "__main__":
    main()

"""Select checkpoint-selection analysis candidates and build the reproducible checkpoint-selection benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from quadcopter_waypoint.imitation.checkpoint_sweep import (
    aggregate_checkpoint_rows,
    select_screening_candidates,
    select_validation_best,
)

RATE_METRICS = (
    "settled_landing_rate",
    "contact_success_rate",
    "deck_miss_rate",
    "hard_contact_rate",
    "ground_crash_rate",
    "timeout_rate",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest = _read_json(path)
    rows = []
    for entry in manifest.get("entries", {}).values():
        if entry.get("status") != "completed":
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"completed manifest entry has no metrics: {entry.get('checkpoint_path')}")
        rows.append(
            {
                "checkpoint_path": entry["checkpoint_path"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "actor_sha256": entry["actor_sha256"],
                "train_seed": entry.get("train_seed"),
                "epoch": int(entry["epoch"]),
                "training_reward": entry.get("training_reward"),
                "kind": entry["kind"],
                "eval_seed": int(entry["eval_seed"]),
                "episodes": int(entry["episodes"]),
                "num_envs": int(entry["num_envs"]),
                "output_csv": entry["output_csv"],
                **{key: value for key, value in metrics.items() if key != "csv"},
            }
        )
    if not rows:
        raise ValueError(f"manifest has no completed entries: {path}")
    return sorted(rows, key=lambda row: (row["train_seed"] is None, row["train_seed"] or -1, row["epoch"], row["kind"], row["eval_seed"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _family_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("method family cannot be empty")
    summary: dict[str, Any] = {
        "evaluations": len(values),
        "episodes": sum(int(row["episodes"]) for row in values),
        "checkpoints": len({str(row["checkpoint_sha256"]) for row in values}),
        "eval_seeds": sorted({int(row["eval_seed"]) for row in values}),
    }
    for metric in (*RATE_METRICS, "touchdown_distance_mean_m"):
        metric_values = [float(row[metric]) for row in values if row.get(metric) is not None]
        if metric_values:
            summary[metric] = mean(metric_values)
            summary[f"{metric}_std"] = pstdev(metric_values)
    return summary


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _drift_correlations(screening_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]]) -> dict[str, Any]:
    drift_by_sha = {row["checkpoint_sha256"]: row for row in drift_rows}
    paired = [(row, drift_by_sha[row["checkpoint_sha256"]]) for row in screening_rows if row["checkpoint_sha256"] in drift_by_sha]
    x = [float(drift["action_mse_vs_bc"]) for _, drift in paired]
    settled = [float(row["settled_landing_rate"]) for row, _ in paired]
    miss = [float(row["deck_miss_rate"]) for row, _ in paired]
    hard = [float(row["hard_contact_rate"]) for row, _ in paired]
    result = {
        "paired_checkpoints": len(paired),
        "pearson_action_drift_vs_settled_landing": _correlation(x, settled),
        "pearson_action_drift_vs_deck_miss": _correlation(x, miss),
        "pearson_action_drift_vs_hard_contact": _correlation(x, hard),
    }
    correlations = [
        abs(value)
        for value in (
            result["pearson_action_drift_vs_settled_landing"],
            result["pearson_action_drift_vs_deck_miss"],
            result["pearson_action_drift_vs_hard_contact"],
        )
        if value is not None
    ]
    maximum = max(correlations, default=0.0)
    result["association_strength"] = "strong" if maximum >= 0.7 else "moderate" if maximum >= 0.4 else "weak"
    result["clear_statistical_association"] = len(paired) >= 10 and maximum >= 0.6
    return result


def _plot_epoch(rows: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    figure = plt.figure(figsize=(8, 5))
    for seed in sorted({row["train_seed"] for row in rows if row["train_seed"] is not None}):
        selected = sorted(
            [row for row in rows if row["train_seed"] == seed and row["kind"] == "periodic"],
            key=lambda row: row["epoch"],
        )
        plt.plot([row["epoch"] for row in selected], [row[metric] for row in selected], marker="o", label=f"train seed {seed}")
    plt.xlabel("PPO epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_scatter(rows: list[dict[str, Any]], x: str, y: str, xlabel: str, ylabel: str, path: Path) -> None:
    selected = [row for row in rows if row.get(x) is not None and row.get(y) is not None]
    figure = plt.figure(figsize=(7, 5))
    for seed in sorted({row.get("train_seed") for row in selected if row.get("train_seed") is not None}):
        seed_rows = [row for row in selected if row.get("train_seed") == seed]
        plt.scatter([row[x] for row in seed_rows], [row[y] for row in seed_rows], label=f"train seed {seed}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_validation_error(aggregated: list[dict[str, Any]], path: Path) -> None:
    figure = plt.figure(figsize=(8, 5))
    for seed in sorted({row["train_seed"] for row in aggregated if row["train_seed"] is not None}):
        selected = sorted([row for row in aggregated if row["train_seed"] == seed], key=lambda row: row["epoch"])
        plt.errorbar(
            [row["epoch"] for row in selected],
            [row["settled_landing_rate"] for row in selected],
            yerr=[row["settled_landing_rate_std"] for row in selected],
            marker="o",
            label=f"train seed {seed}",
        )
    plt.xlabel("PPO epoch")
    plt.ylabel("Validation settled landing rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _copy_text_normalized(source: str | Path, target: Path) -> None:
    """Copy a UTF-8 text artifact while normalizing line endings for Git."""
    value = Path(source).read_text(encoding="utf-8")
    target.write_text(value.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")


def _copy_formal_csvs(output: Path, rows: list[dict[str, Any]]) -> None:
    destination = output / "formal_evaluations"
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        seed = "global" if row["train_seed"] is None else f"train_seed{row['train_seed']}"
        target_dir = destination / f"{row['kind']}_{seed}_ep{row['epoch']}"
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_text_normalized(row["output_csv"], target_dir / f"test_seed{row['eval_seed']}.csv")


def _screening_mode(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    rows = _manifest_rows(args.screening_manifest)
    _write_csv(output / "screening_results.csv", rows)
    candidates = select_screening_candidates(rows, top_k=5)
    paths = list(dict.fromkeys(path for seed_paths in candidates.values() for path in seed_paths))
    _write_json(output / "screening_candidates.json", {"per_train_seed": candidates, "checkpoint_paths": paths})
    print(json.dumps({"screening_rows": len(rows), "candidate_checkpoints": len(paths)}, indent=2))


def _validation_mode(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    rows = _manifest_rows(args.validation_manifest)
    _write_csv(output / "validation_results.csv", rows)
    selected = select_validation_best(rows)
    _write_json(output / "validation_selection.json", {str(seed): value for seed, value in selected.items()})
    print(json.dumps({"validation_rows": len(rows), "selected": {seed: row["checkpoint_path"] for seed, row in selected.items()}}, indent=2))


def _finalize_mode(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    screening_rows = _manifest_rows(args.screening_manifest)
    validation_rows = _manifest_rows(args.validation_manifest)
    formal_rows = _manifest_rows(args.formal_manifest)
    drift_payload = _read_json(args.drift_json)
    drift_rows = drift_payload["records"]
    inventory_payload = _read_json(args.inventory)
    selection = select_validation_best(validation_rows)
    selected_paths = {row["checkpoint_path"] for row in selection.values()}
    teacher_path = str(Path(args.teacher_checkpoint).expanduser().resolve())
    bc_path = str(Path(args.bc_checkpoint).expanduser().resolve())

    for row in formal_rows:
        if row["checkpoint_path"] == teacher_path:
            row["method"] = "frozen PPO teacher"
        elif row["checkpoint_path"] == bc_path:
            row["method"] = "BC epoch 0"
        elif row["checkpoint_path"] in selected_paths:
            row["method"] = "metric-selected BC+PPO"
        elif row["kind"] == "reward_selected":
            row["method"] = "reward-selected BC+PPO"
        else:
            raise ValueError(f"unclassified formal checkpoint: {row['checkpoint_path']}")

    _write_csv(output / "screening_results.csv", screening_rows)
    _write_csv(output / "validation_results.csv", validation_rows)
    _write_csv(output / "formal_results.csv", formal_rows)
    shutil.copy2(args.inventory, output / "checkpoint_inventory.json")
    shutil.copy2(args.drift_json, output / "checkpoint_drift.json")
    _copy_text_normalized(Path(args.drift_json).with_suffix(".csv"), output / "checkpoint_drift.csv")
    _copy_formal_csvs(output, formal_rows)

    combined_manifest = {
        "screening": _read_json(args.screening_manifest),
        "validation": _read_json(args.validation_manifest),
        "formal": _read_json(args.formal_manifest),
    }
    _write_json(output / "sweep_manifest.json", combined_manifest)

    methods = sorted({row["method"] for row in formal_rows})
    comparison_rows = []
    method_summaries = {}
    for method in methods:
        summary = _family_summary(row for row in formal_rows if row["method"] == method)
        method_summaries[method] = summary
        comparison_rows.append({"method": method, **summary})
    _write_csv(output / "comparison.csv", comparison_rows)
    lines = [
        "| Method | Episodes | Settled landing | Deck miss | Hard contact | Touchdown distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['method']} | {row['episodes']} | {100 * row['settled_landing_rate']:.2f}% ± "
            f"{100 * row['settled_landing_rate_std']:.2f}% | {100 * row['deck_miss_rate']:.2f}% | "
            f"{100 * row['hard_contact_rate']:.2f}% | {row['touchdown_distance_mean_m']:.4f} m |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation_aggregated = aggregate_checkpoint_rows(validation_rows)
    formal_checkpoint_aggregated = aggregate_checkpoint_rows(formal_rows)
    best_validation = max(
        (row for row in validation_aggregated if row["kind"] == "periodic"),
        key=lambda row: row["settled_landing_rate"],
    )
    best_test = max(
        (
            row
            for row in formal_checkpoint_aggregated
            if row["train_seed"] is not None and row["kind"] in {"periodic", "reward_selected"}
        ),
        key=lambda row: row["settled_landing_rate"],
    )
    reward_by_seed = {
        int(row["train_seed"]): row
        for row in validation_aggregated
        if row["kind"] == "reward_selected" and row["train_seed"] is not None
    }
    equality = {
        str(seed): {
            "metric_selected_epoch": selection[seed]["epoch"],
            "reward_selected_epoch": reward_by_seed[seed]["epoch"],
            "same_checkpoint_sha256": selection[seed]["checkpoint_sha256"] == reward_by_seed[seed]["checkpoint_sha256"],
            "same_actor_sha256": selection[seed]["actor_sha256"] == reward_by_seed[seed]["actor_sha256"],
        }
        for seed in sorted(selection)
    }
    correlations = _drift_correlations(screening_rows, drift_rows)
    drift_by_seed_epoch = {
        (row.get("train_seed"), int(row["epoch"])): row for row in drift_rows if row["kind"] == "periodic"
    }
    bc_screen = next(row for row in screening_rows if row["kind"] == "bc_init")
    first_saved = []
    for seed in sorted(selection):
        row = next(row for row in screening_rows if row["train_seed"] == seed and row["kind"] == "periodic" and row["epoch"] == 10)
        drift = drift_by_seed_epoch[(seed, 10)]
        first_saved.append(
            {
                "train_seed": seed,
                "epoch": 10,
                "settled_landing_rate": row["settled_landing_rate"],
                "delta_vs_bc": row["settled_landing_rate"] - bc_screen["settled_landing_rate"],
                "action_mse_vs_bc": drift["action_mse_vs_bc"],
            }
        )

    bc_summary = method_summaries["BC epoch 0"]
    metric_summary = method_summaries["metric-selected BC+PPO"]
    improvement_pp = 100 * (metric_summary["settled_landing_rate"] - bc_summary["settled_landing_rate"])
    hard_change_pp = 100 * (metric_summary["hard_contact_rate"] - bc_summary["hard_contact_rate"])
    positive_case = improvement_pp >= 1.0 and hard_change_pp <= 1.0
    summary = {
        "benchmark": "checkpoint-selection analysis periodic checkpoint selection and policy drift diagnosis",
        "checkpoint_selection": {
            "validation_selected": {str(seed): row for seed, row in selection.items()},
            "reward_vs_metric_selected": equality,
            "all_reward_selected_equal_metric_selected": all(value["same_actor_sha256"] for value in equality.values()),
            "best_periodic_validation": best_validation,
            "best_evaluated_bc_ppo_test": best_test,
        },
        "execution": {
            "physical_checkpoint_files": inventory_payload["checkpoint_count"],
            "canonical_policies": inventory_payload["canonical_count"],
            "screening_evaluations": len(screening_rows),
            "screening_episodes": sum(int(row["episodes"]) for row in screening_rows),
            "validation_evaluations": len(validation_rows),
            "validation_episodes": sum(int(row["episodes"]) for row in validation_rows),
            "formal_evaluations": len(formal_rows),
            "formal_episodes": sum(int(row["episodes"]) for row in formal_rows),
            "closed_loop_episodes_total": sum(
                int(row["episodes"]) for row in screening_rows + validation_rows + formal_rows
            ),
            "drift_checkpoints": len(drift_rows),
            "drift_test_transitions_per_checkpoint": drift_payload["transitions"],
        },
        "formal_comparison": method_summaries,
        "answers": {
            "metric_selected_improvement_over_bc_percentage_points": improvement_pp,
            "metric_selected_hard_contact_change_percentage_points": hard_change_pp,
            "existing_periodic_checkpoint_clearly_exceeds_bc": positive_case,
            "metric_selected_reaches_90_percent": metric_summary["settled_landing_rate"] >= 0.90,
            "metric_selected_reaches_92_percent": metric_summary["settled_landing_rate"] >= 0.92,
            "reward_and_settled_selection_are_consistent": all(value["same_actor_sha256"] for value in equality.values()),
            "policy_drift_has_clear_statistical_association": correlations["clear_statistical_association"],
            "degradation_timing": (
                "The first available periodic snapshot is epoch 10, not the first PPO update. "
                "Epoch-10 results quantify whether degradation was already present by the earliest saved checkpoint."
            ),
            "first_saved_checkpoint_vs_bc": first_saved,
            "decision_case": "A" if positive_case else "B",
            "checkpoint_reselection_sufficient_for_positive_checkpoint_selection": positive_case,
            "actor_preserving_ppo_recommended_for_further_improvement": (
                not positive_case or metric_summary["settled_landing_rate"] < 0.92
            ),
        },
        "policy_drift_correlations": correlations,
        "drift_analysis": {
            key: value for key, value in drift_payload.items() if key != "records"
        },
        "limitations": [
            "Checkpoint selection uses validation seeds 145/146/147; test seeds 245/246/247 are used only after selection.",
            "Policy-drift correlations are observational statistics and do not establish strict causality.",
            "The earliest periodic checkpoint is epoch 10, so the exact first-update transition is not directly observed.",
            "checkpoint-selection analysis does not retrain or modify the frozen physical-deck-attitude task environment semantics.",
        ],
    }
    _write_json(output / "summary.json", summary)

    drift_lookup = {row["checkpoint_sha256"]: row for row in drift_rows}
    screening_with_drift = [
        {**row, **{key: value for key, value in drift_lookup[row["checkpoint_sha256"]].items() if key.startswith("action_") or key.startswith("actor_")}}
        for row in screening_rows
        if row["checkpoint_sha256"] in drift_lookup
    ]
    _plot_epoch(screening_rows, "settled_landing_rate", "Settled landing rate", output / "settled_landing_vs_epoch.png")
    _plot_epoch(screening_rows, "deck_miss_rate", "Deck miss rate", output / "deck_miss_vs_epoch.png")
    _plot_scatter(screening_rows, "training_reward", "settled_landing_rate", "Training rolling mean reward", "Settled landing rate", output / "training_reward_vs_settled_landing.png")
    _plot_epoch(screening_with_drift, "action_mse_vs_bc", "Action MSE vs BC", output / "bc_action_drift_vs_epoch.png")
    _plot_scatter(screening_with_drift, "action_mse_vs_bc", "settled_landing_rate", "Action MSE vs BC", "Settled landing rate", output / "action_drift_vs_settled_landing.png")
    _plot_validation_error(validation_aggregated, output / "validation_settled_landing_error_bars.png")

    commands = []
    for manifest in combined_manifest.values():
        commands.extend(entry["command"] for entry in manifest["entries"].values() if entry.get("status") == "completed")
    (output / "commands.txt").write_text("\n".join(dict.fromkeys(commands)) + "\n", encoding="utf-8")

    conclusion = (
        "Existing checkpoint reselection produced a positive checkpoint-selection analysis result." if positive_case else
        "The best existing periodic checkpoint still did not clearly exceed BC-only; checkpoint reselection alone is insufficient."
    )
    readme = f"""# checkpoint-selection analysis Checkpoint Selection and Policy Drift Diagnosis

checkpoint-selection analysis keeps the physical-deck-attitude task environment, reward, observation, action, termination, and contact semantics frozen. It does not retrain.

## Result

{conclusion}

- BC epoch 0 settled landing: {100 * bc_summary['settled_landing_rate']:.2f}%
- Metric-selected BC+PPO settled landing: {100 * metric_summary['settled_landing_rate']:.2f}%
- Improvement over BC: {improvement_pp:.2f} percentage points
- Metric-selected hard-contact change: {hard_change_pp:.2f} percentage points
- Reaches 90%: {metric_summary['settled_landing_rate'] >= 0.90}
- Reaches 92%: {metric_summary['settled_landing_rate'] >= 0.92}
- Reward-selected actor equals metric-selected actor for every train seed: {summary['checkpoint_selection']['all_reward_selected_equal_metric_selected']}

## Protocol

1. Screening: eval seed 145, 64 episodes, all epoch-0/10/.../200 and reward-selected checkpoints.
2. Validation: seeds 145/146/147, 128 episodes per seed, screening Top-5 plus reward-selected and BC.
3. Independent test: seeds 245/246/247, 256 episodes per seed, teacher, BC, validation-selected, and reward-selected policies.
4. Drift: deterministic checkpoint-specific normalized actions on the frozen imitation-learning benchmark test split.

## Interpretation

Policy-drift correlations are observational and are not causal claims. The measured association is {correlations['association_strength']}, mainly through lower settled landing and higher deck miss as action drift grows. The earliest saved PPO snapshot is epoch 10, so checkpoint-selection analysis can determine whether degradation is already visible by epoch 10, but cannot directly observe the first gradient update.

## actor-preserving PPO Recommendation

Checkpoint reselection is sufficient for the positive checkpoint-selection analysis result, but the selected policies still do not reach 92% and drift continues after early PPO updates. For further improvement, use actor-preserving PPO: separate actor/critic handling, critic warm-up, temporary actor freezing, a BC actor anchor (KL or L2), and validation settled landing for checkpoint selection. Do not modify the frozen environment reward merely to improve this benchmark.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "decision_case": summary["answers"]["decision_case"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    screening = subparsers.add_parser("screening")
    screening.add_argument("--screening_manifest", required=True)
    screening.add_argument("--output_dir", required=True)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--validation_manifest", required=True)
    validation.add_argument("--output_dir", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--screening_manifest", required=True)
    finalize.add_argument("--validation_manifest", required=True)
    finalize.add_argument("--formal_manifest", required=True)
    finalize.add_argument("--inventory", required=True)
    finalize.add_argument("--drift_json", required=True)
    finalize.add_argument("--teacher_checkpoint", required=True)
    finalize.add_argument("--bc_checkpoint", required=True)
    finalize.add_argument("--output_dir", default="benchmarks/checkpoint_selection")

    args = parser.parse_args()
    if args.mode == "screening":
        _screening_mode(args)
    elif args.mode == "validation":
        _validation_mode(args)
    else:
        _finalize_mode(args)


if __name__ == "__main__":
    main()

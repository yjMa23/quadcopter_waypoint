"""Build the committed imitation-learning benchmark benchmark, tables, and figures from raw experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import torch

from quadcopter_waypoint.imitation.benchmark import (
    aggregate_seed_summaries,
    summarize_evaluation_csv,
    threshold_crossing_steps,
)
from quadcopter_waypoint.imitation.dataset import sha256_file

SEEDS = (42, 43, 44)
CURVE_EPOCHS = (20, 50, 100, 150, 200)
STEPS_PER_EPOCH = 256 * 24
TASK_ID = "Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0"
TEACHER_RELATIVE = Path(
    "logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/expanded_from_physical_deck_ep990_16to22.pth"
)
DATASET_MANIFEST_RELATIVE = Path("logs/imitation/expert_dataset/manifest.json")
BC_RELATIVE = Path("logs/imitation/behavior_cloning/best_bc.pth")
BC_INIT_RELATIVE = Path("logs/imitation/behavior_cloning/bc_init_rlgames.pth")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _copy_csvs(output: Path, name: str, sources: dict[str, Path]) -> dict[str, Path]:
    destination = output / "formal_evaluations" / name
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for seed, source in sources.items():
        target = destination / f"seed{seed}.csv"
        shutil.copy2(source, target)
        copied[seed] = target
    return copied


def _summarize_method(paths: dict[str, Path]) -> dict[str, Any]:
    per_seed = {seed: summarize_evaluation_csv(path) for seed, path in paths.items()}
    return {"per_seed": per_seed, "aggregate": aggregate_seed_summaries(per_seed)}


def _curve(repo: Path, method: str, include_bc_step_zero: float | None) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if include_bc_step_zero is not None:
        values.append(
            {
                "epoch": 0,
                "environment_steps": 0,
                "settled_landing_rate": include_bc_step_zero,
                "contact_success_rate": None,
                "hard_contact_rate": None,
                "touchdown_distance_mean_m": None,
                "evaluation_episodes": 768,
                "source": "BC initialization before online PPO updates",
            }
        )
    for epoch in CURVE_EPOCHS:
        summaries = [
            summarize_evaluation_csv(
                repo / f"logs/imitation/learning_curves/{method}/seed{seed}_epoch{epoch}.csv"
            )
            for seed in SEEDS
        ]
        row: dict[str, Any] = {
            "epoch": epoch,
            "environment_steps": epoch * STEPS_PER_EPOCH,
            "evaluation_episodes": 128 * len(SEEDS),
            "source": "fixed checkpoint interval, 128 episodes per seed",
        }
        for metric in (
            "settled_landing_rate",
            "contact_success_rate",
            "hard_contact_rate",
            "touchdown_distance_mean_m",
        ):
            metric_values = [float(summary[metric]) for summary in summaries if summary.get(metric) is not None]
            row[metric] = mean(metric_values) if metric_values else None
        values.append(row)
    return values


def _training_run(repo: Path, method: str, seed: int, learning_rate: float) -> dict[str, Any]:
    checkpoint = repo / f"logs/rl_games/{method}/seed{seed}/nn/{method}.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    timing = _read_json(repo / f"logs/rl_games/{method}/seed{seed}_time.json")
    return {
        "method": method,
        "seed": seed,
        "num_envs": 256,
        "horizon_length": 24,
        "epochs_budget": 200,
        "maximum_environment_steps": 200 * STEPS_PER_EPOCH,
        "selected_checkpoint": str(checkpoint.relative_to(repo)),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_selection_rule": "highest RL-Games rolling mean episode reward saved by the common training loop",
        "selected_epoch": int(payload.get("epoch", -1)),
        "selected_environment_steps": int(payload.get("frame", -1)),
        "selected_mean_reward": float(payload.get("last_mean_rewards", float("nan"))),
        "learning_rate": learning_rate,
        "wall_time_s": float(timing["wall_time_s"]),
        "maximum_rss_kb": int(timing["max_rss_kb"]),
    }


def _write_comparison(output: Path, methods: dict[str, dict[str, Any]]) -> None:
    rows = []
    for display_name, value in methods.items():
        aggregate = value["aggregate"]
        rows.append(
            {
                "method": display_name,
                "episodes": aggregate["episodes"],
                "contact_success_percent": 100 * aggregate["contact_success_rate"]["mean"],
                "settled_landing_percent": 100 * aggregate["settled_landing_rate"]["mean"],
                "settled_landing_std_percent": 100 * aggregate["settled_landing_rate"]["std"],
                "hard_contact_percent": 100 * aggregate["hard_contact_rate"]["mean"],
                "ground_crash_percent": 100 * aggregate["ground_crash_rate"]["mean"],
                "deck_miss_percent": 100 * aggregate["deck_miss_rate"]["mean"],
                "timeout_percent": 100 * aggregate["timeout_rate"]["mean"],
                "touchdown_distance_mean_m": aggregate.get("touchdown_distance_mean_m", {}).get("mean"),
                "first_contact_xy_error_mean_m": aggregate.get("first_contact_xy_error_mean_m", {}).get("mean"),
            }
        )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| 方法 | settled landing | contact | hard contact | deck miss | timeout |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['settled_landing_percent']:.2f}% ± {row['settled_landing_std_percent']:.2f}% "
            f"| {row['contact_success_percent']:.2f}% | {row['hard_contact_percent']:.2f}% "
            f"| {row['deck_miss_percent']:.2f}% | {row['timeout_percent']:.2f}% |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_learning_curve(output: Path, curves: dict[str, list[dict[str, Any]]], metric: str, ylabel: str, filename: str) -> None:
    figure = plt.figure(figsize=(8, 5))
    for name, rows in curves.items():
        selected = [row for row in rows if row.get(metric) is not None]
        plt.plot(
            [row["environment_steps"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            label=name,
        )
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output / filename, dpi=180)
    plt.close(figure)


def _plot_bc_loss(repo: Path, output: Path) -> None:
    rows: list[dict[str, str]]
    with (repo / "logs/imitation/behavior_cloning/loss_curves.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    figure = plt.figure(figsize=(8, 5))
    epochs = [float(row["epoch"]) for row in rows]
    plt.plot(epochs, [float(row["train_weighted_mse"]) for row in rows], label="train")
    plt.plot(epochs, [float(row["validation_weighted_mse"]) for row in rows], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted action MSE")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output / "bc_train_validation_loss.png", dpi=180)
    plt.close(figure)
    shutil.copy2(repo / "logs/imitation/behavior_cloning/loss_curves.csv", output / "bc_loss_curves.csv")


def _plot_final_metrics(output: Path, methods: dict[str, dict[str, Any]]) -> None:
    names = list(methods)
    metric_names = ["settled_landing_rate", "contact_success_rate", "hard_contact_rate", "deck_miss_rate"]
    labels = ["settled", "contact", "hard contact", "deck miss"]
    width = 0.18
    x = list(range(len(names)))
    figure = plt.figure(figsize=(10, 5))
    for index, (metric, label) in enumerate(zip(metric_names, labels)):
        values = [100 * methods[name]["aggregate"][metric]["mean"] for name in names]
        positions = [value + (index - 1.5) * width for value in x]
        plt.bar(positions, values, width=width, label=label)
    plt.xticks(x, names, rotation=12)
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.tight_layout()
    figure.savefig(output / "final_metrics_comparison.png", dpi=180)
    plt.close(figure)


def _plot_failures(output: Path, methods: dict[str, dict[str, Any]]) -> None:
    names = list(methods)
    categories = ("hard_contact", "ground_crash", "deck_miss", "timeout", "other")
    bottom = [0.0] * len(names)
    figure = plt.figure(figsize=(9, 5))
    for category in categories:
        values = []
        for name in names:
            aggregate = methods[name]["aggregate"]
            values.append(100 * aggregate["failure_counts"].get(category, 0) / aggregate["episodes"])
        plt.bar(names, values, bottom=bottom, label=category)
        bottom = [old + new for old, new in zip(bottom, values)]
    plt.ylabel("Episodes (%)")
    plt.xticks(rotation=12)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output / "failure_distribution.png", dpi=180)
    plt.close(figure)


def _commands() -> str:
    overrides = (
        "env.deck_roll_amplitude_min_deg=0.0 env.deck_roll_amplitude_max_deg=5.0 "
        "env.deck_pitch_amplitude_min_deg=0.0 env.deck_pitch_amplitude_max_deg=5.0 "
        "env.deck_roll_frequency_min=0.08 env.deck_roll_frequency_max=0.15 "
        "env.deck_pitch_frequency_min=0.08 env.deck_pitch_frequency_max=0.15"
    )
    return f"""# imitation-learning benchmark reproducibility commands (run from repository root)
export PYTHONPATH=source/quadcopter_waypoint
PY=/home/j/anaconda3/envs/env_isaaclab/bin/python
TASK={TASK_ID}
TEACHER={TEACHER_RELATIVE}

# Collection was run independently with seeds 42, 43, and 44.
$PY scripts/imitation/collect_teacher.py --task=$TASK --checkpoint=$TEACHER \\
  --output_dir logs/imitation/expert_dataset/seed_42 --seed=42 --num_envs=64 \\
  --successful_episodes=700 --transitions=180000 --episodes_per_shard=100 --max_steps=200000 --headless {overrides}

$PY scripts/imitation/finalize_dataset.py --dataset_dir logs/imitation/expert_dataset \\
  --split_seed=2026 --min_successful_episodes=2000 --min_transitions=500000

$PY scripts/imitation/train_bc.py --manifest={DATASET_MANIFEST_RELATIVE} --teacher_checkpoint=$TEACHER \\
  --output_dir=logs/imitation/behavior_cloning --seed=42 --epochs=50 --batch_size=4096 --learning_rate=1e-3 --patience=10

$PY scripts/imitation/create_bc_init_checkpoint.py --bc_checkpoint={BC_RELATIVE} \\
  --template_checkpoint=$TEACHER --manifest={DATASET_MANIFEST_RELATIVE} \\
  --output={BC_INIT_RELATIVE} --value_seed=2026

# Fair online runs: same task, 256 envs, seed set, 200 epochs, and PPO config.
$PY scripts/rl_games/train.py --task=$TASK --num_envs=256 --seed=42 --headless --max_iterations=200 \\
  agent.params.config.name=ppo_scratch +agent.params.config.full_experiment_name=seed42
$PY scripts/rl_games/train.py --task=$TASK --num_envs=256 --seed=42 --headless --max_iterations=200 \\
  --checkpoint={BC_INIT_RELATIVE} agent.params.config.name=bc_ppo \\
  +agent.params.config.full_experiment_name=seed42

# Formal evaluation pattern; repeat for seeds 42/43/44 and each selected checkpoint.
$PY scripts/rl_games/eval_metrics.py --task=$TASK --checkpoint=<CHECKPOINT> --num_envs=64 \\
  --episodes=256 --seed=42 --csv=<OUTPUT.csv> --headless {overrides}

$PY scripts/imitation/build_benchmark.py
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/imitation_hybrid")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    tree_dirty_before_generation = bool(_git(repo, "status", "--porcelain"))
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "PPO teacher": {
            str(seed): repo / f"logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/physical_deck_attitude_final_seed{seed}.csv"
            for seed in SEEDS
        },
        "BC only": {str(seed): repo / f"logs/imitation/behavior_cloning/formal_seed{seed}.csv" for seed in SEEDS},
        "PPO scratch": {
            str(seed): repo / f"logs/imitation/formal_evaluations/ppo_scratch/seed{seed}.csv" for seed in SEEDS
        },
        "BC+PPO": {
            str(seed): repo / f"logs/imitation/formal_evaluations/bc_ppo/seed{seed}.csv" for seed in SEEDS
        },
        "BC+PPO lr1e-5 diagnostic": {
            str(seed): repo / f"logs/imitation/formal_evaluations/bc_ppo_lr1e5/seed{seed}.csv" for seed in SEEDS
        },
    }
    copied_paths = {
        name: _copy_csvs(output, name.lower().replace(" ", "_").replace("+", "_plus_"), paths)
        for name, paths in source_paths.items()
    }
    rollout_output = output / "rollout_cases"
    rollout_output.mkdir(parents=True, exist_ok=True)
    rollout_cases: dict[str, Any] = {}
    for source in sorted((repo / "logs/imitation/rollout_cases").glob("*.npz")):
        target = rollout_output / source.name
        metadata_source = source.with_suffix(source.suffix + ".json")
        metadata_target = rollout_output / metadata_source.name
        shutil.copy2(source, target)
        shutil.copy2(metadata_source, metadata_target)
        metadata = _read_json(metadata_source)
        metadata["video_generated"] = False
        metadata["video_note"] = (
            "No interactive display was available. This script records an objective state/action trajectory only; "
            "headless offscreen video would require a separate render-enabled recorder."
        )
        _write_json(metadata_target, metadata)
        metadata["committed_trajectory"] = str(target.relative_to(output))
        metadata["committed_metadata"] = str(metadata_target.relative_to(output))
        rollout_cases[source.stem] = metadata
    method_summaries = {name: _summarize_method(paths) for name, paths in copied_paths.items()}
    primary = {name: method_summaries[name] for name in ("PPO teacher", "PPO scratch", "BC only", "BC+PPO")}

    bc_rate = primary["BC only"]["aggregate"]["settled_landing_rate"]["mean"]
    curves = {
        "PPO scratch": _curve(repo, "ppo_scratch", include_bc_step_zero=None),
        "BC+PPO": _curve(repo, "bc_ppo", include_bc_step_zero=bc_rate),
    }
    thresholds = {name: threshold_crossing_steps(rows) for name, rows in curves.items()}
    thresholds["BC only"] = {"80%": 0 if bc_rate >= 0.8 else None, "90%": 0 if bc_rate >= 0.9 else None, "92%": 0 if bc_rate >= 0.92 else None}
    thresholds["PPO teacher/reference"] = {"80%": 0, "90%": 0, "92%": 0}

    with (output / "learning_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "method",
            "epoch",
            "environment_steps",
            "evaluation_episodes",
            "settled_landing_rate",
            "contact_success_rate",
            "hard_contact_rate",
            "touchdown_distance_mean_m",
            "source",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, rows in curves.items():
            for row in rows:
                writer.writerow({"method": name, **row})

    dataset_summary = _read_json(repo / "logs/imitation/expert_dataset/dataset_summary.json")
    _write_json(output / "dataset_summary.json", dataset_summary)
    bc_metrics = _read_json(repo / "logs/imitation/behavior_cloning/metrics.json")
    _write_json(output / "bc_metrics.json", bc_metrics)

    training_runs = {
        "shared": {
            "task_id": TASK_ID,
            "num_envs": 256,
            "horizon_length": 24,
            "minibatch_size": 384,
            "epochs": 200,
            "maximum_environment_steps": 200 * STEPS_PER_EPOCH,
            "training_seeds": list(SEEDS),
            "checkpoint_selection_rule": "highest RL-Games rolling mean episode reward",
        },
        "ppo_scratch": [_training_run(repo, "ppo_scratch", seed, 1.0e-4) for seed in SEEDS],
        "bc_ppo": [_training_run(repo, "bc_ppo", seed, 1.0e-4) for seed in SEEDS],
        "diagnostic_bc_ppo_lr1e5": [
            _training_run(repo, "bc_ppo_lr1e5", seed, 1.0e-5) for seed in SEEDS
        ],
        "diagnostic_note": (
            "The one permitted targeted correction reduced the configured initial learning rate from 1e-4 to 1e-5. "
            "It did not improve aggregate closed-loop settled landing and is retained as a negative result rather than selected."
        ),
    }
    _write_json(output / "training_runs.json", training_runs)

    teacher = repo / TEACHER_RELATIVE
    dataset_manifest = repo / DATASET_MANIFEST_RELATIVE
    bc_checkpoint = repo / BC_RELATIVE
    bc_init = repo / BC_INIT_RELATIVE
    summary = {
        "benchmark": "imitation-learning benchmark expert demonstrations, behavior cloning, and BC-initialized PPO",
        "code_commit": _git(repo, "rev-parse", "HEAD"),
        "working_tree_dirty_before_generation": tree_dirty_before_generation,
        "task_id": TASK_ID,
        "state_based_policy": True,
        "visual_input": False,
        "artifacts": {
            "teacher_checkpoint": str(TEACHER_RELATIVE),
            "teacher_checkpoint_sha256": sha256_file(teacher),
            "dataset_manifest": str(DATASET_MANIFEST_RELATIVE),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "bc_checkpoint": str(BC_RELATIVE),
            "bc_checkpoint_sha256": sha256_file(bc_checkpoint),
            "bc_init_rlgames_checkpoint": str(BC_INIT_RELATIVE),
            "bc_init_rlgames_checkpoint_sha256": sha256_file(bc_init),
        },
        "dataset": dataset_summary,
        "bc_offline": bc_metrics,
        "formal_evaluation": method_summaries,
        "primary_comparison_methods": list(primary),
        "learning_curves": curves,
        "threshold_crossing_environment_steps": thresholds,
        "acceptance_criteria": {
            "dataset_successful_episodes_at_least_2000": dataset_summary["successful_episodes"] >= 2000,
            "dataset_transitions_at_least_500000": dataset_summary["transitions"] >= 500000,
            "bc_only_settled_landing_at_least_80_percent": bc_rate >= 0.8,
            "bc_ppo_final_settled_landing_at_least_92_percent": primary["BC+PPO"]["aggregate"]["settled_landing_rate"]["mean"] >= 0.92,
            "bc_ppo_reaches_90_percent_before_scratch": (
                thresholds["BC+PPO"]["90%"] is not None
                and (
                    thresholds["PPO scratch"]["90%"] is None
                    or thresholds["BC+PPO"]["90%"] < thresholds["PPO scratch"]["90%"]
                )
            ),
        },
        "acceptance_result": "PARTIAL",
        "failure_analysis": [
            "BC reproduces the teacher well offline and exceeds the closed-loop 80% target, but residual covariate shift appears mainly as deck misses.",
            "Online PPO updates repeatedly move the actor away from the strong BC solution; contact remains high while settled landing drops.",
            "A cold critic and reward-based checkpoint selection do not reliably preserve the touchdown/settle objective.",
            "Reducing the configured initial learning rate to 1e-5 did not fix the drift and increased seed variance.",
        ],
        "rollout_cases": rollout_cases,
        "known_limitations": [
            "The policy is state based and does not use camera images or real visual projection features.",
            "The teacher dataset contains successful episodes only, so recovery behavior after off-distribution mistakes is underrepresented.",
            "The online budget is 1,228,800 environment steps per seed; PPO from scratch did not converge within this budget.",
            "The current motion distribution is xy translation, sinusoidal heave, and roll/pitch up to 5 degrees; it does not cover yaw, wave spectra, hydrodynamics, or full vessel six-DoF motion.",
            "No interactive display was available and the current rollout recorder is numeric-only, so GUI acceptance and video artifacts are not claimed; headless offscreen video would require a separate render-enabled recorder.",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_comparison(output, primary)
    _plot_learning_curve(output, curves, "settled_landing_rate", "Settled landing rate", "settled_landing_vs_environment_steps.png")
    _plot_learning_curve(output, curves, "contact_success_rate", "Contact success rate", "contact_success_vs_environment_steps.png")
    _plot_learning_curve(output, curves, "touchdown_distance_mean_m", "Touchdown distance mean (m)", "touchdown_distance_vs_environment_steps.png")
    _plot_learning_curve(output, curves, "hard_contact_rate", "Hard-contact rate", "hard_contact_vs_environment_steps.png")
    _plot_bc_loss(repo, output)
    _plot_final_metrics(output, primary)
    _plot_failures(output, primary)
    (output / "commands.txt").write_text(_commands(), encoding="utf-8")

    readme = f"""# imitation-learning benchmark Imitation + Hybrid Benchmark

This directory is generated from raw CSV/JSON artifacts by `scripts/imitation/build_benchmark.py`.

- Task: `{TASK_ID}`
- Dataset: {dataset_summary['successful_episodes']} successful episodes / {dataset_summary['transitions']} transitions
- Episode split: train {dataset_summary['split']['train']['episodes']}, validation {dataset_summary['split']['validation']['episodes']}, test {dataset_summary['split']['test']['episodes']}
- BC-only settled landing: {100 * primary['BC only']['aggregate']['settled_landing_rate']['mean']:.2f}%
- PPO-from-scratch settled landing: {100 * primary['PPO scratch']['aggregate']['settled_landing_rate']['mean']:.2f}%
- BC+PPO settled landing: {100 * primary['BC+PPO']['aggregate']['settled_landing_rate']['mean']:.2f}%
- Frozen PPO teacher settled landing: {100 * primary['PPO teacher']['aggregate']['settled_landing_rate']['mean']:.2f}%

The BC-only target was met. The 92% BC+PPO target and the 90% sample-efficiency target were not met. The exact negative result, including the one conservative-learning-rate correction, is retained in `summary.json` and `training_runs.json`.

The current policy is state based. It contains no camera image input and no real visual projection features.

Paper-style formulation, equations, algorithm design, implementation traceability, and discussion:

```text
docs/imitation_hybrid_paper.md
```

Interactive display and headless-rendering diagnosis:

```text
docs/runtime_display_troubleshooting.md
```

## Files

- `summary.json`: checksums, per-seed metrics, aggregate metrics, thresholds, acceptance, and limitations.
- `dataset_summary.json`: dataset scale, split, phase coverage, and action statistics.
- `training_runs.json`: fair training budgets, selected checkpoints, wall time, and diagnostic rerun.
- `formal_evaluations/`: copied 256-episode-per-seed raw CSVs.
- `comparison.csv` / `comparison.md`: final primary-method table.
- `learning_curves.csv`: fixed-interval 128-episode-per-seed evaluation points.
- `rollout_cases/`: teacher, BC, BC+PPO success trajectories plus one PPO-scratch timeout failure trace.
- `*.png`: figures generated from raw artifacts.
- `commands.txt`: reproducibility commands.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "acceptance": summary["acceptance_criteria"]}, indent=2))


if __name__ == "__main__":
    main()

"""Analyze P8B actor, critic, action, RMS, and fixed-sigma drift on one frozen observation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from quadcopter_waypoint.imitation.checkpoint_sweep import discover_checkpoints, write_inventory
from quadcopter_waypoint.imitation.dataset import load_split_transitions, sha256_file
from quadcopter_waypoint.imitation.p8b_drift import compute_p8b_drift_metrics


def _load_model(path: str | Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"checkpoint has no model mapping: {path}")
    return model


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy_drift.json").write_text(
        json.dumps({**metadata, "records": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, list)}
        for index, value in enumerate(row["action_dim_mse_vs_reference"]):
            flat[f"action_dim_mse_vs_reference_{index}"] = value
        flat_rows.append(flat)
    if flat_rows:
        with (output_dir / "policy_drift.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--bc_checkpoint", required=True)
    parser.add_argument("--checkpoint_paths", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_glob", default="last_*_ep_*_rew_*.pth")
    parser.add_argument("--include_reward_selected", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_observations", type=int, default=0)
    args = parser.parse_args()
    if args.max_observations < 0:
        raise ValueError("max_observations must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    records = discover_checkpoints(
        args.run_dirs,
        args.bc_checkpoint,
        checkpoint_glob=args.checkpoint_glob,
        include_reward_selected=args.include_reward_selected,
        checkpoint_paths=args.checkpoint_paths,
    )
    write_inventory(output_dir / "checkpoint_inventory.json", records)
    canonical = [record for record in records if record.canonical]
    reference_records = [record for record in canonical if record.kind == "bc_init"]
    if len(reference_records) != 1:
        raise ValueError("P8B drift analysis requires exactly one canonical BC initialization")

    split = load_split_transitions(args.manifest, args.split, fields=("raw_observation",))
    observations = torch.from_numpy(split["raw_observation"]).float()
    if args.max_observations:
        observations = observations[: args.max_observations]
    reference = reference_records[0]
    reference_model = _load_model(reference.path)

    rows: list[dict[str, Any]] = []
    for record in canonical:
        model = _load_model(record.path)
        metrics = compute_p8b_drift_metrics(reference_model, model, observations)
        row = {
            "checkpoint_path": record.path,
            "checkpoint_sha256": record.sha256,
            "actor_sha256": metrics.pop("actor_sha256"),
            "critic_sha256": metrics.pop("critic_sha256"),
            "observation_rms_sha256": metrics.pop("observation_rms_sha256"),
            "train_seed": record.train_seed,
            "epoch": record.epoch,
            "training_reward": record.training_reward,
            "kind": record.kind,
            "observations": int(observations.shape[0]),
            **metrics,
        }
        rows.append(row)
        print(
            f"[P8B_DRIFT] seed={record.train_seed} kind={record.kind} epoch={record.epoch} "
            f"action_mse={row['action_mse_vs_reference']:.8f} "
            f"actor_l2={row['actor_parameter_relative_l2']:.8f}"
        )
    rows.sort(
        key=lambda row: (
            row["train_seed"] is None,
            row["train_seed"] if row["train_seed"] is not None else -1,
            row["epoch"],
            row["kind"],
            row["checkpoint_path"],
        )
    )
    _write_outputs(
        output_dir,
        rows,
        {
            "schema_version": 1,
            "dataset_manifest": str(Path(args.manifest).expanduser().resolve()),
            "dataset_manifest_sha256": sha256_file(args.manifest),
            "split": args.split,
            "observations": int(observations.shape[0]),
            "reference_checkpoint": reference.path,
            "reference_checkpoint_sha256": reference.sha256,
            "deterministic_action": "checkpoint-specific frozen RL-Games normalization, pre-sampling mean, then [-1,1] clamp",
        },
    )
    print(json.dumps({"checkpoints": len(rows), "observations": int(observations.shape[0])}, indent=2))


if __name__ == "__main__":
    main()

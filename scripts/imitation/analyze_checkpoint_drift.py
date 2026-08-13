"""Analyze deterministic actor and observation-normalization drift on the frozen imitation-learning benchmark test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from quadcopter_waypoint.imitation.checkpoint import deterministic_action_from_rlgames_model
from quadcopter_waypoint.imitation.checkpoint_sweep import (
    compute_drift_metrics,
    discover_checkpoints,
    write_inventory,
)
from quadcopter_waypoint.imitation.dataset import load_split_transitions, sha256_file


def _actions(model: Mapping[str, torch.Tensor], observations: torch.Tensor, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, observations.shape[0], batch_size):
            outputs.append(deterministic_action_from_rlgames_model(model, observations[start : start + batch_size]))
    return torch.cat(outputs, dim=0)


def _load_model(path: str | Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"checkpoint has no model mapping: {path}")
    return model


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {**metadata, "records": rows}
    (output_dir / "checkpoint_drift.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, list)}
        for prefix in ("action_dim_mse_vs_bc", "action_dim_mse_vs_teacher"):
            for index, value in enumerate(row[prefix]):
                flat[f"{prefix}_{index}"] = value
        flat_rows.append(flat)
    with (output_dir / "checkpoint_drift.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--bc_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_glob", default="last_*_ep_*_rew_*.pth")
    parser.add_argument("--include_reward_selected", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    records = discover_checkpoints(
        args.run_dirs,
        args.bc_checkpoint,
        checkpoint_glob=args.checkpoint_glob,
        include_reward_selected=args.include_reward_selected,
    )
    write_inventory(output_dir / "checkpoint_inventory.json", records)
    canonical = [record for record in records if record.canonical]
    bc_records = [record for record in canonical if record.kind == "bc_init"]
    if len(bc_records) != 1:
        raise ValueError("drift analysis requires exactly one BC initialization checkpoint")

    split = load_split_transitions(args.manifest, "test", fields=("raw_observation", "teacher_action"))
    observations = torch.from_numpy(split["raw_observation"]).float()
    teacher_actions = torch.from_numpy(split["teacher_action"]).float()
    reference_model = _load_model(bc_records[0].path)
    reference_actions = _actions(reference_model, observations, args.batch_size)

    rows: list[dict[str, Any]] = []
    for record in canonical:
        model = _load_model(record.path)
        candidate_actions = _actions(model, observations, args.batch_size)
        metrics = compute_drift_metrics(
            reference_actions,
            teacher_actions,
            candidate_actions,
            reference_model,
            model,
        )
        rows.append(
            {
                "checkpoint_path": record.path,
                "checkpoint_sha256": record.sha256,
                "actor_sha256": record.actor_sha256,
                "train_seed": record.train_seed,
                "epoch": record.epoch,
                "training_reward": record.training_reward,
                "kind": record.kind,
                "test_transitions": int(observations.shape[0]),
                **metrics,
            }
        )
        print(
            f"[DRIFT] seed={record.train_seed} kind={record.kind} epoch={record.epoch} "
            f"action_mse_vs_bc={metrics['action_mse_vs_bc']:.8f}"
        )
    rows.sort(key=lambda row: (row["train_seed"] is None, row["train_seed"] or -1, row["epoch"], row["kind"]))
    _write_outputs(
        output_dir,
        rows,
        {
            "dataset_manifest": str(Path(args.manifest).expanduser().resolve()),
            "dataset_manifest_sha256": sha256_file(args.manifest),
            "split": "test",
            "transitions": int(observations.shape[0]),
            "bc_checkpoint": bc_records[0].path,
            "bc_checkpoint_sha256": bc_records[0].sha256,
            "deterministic_action": "checkpoint-specific RL-Games normalization, actor mean, then [-1, 1] clamp",
        },
    )
    print(json.dumps({"checkpoints": len(rows), "transitions": int(observations.shape[0])}, indent=2))


if __name__ == "__main__":
    main()

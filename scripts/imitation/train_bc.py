"""Train and evaluate the P7 behavior-cloning actor on whole-episode dataset splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quadcopter_waypoint.imitation.dataset import (
    PHASE_NAMES,
    load_split_transitions,
    phase_sample_weights,
    sha256_file,
    validate_dataset_manifest,
)
from quadcopter_waypoint.imitation.policy import (
    BCActor,
    BCNetworkConfig,
    teacher_normalization_from_rlgames_checkpoint,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
    weighted: bool,
) -> DataLoader:
    observations = torch.from_numpy(arrays["raw_observation"])
    actions = torch.from_numpy(arrays["teacher_action"])
    phases = torch.from_numpy(arrays["flight_phase"].astype(np.int64, copy=False))
    weights = torch.from_numpy(
        phase_sample_weights(arrays["flight_phase"]) if weighted else np.ones(len(phases), dtype=np.float32)
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(observations, actions, phases, weights),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _evaluate(actor: BCActor, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    actor.eval()
    total_weighted_loss = 0.0
    total_weight = 0.0
    action_sq_sum = torch.zeros(4, dtype=torch.float64)
    action_count = 0
    phase_sq_sum = {phase: 0.0 for phase in PHASE_NAMES}
    phase_count = {phase: 0 for phase in PHASE_NAMES}
    with torch.inference_mode():
        for observations, targets, phases, weights in loader:
            observations = observations.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            predictions = actor(observations)
            squared = torch.square(predictions - targets)
            per_sample = squared.mean(dim=1)
            total_weighted_loss += float(torch.sum(per_sample * weights).item())
            total_weight += float(weights.sum().item())
            action_sq_sum += squared.sum(dim=0).double().cpu()
            action_count += squared.shape[0]
            for phase in PHASE_NAMES:
                mask = phases == phase
                count = int(mask.sum().item())
                if count:
                    phase_sq_sum[phase] += float(squared[mask].mean(dim=1).sum().item())
                    phase_count[phase] += count
    return {
        "weighted_mse": total_weighted_loss / total_weight,
        "action_mse": (action_sq_sum / action_count).tolist(),
        "phase_mse": {
            PHASE_NAMES[phase]: phase_sq_sum[phase] / phase_count[phase] if phase_count[phase] else None
            for phase in PHASE_NAMES
        },
        "transitions": action_count,
    }


def _checkpoint_payload(
    actor: BCActor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    dataset_manifest: Path,
    dataset_hash: str,
    teacher_checkpoint: Path,
    seed: int,
    best_validation_loss: float,
    history: list[dict[str, float]],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "quadcopter_waypoint_p7_bc_v1",
        "model_state_dict": actor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "network_config": actor.export_config(),
        "observation_normalization": {
            "running_mean": actor.running_mean.detach().cpu(),
            "running_var": actor.running_var.detach().cpu(),
            "count": actor.running_count.detach().cpu(),
            "epsilon": actor.config.observation_epsilon,
            "clip": actor.config.observation_clip,
        },
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": dataset_hash,
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "training_seed": int(seed),
        "best_validation_loss": float(best_validation_loss),
        "history": history,
        "training_config": training_config,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no_phase_weighting", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_transitions", type=int, default=None, help="Smoke-only deterministic truncation.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    teacher_path = Path(args.teacher_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty BC output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = validate_dataset_manifest(manifest_path, verify_hashes=True)
    if sha256_file(teacher_path) != manifest["teacher_checkpoint_sha256"]:
        raise RuntimeError("teacher checkpoint does not match the dataset manifest")
    _set_seed(args.seed)
    device = torch.device(args.device)

    fields = ("raw_observation", "teacher_action", "flight_phase", "episode_id")
    train_arrays = load_split_transitions(manifest_path, "train", fields)
    validation_arrays = load_split_transitions(manifest_path, "validation", fields)
    test_arrays = load_split_transitions(manifest_path, "test", fields)
    if args.max_train_transitions is not None:
        limit = min(args.max_train_transitions, len(train_arrays["episode_id"]))
        for name in train_arrays:
            train_arrays[name] = train_arrays[name][:limit]
    weighted = not args.no_phase_weighting
    train_loader = _make_loader(train_arrays, args.batch_size, True, args.seed, weighted)
    validation_loader = _make_loader(validation_arrays, args.batch_size, False, args.seed, weighted)
    test_loader = _make_loader(test_arrays, args.batch_size, False, args.seed, False)

    normalization = teacher_normalization_from_rlgames_checkpoint(teacher_path)
    actor = BCActor(
        normalization["running_mean"],
        normalization["running_var"],
        normalization["count"],
        config=BCNetworkConfig(),
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "phase_weighting": weighted,
        "train_transitions": int(len(train_arrays["episode_id"])),
        "validation_transitions": int(len(validation_arrays["episode_id"])),
        "test_transitions": int(len(test_arrays["episode_id"])),
        "device": str(device),
    }
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    dataset_hash = sha256_file(manifest_path)
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        actor.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for observations, targets, _, weights in train_loader:
            observations = observations.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = actor(observations)
            per_sample = torch.square(predictions - targets).mean(dim=1)
            loss = torch.sum(per_sample * weights) / weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("BC loss became NaN or Inf")
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
            optimizer.step()
            train_loss_sum += float(torch.sum(per_sample.detach() * weights).item())
            train_weight_sum += float(weights.sum().item())
        validation = _evaluate(actor, validation_loader, device)
        record = {
            "epoch": float(epoch),
            "train_weighted_mse": train_loss_sum / train_weight_sum,
            "validation_weighted_mse": float(validation["weighted_mse"]),
            "wall_time_s": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if validation["weighted_mse"] < best_validation:
            best_validation = float(validation["weighted_mse"])
            best_epoch = epoch
            epochs_without_improvement = 0
            payload = _checkpoint_payload(
                actor,
                optimizer,
                epoch,
                manifest_path,
                dataset_hash,
                teacher_path,
                args.seed,
                best_validation,
                history,
                training_config,
            )
            torch.save(payload, output_dir / "best_bc.pth")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    last_payload = _checkpoint_payload(
        actor,
        optimizer,
        int(history[-1]["epoch"]),
        manifest_path,
        dataset_hash,
        teacher_path,
        args.seed,
        best_validation,
        history,
        training_config,
    )
    torch.save(last_payload, output_dir / "last_bc.pth")
    best_actor, best_payload = BCActor.from_checkpoint(output_dir / "best_bc.pth", map_location=device)
    best_actor = best_actor.to(device)
    train_metrics = _evaluate(best_actor, _make_loader(train_arrays, args.batch_size, False, args.seed, False), device)
    validation_metrics = _evaluate(best_actor, _make_loader(validation_arrays, args.batch_size, False, args.seed, False), device)
    test_metrics = _evaluate(best_actor, test_loader, device)
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_validation,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "best_checkpoint": str(output_dir / "best_bc.pth"),
        "best_checkpoint_sha256": sha256_file(output_dir / "best_bc.pth"),
        "last_checkpoint": str(output_dir / "last_bc.pth"),
        "last_checkpoint_sha256": sha256_file(output_dir / "last_bc.pth"),
        "dataset_manifest_sha256": dataset_hash,
        "wall_time_s": time.perf_counter() - started,
        "training_config": training_config,
        "normalization_count": float(best_payload["observation_normalization"]["count"]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "loss_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

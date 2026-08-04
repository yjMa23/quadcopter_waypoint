"""Checkpoint inventory, resumable sweep, selection, and offline drift helpers for P8A."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import torch

from .benchmark import summarize_evaluation_csv
from .dataset import sha256_file

_PERIODIC_RE = re.compile(r"^last_.+_ep_(?P<epoch>\d+)_rew_(?P<reward>_?-?\d+(?:\.\d+)?_?)\.pth$")
_SEED_RE = re.compile(r"^seed(?P<seed>\d+)$")
_ACTOR_KEYS = (
    "a2c_network.actor_mlp.0.weight",
    "a2c_network.actor_mlp.0.bias",
    "a2c_network.actor_mlp.2.weight",
    "a2c_network.actor_mlp.2.bias",
    "a2c_network.mu.weight",
    "a2c_network.mu.bias",
)
_NORMALIZATION_KEYS = (
    "running_mean_std.running_mean",
    "running_mean_std.running_var",
    "running_mean_std.count",
)


@dataclass(frozen=True)
class CheckpointRecord:
    """Read-only checkpoint metadata used by sweeps and drift analysis."""

    path: str
    sha256: str
    actor_sha256: str
    size_bytes: int
    train_seed: int | None
    epoch: int
    training_reward: float | None
    kind: str
    canonical: bool = True
    duplicate_of: str | None = None

    @property
    def checkpoint_id(self) -> str:
        seed = "global" if self.train_seed is None else f"seed{self.train_seed}"
        return f"{seed}_{self.kind}_ep{self.epoch}_{self.sha256[:12]}"


def _train_seed(path: Path) -> int | None:
    for part in path.parts:
        match = _SEED_RE.match(part)
        if match:
            return int(match.group("seed"))
    return None


def parse_periodic_filename(path: str | Path) -> tuple[int, float]:
    """Parse the epoch and current rolling reward encoded by an RL-Games periodic filename."""
    name = Path(path).name
    match = _PERIODIC_RE.match(name)
    if not match:
        raise ValueError(f"not a periodic checkpoint filename: {name}")
    reward_text = match.group("reward").strip("_")
    return int(match.group("epoch")), float(reward_text)


def actor_state_sha256(model_state: Mapping[str, torch.Tensor]) -> str:
    """Hash deterministic actor and observation-normalization tensors independent of pickle metadata."""
    digest = hashlib.sha256()
    for key in (*_ACTOR_KEYS, *_NORMALIZATION_KEYS):
        if key not in model_state:
            raise KeyError(f"checkpoint model is missing {key}")
        tensor = model_state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def inspect_checkpoint(path: str | Path, kind: str | None = None) -> CheckpointRecord:
    """Validate one RL-Games checkpoint and return immutable inventory metadata."""
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"checkpoint has no RL-Games model mapping: {checkpoint}")
    epoch = int(payload.get("epoch", -1))
    inferred_kind = kind
    reward: float | None
    if checkpoint.name.startswith("last_"):
        parsed_epoch, reward = parse_periodic_filename(checkpoint)
        if epoch != parsed_epoch:
            raise ValueError(f"checkpoint epoch mismatch: filename={parsed_epoch}, payload={epoch}: {checkpoint}")
        inferred_kind = inferred_kind or "periodic"
    elif epoch == 0:
        reward = 0.0
        inferred_kind = inferred_kind or "bc_init"
    else:
        value = payload.get("last_mean_rewards")
        reward = float(value) if value is not None and math.isfinite(float(value)) else None
        inferred_kind = inferred_kind or "reward_selected"
    return CheckpointRecord(
        path=str(checkpoint),
        sha256=sha256_file(checkpoint),
        actor_sha256=actor_state_sha256(payload["model"]),
        size_bytes=checkpoint.stat().st_size,
        train_seed=_train_seed(checkpoint),
        epoch=epoch,
        training_reward=reward,
        kind=inferred_kind,
    )


def discover_checkpoints(
    run_dirs: Sequence[str | Path],
    bc_checkpoint: str | Path | None,
    checkpoint_glob: str = "last_*_ep_*_rew_*.pth",
    include_reward_selected: bool = False,
    checkpoint_paths: Sequence[str | Path] = (),
) -> list[CheckpointRecord]:
    """Build a deterministic inventory and mark same-seed/same-epoch duplicate actor snapshots."""
    records: list[CheckpointRecord] = []
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value).expanduser().resolve()
        nn_dir = run_dir / "nn" if (run_dir / "nn").is_dir() else run_dir
        for path in sorted(nn_dir.glob(checkpoint_glob)):
            records.append(inspect_checkpoint(path, kind="periodic"))
        if include_reward_selected:
            selected = nn_dir / f"{run_dir.name}.pth"
            if not selected.exists():
                selected_candidates = sorted(path for path in nn_dir.glob("*.pth") if not path.name.startswith("last_"))
                if len(selected_candidates) != 1:
                    raise FileNotFoundError(f"cannot identify reward-selected checkpoint in {nn_dir}")
                selected = selected_candidates[0]
            records.append(inspect_checkpoint(selected, kind="reward_selected"))
    if bc_checkpoint is not None:
        records.append(inspect_checkpoint(bc_checkpoint, kind="bc_init"))
    for path in checkpoint_paths:
        resolved = Path(path).expanduser().resolve()
        if any(record.path == str(resolved) for record in records):
            continue
        records.append(inspect_checkpoint(resolved))
    if not records:
        raise ValueError("checkpoint inventory is empty")

    grouped: dict[tuple[int | None, int, str], list[int]] = {}
    for index, record in enumerate(records):
        if record.kind == "periodic":
            grouped.setdefault((record.train_seed, record.epoch, record.actor_sha256), []).append(index)
    mutable = list(records)
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        canonical_index = min(indices, key=lambda index: (len(Path(mutable[index].path).name), mutable[index].path))
        canonical_path = mutable[canonical_index].path
        for index in indices:
            if index == canonical_index:
                continue
            value = mutable[index]
            mutable[index] = CheckpointRecord(**{**asdict(value), "canonical": False, "duplicate_of": canonical_path})
    return sorted(mutable, key=lambda item: (item.train_seed is None, item.train_seed or -1, item.epoch, item.kind, item.path))


def write_inventory(path: str | Path, records: Sequence[CheckpointRecord]) -> None:
    """Write inventory JSON without touching checkpoints."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_count": len(records),
        "canonical_count": sum(record.canonical for record in records),
        "records": [{**asdict(record), "checkpoint_id": record.checkpoint_id} for record in records],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_evaluation_csv(path: str | Path, episodes: int) -> dict[str, Any]:
    """Reject missing, truncated, or metric-incomplete evaluator CSVs."""
    csv_path = Path(path)
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        raise FileNotFoundError(f"evaluation CSV is missing or empty: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"settled_landing", "deck_miss", "hard_contact", "touchdown_distance", "time_out"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"evaluation CSV is missing required metrics: {csv_path}")
        rows = list(reader)
    if len(rows) != episodes:
        raise ValueError(f"evaluation CSV has {len(rows)} episodes, expected {episodes}: {csv_path}")
    summary = summarize_evaluation_csv(csv_path)
    if int(summary["episodes"]) != episodes:
        raise ValueError(f"summary episode count mismatch: {csv_path}")
    return summary


def resume_key(record: CheckpointRecord, task: str, eval_seed: int, episodes: int, num_envs: int) -> str:
    """Return the exact identity used for resumable evaluation entries."""
    value = f"{record.sha256}|{task}|{eval_seed}|{episodes}|{num_envs}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selection_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    """Sort best-first by settled landing and the frozen P8A tie-break rules."""
    touchdown = row.get("touchdown_distance_mean_m")
    touchdown_value = float("inf") if touchdown is None else float(touchdown)
    return (
        -float(row["settled_landing_rate"]),
        float(row["deck_miss_rate"]),
        float(row["hard_contact_rate"]),
        touchdown_value,
        int(row["epoch"]),
    )


def select_screening_candidates(rows: Iterable[Mapping[str, Any]], top_k: int = 5) -> dict[int, list[str]]:
    """Select Top-K metric checkpoints plus reward-selected and BC for each training seed."""
    values = list(rows)
    seeds = sorted({int(row["train_seed"]) for row in values if row.get("train_seed") is not None})
    bc_paths = [str(row["checkpoint_path"]) for row in values if row["kind"] == "bc_init"]
    result: dict[int, list[str]] = {}
    for seed in seeds:
        seed_rows = [row for row in values if row.get("train_seed") == seed and row["kind"] in {"periodic", "reward_selected"}]
        periodic = sorted((row for row in seed_rows if row["kind"] == "periodic"), key=selection_sort_key)[:top_k]
        reward = [row for row in seed_rows if row["kind"] == "reward_selected"]
        ordered = [str(row["checkpoint_path"]) for row in periodic + reward] + bc_paths
        result[seed] = list(dict.fromkeys(ordered))
    return result


def aggregate_checkpoint_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate equal-sized evaluation seeds for each checkpoint SHA."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["checkpoint_sha256"]), []).append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "settled_landing_rate",
        "deck_miss_rate",
        "hard_contact_rate",
        "contact_success_rate",
        "timeout_rate",
        "touchdown_distance_mean_m",
    )
    for group in groups.values():
        first = group[0]
        episode_counts = {int(row["episodes"]) for row in group}
        if len(episode_counts) != 1:
            raise ValueError("checkpoint aggregation requires equal episodes per eval seed")
        aggregate = {
            "checkpoint_path": first["checkpoint_path"],
            "checkpoint_sha256": first["checkpoint_sha256"],
            "actor_sha256": first["actor_sha256"],
            "train_seed": first.get("train_seed"),
            "epoch": int(first["epoch"]),
            "training_reward": first.get("training_reward"),
            "kind": first["kind"],
            "eval_seeds": sorted(int(row["eval_seed"]) for row in group),
            "episodes_per_seed": next(iter(episode_counts)),
            "episodes": sum(int(row["episodes"]) for row in group),
        }
        for metric in metrics:
            metric_values = [float(row[metric]) for row in group if row.get(metric) is not None]
            aggregate[metric] = mean(metric_values) if metric_values else None
            aggregate[f"{metric}_std"] = pstdev(metric_values) if metric_values else None
        output.append(aggregate)
    return sorted(output, key=lambda row: (row.get("train_seed") is None, row.get("train_seed") or -1, row["epoch"], row["kind"]))


def select_validation_best(rows: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Choose exactly one validation checkpoint per training seed using P8A rules."""
    aggregated = aggregate_checkpoint_rows(rows)
    seeds = sorted({int(row["train_seed"]) for row in aggregated if row.get("train_seed") is not None})
    result: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        candidates = [row for row in aggregated if row.get("train_seed") == seed and row["kind"] != "bc_init"]
        if not candidates:
            raise ValueError(f"no validation candidates for train seed {seed}")
        result[seed] = min(candidates, key=selection_sort_key)
    return result


def compute_drift_metrics(
    reference_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    candidate_actions: torch.Tensor,
    reference_model: Mapping[str, torch.Tensor],
    candidate_model: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute deterministic action, actor-parameter, and running-statistic drift."""
    if reference_actions.shape != candidate_actions.shape or teacher_actions.shape != candidate_actions.shape:
        raise ValueError("action tensors must have identical shapes")
    if candidate_actions.ndim != 2:
        raise ValueError("action tensors must be two-dimensional")
    bc_error = torch.square(candidate_actions.float() - reference_actions.float())
    teacher_error = torch.square(candidate_actions.float() - teacher_actions.float())
    parameter_sq = torch.zeros((), dtype=torch.float64)
    reference_sq = torch.zeros((), dtype=torch.float64)
    for key in _ACTOR_KEYS:
        delta = candidate_model[key].double() - reference_model[key].double()
        parameter_sq += torch.sum(torch.square(delta))
        reference_sq += torch.sum(torch.square(reference_model[key].double()))
    mean_delta = candidate_model[_NORMALIZATION_KEYS[0]].double() - reference_model[_NORMALIZATION_KEYS[0]].double()
    var_delta = candidate_model[_NORMALIZATION_KEYS[1]].double() - reference_model[_NORMALIZATION_KEYS[1]].double()
    count_delta = candidate_model[_NORMALIZATION_KEYS[2]].double() - reference_model[_NORMALIZATION_KEYS[2]].double()
    parameter_l2 = float(torch.sqrt(parameter_sq).item())
    reference_l2 = float(torch.sqrt(reference_sq).item())
    return {
        "action_mse_vs_bc": float(bc_error.mean().item()),
        "action_mse_vs_teacher": float(teacher_error.mean().item()),
        "action_dim_mse_vs_bc": [float(value) for value in bc_error.mean(dim=0).tolist()],
        "action_dim_mse_vs_teacher": [float(value) for value in teacher_error.mean(dim=0).tolist()],
        "actor_parameter_l2": parameter_l2,
        "actor_parameter_relative_l2": parameter_l2 / max(reference_l2, 1.0e-12),
        "running_mean_l2": float(torch.linalg.vector_norm(mean_delta).item()),
        "running_mean_mse": float(torch.square(mean_delta).mean().item()),
        "running_var_l2": float(torch.linalg.vector_norm(var_delta).item()),
        "running_var_mse": float(torch.square(var_delta).mean().item()),
        "running_count_delta": float(count_delta.item()),
    }

"""Dataset schema, validation, splitting, and loading for imitation-learning benchmark expert trajectories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

OBSERVATION_DIM = 22
ACTION_DIM = 4
SCHEMA_VERSION = 1
PHASE_NAMES = {0: "approach", 1: "align", 2: "descent", 3: "contact_settle"}

REQUIRED_ARRAYS: dict[str, tuple[np.dtype, tuple[int, ...] | None]] = {
    "episode_id": (np.dtype(np.int64), None),
    "step_id": (np.dtype(np.int32), None),
    "seed": (np.dtype(np.int32), None),
    "raw_observation": (np.dtype(np.float32), (OBSERVATION_DIM,)),
    "teacher_action": (np.dtype(np.float32), (ACTION_DIM,)),
    "reward": (np.dtype(np.float32), None),
    "terminated": (np.dtype(np.bool_), None),
    "time_out": (np.dtype(np.bool_), None),
    "flight_phase": (np.dtype(np.int8), None),
    "contact_success": (np.dtype(np.bool_), None),
    "settled_landing": (np.dtype(np.bool_), None),
    "hard_contact": (np.dtype(np.bool_), None),
    "ground_crash": (np.dtype(np.bool_), None),
    "deck_miss": (np.dtype(np.bool_), None),
    "touchdown_distance": (np.dtype(np.float32), None),
    "first_contact_xy_error": (np.dtype(np.float32), None),
    "first_contact_normal_relative_speed": (np.dtype(np.float32), None),
    "first_contact_tangential_relative_speed": (np.dtype(np.float32), None),
    "first_contact_body_deck_normal_angle": (np.dtype(np.float32), None),
    "maximum_penetration": (np.dtype(np.float32), None),
    "deck_xy_velocity": (np.dtype(np.float32), (2,)),
    "deck_heave_amplitude": (np.dtype(np.float32), None),
    "deck_heave_omega": (np.dtype(np.float32), None),
    "deck_roll_amplitude": (np.dtype(np.float32), None),
    "deck_roll_omega": (np.dtype(np.float32), None),
    "deck_pitch_amplitude": (np.dtype(np.float32), None),
    "deck_pitch_omega": (np.dtype(np.float32), None),
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with deterministic serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_episode_split(
    episode_ids: Iterable[int],
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[int]]:
    """Create a reproducible train/validation/test split over whole episodes."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must contain three values summing to one, got {ratios}")
    unique_ids = np.asarray(sorted({int(value) for value in episode_ids}), dtype=np.int64)
    if unique_ids.size < 3:
        raise ValueError("at least three episodes are required for train/validation/test splitting")
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)
    train_end = int(np.floor(shuffled.size * ratios[0]))
    validation_end = train_end + int(np.floor(shuffled.size * ratios[1]))
    train_end = max(1, min(train_end, shuffled.size - 2))
    validation_end = max(train_end + 1, min(validation_end, shuffled.size - 1))
    result = {
        "train": sorted(int(value) for value in shuffled[:train_end]),
        "validation": sorted(int(value) for value in shuffled[train_end:validation_end]),
        "test": sorted(int(value) for value in shuffled[validation_end:]),
    }
    validate_episode_split(result)
    return result


def validate_episode_split(split: Mapping[str, Iterable[int]]) -> None:
    """Raise when split names are missing or episode leakage is present."""
    expected = {"train", "validation", "test"}
    if set(split) != expected:
        raise ValueError(f"split keys must be {sorted(expected)}, got {sorted(split)}")
    sets = {name: {int(value) for value in values} for name, values in split.items()}
    if any(not values for values in sets.values()):
        raise ValueError("train, validation, and test episode sets must all be non-empty")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("episode leakage detected between dataset splits")


def validate_shard_arrays(arrays: Mapping[str, np.ndarray], action_tolerance: float = 1.0e-6) -> dict[str, int]:
    """Validate one trajectory shard and return its transition/episode counts."""
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise ValueError(f"missing required arrays: {missing}")
    lengths = {name: int(np.asarray(arrays[name]).shape[0]) for name in REQUIRED_ARRAYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"array length mismatch: {lengths}")
    transition_count = next(iter(lengths.values()))
    if transition_count <= 0:
        raise ValueError("trajectory shard must contain at least one transition")

    for name, (expected_dtype, trailing_shape) in REQUIRED_ARRAYS.items():
        array = np.asarray(arrays[name])
        if array.dtype != expected_dtype:
            raise ValueError(f"{name} dtype must be {expected_dtype}, got {array.dtype}")
        if trailing_shape is not None and array.shape[1:] != trailing_shape:
            raise ValueError(f"{name} shape must be (N, {trailing_shape}), got {array.shape}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")

    actions = np.asarray(arrays["teacher_action"])
    if np.any(actions < -1.0 - action_tolerance) or np.any(actions > 1.0 + action_tolerance):
        minimum = float(actions.min())
        maximum = float(actions.max())
        raise ValueError(f"teacher_action outside [-1, 1]: min={minimum}, max={maximum}")
    phases = np.asarray(arrays["flight_phase"])
    valid_phases = np.asarray(sorted(PHASE_NAMES), dtype=np.int8)
    if not np.isin(phases, valid_phases).all():
        raise ValueError(f"flight_phase contains values outside {sorted(PHASE_NAMES)}")
    episode_ids = np.asarray(arrays["episode_id"])
    step_ids = np.asarray(arrays["step_id"])
    if np.any(episode_ids < 0) or np.any(step_ids < 0):
        raise ValueError("episode_id and step_id must be non-negative")
    for episode_id in np.unique(episode_ids):
        episode_steps = step_ids[episode_ids == episode_id]
        if not np.array_equal(episode_steps, np.arange(episode_steps.size, dtype=np.int32)):
            raise ValueError(f"episode {int(episode_id)} step_id must be contiguous from zero")
    return {
        "transitions": transition_count,
        "episodes": int(np.unique(episode_ids).size),
    }


def save_shard(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate and save a compressed shard without overwriting an existing file."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {output}")
    stats = validate_shard_arrays(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "path": output.name,
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        **stats,
        "episode_id_min": int(np.asarray(arrays["episode_id"]).min()),
        "episode_id_max": int(np.asarray(arrays["episode_id"]).max()),
    }


def load_shard(path: str | Path) -> dict[str, np.ndarray]:
    """Load all arrays from one NPZ shard into memory."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    validate_shard_arrays(arrays)
    return arrays


def _manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.json"
    return path


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Read a manifest and attach its path for downstream relative-file resolution."""
    path = _manifest_path(manifest_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_manifest_path"] = str(path.resolve())
    return value


def write_manifest(path: str | Path, manifest: Mapping[str, Any], overwrite: bool = False) -> str:
    """Write a deterministic JSON manifest and return its SHA256."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in manifest.items() if not key.startswith("_")}
    output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(output)


def validate_dataset_manifest(manifest_path: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate manifest metadata, every shard, and episode-level split integrity."""
    manifest = read_manifest(manifest_path)
    path = Path(manifest["_manifest_path"])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {manifest.get('schema_version')}")
    if manifest.get("observation_shape") != [OBSERVATION_DIM]:
        raise ValueError(f"manifest observation_shape must be [{OBSERVATION_DIM}]")
    if manifest.get("action_shape") != [ACTION_DIM]:
        raise ValueError(f"manifest action_shape must be [{ACTION_DIM}]")
    if manifest.get("observation_dtype") != "float32" or manifest.get("action_dtype") != "float32":
        raise ValueError("manifest observation/action dtype must be float32")
    split = manifest.get("episode_split")
    if not isinstance(split, dict):
        raise ValueError("manifest is missing episode_split")
    validate_episode_split(split)

    total_transitions = 0
    all_episode_ids: set[int] = set()
    shard_records = manifest.get("shards", [])
    if not shard_records:
        raise ValueError("manifest contains no shards")
    for record in shard_records:
        shard_path = path.parent / record["path"]
        if not shard_path.is_file():
            raise FileNotFoundError(f"manifest shard not found: {shard_path}")
        if verify_hashes and sha256_file(shard_path) != record["sha256"]:
            raise ValueError(f"SHA256 mismatch for shard: {shard_path}")
        arrays = load_shard(shard_path)
        stats = validate_shard_arrays(arrays)
        if stats["transitions"] != int(record["transitions"]) or stats["episodes"] != int(record["episodes"]):
            raise ValueError(f"manifest count mismatch for shard: {shard_path}")
        total_transitions += stats["transitions"]
        episode_ids = {int(value) for value in np.unique(arrays["episode_id"])}
        if all_episode_ids & episode_ids:
            raise ValueError(f"episode IDs repeated across shards: {sorted(all_episode_ids & episode_ids)[:5]}")
        all_episode_ids.update(episode_ids)

    split_ids = {int(value) for values in split.values() for value in values}
    if split_ids != all_episode_ids:
        missing = sorted(all_episode_ids - split_ids)
        extra = sorted(split_ids - all_episode_ids)
        raise ValueError(f"episode_split does not match shard episodes; missing={missing[:5]}, extra={extra[:5]}")
    if total_transitions != int(manifest.get("transition_count", -1)):
        raise ValueError("manifest transition_count does not match shards")
    if len(all_episode_ids) != int(manifest.get("successful_episode_count", -1)):
        raise ValueError("manifest successful_episode_count does not match shards")
    return manifest


def iter_manifest_shards(manifest_path: str | Path) -> Iterable[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Yield each manifest shard record and its validated arrays."""
    manifest = read_manifest(manifest_path)
    root = Path(manifest["_manifest_path"]).parent
    for record in manifest["shards"]:
        yield record, load_shard(root / record["path"])


def load_split_transitions(
    manifest_path: str | Path,
    split_name: str,
    fields: Iterable[str] = ("raw_observation", "teacher_action", "flight_phase", "episode_id"),
) -> dict[str, np.ndarray]:
    """Load selected fields for one whole-episode split."""
    manifest = validate_dataset_manifest(manifest_path, verify_hashes=False)
    if split_name not in manifest["episode_split"]:
        raise KeyError(f"unknown split: {split_name}")
    selected_ids = np.asarray(manifest["episode_split"][split_name], dtype=np.int64)
    requested = tuple(fields)
    unknown = sorted(set(requested) - set(REQUIRED_ARRAYS))
    if unknown:
        raise KeyError(f"unknown dataset fields: {unknown}")
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in requested}
    for _, arrays in iter_manifest_shards(manifest_path):
        mask = np.isin(arrays["episode_id"], selected_ids)
        if not mask.any():
            continue
        for field in requested:
            chunks[field].append(arrays[field][mask])
    if not chunks[requested[0]]:
        raise ValueError(f"split {split_name} has no transitions")
    return {field: np.concatenate(parts, axis=0) for field, parts in chunks.items()}


def phase_sample_weights(phases: np.ndarray, maximum_weight: float = 8.0) -> np.ndarray:
    """Return inverse-frequency phase weights normalized to mean one."""
    phase_values = np.asarray(phases, dtype=np.int64)
    if phase_values.ndim != 1 or phase_values.size == 0:
        raise ValueError("phases must be a non-empty one-dimensional array")
    counts = np.bincount(phase_values, minlength=len(PHASE_NAMES)).astype(np.float64)
    present = counts > 0
    inverse = np.zeros_like(counts)
    inverse[present] = phase_values.size / (present.sum() * counts[present])
    inverse = np.minimum(inverse, maximum_weight)
    weights = inverse[phase_values]
    weights /= weights.mean()
    return weights.astype(np.float32)


def compute_dataset_statistics(manifest_path: str | Path) -> dict[str, Any]:
    """Compute transition, episode, seed, phase, and action statistics from raw shards."""
    manifest = validate_dataset_manifest(manifest_path, verify_hashes=True)
    phase_counts: Counter[int] = Counter()
    seed_episode_ids: dict[int, set[int]] = {}
    action_min = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    action_max = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    action_sq_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    total = 0
    episode_lengths: list[int] = []
    for _, arrays in iter_manifest_shards(manifest_path):
        phases = arrays["flight_phase"]
        phase_counts.update(int(value) for value in phases)
        actions = arrays["teacher_action"].astype(np.float64)
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        action_sum += actions.sum(axis=0)
        action_sq_sum += np.square(actions).sum(axis=0)
        total += actions.shape[0]
        for episode_id in np.unique(arrays["episode_id"]):
            mask = arrays["episode_id"] == episode_id
            episode_lengths.append(int(mask.sum()))
            seed = int(arrays["seed"][np.flatnonzero(mask)[0]])
            seed_episode_ids.setdefault(seed, set()).add(int(episode_id))
    action_mean = action_sum / total
    action_var = np.maximum(action_sq_sum / total - np.square(action_mean), 0.0)
    split_stats = {}
    episode_lengths_array = np.asarray(episode_lengths, dtype=np.float64)
    for name, ids in manifest["episode_split"].items():
        id_set = {int(value) for value in ids}
        transitions = 0
        for _, arrays in iter_manifest_shards(manifest_path):
            transitions += int(np.isin(arrays["episode_id"], list(id_set)).sum())
        split_stats[name] = {"episodes": len(id_set), "transitions": transitions}
    return {
        "manifest": str(Path(manifest["_manifest_path"])),
        "manifest_sha256": sha256_file(manifest["_manifest_path"]),
        "successful_episodes": int(manifest["successful_episode_count"]),
        "transitions": total,
        "episodes_by_seed": {str(seed): len(ids) for seed, ids in sorted(seed_episode_ids.items())},
        "split": split_stats,
        "phase_counts": {PHASE_NAMES[index]: int(phase_counts[index]) for index in PHASE_NAMES},
        "phase_fractions": {
            PHASE_NAMES[index]: float(phase_counts[index] / total) for index in PHASE_NAMES
        },
        "episode_length": {
            "mean": float(episode_lengths_array.mean()),
            "std": float(episode_lengths_array.std()),
            "min": int(episode_lengths_array.min()),
            "max": int(episode_lengths_array.max()),
            "p95": float(np.percentile(episode_lengths_array, 95)),
        },
        "action": {
            "min": action_min.tolist(),
            "max": action_max.tolist(),
            "mean": action_mean.tolist(),
            "std": np.sqrt(action_var).tolist(),
        },
    }

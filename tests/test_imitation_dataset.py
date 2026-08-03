from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quadcopter_waypoint.imitation.dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    compute_dataset_statistics,
    create_episode_split,
    save_shard,
    validate_dataset_manifest,
    validate_shard_arrays,
    write_manifest,
)


def _valid_arrays(episode_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> dict[str, np.ndarray]:
    per_episode = 3
    n = len(episode_ids) * per_episode
    repeated = np.repeat(np.asarray(episode_ids, dtype=np.int64), per_episode)
    steps = np.tile(np.arange(per_episode, dtype=np.int32), len(episode_ids))
    rng = np.random.default_rng(7)
    arrays = {
        "episode_id": repeated,
        "step_id": steps,
        "seed": np.full(n, 42, dtype=np.int32),
        "raw_observation": rng.normal(size=(n, OBSERVATION_DIM)).astype(np.float32),
        "teacher_action": rng.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32),
        "reward": rng.normal(size=n).astype(np.float32),
        "terminated": np.zeros(n, dtype=np.bool_),
        "time_out": np.zeros(n, dtype=np.bool_),
        "flight_phase": np.tile(np.asarray([0, 2, 3], dtype=np.int8), len(episode_ids)),
        "contact_success": np.ones(n, dtype=np.bool_),
        "settled_landing": np.ones(n, dtype=np.bool_),
        "hard_contact": np.zeros(n, dtype=np.bool_),
        "ground_crash": np.zeros(n, dtype=np.bool_),
        "deck_miss": np.zeros(n, dtype=np.bool_),
        "touchdown_distance": np.full(n, 0.1, dtype=np.float32),
        "first_contact_xy_error": np.full(n, 0.08, dtype=np.float32),
        "first_contact_normal_relative_speed": np.full(n, -0.2, dtype=np.float32),
        "first_contact_tangential_relative_speed": np.full(n, 0.1, dtype=np.float32),
        "first_contact_body_deck_normal_angle": np.full(n, 0.05, dtype=np.float32),
        "maximum_penetration": np.full(n, 0.01, dtype=np.float32),
        "deck_xy_velocity": np.zeros((n, 2), dtype=np.float32),
        "deck_heave_amplitude": np.zeros(n, dtype=np.float32),
        "deck_heave_omega": np.zeros(n, dtype=np.float32),
        "deck_roll_amplitude": np.zeros(n, dtype=np.float32),
        "deck_roll_omega": np.zeros(n, dtype=np.float32),
        "deck_pitch_amplitude": np.zeros(n, dtype=np.float32),
        "deck_pitch_omega": np.zeros(n, dtype=np.float32),
    }
    for index in range(per_episode - 1, n, per_episode):
        arrays["terminated"][index] = True
    return arrays


def _write_dataset(tmp_path: Path) -> Path:
    arrays = _valid_arrays()
    record = save_shard(tmp_path / "shard_00000.npz", arrays)
    split = create_episode_split(np.unique(arrays["episode_id"]), seed=9)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "test-task",
        "teacher_checkpoint": "teacher.pth",
        "teacher_checkpoint_sha256": "0" * 64,
        "observation_shape": [OBSERVATION_DIM],
        "action_shape": [ACTION_DIM],
        "observation_dtype": "float32",
        "action_dtype": "float32",
        "action_semantics": "clamped",
        "successful_episode_count": 6,
        "transition_count": 18,
        "episode_split": split,
        "shards": [record],
    }
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    return path


def test_episode_split_is_reproducible_and_has_no_leakage():
    ids = range(100)
    first = create_episode_split(ids, seed=123)
    second = create_episode_split(ids, seed=123)
    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set(first["train"] + first["validation"] + first["test"]) == set(ids)


def test_shard_schema_shape_dtype_bounds_and_non_finite_checks():
    arrays = _valid_arrays()
    assert validate_shard_arrays(arrays) == {"transitions": 18, "episodes": 6}
    bad_shape = dict(arrays)
    bad_shape["raw_observation"] = bad_shape["raw_observation"][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        validate_shard_arrays(bad_shape)
    bad_action = dict(arrays)
    bad_action["teacher_action"] = bad_action["teacher_action"].copy()
    bad_action["teacher_action"][0, 0] = 1.1
    with pytest.raises(ValueError, match="outside"):
        validate_shard_arrays(bad_action)
    bad_nan = dict(arrays)
    bad_nan["reward"] = bad_nan["reward"].copy()
    bad_nan["reward"][0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_shard_arrays(bad_nan)


def test_manifest_hash_counts_and_statistics(tmp_path: Path):
    manifest_path = _write_dataset(tmp_path)
    manifest = validate_dataset_manifest(manifest_path, verify_hashes=True)
    assert manifest["transition_count"] == 18
    stats = compute_dataset_statistics(manifest_path)
    assert stats["successful_episodes"] == 6
    assert stats["transitions"] == 18
    assert sum(stats["phase_counts"].values()) == 18


def test_manifest_rejects_shard_hash_mismatch(tmp_path: Path):
    manifest_path = _write_dataset(tmp_path)
    value = json.loads(manifest_path.read_text())
    value["shards"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="SHA256"):
        validate_dataset_manifest(manifest_path, verify_hashes=True)

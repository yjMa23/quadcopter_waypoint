"""Finalize per-seed imitation-learning benchmark collection shards into one episode-level-split dataset manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quadcopter_waypoint.imitation.dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    compute_dataset_statistics,
    create_episode_split,
    load_shard,
    sha256_file,
    validate_dataset_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--min_successful_episodes", type=int, default=2000)
    parser.add_argument("--min_transitions", type=int, default=500000)
    args = parser.parse_args()

    root = Path(args.dataset_dir).resolve()
    output = root / "manifest.json"
    if output.exists():
        validate_dataset_manifest(output, verify_hashes=True)
        print(f"[INFO] Existing manifest is complete and valid: {output}")
        return

    partial_paths = sorted(root.glob("seed_*/partial_manifest.json"))
    if len(partial_paths) < 2:
        raise RuntimeError("at least two collection seeds are required before finalizing the dataset")
    partials = [json.loads(path.read_text(encoding="utf-8")) for path in partial_paths]
    teacher_hashes = {value["teacher_checkpoint_sha256"] for value in partials}
    task_ids = {value["task_id"] for value in partials}
    if len(teacher_hashes) != 1 or len(task_ids) != 1:
        raise RuntimeError("partial manifests do not share one teacher checkpoint and task")

    shard_records = []
    episode_ids: set[int] = set()
    total_transitions = 0
    total_episodes = 0
    rejected = 0
    collection_commands = []
    versions = []
    seeds = []
    for partial_path, partial in zip(partial_paths, partials):
        seed_root = partial_path.parent
        seeds.append(int(partial["seed"]))
        collection_commands.append(partial["collection_command"])
        versions.append(partial["versions"])
        rejected += int(partial.get("rejected_episode_count", 0))
        for record in partial["shards"]:
            shard_path = seed_root / record["path"]
            if sha256_file(shard_path) != record["sha256"]:
                raise RuntimeError(f"corrupt shard: {shard_path}")
            arrays = load_shard(shard_path)
            shard_episode_ids = {int(value) for value in arrays["episode_id"]}
            if episode_ids & shard_episode_ids:
                raise RuntimeError("episode IDs overlap across collection shards")
            episode_ids.update(shard_episode_ids)
            new_record = dict(record)
            new_record["path"] = str(shard_path.relative_to(root))
            shard_records.append(new_record)
            total_transitions += int(record["transitions"])
            total_episodes += int(record["episodes"])

    if total_episodes < args.min_successful_episodes or total_transitions < args.min_transitions:
        raise RuntimeError(
            f"dataset target not reached: episodes={total_episodes}/{args.min_successful_episodes}, "
            f"transitions={total_transitions}/{args.min_transitions}"
        )
    split = create_episode_split(episode_ids, seed=args.split_seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": next(iter(task_ids)),
        "teacher_checkpoint": partials[0]["teacher_checkpoint"],
        "teacher_checkpoint_sha256": next(iter(teacher_hashes)),
        "observation_shape": [OBSERVATION_DIM],
        "action_shape": [ACTION_DIM],
        "observation_dtype": "float32",
        "action_dtype": "float32",
        "action_semantics": partials[0]["action_semantics"],
        "phase_names": partials[0]["phase_names"],
        "successful_episode_count": total_episodes,
        "transition_count": total_transitions,
        "rejected_episode_count": rejected,
        "collection_seeds": sorted(seeds),
        "split_seed": args.split_seed,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "episode_split": split,
        "shards": shard_records,
        "collection_commands": collection_commands,
        "versions_by_collection": versions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_sha256 = write_manifest(output, manifest)
    validate_dataset_manifest(output, verify_hashes=True)
    statistics = compute_dataset_statistics(output)
    statistics["manifest_sha256"] = manifest_sha256
    (root / "dataset_summary.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(statistics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

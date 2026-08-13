"""Run a resumable checkpoint-selection analysis checkpoint sweep through the existing formal evaluator."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quadcopter_waypoint.imitation.checkpoint_sweep import (
    CheckpointRecord,
    discover_checkpoints,
    resume_key,
    validate_evaluation_csv,
    write_inventory,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_manifest(path: Path, resume: bool) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "created_at_utc": _now(), "entries": {}}
    if not resume:
        raise FileExistsError(f"manifest already exists; pass --resume to continue: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("entries"), dict):
        raise ValueError(f"unsupported or damaged sweep manifest: {path}")
    return value


def _entry_matches(entry: dict[str, Any], record: CheckpointRecord, args: argparse.Namespace, eval_seed: int) -> bool:
    return (
        entry.get("checkpoint_sha256") == record.sha256
        and entry.get("task") == args.task
        and entry.get("agent") == args.agent
        and int(entry.get("eval_seed", -1)) == eval_seed
        and int(entry.get("episodes", -1)) == args.episodes
        and int(entry.get("num_envs", -1)) == args.num_envs
    )


def _evaluate(
    repo: Path,
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    record: CheckpointRecord,
    eval_seed: int,
    args: argparse.Namespace,
) -> None:
    key = resume_key(record, args.task, eval_seed, args.episodes, args.num_envs)
    existing = manifest["entries"].get(key)
    if existing is not None and existing.get("status") == "completed" and _entry_matches(existing, record, args, eval_seed):
        validate_evaluation_csv(existing["output_csv"], args.episodes)
        print(f"[RESUME] {record.checkpoint_id} eval_seed={eval_seed}")
        return

    raw_dir = output_dir / "raw" / record.checkpoint_id
    csv_path = raw_dir / f"eval_seed{eval_seed}.csv"
    log_path = raw_dir / f"eval_seed{eval_seed}.log"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo / "scripts/rl_games/eval_metrics.py"),
        f"--task={args.task}",
        f"--agent={args.agent}",
        f"--checkpoint={record.path}",
        f"--num_envs={args.num_envs}",
        f"--episodes={args.episodes}",
        f"--seed={eval_seed}",
        f"--csv={csv_path}",
    ]
    if args.headless:
        command.append("--headless")
    entry = {
        "checkpoint_path": record.path,
        "checkpoint_sha256": record.sha256,
        "actor_sha256": record.actor_sha256,
        "train_seed": record.train_seed,
        "epoch": record.epoch,
        "training_reward": record.training_reward,
        "kind": record.kind,
        "task": args.task,
        "agent": args.agent,
        "eval_seed": eval_seed,
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "command": shlex.join(command),
        "status": "running",
        "started_at_utc": _now(),
        "output_csv": str(csv_path.resolve()),
        "output_log": str(log_path.resolve()),
    }
    manifest["entries"][key] = entry
    manifest["updated_at_utc"] = _now()
    _write_json_atomic(manifest_path, manifest)
    print(f"[RUN] {record.checkpoint_id} eval_seed={eval_seed}")
    try:
        with log_path.open("w", encoding="utf-8") as log_stream:
            result = subprocess.run(command, cwd=repo, stdout=log_stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"formal evaluator exited with code {result.returncode}; see {log_path}")
        summary = validate_evaluation_csv(csv_path, args.episodes)
    except Exception as error:
        entry["status"] = "failed"
        entry["completed_at_utc"] = _now()
        entry["error"] = str(error)
        manifest["updated_at_utc"] = _now()
        _write_json_atomic(manifest_path, manifest)
        raise
    entry["status"] = "completed"
    entry["completed_at_utc"] = _now()
    entry["metrics"] = summary
    manifest["updated_at_utc"] = _now()
    _write_json_atomic(manifest_path, manifest)
    print(
        f"[DONE] {record.checkpoint_id} eval_seed={eval_seed} "
        f"settled={summary['settled_landing_rate']:.4f} deck_miss={summary['deck_miss_rate']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", default="rl_games_cfg_entry_point")
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--bc_checkpoint", default=None)
    parser.add_argument("--checkpoint_paths", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval_seeds", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint_glob", default="last_*_ep_*_rew_*.pth")
    parser.add_argument("--include_reward_selected", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("eval_seeds must be unique")

    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_checkpoints(
        args.run_dirs,
        args.bc_checkpoint,
        checkpoint_glob=args.checkpoint_glob,
        include_reward_selected=args.include_reward_selected,
        checkpoint_paths=args.checkpoint_paths,
    )
    write_inventory(output_dir / "checkpoint_inventory.json", records)
    canonical = [record for record in records if record.canonical]
    manifest_path = output_dir / "sweep_manifest.json"
    manifest = _load_manifest(manifest_path, args.resume)
    manifest["configuration"] = {
        "task": args.task,
        "agent": args.agent,
        "num_envs": args.num_envs,
        "episodes": args.episodes,
        "eval_seeds": args.eval_seeds,
        "headless": args.headless,
        "checkpoint_glob": args.checkpoint_glob,
    }
    manifest["inventory"] = str((output_dir / "checkpoint_inventory.json").resolve())
    _write_json_atomic(manifest_path, manifest)

    for record in canonical:
        for eval_seed in args.eval_seeds:
            _evaluate(repo, output_dir, manifest_path, manifest, record, eval_seed, args)

    completed = sum(entry.get("status") == "completed" for entry in manifest["entries"].values())
    expected = len(canonical) * len(args.eval_seeds)
    if completed < expected:
        raise RuntimeError(f"manifest contains only {completed} completed entries; expected at least {expected}")
    print(json.dumps({"canonical_checkpoints": len(canonical), "completed_evaluations": completed}, indent=2))


if __name__ == "__main__":
    main()

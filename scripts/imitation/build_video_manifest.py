"""Validate targeted P8B rollout videos and build the formal video manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quadcopter_waypoint.imitation.dataset import sha256_file


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecars", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite video manifest: {output}")

    entries: list[dict[str, Any]] = []
    for sidecar_value in args.sidecars:
        sidecar = Path(sidecar_value).expanduser().resolve()
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("video_generated") is not True:
            raise ValueError(f"sidecar does not describe a generated video: {sidecar}")
        video = sidecar.parent / metadata["video_path"]
        trajectory = sidecar.parent / metadata["trajectory"]
        if not video.is_file() or video.stat().st_size <= 0:
            raise FileNotFoundError(f"missing or empty video: {video}")
        if not trajectory.is_file() or trajectory.stat().st_size <= 0:
            raise FileNotFoundError(f"missing or empty trajectory: {trajectory}")
        if sha256_file(video) != metadata["video_sha256"]:
            raise ValueError(f"video SHA256 mismatch: {video}")
        if sha256_file(trajectory) != metadata["trajectory_sha256"]:
            raise ValueError(f"trajectory SHA256 mismatch: {trajectory}")
        if metadata.get("human_review_completed") is not False:
            raise ValueError("automated headless generation must not claim human review")
        entries.append(
            {
                "file_path": str(video),
                "sha256": metadata["video_sha256"],
                "git_commit": _git(repo, "rev-parse", "HEAD"),
                "checkpoint_path": metadata["checkpoint"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "actor_sha256": metadata["actor_sha256"],
                "training_seed": metadata["training_seed"],
                "evaluation_seed": metadata["evaluation_seed"],
                "episode_id": metadata["episode_id"],
                "scenario": metadata["scenario"],
                "success": bool(metadata["terminal"]["settled_landing"]),
                "failure_type": None if metadata["terminal"]["settled_landing"] else metadata["outcome"],
                "generation_command": metadata["generation_command"],
                "resolution": metadata["video_resolution"],
                "fps": metadata["video_fps"],
                "duration_seconds": metadata["video_duration_seconds"],
                "frames": metadata["video_frames"],
                "headless": bool(metadata["headless"]),
                "human_review_completed": False,
                "trajectory_path": str(trajectory),
                "trajectory_sha256": metadata["trajectory_sha256"],
                "sidecar_path": str(sidecar),
                "terminal": metadata["terminal"],
            }
        )

    if sum(bool(entry["success"]) for entry in entries) < 1:
        raise ValueError("video manifest requires at least one settled-landing success")
    if sum(not bool(entry["success"]) for entry in entries) < 1:
        raise ValueError("video manifest requires at least one real failure")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git(repo, "rev-parse", "HEAD"),
                "video_generation_completed": True,
                "headless_generation_validated": True,
                "interactive_gui_available": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
                "interactive_gui_review_performed": False,
                "human_review_completed": False,
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "videos": len(entries)}, indent=2))


if __name__ == "__main__":
    main()

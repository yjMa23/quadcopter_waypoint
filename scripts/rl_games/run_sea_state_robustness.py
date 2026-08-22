#!/usr/bin/env python3
"""Run reproducible zero-shot Sea-State profile sweeps through the frozen eval_metrics.py contract."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from quadcopter_waypoint.utils.sea_state_profiles import hydra_env_overrides, load_sea_state_profiles

TASK_ID = "Isaac-Quadcopter-ShipLanding-SeaState-Direct-v0"


def select_profiles(profiles: dict, names: list[str], families: list[str]) -> list[str]:
    if names:
        unknown = set(names) - profiles.keys()
        if unknown:
            raise ValueError(f"Unknown profiles: {sorted(unknown)}")
        selected = names
    elif families:
        selected = [name for name, profile in profiles.items() if profile["family"] in set(families)]
    else:
        selected = list(profiles)
    return sorted(selected, key=lambda name: (profiles[name]["family"], profiles[name]["severity_rank"], name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("benchmarks/sea_state/profiles.yaml"))
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy_label", type=str, default="teacher")
    parser.add_argument("--actor_preserving", action="store_true")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, default=Path("benchmarks/sea_state/pilot_raw"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    profiles = load_sea_state_profiles(args.profiles)
    selected = select_profiles(profiles, args.profile, args.family)
    seeds = args.seed or [245]
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for profile_name in selected:
        profile = profiles[profile_name]
        for seed in seeds:
            stem = f"{args.policy_label}__{profile_name}__seed{seed}"
            csv_path = args.output_dir / f"{stem}.csv"
            log_path = args.output_dir / f"{stem}.log"
            command = [
                sys.executable,
                "scripts/rl_games/eval_metrics.py",
                f"--task={TASK_ID}",
                f"--num_envs={args.num_envs}",
                f"--episodes={args.episodes}",
                f"--seed={seed}",
                f"--checkpoint={checkpoint}",
                f"--csv={csv_path}",
                "--headless",
            ]
            if args.actor_preserving:
                command.append("--agent=rl_games_actor_preserving_cfg_entry_point")
            command.extend(hydra_env_overrides(profile_name, profile))
            print("[RUN] " + shlex.join(command), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log_path.write_text(result.stdout)
            print(result.stdout, end="")
            if result.returncode != 0:
                raise RuntimeError(f"Profile {profile_name} seed {seed} failed; see {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Expand an rl_games policy checkpoint from a 16-D to a 22-D observation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quadcopter_waypoint.utils.checkpoint_observation import (
    expand_checkpoint_observation_state,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source rl_games checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Destination checkpoint; must not already exist.")
    parser.add_argument("--old-dim", type=int, default=16)
    parser.add_argument("--new-dim", type=int, default=22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if source == output:
        raise ValueError("Refusing to overwrite the source checkpoint.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_sha256 = sha256_file(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    expanded, changed = expand_checkpoint_observation_state(checkpoint, args.old_dim, args.new_dim)
    expanded["observation_expansion"]["source_checkpoint"] = str(source)
    expanded["observation_expansion"]["source_sha256"] = source_sha256

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded, output)
    output_sha256 = sha256_file(output)
    manifest = {
        "source_checkpoint": str(source),
        "source_sha256": source_sha256,
        "output_checkpoint": str(output),
        "output_sha256": output_sha256,
        "old_observation_dim": args.old_dim,
        "new_observation_dim": args.new_dim,
        "changed": changed,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"source checkpoint: {source}")
    print(f"source SHA256:    {source_sha256}")
    for line in changed:
        print(f"changed: {line}")
    print(f"output checkpoint: {output}")
    print(f"output SHA256:     {output_sha256}")
    print(f"manifest:          {manifest_path}")


if __name__ == "__main__":
    main()

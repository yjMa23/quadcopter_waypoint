"""Create and verify a fresh PPO checkpoint initialized from the standalone imitation-learning benchmark BC actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quadcopter_waypoint.imitation.checkpoint import (
    bc_rlgames_parity_error,
    build_bc_initialized_rlgames_checkpoint,
)
from quadcopter_waypoint.imitation.dataset import sha256_file, validate_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc_checkpoint", required=True)
    parser.add_argument("--template_checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--value_seed", type=int, default=2026)
    parser.add_argument("--parity_samples", type=int, default=1024)
    parser.add_argument("--parity_tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    validate_dataset_manifest(manifest_path, verify_hashes=True)
    metadata = build_bc_initialized_rlgames_checkpoint(
        args.bc_checkpoint,
        args.template_checkpoint,
        args.output,
        dataset_manifest_sha256=sha256_file(manifest_path),
        value_seed=args.value_seed,
    )
    generator = torch.Generator().manual_seed(123456)
    observations = torch.randn(args.parity_samples, 22, generator=generator)
    error = bc_rlgames_parity_error(args.bc_checkpoint, args.output, observations)
    metadata["parity_samples"] = args.parity_samples
    metadata["maximum_absolute_action_error"] = error
    metadata["parity_tolerance"] = args.parity_tolerance
    metadata["parity_passed"] = error < args.parity_tolerance
    if not metadata["parity_passed"]:
        Path(args.output).unlink(missing_ok=True)
        raise RuntimeError(f"BC→RL-Games parity failed: max_abs_error={error}")
    sidecar = Path(args.output).with_suffix(Path(args.output).suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

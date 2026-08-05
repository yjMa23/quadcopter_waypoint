#!/usr/bin/env python3
"""Create a fresh separate actor/critic P8B checkpoint from the frozen P7 BC initialization."""

from __future__ import annotations

import argparse
import json

from quadcopter_waypoint.imitation.p8b_checkpoint import build_p8b_separate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--critic_seed", type=int, default=2026)
    args = parser.parse_args()
    result = build_p8b_separate_checkpoint(
        source_checkpoint=args.source_checkpoint,
        config_path=args.config,
        output_path=args.output,
        dataset_manifest=args.dataset_manifest,
        critic_seed=args.critic_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

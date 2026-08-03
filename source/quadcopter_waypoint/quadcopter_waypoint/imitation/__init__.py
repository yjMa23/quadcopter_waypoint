"""Phase 7 imitation-learning utilities for the physical-deck landing task."""

from .checkpoint import build_bc_initialized_rlgames_checkpoint
from .dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PHASE_NAMES,
    compute_dataset_statistics,
    create_episode_split,
    load_split_transitions,
    sha256_file,
    validate_dataset_manifest,
    validate_shard_arrays,
)
from .policy import BCActor, normalize_observations

__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "PHASE_NAMES",
    "BCActor",
    "build_bc_initialized_rlgames_checkpoint",
    "compute_dataset_statistics",
    "create_episode_split",
    "load_split_transitions",
    "normalize_observations",
    "sha256_file",
    "validate_dataset_manifest",
    "validate_shard_arrays",
]

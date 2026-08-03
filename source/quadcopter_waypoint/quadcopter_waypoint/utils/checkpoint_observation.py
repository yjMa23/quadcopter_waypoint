"""rl_games checkpoint observation-expansion helpers."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

POLICY_FIRST_LAYER_KEY = "a2c_network.actor_mlp.0.weight"
OBS_MEAN_KEY = "running_mean_std.running_mean"
OBS_VAR_KEY = "running_mean_std.running_var"
OBS_COUNT_KEY = "running_mean_std.count"


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_last_dimension(tensor: torch.Tensor, old_dim: int, new_dim: int, fill: float) -> torch.Tensor:
    if tensor.shape[-1] != old_dim:
        raise ValueError(f"Expected final dimension {old_dim}, got shape {tuple(tensor.shape)}")
    expanded = torch.full((*tensor.shape[:-1], new_dim), fill, dtype=tensor.dtype, device=tensor.device)
    expanded[..., :old_dim] = tensor
    return expanded


def expand_checkpoint_observation_state(
    checkpoint: dict[str, Any], old_dim: int = 16, new_dim: int = 22
) -> tuple[dict[str, Any], list[str]]:
    """Expand policy input and observation normalization while preserving all unrelated tensors.

    Optimizer first/second moments matching the policy first-layer shape are expanded as well, which allows
    rl_games to resume fine-tuning instead of loading weights only with incompatible optimizer state.
    """
    if new_dim <= old_dim:
        raise ValueError(f"new_dim must exceed old_dim, got {old_dim} -> {new_dim}")
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise ValueError("Checkpoint does not contain a model state dictionary.")

    expanded_checkpoint = copy.deepcopy(checkpoint)
    model = expanded_checkpoint["model"]
    required = (POLICY_FIRST_LAYER_KEY, OBS_MEAN_KEY, OBS_VAR_KEY, OBS_COUNT_KEY)
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    first_weight = model[POLICY_FIRST_LAYER_KEY]
    if first_weight.ndim != 2 or first_weight.shape[1] != old_dim:
        raise ValueError(
            f"Expected {POLICY_FIRST_LAYER_KEY} shape (*, {old_dim}), got {tuple(first_weight.shape)}"
        )
    obs_mean = model[OBS_MEAN_KEY]
    obs_var = model[OBS_VAR_KEY]
    if tuple(obs_mean.shape) != (old_dim,):
        raise ValueError(f"Expected {OBS_MEAN_KEY} shape ({old_dim},), got {tuple(obs_mean.shape)}")
    if tuple(obs_var.shape) != (old_dim,):
        raise ValueError(f"Expected {OBS_VAR_KEY} shape ({old_dim},), got {tuple(obs_var.shape)}")
    if model[OBS_COUNT_KEY].ndim != 0:
        raise ValueError(f"Expected scalar {OBS_COUNT_KEY}, got {tuple(model[OBS_COUNT_KEY].shape)}")

    model[POLICY_FIRST_LAYER_KEY] = _expand_last_dimension(first_weight, old_dim, new_dim, 0.0)
    model[OBS_MEAN_KEY] = _expand_last_dimension(obs_mean, old_dim, new_dim, 0.0)
    model[OBS_VAR_KEY] = _expand_last_dimension(obs_var, old_dim, new_dim, 1.0)
    changed = [
        f"model.{POLICY_FIRST_LAYER_KEY}: {tuple(first_weight.shape)} -> {tuple(model[POLICY_FIRST_LAYER_KEY].shape)}",
        f"model.{OBS_MEAN_KEY}: {tuple(obs_mean.shape)} -> {tuple(model[OBS_MEAN_KEY].shape)}",
        f"model.{OBS_VAR_KEY}: {tuple(obs_var.shape)} -> {tuple(model[OBS_VAR_KEY].shape)}",
        f"model.{OBS_COUNT_KEY}: preserved scalar",
    ]

    optimizer = expanded_checkpoint.get("optimizer")
    if isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict):
        matching_optimizer_tensors = 0
        for parameter_id, state in optimizer["state"].items():
            if not isinstance(state, dict):
                continue
            for state_name, tensor in list(state.items()):
                if isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(first_weight.shape):
                    state[state_name] = _expand_last_dimension(tensor, old_dim, new_dim, 0.0)
                    changed.append(
                        f"optimizer.state[{parameter_id!r}].{state_name}: "
                        f"{tuple(tensor.shape)} -> {tuple(state[state_name].shape)}"
                    )
                    matching_optimizer_tensors += 1
        if matching_optimizer_tensors not in (0, 2):
            raise ValueError(
                "Expected zero optimizer tensors (weights-only checkpoint) or Adam's two first-layer moments, "
                f"found {matching_optimizer_tensors}."
            )

    expanded_checkpoint["observation_expansion"] = {
        "old_dim": old_dim,
        "new_dim": new_dim,
        "new_policy_columns_initialization": "zeros",
        "new_observation_mean_initialization": "zeros",
        "new_observation_variance_initialization": "ones",
    }
    return expanded_checkpoint, changed

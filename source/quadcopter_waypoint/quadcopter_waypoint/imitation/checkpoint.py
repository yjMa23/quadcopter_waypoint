"""Reliable migration from a standalone BC actor into an RL-Games PPO checkpoint."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .dataset import sha256_file
from .policy import BCActor, normalize_observations

_ACTOR_KEY_MAP = {
    "actor_mlp.0.weight": "a2c_network.actor_mlp.0.weight",
    "actor_mlp.0.bias": "a2c_network.actor_mlp.0.bias",
    "actor_mlp.2.weight": "a2c_network.actor_mlp.2.weight",
    "actor_mlp.2.bias": "a2c_network.actor_mlp.2.bias",
    "mu.weight": "a2c_network.mu.weight",
    "mu.bias": "a2c_network.mu.bias",
}


def _validate_actor_mapping(bc_state: Mapping[str, torch.Tensor], model_state: Mapping[str, torch.Tensor]) -> None:
    missing_bc = sorted(set(_ACTOR_KEY_MAP) - set(bc_state))
    missing_rl = sorted(set(_ACTOR_KEY_MAP.values()) - set(model_state))
    if missing_bc or missing_rl:
        raise KeyError(f"actor checkpoint keys missing; bc={missing_bc}, rl_games={missing_rl}")
    for bc_key, rl_key in _ACTOR_KEY_MAP.items():
        if bc_state[bc_key].shape != model_state[rl_key].shape:
            raise ValueError(
                f"shape mismatch for {bc_key}->{rl_key}: {tuple(bc_state[bc_key].shape)} != "
                f"{tuple(model_state[rl_key].shape)}"
            )


def build_bc_initialized_rlgames_checkpoint(
    bc_checkpoint: str | Path,
    template_checkpoint: str | Path,
    output_path: str | Path,
    dataset_manifest_sha256: str,
    value_seed: int = 2026,
) -> dict[str, Any]:
    """Create a fresh-training PPO checkpoint whose deterministic actor exactly matches BC.

    The teacher/template contributes only RL-Games key structure and fixed sigma. Training progress,
    optimizer moments, environment state, value normalization, and teacher value-head parameters are reset.
    """
    bc_path = Path(bc_checkpoint)
    template_path = Path(template_checkpoint)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing BC-init checkpoint: {output}")
    actor, bc_payload = BCActor.from_checkpoint(bc_path, map_location="cpu")
    template = torch.load(template_path, map_location="cpu", weights_only=False)
    if "model" not in template or "optimizer" not in template:
        raise KeyError("template checkpoint must contain model and optimizer")
    checkpoint = copy.deepcopy(template)
    model_state = checkpoint["model"]
    bc_state = actor.state_dict()
    _validate_actor_mapping(bc_state, model_state)

    for bc_key, rl_key in _ACTOR_KEY_MAP.items():
        model_state[rl_key] = bc_state[bc_key].detach().clone().to(model_state[rl_key].dtype)
    model_state["running_mean_std.running_mean"] = actor.running_mean.detach().clone()
    model_state["running_mean_std.running_var"] = actor.running_var.detach().clone()
    model_state["running_mean_std.count"] = actor.running_count.detach().clone()

    # RL-Games uses the shared actor MLP for critic features. Only the scalar value head is independently reset.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(value_seed)
        value_head = nn.Linear(model_state["a2c_network.value.weight"].shape[1], 1)
    model_state["a2c_network.value.weight"] = value_head.weight.detach().clone().to(
        model_state["a2c_network.value.weight"].dtype
    )
    model_state["a2c_network.value.bias"] = value_head.bias.detach().clone().to(
        model_state["a2c_network.value.bias"].dtype
    )
    model_state["value_mean_std.running_mean"] = torch.zeros_like(model_state["value_mean_std.running_mean"])
    model_state["value_mean_std.running_var"] = torch.ones_like(model_state["value_mean_std.running_var"])
    model_state["value_mean_std.count"] = torch.ones_like(model_state["value_mean_std.count"])

    optimizer = copy.deepcopy(checkpoint["optimizer"])
    optimizer["state"] = {}
    checkpoint["optimizer"] = optimizer
    checkpoint["epoch"] = 0
    checkpoint["frame"] = 0
    checkpoint["last_mean_rewards"] = torch.zeros((), dtype=torch.float32)
    checkpoint["env_state"] = None
    checkpoint.pop("observation_expansion", None)
    checkpoint["bc_initialization"] = {
        "source_bc_checkpoint": str(bc_path),
        "source_bc_checkpoint_sha256": sha256_file(bc_path),
        "source_template_checkpoint": str(template_path),
        "source_template_checkpoint_sha256": sha256_file(template_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "value_head_initialization": "torch.nn.Linear default initialization",
        "value_head_seed": int(value_seed),
        "optimizer_state": "cleared",
        "epoch_frame_history": "reset_to_zero",
        "observation_normalization": "copied_from_bc",
        "fixed_sigma": "preserved_from_template",
        "network_config": bc_payload["network_config"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "bc_checkpoint_sha256": sha256_file(bc_path),
        "template_checkpoint_sha256": sha256_file(template_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "value_seed": int(value_seed),
    }


def deterministic_action_from_rlgames_model(
    model_state: Mapping[str, torch.Tensor], raw_observation: torch.Tensor
) -> torch.Tensor:
    """Evaluate the restored RL-Games deterministic actor from its serialized tensors."""
    normalized = normalize_observations(
        raw_observation,
        model_state["running_mean_std.running_mean"],
        model_state["running_mean_std.running_var"],
    )
    hidden = torch.nn.functional.elu(
        torch.nn.functional.linear(
            normalized,
            model_state["a2c_network.actor_mlp.0.weight"],
            model_state["a2c_network.actor_mlp.0.bias"],
        )
    )
    hidden = torch.nn.functional.elu(
        torch.nn.functional.linear(
            hidden,
            model_state["a2c_network.actor_mlp.2.weight"],
            model_state["a2c_network.actor_mlp.2.bias"],
        )
    )
    action = torch.nn.functional.linear(
        hidden,
        model_state["a2c_network.mu.weight"],
        model_state["a2c_network.mu.bias"],
    )
    return torch.clamp(action, -1.0, 1.0)


def bc_rlgames_parity_error(
    bc_checkpoint: str | Path,
    rlgames_checkpoint: str | Path,
    raw_observation: torch.Tensor,
) -> float:
    """Return maximum absolute deterministic-action error after identical normalization and clamp."""
    actor, _ = BCActor.from_checkpoint(bc_checkpoint, map_location="cpu")
    payload = torch.load(Path(rlgames_checkpoint), map_location="cpu", weights_only=False)
    with torch.inference_mode():
        bc_action = actor(raw_observation.cpu())
        rl_action = deterministic_action_from_rlgames_model(payload["model"], raw_observation.cpu())
    return float(torch.max(torch.abs(bc_action - rl_action)).item())

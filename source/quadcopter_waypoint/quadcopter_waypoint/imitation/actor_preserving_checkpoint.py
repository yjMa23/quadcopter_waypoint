"""actor-preserving PPO shared-to-separate checkpoint migration and deterministic parity helpers."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from rl_games.algos_torch import model_builder
from torch import nn

from .dataset import sha256_file
from .policy import normalize_observations

SCHEMA_VERSION = "actor-preserving-separate-v1"
LEGACY_BC_INITIALIZATION_KEY = "p7_bc_initialization"
LEGACY_ACTOR_PRESERVING_KEY = "p8b_actor_preserving"
LEGACY_SCHEMA_VERSION = "p8b-separate-v1"
ACTOR_KEYS = (
    "a2c_network.actor_mlp.0.weight",
    "a2c_network.actor_mlp.0.bias",
    "a2c_network.actor_mlp.2.weight",
    "a2c_network.actor_mlp.2.bias",
    "a2c_network.mu.weight",
    "a2c_network.mu.bias",
)
CRITIC_KEYS = (
    "a2c_network.critic_mlp.0.weight",
    "a2c_network.critic_mlp.0.bias",
    "a2c_network.critic_mlp.2.weight",
    "a2c_network.critic_mlp.2.bias",
    "a2c_network.value.weight",
    "a2c_network.value.bias",
)
OBS_RMS_KEYS = (
    "running_mean_std.running_mean",
    "running_mean_std.running_var",
    "running_mean_std.count",
)
VALUE_RMS_KEYS = (
    "value_mean_std.running_mean",
    "value_mean_std.running_var",
    "value_mean_std.count",
)
REFERENCE_KEY_MAP = {
    "a2c_network.actor_mlp.0.weight": "actor_mlp.0.weight",
    "a2c_network.actor_mlp.0.bias": "actor_mlp.0.bias",
    "a2c_network.actor_mlp.2.weight": "actor_mlp.2.weight",
    "a2c_network.actor_mlp.2.bias": "actor_mlp.2.bias",
    "a2c_network.mu.weight": "mu.weight",
    "a2c_network.mu.bias": "mu.bias",
}


def tensor_mapping_sha256(state: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> str:
    """Hash an ordered tensor subset independently of pickle metadata."""
    digest = hashlib.sha256()
    for key in keys:
        if key not in state:
            raise KeyError(f"state is missing {key}")
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def actor_weights_sha256(model_state: Mapping[str, torch.Tensor]) -> str:
    return tensor_mapping_sha256(model_state, ACTOR_KEYS)


def critic_weights_sha256(model_state: Mapping[str, torch.Tensor]) -> str:
    return tensor_mapping_sha256(model_state, CRITIC_KEYS)


def obs_rms_sha256(model_state: Mapping[str, torch.Tensor]) -> str:
    return tensor_mapping_sha256(model_state, OBS_RMS_KEYS)


class FrozenReferenceActor(nn.Module):
    """Independent deterministic BC mean actor operating on normalized observations."""

    def __init__(self, observation_dim: int = 22, hidden_units: tuple[int, int] = (64, 64), action_dim: int = 4):
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(observation_dim, hidden_units[0]),
            nn.ELU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden_units[-1], action_dim)
        self.requires_grad_(False)
        self.eval()

    def forward(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        return self.mu(self.actor_mlp(normalized_observation))


def reference_state_from_model(model_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        reference_key: model_state[model_key].detach().cpu().clone()
        for model_key, reference_key in REFERENCE_KEY_MAP.items()
    }


def load_reference_actor(
    state: Mapping[str, torch.Tensor], device: torch.device | str = "cpu"
) -> FrozenReferenceActor:
    actor = FrozenReferenceActor().to(device)
    actor.load_state_dict(state, strict=True)
    actor.requires_grad_(False)
    actor.eval()
    return actor


def load_actor_preserving_params(config_path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "params" not in payload:
        raise ValueError("actor-preserving PPO config must contain a params mapping")
    params = copy.deepcopy(payload["params"])
    if params["network"].get("separate") is not True:
        raise ValueError("actor-preserving PPO network.separate must be true")
    if params["algo"].get("name") != "actor_preserving_ppo":
        raise ValueError("actor-preserving PPO algo.name must be actor_preserving_ppo")
    return params


def actor_preserving_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Read current or frozen legacy actor-preserving checkpoint metadata."""
    metadata = checkpoint.get("actor_preserving_ppo")
    if not isinstance(metadata, dict):
        metadata = checkpoint.get(LEGACY_ACTOR_PRESERVING_KEY)
    if not isinstance(metadata, dict) or metadata.get("schema_version") not in {
        SCHEMA_VERSION,
        LEGACY_SCHEMA_VERSION,
    }:
        raise ValueError("checkpoint is missing valid actor-preserving PPO metadata")
    return metadata


def build_actor_preserving_model(params: Mapping[str, Any], seed: int = 2026) -> nn.Module:
    """Build the exact pure-Python RL-Games model used by actor-preserving PPO."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        builder = model_builder.ModelBuilder()
        factory = builder.load(copy.deepcopy(dict(params)))
        model = factory.build(
            {
                "actions_num": 4,
                "input_shape": (22,),
                "num_seqs": 1,
                "value_size": 1,
                "normalize_value": bool(params["config"]["normalize_value"]),
                "normalize_input": bool(params["config"]["normalize_input"]),
            }
        )
    return model


def _validate_source_model(source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> None:
    for key in (*ACTOR_KEYS, *OBS_RMS_KEYS, "a2c_network.sigma"):
        if key not in source:
            raise KeyError(f"source checkpoint model is missing {key}")
        if key not in target:
            raise KeyError(f"target separate model is missing {key}")
        if source[key].shape != target[key].shape:
            raise ValueError(f"shape mismatch for {key}: {tuple(source[key].shape)} != {tuple(target[key].shape)}")
    for key in CRITIC_KEYS:
        if key not in target:
            raise KeyError(f"target separate model is missing {key}")


def build_actor_preserving_separate_checkpoint(
    source_checkpoint: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    dataset_manifest: str | Path,
    critic_seed: int = 2026,
) -> dict[str, Any]:
    """Create a fresh actor-preserving PPO checkpoint while preserving the BC actor and observation RMS exactly."""
    source_path = Path(source_checkpoint).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest = Path(dataset_manifest).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing actor-preserving PPO checkpoint: {output}")
    if not source_path.is_file() or not manifest.is_file() or not config.is_file():
        raise FileNotFoundError("source checkpoint, dataset manifest, and actor-preserving PPO config must exist")

    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    initialization = source_payload.get("bc_initialization")
    if not isinstance(initialization, dict):
        initialization = source_payload.get(LEGACY_BC_INITIALIZATION_KEY)
    if "model" not in source_payload or not isinstance(initialization, dict):
        raise ValueError("source must be the fresh imitation-learning benchmark BC-initialized RL-Games checkpoint")
    source_model = source_payload["model"]
    params = load_actor_preserving_params(config)
    model = build_actor_preserving_model(params, seed=critic_seed)
    target_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _validate_source_model(source_model, target_model)

    for key in (*ACTOR_KEYS, *OBS_RMS_KEYS, "a2c_network.sigma"):
        target_model[key] = source_model[key].detach().cpu().clone().to(target_model[key].dtype)
    target_model["value_mean_std.running_mean"] = torch.zeros_like(target_model["value_mean_std.running_mean"])
    target_model["value_mean_std.running_var"] = torch.ones_like(target_model["value_mean_std.running_var"])
    target_model["value_mean_std.count"] = torch.ones_like(target_model["value_mean_std.count"])

    model.load_state_dict(target_model, strict=True)
    learning_rate = float(params["config"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, eps=1.0e-8)
    source_sha = sha256_file(source_path)
    manifest_sha = sha256_file(manifest)
    reference_state = reference_state_from_model(target_model)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_bc_checkpoint": initialization.get("source_bc_checkpoint"),
        "source_bc_checkpoint_sha256": initialization.get("source_bc_checkpoint_sha256"),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": manifest_sha,
        "config_path": str(config),
        "critic_initialization": "RL-Games separate network default initialization",
        "critic_seed": int(critic_seed),
        "optimizer_state": "fresh_empty_adam_for_separate_model",
        "epoch_frame_history": "reset_to_zero",
        "observation_normalization": "copied_from_behavior_cloning_init_and_frozen",
        "fixed_sigma": "copied_from_behavior_cloning_init",
        "warmup_epochs": int(params["config"]["actor_preserving"]["warmup_epochs"]),
        "freeze_lr_scheduler_during_warmup": bool(
            params["config"]["actor_preserving"]["freeze_lr_scheduler_during_warmup"]
        ),
        "base_learning_rate": learning_rate,
        "freeze_observation_rms": bool(params["config"]["actor_preserving"]["freeze_observation_rms"]),
        "bc_anchor_type": params["config"]["actor_preserving"]["bc_anchor_type"],
        "bc_anchor_coefficient": float(params["config"]["actor_preserving"]["bc_anchor_coefficient"]),
        "actor_weights_sha256": actor_weights_sha256(target_model),
        "critic_weights_sha256": critic_weights_sha256(target_model),
        "observation_rms_sha256": obs_rms_sha256(target_model),
        "reference_actor_state": reference_state,
    }
    checkpoint = {
        "model": target_model,
        "epoch": 0,
        "frame": 0,
        "optimizer": optimizer.state_dict(),
        "last_mean_rewards": torch.zeros((), dtype=torch.float32),
        "env_state": None,
        "actor_preserving_ppo": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "source_checkpoint_sha256": source_sha,
        "dataset_manifest_sha256": manifest_sha,
        "actor_weights_sha256": metadata["actor_weights_sha256"],
        "critic_weights_sha256": metadata["critic_weights_sha256"],
        "observation_rms_sha256": metadata["observation_rms_sha256"],
        "schema_version": SCHEMA_VERSION,
    }


def deterministic_mean_from_model_state(
    model_state: Mapping[str, torch.Tensor], raw_observation: torch.Tensor, clamp: bool = True
) -> torch.Tensor:
    """Evaluate shared or separate serialized actor tensors with RL-Games normalization semantics."""
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
    mean = torch.nn.functional.linear(
        hidden,
        model_state["a2c_network.mu.weight"],
        model_state["a2c_network.mu.bias"],
    )
    return torch.clamp(mean, -1.0, 1.0) if clamp else mean


def checkpoint_parity_error(
    source_checkpoint: str | Path, actor_preserving_checkpoint: str | Path, raw_observation: torch.Tensor
) -> float:
    source = torch.load(Path(source_checkpoint), map_location="cpu", weights_only=False)
    target = torch.load(Path(actor_preserving_checkpoint), map_location="cpu", weights_only=False)
    with torch.inference_mode():
        source_action = deterministic_mean_from_model_state(source["model"], raw_observation.cpu())
        target_action = deterministic_mean_from_model_state(target["model"], raw_observation.cpu())
    return float(torch.max(torch.abs(source_action - target_action)).item())

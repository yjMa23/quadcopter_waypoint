"""Standalone behavior-cloning actor with RL-Games-compatible observation normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .dataset import ACTION_DIM, OBSERVATION_DIM


@dataclass(frozen=True)
class BCNetworkConfig:
    """Serializable architecture contract shared by BC and the PPO actor."""

    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM
    hidden_units: tuple[int, int] = (64, 64)
    activation: str = "elu"
    observation_epsilon: float = 1.0e-5
    observation_clip: float = 5.0
    action_clip: float = 1.0


def normalize_observations(
    observations: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    epsilon: float = 1.0e-5,
    clip: float = 5.0,
) -> torch.Tensor:
    """Apply the exact inference transform used by RL-Games RunningMeanStd."""
    if observations.shape[-1] != running_mean.numel() or running_mean.shape != running_var.shape:
        raise ValueError(
            f"normalization shape mismatch: observations={tuple(observations.shape)}, "
            f"mean={tuple(running_mean.shape)}, var={tuple(running_var.shape)}"
        )
    if not torch.isfinite(observations).all():
        raise ValueError("observations contain NaN or Inf")
    if not torch.isfinite(running_mean).all() or not torch.isfinite(running_var).all():
        raise ValueError("running statistics contain NaN or Inf")
    if torch.any(running_var < 0):
        raise ValueError("running variance contains negative values")
    normalized = (observations - running_mean.float()) / torch.sqrt(running_var.float() + epsilon)
    return torch.clamp(normalized, min=-clip, max=clip)


class BCActor(nn.Module):
    """22→64→64→4 ELU actor using frozen teacher observation statistics."""

    def __init__(
        self,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        running_count: torch.Tensor | float,
        config: BCNetworkConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or BCNetworkConfig()
        if self.config.observation_dim != OBSERVATION_DIM or self.config.action_dim != ACTION_DIM:
            raise ValueError("imitation-learning benchmark BC actor requires 22 observations and 4 actions")
        if tuple(self.config.hidden_units) != (64, 64) or self.config.activation.lower() != "elu":
            raise ValueError("imitation-learning benchmark BC actor must match PPO actor architecture [64, 64] with ELU")
        mean = torch.as_tensor(running_mean, dtype=torch.float64).reshape(-1)
        var = torch.as_tensor(running_var, dtype=torch.float64).reshape(-1)
        count = torch.as_tensor(running_count, dtype=torch.float64).reshape(())
        if mean.numel() != OBSERVATION_DIM or var.numel() != OBSERVATION_DIM:
            raise ValueError("observation running statistics must contain 22 values")
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or not torch.isfinite(count):
            raise ValueError("observation running statistics contain NaN or Inf")
        if torch.any(var < 0) or count <= 0:
            raise ValueError("observation running variance/count are invalid")
        self.register_buffer("running_mean", mean.clone())
        self.register_buffer("running_var", var.clone())
        self.register_buffer("running_count", count.clone())
        self.actor_mlp = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.mu = nn.Linear(64, ACTION_DIM)

    def normalized_observation(self, raw_observation: torch.Tensor) -> torch.Tensor:
        """Normalize raw task observations without updating frozen running statistics."""
        return normalize_observations(
            raw_observation,
            self.running_mean,
            self.running_var,
            epsilon=self.config.observation_epsilon,
            clip=self.config.observation_clip,
        )

    def raw_action(self, raw_observation: torch.Tensor) -> torch.Tensor:
        """Return the unclamped deterministic mean action used for supervised fitting."""
        normalized = self.normalized_observation(raw_observation)
        return self.mu(self.actor_mlp(normalized))

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action after the environment's [-1, 1] clamp."""
        return torch.clamp(
            self.raw_action(raw_observation),
            min=-self.config.action_clip,
            max=self.config.action_clip,
        )

    def export_config(self) -> dict[str, Any]:
        """Return a JSON/checkpoint-friendly architecture mapping."""
        value = asdict(self.config)
        value["hidden_units"] = list(self.config.hidden_units)
        return value

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | Mapping[str, Any],
        map_location: str | torch.device = "cpu",
    ) -> tuple["BCActor", dict[str, Any]]:
        """Restore a standalone BC actor and return its complete checkpoint metadata."""
        payload = (
            torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
            if isinstance(checkpoint, (str, Path))
            else dict(checkpoint)
        )
        config_value = payload.get("network_config", {})
        config = BCNetworkConfig(
            observation_dim=int(config_value.get("observation_dim", OBSERVATION_DIM)),
            action_dim=int(config_value.get("action_dim", ACTION_DIM)),
            hidden_units=tuple(config_value.get("hidden_units", (64, 64))),
            activation=str(config_value.get("activation", "elu")),
            observation_epsilon=float(config_value.get("observation_epsilon", 1.0e-5)),
            observation_clip=float(config_value.get("observation_clip", 5.0)),
            action_clip=float(config_value.get("action_clip", 1.0)),
        )
        normalization = payload["observation_normalization"]
        actor = cls(
            normalization["running_mean"],
            normalization["running_var"],
            normalization["count"],
            config=config,
        )
        actor.load_state_dict(payload["model_state_dict"], strict=True)
        actor.eval()
        return actor, payload


def teacher_normalization_from_rlgames_checkpoint(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Extract and validate the frozen input statistics from an RL-Games checkpoint."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model = payload.get("model", {})
    required = (
        "running_mean_std.running_mean",
        "running_mean_std.running_var",
        "running_mean_std.count",
    )
    missing = [name for name in required if name not in model]
    if missing:
        raise KeyError(f"RL-Games checkpoint missing normalization tensors: {missing}")
    mean = model[required[0]].detach().cpu().to(torch.float64)
    var = model[required[1]].detach().cpu().to(torch.float64)
    count = model[required[2]].detach().cpu().to(torch.float64)
    if mean.shape != (OBSERVATION_DIM,) or var.shape != (OBSERVATION_DIM,) or count.ndim != 0:
        raise ValueError(
            f"unexpected normalization shapes: mean={tuple(mean.shape)}, var={tuple(var.shape)}, count={tuple(count.shape)}"
        )
    return {"running_mean": mean, "running_var": var, "count": count}

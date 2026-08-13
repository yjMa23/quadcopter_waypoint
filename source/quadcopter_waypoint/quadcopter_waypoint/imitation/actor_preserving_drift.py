"""Extended actor-preserving PPO checkpoint hash and policy-drift diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .actor_preserving_checkpoint import (
    ACTOR_KEYS,
    CRITIC_KEYS,
    OBS_RMS_KEYS,
    actor_weights_sha256,
    critic_weights_sha256,
    deterministic_mean_from_model_state,
    obs_rms_sha256,
)


def _relative_l2(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor], keys: tuple[str, ...]
) -> float:
    delta_sq = torch.zeros((), dtype=torch.float64)
    reference_sq = torch.zeros((), dtype=torch.float64)
    for key in keys:
        if key not in reference or key not in candidate:
            raise KeyError(f"drift state is missing {key}")
        delta = candidate[key].double().cpu() - reference[key].double().cpu()
        delta_sq += torch.sum(torch.square(delta))
        reference_sq += torch.sum(torch.square(reference[key].double().cpu()))
    return float(torch.sqrt(delta_sq).item() / max(torch.sqrt(reference_sq).item(), 1.0e-12))


def compute_actor_preserving_drift_metrics(
    reference_model: Mapping[str, torch.Tensor],
    candidate_model: Mapping[str, torch.Tensor],
    raw_observation: torch.Tensor,
) -> dict[str, Any]:
    """Compute actor/critic/action/RMS/sigma drift on one fixed raw-observation batch."""
    if raw_observation.ndim != 2 or raw_observation.shape[1] != 22:
        raise ValueError("raw_observation must have shape [batch, 22]")
    if not torch.isfinite(raw_observation).all():
        raise FloatingPointError("raw_observation contains NaN or Inf")
    with torch.inference_mode():
        reference_action = deterministic_mean_from_model_state(reference_model, raw_observation.cpu())
        candidate_action = deterministic_mean_from_model_state(candidate_model, raw_observation.cpu())
    error = candidate_action.float() - reference_action.float()
    if not torch.isfinite(error).all():
        raise FloatingPointError("deterministic action drift contains NaN or Inf")

    mean_delta = candidate_model[OBS_RMS_KEYS[0]].double().cpu() - reference_model[OBS_RMS_KEYS[0]].double().cpu()
    var_delta = candidate_model[OBS_RMS_KEYS[1]].double().cpu() - reference_model[OBS_RMS_KEYS[1]].double().cpu()
    count_delta = candidate_model[OBS_RMS_KEYS[2]].double().cpu() - reference_model[OBS_RMS_KEYS[2]].double().cpu()
    sigma_delta = candidate_model["a2c_network.sigma"].double().cpu() - reference_model[
        "a2c_network.sigma"
    ].double().cpu()
    return {
        "actor_sha256": actor_weights_sha256(candidate_model),
        "critic_sha256": critic_weights_sha256(candidate_model),
        "observation_rms_sha256": obs_rms_sha256(candidate_model),
        "action_mse_vs_reference": float(torch.square(error).mean().item()),
        "action_max_abs_error_vs_reference": float(torch.max(torch.abs(error)).item()),
        "action_dim_mse_vs_reference": [float(value) for value in torch.square(error).mean(dim=0).tolist()],
        "actor_parameter_relative_l2": _relative_l2(reference_model, candidate_model, ACTOR_KEYS),
        "critic_parameter_relative_l2": _relative_l2(reference_model, candidate_model, CRITIC_KEYS),
        "observation_mean_l2": float(torch.linalg.vector_norm(mean_delta).item()),
        "observation_variance_l2": float(torch.linalg.vector_norm(var_delta).item()),
        "observation_count_delta": float(count_delta.item()),
        "fixed_sigma_l2": float(torch.linalg.vector_norm(sigma_delta).item()),
        "fixed_sigma_max_abs": float(torch.max(torch.abs(sigma_delta)).item()),
    }

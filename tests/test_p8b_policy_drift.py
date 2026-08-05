"""Extended P8B policy-drift metrics and hash semantics."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from quadcopter_waypoint.imitation.p8b_checkpoint import build_p8b_model, load_p8b_params
from quadcopter_waypoint.imitation.p8b_drift import compute_p8b_drift_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_p8b_ppo_cfg.yaml"
)


def _state() -> dict[str, torch.Tensor]:
    model = build_p8b_model(load_p8b_params(CONFIG))
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def test_zero_drift_is_exact_for_identical_state() -> None:
    state = _state()
    metrics = compute_p8b_drift_metrics(state, copy.deepcopy(state), torch.randn(128, 22))
    assert metrics["action_mse_vs_reference"] == pytest.approx(0.0)
    assert metrics["action_max_abs_error_vs_reference"] == pytest.approx(0.0)
    assert metrics["actor_parameter_relative_l2"] == pytest.approx(0.0)
    assert metrics["critic_parameter_relative_l2"] == pytest.approx(0.0)
    assert metrics["observation_mean_l2"] == pytest.approx(0.0)
    assert metrics["observation_variance_l2"] == pytest.approx(0.0)
    assert metrics["observation_count_delta"] == pytest.approx(0.0)
    assert metrics["fixed_sigma_l2"] == pytest.approx(0.0)


def test_actor_critic_rms_and_sigma_drift_are_distinguished() -> None:
    reference = _state()
    candidate = copy.deepcopy(reference)
    candidate["a2c_network.mu.bias"] += 0.1
    candidate["a2c_network.value.bias"] += 0.2
    candidate["running_mean_std.running_mean"] += 0.01
    candidate["running_mean_std.running_var"] += 0.02
    candidate["running_mean_std.count"] += 10
    candidate["a2c_network.sigma"] += 0.03
    metrics = compute_p8b_drift_metrics(reference, candidate, torch.randn(128, 22))
    assert metrics["action_mse_vs_reference"] > 0.0
    assert metrics["action_max_abs_error_vs_reference"] > 0.0
    assert metrics["actor_parameter_relative_l2"] > 0.0
    assert metrics["critic_parameter_relative_l2"] > 0.0
    assert metrics["observation_mean_l2"] > 0.0
    assert metrics["observation_variance_l2"] > 0.0
    assert metrics["observation_count_delta"] == pytest.approx(10.0)
    assert metrics["fixed_sigma_l2"] > 0.0


def test_drift_rejects_invalid_observation_batch() -> None:
    state = _state()
    with pytest.raises(ValueError):
        compute_p8b_drift_metrics(state, state, torch.randn(8, 21))
    invalid = torch.randn(8, 22)
    invalid[0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        compute_p8b_drift_metrics(state, state, invalid)

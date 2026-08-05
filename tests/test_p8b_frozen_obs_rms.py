"""Observation RMS must remain frozen independently of value RMS."""

from __future__ import annotations

from pathlib import Path

import torch

from quadcopter_waypoint.imitation.p8b_checkpoint import (
    build_p8b_model,
    deterministic_mean_from_model_state,
    load_p8b_params,
    obs_rms_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_p8b_ppo_cfg.yaml"
)


def _model() -> torch.nn.Module:
    return build_p8b_model(load_p8b_params(CONFIG))


def test_eval_rms_does_not_update_mean_variance_or_count() -> None:
    model = _model()
    model.train()
    model.running_mean_std.eval()
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    rms_before = obs_rms_sha256(state_before)
    raw = torch.randn(128, 22) * 7.0 + 3.0
    action_before = deterministic_mean_from_model_state(state_before, raw)

    for _ in range(5):
        model({"is_train": True, "prev_actions": torch.zeros(128, 4), "obs": raw})

    state_after = model.state_dict()
    assert obs_rms_sha256(state_after) == rms_before
    action_after = deterministic_mean_from_model_state(state_after, raw)
    assert torch.equal(action_before, action_after)


def test_value_rms_can_update_while_observation_rms_is_frozen() -> None:
    model = _model()
    model.train()
    model.running_mean_std.eval()
    obs_before = obs_rms_sha256(model.state_dict())
    value_count_before = model.value_mean_std.count.clone()
    model.value_mean_std(torch.randn(64, 1))
    assert obs_rms_sha256(model.state_dict()) == obs_before
    assert model.value_mean_std.count > value_count_before


def test_train_mode_would_update_rms_and_is_therefore_rejected_by_design() -> None:
    model = _model()
    model.running_mean_std.train()
    before = obs_rms_sha256(model.state_dict())
    model.running_mean_std(torch.randn(64, 22))
    assert obs_rms_sha256(model.state_dict()) != before

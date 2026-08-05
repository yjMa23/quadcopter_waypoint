"""Warm-up epoch boundaries and actor/critic hash behavior."""

from __future__ import annotations

from pathlib import Path

import torch

from quadcopter_waypoint.imitation.p8b_agent import warmup_active, warmup_scheduler_frozen
from quadcopter_waypoint.imitation.p8b_checkpoint import (
    actor_weights_sha256,
    build_p8b_model,
    critic_weights_sha256,
    load_p8b_params,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_p8b_ppo_cfg.yaml"
)


def _model() -> torch.nn.Module:
    model = build_p8b_model(load_p8b_params(CONFIG))
    model.running_mean_std.eval()
    return model


def _actor(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    net = model.a2c_network
    return [*net.actor_mlp.parameters(), *net.mu.parameters(), net.sigma]


def _critic(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    net = model.a2c_network
    return [*net.critic_mlp.parameters(), *net.value.parameters()]


def test_warmup_epoch_boundaries() -> None:
    assert not warmup_active(0, 10)
    assert warmup_active(1, 10)
    assert warmup_active(9, 10)
    assert warmup_active(10, 10)
    assert not warmup_active(11, 10)
    assert warmup_scheduler_frozen(1, 10, True)
    assert warmup_scheduler_frozen(10, 10, True)
    assert not warmup_scheduler_frozen(11, 10, True)
    assert not warmup_scheduler_frozen(5, 10, False)


def test_critic_only_update_preserves_actor_hash_and_changes_critic() -> None:
    model = _model()
    for parameter in _actor(model):
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    actor_before = actor_weights_sha256(model.state_dict())
    critic_before = critic_weights_sha256(model.state_dict())

    for _ in range(3):
        optimizer.zero_grad()
        result = model({"is_train": True, "prev_actions": torch.zeros(64, 4), "obs": torch.randn(64, 22)})
        loss = result["values"].square().mean() + result["values"].mean()
        loss.backward()
        optimizer.step()

    assert actor_weights_sha256(model.state_dict()) == actor_before
    assert critic_weights_sha256(model.state_dict()) != critic_before
    assert all(parameter.grad is None for parameter in _actor(model))


def test_first_post_warmup_actor_update_changes_actor() -> None:
    model = _model()
    for parameter in _actor(model):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    actor_before = actor_weights_sha256(model.state_dict())
    optimizer.zero_grad()
    result = model({"is_train": True, "prev_actions": torch.zeros(64, 4), "obs": torch.randn(64, 22)})
    result["mus"].square().mean().backward()
    optimizer.step()
    assert actor_weights_sha256(model.state_dict()) != actor_before


def test_resume_epoch_determines_next_warmup_state() -> None:
    assert warmup_active(9 + 1, 10)
    assert not warmup_active(10 + 1, 10)

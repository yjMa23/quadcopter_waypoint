"""Verify that the actor-preserving PPO RL-Games network has isolated actor and critic gradient paths."""

from __future__ import annotations

from pathlib import Path

import torch

from quadcopter_waypoint.imitation.actor_preserving_checkpoint import build_actor_preserving_model, load_actor_preserving_params


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_actor_preserving_ppo_cfg.yaml"
)


def _model() -> torch.nn.Module:
    model = build_actor_preserving_model(load_actor_preserving_params(CONFIG))
    model.running_mean_std.eval()
    return model


def _actor_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    network = model.a2c_network
    return [*network.actor_mlp.parameters(), *network.mu.parameters()]


def _critic_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    network = model.a2c_network
    return [*network.critic_mlp.parameters(), *network.value.parameters()]


def test_actor_and_critic_have_no_shared_storage() -> None:
    model = _model()
    actor = {parameter.untyped_storage().data_ptr() for parameter in _actor_parameters(model)}
    critic = {parameter.untyped_storage().data_ptr() for parameter in _critic_parameters(model)}
    assert actor.isdisjoint(critic)


def test_value_backward_does_not_create_actor_gradient() -> None:
    model = _model()
    result = model({"is_train": True, "prev_actions": torch.zeros(32, 4), "obs": torch.randn(32, 22)})
    result["values"].square().mean().backward()
    assert all(parameter.grad is None for parameter in _actor_parameters(model))
    assert any(parameter.grad is not None for parameter in _critic_parameters(model))


def test_actor_backward_does_not_create_critic_gradient() -> None:
    model = _model()
    result = model({"is_train": True, "prev_actions": torch.zeros(32, 4), "obs": torch.randn(32, 22)})
    result["mus"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in _actor_parameters(model))
    assert all(parameter.grad is None for parameter in _critic_parameters(model))

"""Gradient and numerical tests for the on-policy BC mean-action anchor."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quadcopter_waypoint.imitation.actor_preserving_agent import bc_anchor_loss
from quadcopter_waypoint.imitation.actor_preserving_checkpoint import (
    build_actor_preserving_model,
    load_actor_preserving_params,
    load_reference_actor,
    reference_state_from_model,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_actor_preserving_ppo_cfg.yaml"
)


def _model() -> torch.nn.Module:
    model = build_actor_preserving_model(load_actor_preserving_params(CONFIG))
    model.running_mean_std.eval()
    return model


def test_identical_actor_has_zero_anchor_and_perturbation_increases_it() -> None:
    model = _model()
    reference = load_reference_actor(reference_state_from_model(model.state_dict()))
    normalized = torch.randn(64, 22)
    current = model.a2c_network.mu(model.a2c_network.actor_mlp(normalized))
    target = reference(normalized)
    baseline = bc_anchor_loss(current, target)
    assert baseline.item() == pytest.approx(0.0, abs=1.0e-12)
    with torch.no_grad():
        model.a2c_network.mu.bias.add_(0.1)
    perturbed = model.a2c_network.mu(model.a2c_network.actor_mlp(normalized))
    assert bc_anchor_loss(perturbed, target) > baseline


def test_anchor_gradient_only_reaches_current_actor() -> None:
    model = _model()
    reference = load_reference_actor(reference_state_from_model(model.state_dict()))
    with torch.no_grad():
        model.a2c_network.mu.bias.add_(0.05)
    normalized = torch.randn(64, 22)
    current = model.a2c_network.mu(model.a2c_network.actor_mlp(normalized))
    loss = bc_anchor_loss(current, reference(normalized))
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.a2c_network.actor_mlp.parameters())
    assert any(parameter.grad is not None for parameter in model.a2c_network.mu.parameters())
    assert all(parameter.grad is None for parameter in model.a2c_network.critic_mlp.parameters())
    assert all(parameter.grad is None for parameter in model.a2c_network.value.parameters())
    assert model.a2c_network.sigma.grad is None
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_zero_anchor_coefficient_is_exact_degenerate_behavior() -> None:
    model = _model()
    reference = load_reference_actor(reference_state_from_model(model.state_dict()))
    normalized = torch.randn(32, 22)
    current = model.a2c_network.mu(model.a2c_network.actor_mlp(normalized))
    base = current.square().mean()
    with_zero_anchor = base + 0.0 * bc_anchor_loss(current, reference(normalized))
    base_grad = torch.autograd.grad(base, tuple(model.a2c_network.mu.parameters()), retain_graph=True)
    zero_grad = torch.autograd.grad(with_zero_anchor, tuple(model.a2c_network.mu.parameters()))
    for expected, actual in zip(base_grad, zero_grad, strict=True):
        assert torch.equal(expected, actual)


def test_anchor_rejects_shape_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        bc_anchor_loss(torch.zeros(8, 4), torch.zeros(8, 3))
    invalid = torch.zeros(8, 4)
    invalid[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        bc_anchor_loss(invalid, torch.zeros(8, 4))

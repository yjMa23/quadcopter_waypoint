from __future__ import annotations

from pathlib import Path

import torch

from quadcopter_waypoint.imitation.checkpoint import (
    bc_rlgames_parity_error,
    build_bc_initialized_rlgames_checkpoint,
)
from quadcopter_waypoint.imitation.policy import BCActor, normalize_observations


def _save_bc(path: Path) -> BCActor:
    torch.manual_seed(3)
    actor = BCActor(torch.arange(22, dtype=torch.float64), torch.full((22,), 4.0), torch.tensor(123.0))
    payload = {
        "model_state_dict": actor.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 2,
        "network_config": actor.export_config(),
        "observation_normalization": {
            "running_mean": actor.running_mean,
            "running_var": actor.running_var,
            "count": actor.running_count,
        },
        "dataset_manifest_sha256": "a" * 64,
    }
    torch.save(payload, path)
    return actor


def _save_template(path: Path) -> dict:
    model = {
        "value_mean_std.running_mean": torch.tensor([9.0], dtype=torch.float64),
        "value_mean_std.running_var": torch.tensor([7.0], dtype=torch.float64),
        "value_mean_std.count": torch.tensor(100.0, dtype=torch.float64),
        "running_mean_std.running_mean": torch.zeros(22, dtype=torch.float64),
        "running_mean_std.running_var": torch.ones(22, dtype=torch.float64),
        "running_mean_std.count": torch.tensor(1.0, dtype=torch.float64),
        "a2c_network.sigma": torch.zeros(4),
        "a2c_network.actor_mlp.0.weight": torch.zeros(64, 22),
        "a2c_network.actor_mlp.0.bias": torch.zeros(64),
        "a2c_network.actor_mlp.2.weight": torch.zeros(64, 64),
        "a2c_network.actor_mlp.2.bias": torch.zeros(64),
        "a2c_network.value.weight": torch.full((1, 64), 9.0),
        "a2c_network.value.bias": torch.full((1,), 9.0),
        "a2c_network.mu.weight": torch.zeros(4, 64),
        "a2c_network.mu.bias": torch.zeros(4),
    }
    payload = {
        "model": model,
        "optimizer": {"state": {1: {"step": torch.tensor(5)}}, "param_groups": [{"params": list(range(9))}]},
        "epoch": 990,
        "frame": 12345,
        "last_mean_rewards": torch.tensor(99.0),
        "env_state": {"should": "clear"},
    }
    torch.save(payload, path)
    return payload


def test_normalization_matches_rlgames_formula_and_clamp():
    observations = torch.tensor([[100.0] * 22, [1.0] * 22])
    mean = torch.ones(22, dtype=torch.float64)
    var = torch.full((22,), 4.0, dtype=torch.float64)
    normalized = normalize_observations(observations, mean, var)
    assert torch.all(normalized[0] == 5.0)
    assert torch.allclose(normalized[1], torch.zeros(22))


def test_bc_actor_shape_bounds_and_save_load(tmp_path: Path):
    path = tmp_path / "bc.pth"
    actor = _save_bc(path)
    observations = torch.randn(17, 22) * 20.0
    actions = actor(observations)
    assert actions.shape == (17, 4)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    restored, _ = BCActor.from_checkpoint(path)
    assert torch.equal(actor.running_mean, restored.running_mean)
    assert torch.allclose(actor(observations), restored(observations))


def test_bc_to_rlgames_migration_resets_training_state_and_has_parity(tmp_path: Path):
    bc_path = tmp_path / "bc.pth"
    template_path = tmp_path / "teacher.pth"
    output_path = tmp_path / "bc_init.pth"
    actor = _save_bc(bc_path)
    template = _save_template(template_path)
    metadata = build_bc_initialized_rlgames_checkpoint(
        bc_path,
        template_path,
        output_path,
        dataset_manifest_sha256="a" * 64,
        value_seed=11,
    )
    assert metadata["output"] == str(output_path)
    migrated = torch.load(output_path, map_location="cpu", weights_only=False)
    assert migrated["epoch"] == 0
    assert migrated["frame"] == 0
    assert migrated["env_state"] is None
    assert migrated["optimizer"]["state"] == {}
    assert torch.equal(migrated["model"]["a2c_network.sigma"], template["model"]["a2c_network.sigma"])
    assert not torch.equal(
        migrated["model"]["a2c_network.value.weight"], template["model"]["a2c_network.value.weight"]
    )
    assert torch.equal(migrated["model"]["running_mean_std.running_mean"], actor.running_mean)
    observations = torch.randn(128, 22)
    assert bc_rlgames_parity_error(bc_path, output_path, observations) < 1.0e-7

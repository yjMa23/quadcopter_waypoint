"""Pure-Python tests for P8B shared-to-separate checkpoint migration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quadcopter_waypoint.imitation.checkpoint import bc_rlgames_parity_error
from quadcopter_waypoint.imitation.p8b_checkpoint import (
    ACTOR_KEYS,
    CRITIC_KEYS,
    OBS_RMS_KEYS,
    SCHEMA_VERSION,
    build_p8b_separate_checkpoint,
    checkpoint_parity_error,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/imitation/p7_bc/bc_init_rlgames.pth"
BC = ROOT / "logs/imitation/p7_bc/best_bc.pth"
MANIFEST = ROOT / "logs/imitation/p7_expert_dataset/manifest.json"
CONFIG = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_p8b_ppo_cfg.yaml"
)


def _migrate(tmp_path: Path) -> Path:
    output = tmp_path / "p8b_init.pth"
    result = build_p8b_separate_checkpoint(SOURCE, CONFIG, output, MANIFEST)
    assert result["schema_version"] == SCHEMA_VERSION
    assert len(result["sha256"]) == 64
    return output


def test_shared_to_separate_migration_preserves_actor_rms_and_parity(tmp_path: Path) -> None:
    output = _migrate(tmp_path)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    target = torch.load(output, map_location="cpu", weights_only=False)
    source_model, target_model = source["model"], target["model"]

    for key in (*ACTOR_KEYS, *OBS_RMS_KEYS, "a2c_network.sigma"):
        assert torch.equal(source_model[key], target_model[key]), key
    for key in CRITIC_KEYS:
        assert key in target_model
    assert "a2c_network.critic_mlp.0.weight" not in source_model
    assert target["epoch"] == 0 and target["frame"] == 0
    assert target["env_state"] is None
    assert target["optimizer"]["state"] == {}
    assert len(target["optimizer"]["param_groups"]) == 1
    assert len(target["optimizer"]["param_groups"][0]["params"]) == 13

    raw = torch.randn(257, 22)
    assert checkpoint_parity_error(SOURCE, output, raw) <= 1.0e-5
    assert bc_rlgames_parity_error(BC, output, raw) <= 1.0e-5

    metadata = target["p8b_actor_preserving"]
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["dataset_manifest_sha256"]
    assert metadata["source_checkpoint_sha256"]
    assert metadata["reference_actor_state"]
    assert metadata["freeze_lr_scheduler_during_warmup"] is True
    assert metadata["base_learning_rate"] == pytest.approx(1.0e-4)
    assert metadata["freeze_observation_rms"] is True


def test_migration_is_deterministic_for_critic_seed(tmp_path: Path) -> None:
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    build_p8b_separate_checkpoint(SOURCE, CONFIG, first, MANIFEST, critic_seed=2026)
    build_p8b_separate_checkpoint(SOURCE, CONFIG, second, MANIFEST, critic_seed=2026)
    a = torch.load(first, map_location="cpu", weights_only=False)
    b = torch.load(second, map_location="cpu", weights_only=False)
    for key in CRITIC_KEYS:
        assert torch.equal(a["model"][key], b["model"][key])


def test_migration_refuses_overwrite_and_invalid_source(tmp_path: Path) -> None:
    output = _migrate(tmp_path)
    with pytest.raises(FileExistsError):
        build_p8b_separate_checkpoint(SOURCE, CONFIG, output, MANIFEST)
    invalid = tmp_path / "invalid.pth"
    torch.save({"model": {}}, invalid)
    with pytest.raises(ValueError):
        build_p8b_separate_checkpoint(invalid, CONFIG, tmp_path / "bad.pth", MANIFEST)

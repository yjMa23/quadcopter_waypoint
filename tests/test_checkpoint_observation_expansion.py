import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from quadcopter_waypoint.utils.checkpoint_observation import (
    OBS_COUNT_KEY,
    OBS_MEAN_KEY,
    OBS_VAR_KEY,
    POLICY_FIRST_LAYER_KEY,
    expand_checkpoint_observation_state,
)


def _checkpoint(input_dim: int = 16):
    weight = torch.arange(64 * input_dim, dtype=torch.float32).reshape(64, input_dim)
    return {
        "model": {
            POLICY_FIRST_LAYER_KEY: weight.clone(),
            "a2c_network.actor_mlp.0.bias": torch.arange(64, dtype=torch.float32),
            OBS_MEAN_KEY: torch.arange(input_dim, dtype=torch.float64),
            OBS_VAR_KEY: torch.arange(input_dim, dtype=torch.float64) + 2.0,
            OBS_COUNT_KEY: torch.tensor(123.0, dtype=torch.float64),
            "value_mean_std.running_mean": torch.tensor([7.0], dtype=torch.float64),
        },
        "optimizer": {
            "state": {
                1: {
                    "step": torch.tensor(9.0),
                    "exp_avg": torch.full((64, input_dim), 3.0),
                    "exp_avg_sq": torch.full((64, input_dim), 4.0),
                }
            },
            "param_groups": [{"params": [1]}],
        },
        "epoch": 10,
    }


def test_policy_and_normalization_are_expanded_without_changing_existing_values():
    source = _checkpoint()
    expanded, changed = expand_checkpoint_observation_state(source, 16, 22)
    weight = expanded["model"][POLICY_FIRST_LAYER_KEY]
    assert weight.shape == (64, 22)
    torch.testing.assert_close(weight[:, :16], source["model"][POLICY_FIRST_LAYER_KEY])
    torch.testing.assert_close(weight[:, 16:], torch.zeros(64, 6))
    torch.testing.assert_close(expanded["model"][OBS_MEAN_KEY][:16], source["model"][OBS_MEAN_KEY])
    torch.testing.assert_close(expanded["model"][OBS_MEAN_KEY][16:], torch.zeros(6, dtype=torch.float64))
    torch.testing.assert_close(expanded["model"][OBS_VAR_KEY][:16], source["model"][OBS_VAR_KEY])
    torch.testing.assert_close(expanded["model"][OBS_VAR_KEY][16:], torch.ones(6, dtype=torch.float64))
    torch.testing.assert_close(expanded["model"][OBS_COUNT_KEY], source["model"][OBS_COUNT_KEY])
    torch.testing.assert_close(
        expanded["model"]["value_mean_std.running_mean"], source["model"]["value_mean_std.running_mean"]
    )
    assert len(changed) >= 6


def test_optimizer_moments_are_expanded_for_resumable_training():
    expanded, _ = expand_checkpoint_observation_state(_checkpoint(), 16, 22)
    state = expanded["optimizer"]["state"][1]
    for key, old_value in (("exp_avg", 3.0), ("exp_avg_sq", 4.0)):
        assert state[key].shape == (64, 22)
        torch.testing.assert_close(state[key][:, :16], torch.full((64, 16), old_value))
        torch.testing.assert_close(state[key][:, 16:], torch.zeros(64, 6))


def test_wrong_input_dimension_fails_clearly():
    with pytest.raises(ValueError, match="Expected .* shape"):
        expand_checkpoint_observation_state(_checkpoint(input_dim=15), 16, 22)


def test_cli_preserves_source_and_refuses_existing_output(tmp_path: Path):
    source = tmp_path / "source.pth"
    output = tmp_path / "expanded.pth"
    torch.save(_checkpoint(), source)
    source_bytes = source.read_bytes()
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "source" / "quadcopter_waypoint")
    command = [
        sys.executable,
        str(repo_root / "scripts" / "rl_games" / "expand_checkpoint_observation.py"),
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, cwd=repo_root, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert source.read_bytes() == source_bytes
    assert output.is_file()
    manifest = json.loads(output.with_suffix(".pth.json").read_text())
    assert manifest["old_observation_dim"] == 16
    assert manifest["new_observation_dim"] == 22
    assert len(manifest["source_sha256"]) == 64
    assert len(manifest["output_sha256"]) == 64

    second = subprocess.run(command, cwd=repo_root, env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert "Refusing to overwrite existing output" in second.stderr

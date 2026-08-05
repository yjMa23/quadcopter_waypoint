"""Keep the P8B theory-first preregistration synchronized with frozen executable contracts."""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/p8b_actor_preserving_ppo.md"
BASE_ENV = ROOT / "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/quadrotor_ship_landing_env.py"
ATTITUDE_ENV = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/quadrotor_ship_landing_physical_deck_attitude_env.py"
)
TASK_INIT = ATTITUDE_ENV.parent / "__init__.py"
BASE_PPO_CONFIG = ATTITUDE_ENV.parent / "agents/rl_games_ppo_cfg.yaml"
PPO_CONFIG = ATTITUDE_ENV.parent / "agents/rl_games_p8b_ppo_cfg.yaml"
P8A_VALIDATION = ROOT / "benchmarks/phase8a_checkpoint_selection/validation_results.csv"
P8A_FORMAL = ROOT / "benchmarks/phase8a_checkpoint_selection/formal_results.csv"


def _class_literals(path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, Any] = {}
            for item in node.body:
                if not isinstance(item, (ast.Assign, ast.AnnAssign)):
                    continue
                if isinstance(item, ast.Assign):
                    if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                        continue
                    name, value = item.targets[0].id, item.value
                else:
                    if not isinstance(item.target, ast.Name) or item.value is None:
                        continue
                    name, value = item.target.id, item.value
                try:
                    values[name] = ast.literal_eval(value)
                except (TypeError, ValueError):
                    continue
            return values
    raise AssertionError(f"missing class {class_name}: {path}")


def _preregistered() -> dict[str, Any]:
    text = DOCUMENT.read_text(encoding="utf-8")
    matches = re.findall(
        r"```yaml\s*\n(p8b_preregistered_config:.*?\n)```",
        text,
        flags=re.DOTALL,
    )
    assert len(matches) == 1, "P8B document must contain exactly one machine-readable parameter block"
    payload = yaml.safe_load(matches[0])
    assert set(payload) == {"p8b_preregistered_config"}
    return payload["p8b_preregistered_config"]


def test_p8b_preregistered_parameters_match_frozen_task_and_ppo() -> None:
    prereg = _preregistered()
    base_values = _class_literals(BASE_ENV, "QuadcopterShipLandingEnvCfg")
    attitude_values = _class_literals(ATTITUDE_ENV, "QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg")
    ppo = yaml.safe_load(PPO_CONFIG.read_text(encoding="utf-8"))["params"]
    base_ppo = yaml.safe_load(BASE_PPO_CONFIG.read_text(encoding="utf-8"))["params"]
    network = ppo["network"]
    config = ppo["config"]

    assert prereg["task_id"] in TASK_INIT.read_text(encoding="utf-8")
    assert prereg["observation_dim"] == attitude_values["observation_space"] == 22
    assert prereg["action_dim"] == base_values["action_space"] == 4
    assert prereg["network"]["separate"] is True
    assert prereg["network"]["units"] == network["mlp"]["units"]
    assert prereg["network"]["activation"] == network["mlp"]["activation"]
    assert prereg["network"]["fixed_sigma"] == network["space"]["continuous"]["fixed_sigma"]
    assert network["separate"] is True
    assert base_ppo["network"]["separate"] is False
    assert ppo["algo"]["name"] == "p8b_actor_preserving"

    assert prereg["warmup_epochs"] == prereg["warmup_active_epoch_max"] == 10
    assert prereg["freeze_lr_scheduler_during_warmup"] is True
    assert prereg["freeze_observation_rms"] is True
    assert prereg["bc_anchor"]["type"] == "mse_mean_action"
    assert prereg["bc_anchor"]["coefficient"] == pytest.approx(10.0)
    assert prereg["bc_anchor"]["pilot_candidates"] == [0.0, 10.0, 50.0]
    assert prereg["bc_anchor"]["reduction"] == "mean_all_elements"
    assert prereg["bc_anchor"]["action_representation"] == "deterministic_pre_clamp_mean"
    p8b_config = config["p8b"]
    assert p8b_config["schema_version"] == prereg["migration"]["schema_version"]
    assert p8b_config["warmup_epochs"] == prereg["warmup_epochs"]
    assert (
        p8b_config["freeze_lr_scheduler_during_warmup"]
        == prereg["freeze_lr_scheduler_during_warmup"]
    )
    assert p8b_config["freeze_observation_rms"] == prereg["freeze_observation_rms"]
    assert p8b_config["bc_anchor_type"] == prereg["bc_anchor"]["type"]
    assert p8b_config["bc_anchor_coefficient"] == pytest.approx(prereg["bc_anchor"]["coefficient"])

    ppo_doc = prereg["ppo"]
    expected = {
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "gae_lambda": config["tau"],
        "clip_epsilon": config["e_clip"],
        "critic_coef": config["critic_coef"],
        "entropy_coef": config["entropy_coef"],
        "bounds_loss_coef": config["bounds_loss_coef"],
        "reward_scale": config["reward_shaper"]["scale_value"],
        "normalize_input": config["normalize_input"],
        "normalize_value": config["normalize_value"],
        "normalize_advantage": config["normalize_advantage"],
        "horizon_length": config["horizon_length"],
        "minibatch_size": config["minibatch_size"],
        "mini_epochs": config["mini_epochs"],
        "max_epochs": config["max_epochs"],
    }
    for key, value in expected.items():
        if key in {"learning_rate", "gamma", "gae_lambda", "clip_epsilon", "critic_coef", "entropy_coef", "bounds_loss_coef", "reward_scale"}:
            assert float(ppo_doc[key]) == pytest.approx(float(value)), f"documentation drift: ppo.{key}"
        else:
            assert ppo_doc[key] == value, f"documentation drift: ppo.{key}"
    assert prereg["checkpoint"]["frequency_epochs"] == config["save_frequency"]


def test_p8b_seed_roles_and_evaluation_match_frozen_p8a_protocol() -> None:
    prereg = _preregistered()
    with P8A_VALIDATION.open(newline="", encoding="utf-8") as stream:
        validation_rows = list(csv.DictReader(stream))
    with P8A_FORMAL.open(newline="", encoding="utf-8") as stream:
        formal_rows = list(csv.DictReader(stream))

    assert prereg["seeds"]["training"] == [42, 43, 44]
    assert prereg["seeds"]["pilot_validation"] == [145, 146, 147]
    assert prereg["seeds"]["formal_validation"] == [145, 146, 147]
    assert prereg["seeds"]["formal_test"] == [245, 246, 247]
    assert set(prereg["seeds"]["training"]).isdisjoint(prereg["seeds"]["formal_test"])
    assert set(prereg["seeds"]["formal_validation"]).isdisjoint(prereg["seeds"]["formal_test"])

    assert sorted({int(row["eval_seed"]) for row in validation_rows}) == prereg["seeds"]["formal_validation"]
    assert sorted({int(row["eval_seed"]) for row in formal_rows}) == prereg["seeds"]["formal_test"]
    assert {int(row["episodes"]) for row in validation_rows} == {prereg["evaluation"]["validation_episodes_per_seed"]}
    assert {int(row["episodes"]) for row in formal_rows} == {prereg["evaluation"]["episodes_per_seed"]}
    assert prereg["evaluation"]["target_settled_landing"] == pytest.approx(0.92)
    assert prereg["evaluation"]["target_ground_crash_max"] == pytest.approx(0.01)
    assert prereg["evaluation"]["target_hard_contact_max"] == pytest.approx(0.02)
    assert prereg["evaluation"]["target_timeout_max"] == pytest.approx(0.03)


def test_p8b_theory_traceability_targets_exist() -> None:
    prereg = _preregistered()
    assert prereg["migration"]["schema_version"] == "p8b-separate-v1"
    assert prereg["migration"]["critic_seed"] == 2026
    assert prereg["migration"]["parity_max_abs_error"] == pytest.approx(1.0e-5)
    for path in (
        ROOT / "docs/p6a_heave_precision_theory.md",
        ROOT / "docs/p6b_physical_deck_theory.md",
        ROOT / "docs/p6c_physical_deck_attitude_theory.md",
        ROOT / "docs/p7_imitation_hybrid_paper.md",
        ROOT / "docs/p8a_checkpoint_selection_and_policy_drift.md",
        DOCUMENT,
        ROOT / "source/quadcopter_waypoint/quadcopter_waypoint/imitation/checkpoint.py",
        ROOT / "source/quadcopter_waypoint/quadcopter_waypoint/imitation/checkpoint_sweep.py",
        P8A_VALIDATION,
        P8A_FORMAL,
        BASE_PPO_CONFIG,
        PPO_CONFIG,
    ):
        assert path.exists(), f"traceability target missing: {path.relative_to(ROOT)}"

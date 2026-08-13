"""Keep the imitation-learning benchmark paper-style theory document synchronized with executable contracts."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/imitation_hybrid_paper.md"
DATASET_SOURCE = ROOT / "source/quadcopter_waypoint/quadcopter_waypoint/imitation/dataset.py"
POLICY_SOURCE = ROOT / "source/quadcopter_waypoint/quadcopter_waypoint/imitation/policy.py"
BASE_ENV_SOURCE = (
    ROOT
    / "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/quadrotor_ship_landing_env.py"
)
PHYSICAL_ENV_SOURCE = (
    ROOT
    / "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck/"
    "quadrotor_ship_landing_physical_deck_env.py"
)
ATTITUDE_ENV_SOURCE = (
    ROOT
    / "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck_attitude/"
    "quadrotor_ship_landing_physical_deck_attitude_env.py"
)
PPO_CONFIG = (
    ROOT
    / "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck_attitude/"
    "agents/rl_games_ppo_cfg.yaml"
)


def _literal(node: ast.AST, names: dict[str, Any] | None = None) -> Any:
    if isinstance(node, ast.Name) and names is not None and node.id in names:
        return names[node.id]
    return ast.literal_eval(node)


def _module_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = _literal(node.value, values)
            except (ValueError, TypeError):
                continue
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = _literal(node.value, values)
            except (ValueError, TypeError):
                continue
    return values


def _class_assignments(path: Path, class_name: str, names: dict[str, Any] | None = None) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, Any] = {}
            for item in node.body:
                target: ast.Name | None = None
                value: ast.AST | None = None
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    target, value = item.targets[0], item.value
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    target, value = item.target, item.value
                if target is None or value is None:
                    continue
                try:
                    values[target.id] = _literal(value, {**(names or {}), **values})
                except (ValueError, TypeError):
                    continue
            return values
    raise AssertionError(f"class not found: {class_name} in {path}")


def _function_default(path: Path, function_name: str, argument_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            positional = node.args.posonlyargs + node.args.args
            defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
            for argument, default in zip(positional, defaults, strict=True):
                if argument.arg == argument_name and default is not None:
                    return ast.literal_eval(default)
    raise AssertionError(f"default not found: {function_name}({argument_name})")


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^#\s]+)", text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"YAML key not found: {key}")
    return match.group(1)


def _sync_values() -> dict[str, str]:
    text = PAPER.read_text(encoding="utf-8")
    match = re.search(r"<!-- CODE_SYNC\n(?P<body>.*?)\n-->", text, flags=re.DOTALL)
    assert match, "imitation-learning benchmark theory document is missing the CODE_SYNC block"
    values: dict[str, str] = {}
    for line in match.group("body").splitlines():
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def _assert_number(sync: dict[str, str], key: str, actual: float) -> None:
    assert key in sync, f"CODE_SYNC missing {key}"
    assert float(sync[key]) == pytest.approx(float(actual)), f"documentation drift for {key}"


def test_imitation_paper_code_sync_contract() -> None:
    dataset_values = _module_assignments(DATASET_SOURCE)
    policy_values = _class_assignments(POLICY_SOURCE, "BCNetworkConfig", dataset_values)
    base_values = _class_assignments(BASE_ENV_SOURCE, "QuadcopterShipLandingEnvCfg")
    physical_values = _class_assignments(PHYSICAL_ENV_SOURCE, "QuadcopterShipLandingPhysicalDeckEnvCfg")
    attitude_values = _class_assignments(
        ATTITUDE_ENV_SOURCE, "QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg"
    )
    phase_weight_cap = _function_default(DATASET_SOURCE, "phase_sample_weights", "maximum_weight")
    yaml_text = PPO_CONFIG.read_text(encoding="utf-8")
    sync = _sync_values()

    _assert_number(sync, "observation_dim", dataset_values["OBSERVATION_DIM"])
    _assert_number(sync, "action_dim", dataset_values["ACTION_DIM"])
    assert sync["hidden_units"] == ",".join(str(value) for value in policy_values["hidden_units"])
    assert sync["activation"] == policy_values["activation"]
    _assert_number(sync, "observation_epsilon", policy_values["observation_epsilon"])
    _assert_number(sync, "observation_clip", policy_values["observation_clip"])
    _assert_number(sync, "action_clip", policy_values["action_clip"])

    _assert_number(sync, "thrust_to_weight", base_values["thrust_to_weight"])
    _assert_number(sync, "moment_scale", base_values["moment_scale"])
    _assert_number(sync, "deck_roll_max_deg", attitude_values["deck_roll_amplitude_max_deg"])
    _assert_number(sync, "deck_pitch_max_deg", attitude_values["deck_pitch_amplitude_max_deg"])
    _assert_number(sync, "deck_roll_frequency_min", attitude_values["deck_roll_frequency_min"])
    _assert_number(sync, "deck_roll_frequency_max", attitude_values["deck_roll_frequency_max"])
    _assert_number(sync, "deck_pitch_frequency_min", attitude_values["deck_pitch_frequency_min"])
    _assert_number(sync, "deck_pitch_frequency_max", attitude_values["deck_pitch_frequency_max"])
    _assert_number(sync, "landing_success_radius", physical_values["landing_success_radius"])
    _assert_number(sync, "settle_hold_steps", physical_values["settle_hold_steps"])
    _assert_number(sync, "phase_weight_cap", phase_weight_cap)

    _assert_number(sync, "ppo_gamma", float(_yaml_scalar(yaml_text, "gamma")))
    _assert_number(sync, "ppo_tau", float(_yaml_scalar(yaml_text, "tau")))
    _assert_number(sync, "ppo_learning_rate", float(_yaml_scalar(yaml_text, "learning_rate")))
    _assert_number(sync, "ppo_clip", float(_yaml_scalar(yaml_text, "e_clip")))
    _assert_number(sync, "horizon_length", float(_yaml_scalar(yaml_text, "horizon_length")))
    _assert_number(sync, "minibatch_size", float(_yaml_scalar(yaml_text, "minibatch_size")))
    _assert_number(sync, "mini_epochs", float(_yaml_scalar(yaml_text, "mini_epochs")))
    _assert_number(sync, "critic_coef", float(_yaml_scalar(yaml_text, "critic_coef")))
    assert sync["fixed_sigma"].lower() == _yaml_scalar(yaml_text, "fixed_sigma").lower()


def test_imitation_paper_traceability_paths_exist() -> None:
    for path in (
        DATASET_SOURCE,
        POLICY_SOURCE,
        BASE_ENV_SOURCE,
        PHYSICAL_ENV_SOURCE,
        ATTITUDE_ENV_SOURCE,
        PPO_CONFIG,
        ROOT / "scripts/imitation/train_bc.py",
        ROOT / "source/quadcopter_waypoint/quadcopter_waypoint/imitation/checkpoint.py",
        ROOT / "benchmarks/imitation_hybrid/summary.json",
        ROOT / "benchmarks/imitation_hybrid/commands.txt",
    ):
        assert path.exists(), f"traceability target missing: {path.relative_to(ROOT)}"

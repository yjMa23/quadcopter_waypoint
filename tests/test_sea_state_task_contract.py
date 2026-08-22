import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEA_ENV = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_sea_state/quadrotor_ship_landing_sea_state_env.py"
)
SEA_INIT = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_sea_state/__init__.py"
)
BASE_ENV = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/quadrotor_ship_landing_physical_deck_attitude_env.py"
)
SHIP_ENV = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/quadrotor_ship_landing_env.py"
)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assigned_literal(class_node: ast.ClassDef, name: str):
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing class assignment {class_node.name}.{name}")


def test_sea_state_inherits_frozen_physical_deck_attitude_contract():
    tree = ast.parse(SEA_ENV.read_text())
    cfg = _class(tree, "QuadcopterShipLandingSeaStateEnvCfg")
    env = _class(tree, "QuadcopterShipLandingSeaStateEnv")

    assert [ast.unparse(base) for base in cfg.bases] == ["QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg"]
    assert [ast.unparse(base) for base in env.bases] == ["QuadcopterShipLandingPhysicalDeckAttitudeEnv"]
    assert _assigned_literal(cfg, "observation_space") == 22

    # SeaState is intentionally a motion-only specialization. These task-contract methods remain inherited.
    overridden = {node.name for node in env.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "_get_observations" not in overridden
    assert "_get_rewards" not in overridden
    assert "_get_dones" not in overridden
    assert "_apply_action" not in overridden


def test_frozen_task_still_declares_22d_observation_and_4d_action():
    base_tree = ast.parse(BASE_ENV.read_text())
    base_cfg = _class(base_tree, "QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg")
    assert _assigned_literal(base_cfg, "observation_space") == 22

    ship_tree = ast.parse(SHIP_ENV.read_text())
    ship_cfg = _class(ship_tree, "QuadcopterShipLandingEnvCfg")
    assert _assigned_literal(ship_cfg, "action_space") == 4


def test_sea_state_registration_is_independent():
    source = SEA_INIT.read_text()
    assert 'id="Isaac-Quadcopter-ShipLanding-SeaState-Direct-v0"' in source
    assert "PhysicalDeckAttitude-Direct-v0" not in source

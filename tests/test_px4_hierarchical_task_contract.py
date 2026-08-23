from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NEW_ENV = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_px4_hierarchical/quadrotor_ship_landing_px4_hierarchical_env.py"
)
NEW_INIT = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_px4_hierarchical/__init__.py"
)
NEW_PPO = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_px4_hierarchical/agents/rl_games_ppo_cfg.yaml"
)
FROZEN_DIRECT = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing/quadrotor_ship_landing_env.py"
)
FROZEN_ATTITUDE = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_physical_deck_attitude/quadrotor_ship_landing_physical_deck_attitude_env.py"
)


def test_new_task_declares_independent_3d_action_and_25hz_reference_rate():
    source = NEW_ENV.read_text()
    assert "class QuadcopterShipLandingPx4HierarchicalEnvCfg" in source
    assert "decimation = 4" in source
    assert "action_space = 3" in source
    assert "observation_space = 22" in source
    assert "class QuadcopterShipLandingPx4HierarchicalEnv(" in source
    assert "QuadcopterShipLandingPhysicalDeckAttitudeEnv" in source


def test_new_task_registration_does_not_alias_frozen_direct_id():
    source = NEW_INIT.read_text()
    assert 'id="Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0"' in source
    assert "PhysicalDeckAttitude-Direct-v0" not in source


def test_frozen_direct_action_semantics_remain_4d_thrust_and_moment():
    source = FROZEN_DIRECT.read_text()
    assert "decimation = 2" in source
    assert "action_space = 4" in source
    assert "self.cfg.thrust_to_weight * self._robot_weight" in source
    assert "self.cfg.moment_scale * self._actions[:, 1:]" in source


def test_new_task_does_not_override_reward_or_success_logic():
    source = NEW_ENV.read_text()
    assert "def _get_rewards" not in source
    assert "def _compute_landing_terms" not in source
    assert "def _get_dones" not in source
    frozen = FROZEN_ATTITUDE.read_text()
    for contract in (
        "safe_contact_body_deck_angle = math.radians(12.0)",
        "hard_contact_impulse_threshold = 0.025",
        "success_max_penetration = 0.025",
        "max_physical_penetration = 0.030",
    ):
        assert contract in frozen


def test_new_ppo_config_starts_from_scratch_and_keeps_small_mlp():
    config = yaml.safe_load(NEW_PPO.read_text())["params"]
    assert config["load_checkpoint"] is False
    assert config["load_path"] == ""
    assert config["network"]["mlp"]["units"] == [64, 64]
    assert config["config"]["name"] == "quadcopter_ship_landing_px4_hierarchical"

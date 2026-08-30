from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_px4_continuous_stage"
)
ENV = TASK_ROOT / "quadrotor_ship_landing_px4_continuous_stage_env.py"
INIT = TASK_ROOT / "__init__.py"
PPO = TASK_ROOT / "agents/rl_games_ppo_cfg.yaml"
OLD_M2 = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing_px4_hierarchical/quadrotor_ship_landing_px4_hierarchical_env.py"
)
OLD_M2_INIT = OLD_M2.parent / "__init__.py"
FROZEN_DIRECT = ROOT / (
    "source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/"
    "quadrotor_ship_landing/quadrotor_ship_landing_env.py"
)


def _source() -> str:
    assert ENV.exists()
    return ENV.read_text()


def test_independent_task_identity_and_spaces_are_frozen():
    source = _source()
    assert "class QuadcopterShipLandingPx4ContinuousStageEnvCfg" in source
    assert "class QuadcopterShipLandingPx4ContinuousStageEnv(" in source
    assert "QuadcopterShipLandingPx4HierarchicalEnv" in source
    assert "decimation = 4" in source
    assert "action_space = 4" in source
    assert "observation_space = 22" in source

    registration = INIT.read_text()
    assert 'id="Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"' in registration
    assert "Px4Hierarchical-Direct-v0" not in registration


def test_old_m2_contract_remains_22d_to_3d_with_same_id():
    source = OLD_M2.read_text()
    registration = OLD_M2_INIT.read_text()
    assert "action_space = 3" in source
    assert "observation_space = 22" in source
    assert 'id="Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0"' in registration


def test_action_stage_state_and_index15_semantics_are_explicit():
    source = _source()
    for required in (
        "actions[..., :3]",
        "actions[..., 3]",
        "normalized_stage_action",
        "filter_landing_stage",
        "map_stage_conditioned_relative_velocity",
        "limit_stage_conditioned_reference_slew",
        "self._landing_stage",
        "self._previous_relative_velocity_ref_d",
        "self._previous_attitude_reference_wxyz",
        "self._previous_deck_heading_w",
        "self._landing_stage.unsqueeze(-1)",
    ):
        assert required in source
    assert "self._align_success.float().unsqueeze(-1)" not in source


def test_reference_path_reuses_contact_point_adapter_functions():
    source = _source()
    for required in (
        "deck_contact_point_velocity(",
        "deck_relative_to_world_velocity(",
        "world_to_ned_velocity(",
    ):
        assert required in source
    assert "torch.cross(deck" not in source


def test_terminal_attitude_pipeline_uses_s2_guidance_and_external_controller_reference():
    source = _source()
    for required in (
        "terminal_alignment_weight(",
        "deck_heading_world(",
        "compute_velocity_attitude_reference(",
        "shortest_quaternion_slerp(",
        "limit_attitude_tilt(",
        "limit_attitude_reference_rate(",
        "attitude_reference_wxyz=self._attitude_reference_wxyz",
    ):
        assert required in source
    for forbidden in ("RL -> roll", "RL -> torque", "motor_action"):
        assert forbidden not in source


def test_new_success_uses_relative_angular_velocity_without_changing_old_m2():
    source = _source()
    assert "relative_angular_velocity(" in source
    assert 'terms["relative_ang_vel_norm"]' in source
    assert "self.cfg.safe_contact_ang_vel" in source
    old = OLD_M2.read_text()
    assert "def _compute_landing_terms" not in old


def test_continuous_reward_does_not_use_hard_landing_decision_gate():
    source = _source()
    reward = source[source.index("    def _get_rewards") : source.index("    def _get_dones")]
    assert "can_land" not in reward
    assert "align_success" not in reward
    assert "terminal_alpha" in reward
    assert "delta_stage" in reward
    assert "relative_reference_delta" in reward


def test_new_ppo_config_starts_from_scratch_and_does_not_reinterpret_m2_checkpoint():
    config = yaml.safe_load(PPO.read_text())["params"]
    assert config["load_checkpoint"] is False
    assert config["load_path"] == ""
    assert config["network"]["mlp"]["units"] == [64, 64]
    assert config["network"]["mlp"]["activation"].lower() == "elu"
    assert config["config"]["name"] == "quadcopter_ship_landing_px4_continuous_stage"


def test_frozen_direct_source_is_not_reinterpreted_as_new_action_contract():
    source = FROZEN_DIRECT.read_text()
    assert "action_space = 4" in source
    assert "self.cfg.thrust_to_weight * self._robot_weight" in source
    assert "continuous_stage" not in source

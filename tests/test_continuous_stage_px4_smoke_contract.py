from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rl_games/check_px4_continuous_stage_smoke.py"
TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"


def _source() -> str:
    assert SCRIPT.exists()
    return SCRIPT.read_text()


def test_continuous_stage_smoke_script_identity_and_single_env_contract():
    source = _source()
    assert f'TASK_ID = "{TASK_ID}"' in source
    assert "QuadcopterShipLandingPx4ContinuousStageEnvCfg" in source
    assert "QuadcopterShipLandingPx4HierarchicalEnvCfg" not in source
    assert 'parser.add_argument("--num_envs", type=int, default=1, choices=(1,))' in source


def test_continuous_stage_smoke_is_deterministic_and_does_not_train_or_load_checkpoint():
    source = _source()
    for required in (
        '"static_hover"',
        '"stage_ramp"',
        '"constant_xy_deck"',
        '"heave_tracking"',
        '"normal_descent_stage_ramp"',
        '"terminal_attitude_blend"',
        '"static_yaw_heading"',
        '"off_center_contact_point"',
        '"recovery"',
        "env.reset(seed=42)",
    ):
        assert required in source
    for forbidden in ("load_checkpoint", "load_path", "Runner(", "runner.run", "PPO", "--checkpoint"):
        assert forbidden not in source


def test_continuous_stage_smoke_checks_frozen_stage_reference_attitude_and_finite_gates():
    source = _source()
    for required in (
        "stage_raw",
        "stage_filtered",
        "delta_stage",
        "V_t",
        "V_down",
        "V_up",
        "relative_velocity_target_d",
        "relative_velocity_reference_d",
        "deck_contact_velocity_w",
        "velocity_reference_w",
        "velocity_reference_ned",
        "terminal_alpha",
        "deck_heading_w",
        "q_vel",
        "q_ref",
        "q_deck",
        "attitude_reference_rate",
        "relative_angular_speed",
        "controller_saturation_ratio",
        "no_nan_inf",
        "basic_ground_crash_zero",
    ):
        assert required in source


def test_continuous_stage_smoke_reuses_production_contact_point_path():
    source = _source()
    assert "_deck_contact_velocity_ref_w" in source
    assert "torch.cross(" not in source
    assert "deck_contact_point_velocity(" not in source
    assert "zip(samples, samples[1:], strict=True)" not in source

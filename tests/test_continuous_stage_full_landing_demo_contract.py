import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rl_games/run_px4_continuous_stage_full_landing_demo.py"
TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"


def _source() -> str:
    assert SCRIPT.exists()
    return SCRIPT.read_text()


def test_full_landing_demo_identity_and_deterministic_single_env_contract():
    source = _source()
    assert f'TASK_ID = "{TASK_ID}"' in source
    assert "SEED = 42" in source
    assert "cfg.scene.num_envs = 1" in source
    assert "cfg.seed = SEED" in source
    assert "env.reset(seed=SEED)" in source
    assert 'parser.add_argument("--initial_clearance", type=float, default=0.25)' in source


def test_full_landing_demo_does_not_train_or_load_checkpoint():
    source = _source()
    for forbidden in ("Runner(", "load_checkpoint", "load_path", "agent.restore", "PPO", "--checkpoint"):
        assert forbidden not in source


def test_full_landing_demo_keeps_success_thresholds_read_only():
    source = _source()
    threshold_names = (
        "landing_success_radius",
        "safe_contact_normal_speed",
        "safe_contact_tangential_speed",
        "safe_contact_ang_vel",
        "hard_contact_normal_speed",
        "settle_hold_steps",
    )
    for name in threshold_names:
        assert re.search(rf"task\.cfg\.{name}\s*=", source) is None
        assert re.search(rf"cfg\.{name}\s*=", source) is None


def test_full_landing_demo_has_contact_latch_settle_and_safety_validation():
    source = _source()
    for required in (
        "contact_latched = False",
        "contact_latched = True",
        'phase = "contact_settle"',
        'result = "settled_landing"',
        'if snapshot["hard_contact"]',
        'if snapshot["ground_crash"]',
        '"settled_landing": result == "settled_landing"',
        'if result != "settled_landing"',
        "raise SystemExit(1)",
    ):
        assert required in source


def test_full_landing_demo_reuses_production_reference_and_controller_path():
    source = _source()
    assert "QuadcopterShipLandingPx4ContinuousStageEnvCfg" in source
    assert "task._pre_physics_step(action)" in source
    assert "task._apply_action()" in source
    assert "task._compute_landing_terms()" in source
    for forbidden in (
        "deck_contact_point_velocity(",
        "torch.cross(",
        "Px4ReferenceAdapter(",
        "VectorizedPx4LikeController(",
    ):
        assert forbidden not in source

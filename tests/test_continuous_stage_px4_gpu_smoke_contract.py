from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rl_games/check_px4_continuous_stage_gpu_smoke.py"
TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"


def _source() -> str:
    assert SCRIPT.exists()
    return SCRIPT.read_text()


def test_gpu_smoke_identity_seed_env_count_and_device_contract():
    source = _source()
    assert f'TASK_ID = "{TASK_ID}"' in source
    assert "SEED = 42" in source
    assert 'parser.add_argument("--num_envs", type=int, default=16, choices=(16,))' in source
    assert 'EXPECTED_DEVICE = "cuda"' in source
    assert "cfg.scene.num_envs = args.num_envs" in source
    assert "cfg.seed = SEED" in source
    assert "env.reset(seed=SEED)" in source


def test_gpu_smoke_does_not_train_or_load_checkpoint():
    source = _source()
    for forbidden in ("Runner(", "load_checkpoint", "load_path", "agent.restore", "PPO", "--checkpoint", "runner.run"):
        assert forbidden not in source


def test_gpu_smoke_freezes_required_vectorized_tensor_contract():
    source = _source()
    for required in (
        "_landing_stage",
        "_stage_raw",
        "_delta_stage",
        "_relative_velocity_target_d",
        "_relative_velocity_ref_d",
        "_deck_contact_velocity_ref_w",
        "_velocity_reference_w",
        "_velocity_reference_ned",
        "_terminal_alpha",
        "_deck_heading_w",
        "_velocity_attitude_reference_wxyz",
        "_attitude_reference_wxyz",
        "_attitude_reference_rate",
        "_relative_angular_velocity_w",
        "_relative_angular_speed",
        "shape_device_finite",
    ):
        assert required in source


def test_gpu_smoke_has_cross_env_partial_reset_stage_frame_attitude_and_controller_gates():
    source = _source()
    for required in (
        "cross_env_contamination",
        "cross_env_isolation",
        "partial_reset_isolated",
        "stage_bounds",
        "stage_rate",
        "reference_frame_signs",
        "attitude_reference",
        "controller_finite",
        "reward_path_finite",
        "ground_crash_zero",
        "per_env",
        "global_max_velocity_tracking_error_mps",
        "global_max_attitude_reference_rate_radps",
        "global_controller_saturation_ratio",
    ):
        assert required in source


def test_gpu_smoke_does_not_replace_vectorization_with_per_env_production_loop():
    source = _source()
    assert "for env_id in range(16)" not in source
    assert "deck_contact_point_velocity(" not in source
    assert "torch.cross(" not in source
    assert "Px4ReferenceAdapter(" not in source
    assert "VectorizedPx4LikeController(" not in source

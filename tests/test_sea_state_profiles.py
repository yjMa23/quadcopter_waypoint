from pathlib import Path

import pytest

from quadcopter_waypoint.utils.sea_state_profiles import hydra_env_overrides, load_sea_state_profiles


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "benchmarks/sea_state/profiles.yaml"


def family(profiles: dict, name: str) -> list[tuple[str, dict]]:
    rows = [(profile_name, profile) for profile_name, profile in profiles.items() if profile["family"] == name]
    return sorted(rows, key=lambda item: item[1]["severity_rank"])


def test_profiles_are_factor_isolated_and_ranked():
    profiles = load_sea_state_profiles(PROFILES)

    frequency = family(profiles, "frequency_shift")
    assert len(frequency) == 7
    assert [profile["severity_rank"] for _, profile in frequency] == list(range(7))
    assert [profile["env"]["sea_state_tp_min_s"] for _, profile in frequency] == [6.0, 5.0, 4.0, 3.2, 2.5, 2.0, 1.6]
    for _, profile in frequency:
        env = profile["env"]
        assert (env["sea_state_hs_min_m"], env["sea_state_hs_max_m"]) == (0.16, 0.20)
        assert (env["sea_state_gamma_min"], env["sea_state_gamma_max"]) == (2.8, 3.4)

    tilt = family(profiles, "tilt_shift")
    assert len(tilt) == 5
    tilt_gains = [profile["env"]["sea_state_roll_gain_deg_per_m"] for _, profile in tilt]
    assert tilt_gains == sorted(tilt_gains)
    for _, profile in tilt:
        env = profile["env"]
        assert env["sea_state_roll_gain_deg_per_m"] == env["sea_state_pitch_gain_deg_per_m"]
        assert (env["sea_state_tp_min_s"], env["sea_state_tp_max_s"]) == (4.4, 5.2)
        assert env["sea_state_heave_gain"] == 0.75

    heave = family(profiles, "heave_rate_shift")
    assert len(heave) == 5
    heave_gains = [profile["env"]["sea_state_heave_gain"] for _, profile in heave]
    assert heave_gains == sorted(heave_gains)
    for _, profile in heave:
        env = profile["env"]
        assert env["sea_state_roll_gain_deg_per_m"] == 15.0
        assert env["sea_state_pitch_gain_deg_per_m"] == 15.0
        assert (env["sea_state_tp_min_s"], env["sea_state_tp_max_s"]) == (2.0, 2.6)

    combined = family(profiles, "combined_shift")
    assert len(combined) == 5
    assert [profile["severity_rank"] for _, profile in combined] == list(range(5))


def test_profile_hydra_overrides_are_explicit():
    profiles = load_sea_state_profiles(PROFILES)
    overrides = hydra_env_overrides("tilt_shift_target4deg", profiles["tilt_shift_target4deg"])
    assert overrides[0] == "env.sea_state_benchmark_profile=tilt_shift_target4deg"
    assert "env.sea_state_mode=stochastic" in overrides
    assert "env.sea_state_roll_gain_deg_per_m=86.0" in overrides
    assert "env.sea_state_pitch_gain_deg_per_m=86.0" in overrides


def test_profile_loader_rejects_core_math_override(tmp_path: Path):
    invalid = tmp_path / "profiles.yaml"
    invalid.write_text(
        "profiles:\n"
        "  invalid:\n"
        "    family: frequency_shift\n"
        "    severity_rank: 0\n"
        "    env:\n"
        "      sea_state_num_components: 48\n"
    )
    with pytest.raises(ValueError, match="unsupported env fields"):
        load_sea_state_profiles(invalid)

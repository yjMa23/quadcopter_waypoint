import math

from scripts.rl_games.analyze_sea_state_robustness import boundary_candidates, summarize_profiles, wilson_upper


def test_profile_summary_keeps_independent_seed_rates() -> None:
    rows = [
        {"policy_label": "teacher", "profile": "candidate", "eval_seed": "245", "settled_landing": "True"},
        {"policy_label": "teacher", "profile": "candidate", "eval_seed": "245", "settled_landing": "False"},
        {"policy_label": "teacher", "profile": "candidate", "eval_seed": "246", "settled_landing": "True"},
        {"policy_label": "teacher", "profile": "candidate", "eval_seed": "246", "settled_landing": "True"},
    ]
    summary = summarize_profiles(rows, {"candidate": {"family": "frequency_shift", "severity_rank": 1}})[0]

    assert summary["eval_seed_count"] == 2
    assert summary["settled_landing_rate_min_by_seed"] == 0.5
    assert summary["settled_landing_rate_max_by_seed"] == 1.0
    assert summary["settled_landing_wilson_upper_max_by_seed"] == 1.0
    assert math.isclose(wilson_upper(61, 64), 0.983930983213707)


def test_boundary_requires_replication_and_non_scaling_dominated_motion() -> None:
    base = {
        "policy_label": "teacher",
        "family": "frequency_shift",
        "profile": "candidate",
        "severity_rank": 1,
        "episodes": 128,
        "settled_landing_rate": 0.85,
        "settled_landing_rate_min_by_seed": 0.84,
        "settled_landing_rate_max_by_seed": 0.86,
        "settled_landing_wilson_upper_max_by_seed": 0.92,
        "deck_miss_rate": 0.15,
        "hard_contact_rate": 0.0,
        "scaling_fraction": 0.10,
        "min_scale_p05": 0.95,
    }

    admitted = boundary_candidates([{**base, "eval_seed_count": 2}])
    assert admitted["candidates"]
    assert admitted["adaptation_training_allowed"]

    prioritized = boundary_candidates(
        [
            {
                **base,
                "profile": "near_target_lower_severity",
                "severity_rank": 0,
                "settled_landing_rate": 0.93,
                "settled_landing_wilson_upper_max_by_seed": 0.94,
                "eval_seed_count": 2,
            },
            {**base, "profile": "target_first", "severity_rank": 1, "eval_seed_count": 2},
            {
                **base,
                "profile": "target_later",
                "severity_rank": 2,
                "settled_landing_rate": 0.80,
                "settled_landing_wilson_upper_max_by_seed": 0.88,
                "eval_seed_count": 2,
            },
        ]
    )
    assert prioritized["candidates"][0]["candidate_profile"] == "target_first"

    blocked = boundary_candidates([{**base, "eval_seed_count": 1}])
    assert not blocked["candidates"]
    assert not blocked["adaptation_training_allowed"]
    assert blocked["adaptation_training_block_reason"] == "no eligible robustness boundary candidate"
    assert not boundary_candidates([{**base, "eval_seed_count": 2, "scaling_fraction": 0.21}])["candidates"]
    assert not boundary_candidates(
        [{**base, "eval_seed_count": 2, "settled_landing_wilson_upper_max_by_seed": 0.96}]
    )["candidates"]

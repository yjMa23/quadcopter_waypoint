from quadcopter_waypoint.imitation.benchmark import (
    aggregate_seed_summaries,
    summarize_episode_rows,
    threshold_crossing_steps,
)


def _row(**overrides):
    value = {
        "contact_success": "True",
        "settled_landing": "True",
        "hard_contact": "False",
        "ground_crash": "False",
        "deck_miss": "False",
        "time_out": "False",
        "first_contact_xy_error_deck_frame": "0.1",
        "first_contact_normal_rel_speed": "-0.2",
        "first_contact_tangential_rel_speed": "0.05",
        "first_contact_body_deck_normal_angle": "0.1",
        "touchdown_distance": "0.11",
        "maximum_penetration": "0.01",
    }
    value.update(overrides)
    return value


def test_episode_and_seed_aggregation():
    summary = summarize_episode_rows([_row(), _row(settled_landing="False", deck_miss="True")])
    assert summary["episodes"] == 2
    assert summary["settled_landing_rate"] == 0.5
    assert summary["deck_miss_rate"] == 0.5
    aggregate = aggregate_seed_summaries({"42": summary, "43": summary})
    assert aggregate["episodes"] == 4
    assert aggregate["settled_landing_rate"]["mean"] == 0.5
    assert aggregate["settled_landing_rate"]["std"] == 0.0


def test_threshold_crossing_does_not_interpolate_or_extrapolate():
    curve = [
        {"environment_steps": 100, "settled_landing_rate": 0.7},
        {"environment_steps": 200, "settled_landing_rate": 0.85},
        {"environment_steps": 300, "settled_landing_rate": 0.91},
    ]
    assert threshold_crossing_steps(curve) == {"80%": 200, "90%": 300, "92%": None}

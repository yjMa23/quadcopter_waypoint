from scripts.rl_games.analyze_m2_reward_compatibility import (
    length_normalized,
    negative_ranks,
    pearson,
    previous_or_exact,
    relative_magnitude_percent,
    restore_episode_contribution,
    spearman,
    summarize_deterministic_rows,
)


def test_restore_episode_contribution_undoes_environment_logging_scale():
    assert restore_episode_contribution(-2.5, max_episode_length_s=10.0) == -25.0


def test_length_normalized_handles_missing_and_nonpositive_lengths():
    assert length_normalized(-20.0, 100.0) == -0.2
    assert length_normalized(-20.0, None) is None
    assert length_normalized(-20.0, 0.0) is None


def test_previous_or_exact_preserves_source_iteration():
    series = {1: 10.0, 3: 30.0, 5: 50.0}
    assert previous_or_exact(series, 3) == (3, 30.0)
    assert previous_or_exact(series, 4) == (3, 30.0)
    assert previous_or_exact(series, 0) == (None, None)


def test_reward_magnitude_percent_and_negative_rank_are_deterministic():
    values = {"a": -6.0, "b": 2.0, "c": -2.0}
    percentages = relative_magnitude_percent(values)
    assert percentages == {"a": 60.0, "b": 20.0, "c": 20.0}
    assert negative_ranks(values) == {"a": 1, "b": None, "c": 2}


def test_correlations_cover_linear_monotonic_and_constant_cases():
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0
    assert spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == -1.0
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_deterministic_summary_detects_terminal_drift_after_latched_alignment():
    rows = [
        {
            "align_success": "True",
            "time_out": "True",
            "crash": "False",
            "settled_landing": "False",
            "terminal_horizontal_error": "0.31",
            "terminal_surface_clearance": "0.40",
        },
        {
            "align_success": "True",
            "time_out": "False",
            "crash": "True",
            "settled_landing": "False",
            "terminal_horizontal_error": "0.10",
            "terminal_surface_clearance": "0.00",
        },
        {
            "align_success": "False",
            "time_out": "True",
            "crash": "False",
            "settled_landing": "False",
            "terminal_horizontal_error": "0.50",
            "terminal_surface_clearance": "0.50",
        },
    ]
    summary = summarize_deterministic_rows(rows, align_radius_m=0.25)
    assert summary["aligned_episodes"] == 2
    assert summary["aligned_terminal_outside_radius"] == 1
    assert summary["aligned_terminal_outside_radius_rate"] == 0.5
    assert summary["aligned_timeout_episodes"] == 1
    assert summary["aligned_timeout_terminal_outside_radius_rate"] == 1.0
    assert summary["aligned_timeout_terminal_clearance_mean_m"] == 0.4

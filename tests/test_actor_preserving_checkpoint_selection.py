"""actor-preserving PPO reuses the frozen checkpoint-selection analysis metric-selection semantics."""

from __future__ import annotations

from quadcopter_waypoint.imitation.checkpoint_sweep import selection_sort_key, select_validation_best


def _row(**overrides):
    row = {
        "checkpoint_path": "candidate.pth",
        "checkpoint_sha256": "a" * 64,
        "actor_sha256": "b" * 64,
        "train_seed": 42,
        "epoch": 20,
        "training_reward": 10.0,
        "kind": "periodic",
        "eval_seed": 145,
        "episodes": 128,
        "settled_landing_rate": 0.9,
        "deck_miss_rate": 0.08,
        "hard_contact_rate": 0.01,
        "contact_success_rate": 0.99,
        "timeout_rate": 0.01,
        "touchdown_distance_mean_m": 0.06,
    }
    row.update(overrides)
    return row


def test_selection_tie_break_order() -> None:
    rows = [
        _row(checkpoint_path="later.pth", checkpoint_sha256="1" * 64, epoch=20),
        _row(checkpoint_path="earlier.pth", checkpoint_sha256="2" * 64, epoch=10),
    ]
    assert min(rows, key=selection_sort_key)["checkpoint_path"] == "earlier.pth"

    rows = [
        _row(checkpoint_path="more_miss.pth", checkpoint_sha256="3" * 64, deck_miss_rate=0.09),
        _row(checkpoint_path="less_miss.pth", checkpoint_sha256="4" * 64, deck_miss_rate=0.07),
    ]
    assert min(rows, key=selection_sort_key)["checkpoint_path"] == "less_miss.pth"

    rows = [
        _row(checkpoint_path="hard.pth", checkpoint_sha256="5" * 64, hard_contact_rate=0.02),
        _row(checkpoint_path="safe.pth", checkpoint_sha256="6" * 64, hard_contact_rate=0.00),
    ]
    assert min(rows, key=selection_sort_key)["checkpoint_path"] == "safe.pth"


def test_validation_selection_ignores_training_reward() -> None:
    rows = [
        _row(checkpoint_path="metric.pth", checkpoint_sha256="7" * 64, settled_landing_rate=0.95, training_reward=1.0),
        _row(checkpoint_path="reward.pth", checkpoint_sha256="8" * 64, settled_landing_rate=0.80, training_reward=100.0),
    ]
    selected = select_validation_best(rows)
    assert selected[42]["checkpoint_path"] == "metric.pth"

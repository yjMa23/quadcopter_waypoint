import csv
from pathlib import Path

import torch

from quadcopter_waypoint.imitation.checkpoint_sweep import (
    CheckpointRecord,
    actor_state_sha256,
    aggregate_checkpoint_rows,
    compute_drift_metrics,
    discover_checkpoints,
    parse_periodic_filename,
    resume_key,
    select_screening_candidates,
    select_validation_best,
    validate_evaluation_csv,
)


def _model(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "a2c_network.actor_mlp.0.weight": torch.full((2, 2), offset),
        "a2c_network.actor_mlp.0.bias": torch.full((2,), offset),
        "a2c_network.actor_mlp.2.weight": torch.full((2, 2), offset),
        "a2c_network.actor_mlp.2.bias": torch.full((2,), offset),
        "a2c_network.mu.weight": torch.full((4, 2), offset),
        "a2c_network.mu.bias": torch.full((4,), offset),
        "running_mean_std.running_mean": torch.full((2,), offset, dtype=torch.float64),
        "running_mean_std.running_var": torch.full((2,), 1.0 + offset, dtype=torch.float64),
        "running_mean_std.count": torch.tensor(10.0 + offset, dtype=torch.float64),
    }


def _save(path: Path, epoch: int, model: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "epoch": epoch, "last_mean_rewards": torch.tensor(3.0)}, path)


def test_parse_periodic_filename_handles_negative_reward_escape() -> None:
    assert parse_periodic_filename("last_bc_ppo_ep_20_rew_4.5.pth") == (20, 4.5)
    assert parse_periodic_filename("last_bc_ppo_ep_200_rew__-2.25_.pth") == (200, -2.25)


def test_discover_marks_duplicate_actor_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed42"
    model = _model()
    first = run_dir / "nn" / "last_run_ep_200_rew_1.0.pth"
    duplicate = run_dir / "nn" / "last_run_ep_200_rew__1.0_.pth"
    selected = run_dir / "nn" / "run.pth"
    bc = tmp_path / "bc_init.pth"
    _save(first, 200, model)
    _save(duplicate, 200, model)
    _save(selected, 150, model)
    _save(bc, 0, model)

    records = discover_checkpoints([run_dir], bc, include_reward_selected=True)
    periodic = [record for record in records if record.kind == "periodic"]
    assert len(periodic) == 2
    assert sum(record.canonical for record in periodic) == 1
    assert periodic[0].actor_sha256 == periodic[1].actor_sha256 == actor_state_sha256(model)
    assert any(record.kind == "reward_selected" and record.epoch == 150 for record in records)
    assert any(record.kind == "bc_init" and record.epoch == 0 for record in records)


def test_resume_key_requires_matching_hash_and_parameters() -> None:
    record = CheckpointRecord(
        path="checkpoint.pth",
        sha256="a" * 64,
        actor_sha256="b" * 64,
        size_bytes=1,
        train_seed=42,
        epoch=10,
        training_reward=1.0,
        kind="periodic",
    )
    baseline = resume_key(record, "task", eval_seed=145, episodes=64, num_envs=64)
    assert baseline == resume_key(record, "task", eval_seed=145, episodes=64, num_envs=64)
    assert baseline != resume_key(record, "task", eval_seed=146, episodes=64, num_envs=64)
    assert baseline != resume_key(record, "task", eval_seed=145, episodes=128, num_envs=64)
    assert baseline != resume_key(record, "task", eval_seed=145, episodes=64, num_envs=32)


def test_validate_evaluation_csv_rejects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    fields = [
        "contact_success",
        "settled_landing",
        "hard_contact",
        "ground_crash",
        "deck_miss",
        "time_out",
        "first_contact_xy_error_deck_frame",
        "first_contact_normal_rel_speed",
        "first_contact_tangential_rel_speed",
        "first_contact_body_deck_normal_angle",
        "touchdown_distance",
        "maximum_penetration",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: "0" for field in fields})
    try:
        validate_evaluation_csv(path, episodes=2)
    except ValueError as error:
        assert "expected 2" in str(error)
    else:
        raise AssertionError("truncated CSV was accepted")


def _row(
    path: str,
    sha: str,
    seed: int | None,
    epoch: int,
    kind: str,
    settled: float,
    miss: float,
    hard: float,
    touchdown: float,
    eval_seed: int = 145,
) -> dict[str, object]:
    return {
        "checkpoint_path": path,
        "checkpoint_sha256": sha,
        "actor_sha256": f"actor-{sha}",
        "train_seed": seed,
        "epoch": epoch,
        "training_reward": float(epoch),
        "kind": kind,
        "eval_seed": eval_seed,
        "episodes": 64,
        "settled_landing_rate": settled,
        "deck_miss_rate": miss,
        "hard_contact_rate": hard,
        "contact_success_rate": 1.0 - miss,
        "timeout_rate": 0.0,
        "touchdown_distance_mean_m": touchdown,
    }


def test_screening_and_validation_selection_follow_tie_break_rules() -> None:
    rows = [
        _row("bc", "bc", None, 0, "bc_init", 0.88, 0.12, 0.0, 0.05),
        _row("early", "a", 42, 10, "periodic", 0.90, 0.08, 0.01, 0.05),
        _row("late", "b", 42, 20, "periodic", 0.90, 0.08, 0.01, 0.05),
        _row("reward", "c", 42, 15, "reward_selected", 0.70, 0.20, 0.02, 0.06),
    ]
    candidates = select_screening_candidates(rows, top_k=1)
    assert candidates[42] == ["early", "reward", "bc"]

    validation_rows = []
    for eval_seed in (145, 146, 147):
        validation_rows.extend(
            [
                {**rows[1], "eval_seed": eval_seed, "episodes": 128},
                {**rows[2], "eval_seed": eval_seed, "episodes": 128},
            ]
        )
    aggregated = aggregate_checkpoint_rows(validation_rows)
    assert len(aggregated) == 2
    assert select_validation_best(validation_rows)[42]["checkpoint_path"] == "early"


def test_compute_drift_metrics_reports_action_parameter_and_stat_changes() -> None:
    reference_model = _model(0.0)
    candidate_model = _model(1.0)
    reference_actions = torch.zeros((2, 4))
    teacher_actions = torch.full((2, 4), 0.5)
    candidate_actions = torch.ones((2, 4))
    metrics = compute_drift_metrics(
        reference_actions,
        teacher_actions,
        candidate_actions,
        reference_model,
        candidate_model,
    )
    assert metrics["action_mse_vs_bc"] == 1.0
    assert metrics["action_mse_vs_teacher"] == 0.25
    assert metrics["action_dim_mse_vs_bc"] == [1.0, 1.0, 1.0, 1.0]
    assert metrics["actor_parameter_l2"] > 0.0
    assert metrics["running_mean_l2"] > 0.0
    assert metrics["running_var_l2"] > 0.0
    assert metrics["running_count_delta"] == 1.0

# imitation-learning benchmark Imitation + Hybrid Benchmark

This directory is generated from raw CSV/JSON artifacts by `scripts/imitation/build_benchmark.py`.

- Task: `Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0`
- Dataset: 3976 successful episodes / 540321 transitions
- Episode split: train 3180, validation 397, test 399
- BC-only settled landing: 88.28%
- PPO-from-scratch settled landing: 0.91%
- BC+PPO settled landing: 76.69%
- Frozen PPO teacher settled landing: 94.66%

The BC-only target was met. The 92% BC+PPO target and the 90% sample-efficiency target were not met. The exact negative result, including the one conservative-learning-rate correction, is retained in `summary.json` and `training_runs.json`.

The current policy is state based. It contains no camera image input and no real visual projection features.

Paper-style formulation, equations, algorithm design, implementation traceability, and discussion:

```text
docs/imitation_hybrid_paper.md
```

Interactive display and headless-rendering diagnosis:

```text
docs/runtime_display_troubleshooting.md
```

## Files

- `summary.json`: checksums, per-seed metrics, aggregate metrics, thresholds, acceptance, and limitations.
- `dataset_summary.json`: dataset scale, split, phase coverage, and action statistics.
- `training_runs.json`: fair training budgets, selected checkpoints, wall time, and diagnostic rerun.
- `formal_evaluations/`: copied 256-episode-per-seed raw CSVs.
- `comparison.csv` / `comparison.md`: final primary-method table.
- `learning_curves.csv`: fixed-interval 128-episode-per-seed evaluation points.
- `rollout_cases/`: teacher, BC, BC+PPO success trajectories plus one PPO-scratch timeout failure trace.
- `*.png`: figures generated from raw artifacts.
- `commands.txt`: reproducibility commands.

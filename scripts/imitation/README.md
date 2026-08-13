# imitation-learning benchmark–actor-preserving pipeline Imitation and Actor-Preserving PPO Scripts

imitation-learning benchmark keeps the frozen physical-deck-attitude task task unchanged and adds expert collection, Behavior Cloning (BC), BC-to-RL-Games migration, fair PPO training/evaluation support, benchmark generation, and objective rollout traces. checkpoint-selection analysis adds checkpoint selection/drift diagnosis; actor-preserving PPO adds a project-local actor-preserving PPO implementation without modifying Isaac Lab or installed RL-Games source.

论文式理论与实现对应关系见：

```text
docs/imitation_hybrid_paper.md
```

该文档顶部包含由 `tests/test_imitation_documentation_sync.py` 校验的 `CODE_SYNC` 参数快照，确保网络、归一化、动作缩放、甲板范围、落地阈值和 PPO 配置与当前代码一致。

## Pipeline

```text
frozen physical-deck-attitude task teacher
  -> collect_teacher.py
  -> finalize_dataset.py
  -> train_bc.py
  -> create_bc_init_checkpoint.py
  -> scripts/rl_games/train.py (scratch / BC+PPO)
  -> scripts/rl_games/eval_metrics.py
  -> build_benchmark.py
```

## Dataset

`collect_teacher.py` records only complete settled-landing expert episodes. Every transition contains the raw 22-dimensional observation, deterministic teacher action after the `[-1, 1]` clamp, reward, terminal flags, phase, deck-motion parameters, and touchdown/contact metrics. Output uses compressed NPZ shards plus a resumable partial manifest; existing valid shards are verified and never silently overwritten.

`finalize_dataset.py` merges collection seeds and creates a reproducible episode-level 80/10/10 train/validation/test split. It rejects episode overlap, missing/corrupt shards, hash mismatches, insufficient successful episodes, and insufficient transitions.

## Behavior Cloning

`train_bc.py` trains the PPO-compatible actor:

```text
22 -> 64 -> ELU -> 64 -> ELU -> 4
```

Raw observations are normalized with the frozen teacher's RL-Games running mean/variance, epsilon, and clamp semantics. Training supports inverse-frequency flight-phase weighting, selects the best checkpoint by validation loss, and reports split/action/phase MSE.

## BC initialization

`create_bc_init_checkpoint.py` copies only the BC actor and observation statistics into the RL-Games checkpoint structure. It resets optimizer state, epoch/frame/history, environment state, value normalization, and initializes the value head reproducibly. Fixed sigma remains identical to the physical-deck-attitude task PPO configuration. A deterministic action parity check must pass before online PPO training.

## Closed-loop evaluation

BC-only is evaluated by loading the BC-initialized RL-Games checkpoint before any PPO update. PPO-from-scratch and BC+PPO use the same task, number of environments, seeds, PPO configuration, epoch/interaction budget, checkpoint-selection rule, and formal evaluator.

`record_rollout_case.py` stores one complete objective state/action trajectory for a requested success or failure outcome. With `--video --enable_cameras --headless --num_envs=1`, it also records the matching episode through Isaac Sim offscreen rendering, writes an MP4 beside the NPZ, and hashes both. For rare outcomes, `--video_start_episode=N` deterministically skips rendering earlier single-environment episodes while retaining the same checkpoint/seed sequence, then caches only the target episode. The sidecar explicitly records `human_review_completed=false`; headless structural/terminal validation is not presented as interactive GUI acceptance.

Display diagnosis and local GUI startup instructions are documented in:

```text
docs/runtime_display_troubleshooting.md
```

## Tests

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

Tests cover schema, dtypes/shapes, action bounds, NaN/Inf rejection, episode split leakage, manifest hashes, statistics, observation normalization, actor save/load, BC-to-RL-Games parity, benchmark aggregation, threshold crossing, and paper-to-code parameter synchronization.

Exact imitation-learning benchmark formal commands and artifact checksums are generated in:

```text
benchmarks/imitation_hybrid/commands.txt
benchmarks/imitation_hybrid/summary.json
```

## checkpoint-selection analysis checkpoint selection and drift

checkpoint-selection analysis reuses the frozen checkpoints and the same formal evaluator; it does not retrain or modify the physical-deck-attitude task task.

- `evaluate_checkpoint_sweep.py` builds a SHA256 inventory, calls `eval_metrics.py`, writes per-checkpoint/per-seed CSVs, and atomically records a resumable manifest. A completed entry is reused only when the checkpoint hash and evaluation parameters match.
- `analyze_checkpoint_drift.py` evaluates checkpoint-specific deterministic mean actions on the frozen imitation-learning benchmark test split using each checkpoint's RL-Games observation normalization. It reports action MSE against BC and teacher, per-action MSE, actor parameter distance, and running-statistic drift.
- `build_checkpoint_selection_benchmark.py` performs screening candidate extraction, validation selection with the fixed tie-break rules, independent-test aggregation, plotting, and benchmark generation.

Pure-Python tests cover checkpoint parsing, duplicate actor snapshots, truncated CSV rejection, resume identity, aggregation, metric selection, tie-breaking, and drift calculations without launching Isaac Sim.

checkpoint-selection analysis generated evidence:

```text
benchmarks/checkpoint_selection/README.md
benchmarks/checkpoint_selection/summary.json
benchmarks/checkpoint_selection/commands.txt
```

## actor-preserving PPO

actor-preserving PPO uses the frozen physical-deck-attitude task task and the frozen imitation-learning benchmark BC initialization, but changes the optimizer path:

```text
shared BC-init checkpoint
  -> create_actor_preserving_checkpoint.py
  -> separate actor/critic checkpoint
  -> actor_preserving_agent.py
       epoch 1..10: critic-only warm-up
       frozen observation RMS
       frozen warm-up LR scheduler
       on-policy BC mean-action MSE anchor
  -> periodic validation selection
  -> independent formal test
  -> analyze_actor_preserving_drift.py
  -> finalize_actor_preserving_benchmark.py
```

Important files:

- `actor_preserving_checkpoint.py`: strict shared→separate migration, source/dataset SHA256, embedded frozen reference actor, deterministic actor parity.
- `actor_preserving_agent.py`: project-local RL-Games custom agent registration, gradient isolation, warm-up/resume semantics, frozen RMS, BC anchor and hash diagnostics.
- `evaluate_checkpoint_sweep.py`: supports an explicit `--agent` so shared checkpoint-selection analysis and separate actor-preserving PPO checkpoints use the matching network schema.
- `analyze_actor_preserving_drift.py`: actor/critic/action/RMS/fixed-sigma drift on the frozen imitation-learning benchmark test observation split.
- `build_actor_preserving_benchmark.py`: pilot aggregation and coefficient selection.
- `finalize_actor_preserving_benchmark.py`: formal validation selection, independent-test aggregation, 95% confidence interval, prediction table and figures.
- `build_video_manifest.py`: validates one real success and one real failure video/trajectory pair.

The machine-readable actor-preserving PPO source of truth is the single YAML block in `docs/actor_preserving_ppo.md`; `tests/test_actor_preserving_documentation_sync.py` compares it with the actor-preserving PPO YAML, frozen task dimensions and frozen seed protocol.

## actor-preserving PPO formal completion

The complete validation grid contains 66 evaluations per training seed (22 checkpoints × seeds 145/146/147, 128 episodes each). Validation selected epochs 91, 30, and 51 for training seeds 42, 43, and 44. The independent formal test completed all 24 planned actor-preserving PPO/BC physical evaluations without failed/running entries.

actor-preserving PPO metric-selected reached 96.7448% settled landing over 2304 episodes, with 3.1684% deck miss, 0.0868% hard contact, 0 ground crash, and 0.0434% timeout. The exact aggregate, Wilson intervals, per-training-seed results, seven preregistered prediction verdicts, drift evidence, figures, and targeted success/deck-miss video manifest are in:

```text
benchmarks/actor_preserving_ppo/
```

The video manifest validates one real `settled_landing` and one real `deck_miss`. Both are automated headless artifacts and keep `human_review_completed=false`.

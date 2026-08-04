# Phase 7 Imitation Learning Scripts

P7 keeps the frozen P6C task unchanged and adds expert collection, Behavior Cloning (BC), BC-to-RL-Games migration, fair PPO training/evaluation support, benchmark generation, and objective rollout traces.

论文式理论与实现对应关系见：

```text
docs/p7_imitation_hybrid_paper.md
```

该文档顶部包含由 `tests/test_p7_documentation_sync.py` 校验的 `CODE_SYNC` 参数快照，确保网络、归一化、动作缩放、甲板范围、落地阈值和 PPO 配置与当前代码一致。

## Pipeline

```text
frozen P6C teacher
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

`create_bc_init_checkpoint.py` copies only the BC actor and observation statistics into the RL-Games checkpoint structure. It resets optimizer state, epoch/frame/history, environment state, value normalization, and initializes the value head reproducibly. Fixed sigma remains identical to the P6C PPO configuration. A deterministic action parity check must pass before online PPO training.

## Closed-loop evaluation

BC-only is evaluated by loading the BC-initialized RL-Games checkpoint before any PPO update. PPO-from-scratch and BC+PPO use the same task, number of environments, seeds, PPO configuration, epoch/interaction budget, checkpoint-selection rule, and formal evaluator.

`record_rollout_case.py` stores one complete objective state/action trajectory for a requested success or failure outcome. It is suitable for numerical replay and inspection; it does not claim GUI acceptance when the current shell has no usable display. This script does not implement offscreen video recording. A missing interactive `DISPLAY` blocks the GUI, but headless MP4 generation would still be possible through a separate render-enabled recorder.

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

Exact formal commands and artifact checksums are generated in:

```text
benchmarks/phase7_imitation_hybrid/commands.txt
benchmarks/phase7_imitation_hybrid/summary.json
```

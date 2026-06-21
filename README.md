# Quadcopter Waypoint RL

This repository is an **external Isaac Lab project** for reproducing and extending the official `Isaac-Quadcopter-Direct-v0` task. The current goal is to keep the training environment clean and reproducible before adding waypoint-style task changes.

The important conclusion from the migration/debugging process is:

> The external project itself can reproduce the official quadcopter hover behavior. The main source of earlier unstable hovering was not reward design or metrics, but inconsistent training settings and checkpoint confusion, especially `num_envs=1024` versus the official-style `num_envs=4096`.

## Current task layout

### 1. Official clone baseline

Task ID:

```bash
Isaac-Quadcopter-OfficialClone-Direct-v0
```

Location:

```bash
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_official_clone/
```

Purpose:

- Exactly clone the official Isaac Lab quadcopter direct environment logic.
- Only rename the Python classes and Gym task ID so it can live in this external project.
- Keep this task as the clean baseline for all future changes.

The cloned logic keeps the official settings:

```text
episode_length_s = 10.0
scene.num_envs = 4096
scene.env_spacing = 2.5
goal x/y range = [-2.0, 2.0]
goal z range = [0.5, 1.5]
height termination = z < 0.1 or z > 2.0
reward = lin_vel penalty + ang_vel penalty + distance_to_goal reward
```

### 2. Waypoint v1

Task ID:

```bash
Isaac-Quadcopter-WaypointV1-Direct-v0
```

Location:

```bash
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_v1_metrics/
```

Purpose:

- Use the same clean `OfficialClone` training environment.
- Use a separate rl_games experiment name: `quadcopter_waypoint_v1`.
- Avoid mixing v1 checkpoints with the official `quadcopter_direct` checkpoints.

At this stage, v1 intentionally does **not** add metrics inside the training environment. Earlier attempts showed that even evaluation-only tensor operations inside the RL environment can change training trajectories. Metrics should be collected later with a separate evaluation script instead of being inserted into the training step.

### 3. Retired experimental waypoint task

An earlier experimental task under `quadrotor_waypoint/` expanded the target range and added in-environment metrics. It was retired because it made the debugging process confusing and could easily be mixed with official-clone checkpoints.

The retired local experiment is ignored by `.gitignore` and should not be used as a base for future work.

## Repository structure

```text
.
├── README.md
├── scripts/
│   └── rl_games/
│       ├── train.py
│       └── play.py
└── source/
    └── quadcopter_waypoint/
        ├── setup.py
        └── quadcopter_waypoint/
            └── tasks/
                └── direct/
                    ├── quadrotor_official_clone/
                    │   ├── __init__.py
                    │   └── quadrotor_official_clone_env.py
                    └── quadrotor_v1_metrics/
                        ├── __init__.py
                        └── agents/
                            ├── __init__.py
                            └── rl_games_ppo_cfg.yaml
```

## Setup

Activate the Isaac Lab environment first:

```bash
conda activate env_isaaclab
```

Install this external project in editable mode:

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
python -m pip install -e source/quadcopter_waypoint
```

## Training commands

### Official clone baseline

Use this to verify that the external project reproduces the official hover task:

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=200
```

### Waypoint v1

Use this as the stable v1 starting point with a separate log directory:

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=200
```

Expected log directory:

```text
logs/rl_games/quadcopter_waypoint_v1/<timestamp>/
```

## Playing checkpoints

Always use an explicit checkpoint path. Avoid `find ... | sort | tail -n 1` across multiple task logs, because different tasks may produce similarly named checkpoint files.

### Play OfficialClone

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_direct/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_direct.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

### Play WaypointV1

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v1/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v1.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games
```

For the stable v1 run, check:

```text
logs/rl_games/quadcopter_waypoint_v1
```

For the official clone run, check:

```text
logs/rl_games/quadcopter_direct
```

## Important debugging notes

### 1. Keep `num_envs=4096` for this PPO config

The official PPO config uses:

```yaml
horizon_length: 24
minibatch_size: 24576
```

With `num_envs=1024`:

```text
batch size = 24 * 1024 = 24576
```

This gives only one minibatch per PPO update and produced worse hovering behavior in testing.

With `num_envs=4096`:

```text
batch size = 24 * 4096 = 98304
```

This gives four minibatches per PPO update and reproduced the official stable hover behavior.

If `num_envs` must be reduced for GPU memory reasons, also retune `minibatch_size`, `horizon_length`, `mini_epochs`, and possibly `learning_rate`. Do not only change `num_envs`.

### 2. Do not mix task IDs and checkpoints

A common mistake is training one task but playing another task with its checkpoint. Always pair them explicitly:

```text
OfficialClone task  + quadcopter_direct checkpoint
WaypointV1 task     + quadcopter_waypoint_v1 checkpoint
```

### 3. Keep metrics out of the training environment

During debugging, adding success-rate or stable-hover metrics inside `_get_rewards()` or `_reset_idx()` caused confusing differences in training behavior. The current strategy is:

```text
training environment: clean and identical to OfficialClone
evaluation metrics: compute in a separate evaluation script later
```

This keeps RL training reproducible.

## Migration process summary

1. Created an external Isaac Lab project instead of modifying the Isaac Lab source tree.
2. Added wrapper scripts under `scripts/rl_games/` that inject this external task package into the official Isaac Lab training/play scripts while preserving Isaac Sim launch order.
3. First attempted a waypoint task with enlarged target range and in-environment metrics.
4. Observed unstable hovering and checkpoint/log confusion.
5. Built `OfficialClone` to exactly reproduce `Isaac-Quadcopter-Direct-v0` inside the external project.
6. Verified that fixed checkpoint paths and `num_envs=4096` reproduce stable hover behavior.
7. Retired the earlier experimental waypoint task.
8. Added `WaypointV1` as a clean v1 task with a separate experiment name while reusing the official-clone training environment.

## Next planned steps

1. Add a standalone evaluation script to compute success rate, final distance, final velocity, and stable hover rate without modifying the training environment.
2. After evaluation is stable, add waypoint progression: when the drone reaches one target, sample the next target without ending the episode.
3. Add curriculum for larger target ranges only after the official-range waypoint behavior is stable.

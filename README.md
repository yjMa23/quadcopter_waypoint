# Quadcopter Waypoint RL

这是一个基于 **Isaac Lab External Project** 方式搭建的四旋翼强化学习项目，用于复现并扩展官方任务 `Isaac-Quadcopter-Direct-v0`。当前阶段的目标不是马上改复杂任务，而是先保证外部项目迁移后的训练环境足够干净、可复现，再在此基础上做航点任务扩展。

本轮迁移和调试得到的关键结论是：

> External project 本身可以稳定复现官方四旋翼悬停效果。前期出现“训练后到目标附近盘旋、不够稳”的主要原因不是 reward、指标统计或 External project 机制，而是训练设置和 checkpoint 使用不一致，尤其是 `num_envs=1024` 与官方风格的 `num_envs=4096` 带来的 PPO batch 结构差异。

## 当前任务结构

### 1. OfficialClone 官方复刻基线

任务 ID：

```bash
Isaac-Quadcopter-OfficialClone-Direct-v0
```

代码位置：

```bash
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_official_clone/
```

用途：

- 完整复刻 Isaac Lab 官方 `Isaac-Quadcopter-Direct-v0` 的环境逻辑。
- 只修改 Python 类名和 Gym 任务 ID，使其可以作为 external project 中的独立任务存在。
- 后续所有改动都应以这个任务作为干净基线。

该任务保留官方设定：

```text
episode_length_s = 10.0
scene.num_envs = 4096
scene.env_spacing = 2.5
goal x/y range = [-2.0, 2.0]
goal z range = [0.5, 1.5]
height termination = z < 0.1 or z > 2.0
reward = 线速度惩罚 + 角速度惩罚 + 目标距离奖励
```

### 2. WaypointV1 稳定起点版本

任务 ID：

```bash
Isaac-Quadcopter-WaypointV1-Direct-v0
```

代码位置：

```bash
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_v1_metrics/
```

用途：

- 训练环境继续复用干净的 `OfficialClone`。
- 使用独立的 rl_games 实验名：`quadcopter_waypoint_v1`。
- 避免 v1 checkpoint 与官方 `quadcopter_direct` checkpoint 混在同一个目录里。

当前阶段的 v1 **不在训练环境内部添加指标统计**。前期调试发现，即使是 evaluation-only 的 tensor 运算，只要放进 RL 环境的 `_get_rewards()` 或 `_reset_idx()`，也可能让训练轨迹发生变化，使结果不再和官方基线一致。因此后续指标统计应该放到单独的 evaluation 脚本中，而不是插入训练环境。

### 3. 已废弃的旧实验任务

早期在 `quadrotor_waypoint/` 下实现过一个实验版本，包含扩大目标范围、环境内 success_rate / stable_hover_rate 指标等改动。该版本在排查问题时容易和官方复刻任务、v1 checkpoint 混淆，因此已经废弃。

旧实验目录已加入 `.gitignore`，不应作为后续开发基础。

## 仓库结构

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

## 环境准备

先激活 Isaac Lab 环境：

```bash
conda activate env_isaaclab
```

进入项目根目录，并以 editable 模式安装 external project：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
python -m pip install -e source/quadcopter_waypoint
```

## 训练命令

### 训练 OfficialClone 基线

用于确认 external project 是否能复现官方悬停任务：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=200
```

### 训练 WaypointV1

这是当前稳定 v1 起点，使用独立日志目录：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=200
```

预期日志目录：

```text
logs/rl_games/quadcopter_waypoint_v1/<timestamp>/
```

## 播放 checkpoint

播放时一定要使用明确的 checkpoint 路径。不要在多个任务日志目录中使用 `find ... | sort | tail -n 1` 这类模糊方式，因为不同任务可能生成类似名称的 checkpoint，容易误加载。

### 播放 OfficialClone

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_direct/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_direct.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

### 播放 WaypointV1

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v1/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v1.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

## 独立评估指标

为了避免指标统计影响训练轨迹，当前项目把评估逻辑放在独立脚本中：

```bash
scripts/rl_games/eval_metrics.py
```

示例：评估最新 WaypointV1 checkpoint：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v1/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v1.pth"

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --checkpoint "$CKPT" \
  --num_envs=64 \
  --episodes=256 \
  --headless
```

如果需要保存逐 episode 结果：

```bash
python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --checkpoint "$CKPT" \
  --num_envs=64 \
  --episodes=256 \
  --csv "$LATEST_RUN/eval_metrics.csv" \
  --headless
```

该脚本会输出：

```text
success_rate
strict_success_rate
stable_hover_rate
final_stable_hover_rate
termination_rate
timeout_rate
mean_final_distance
mean_min_distance
mean_final_lin_vel
mean_final_ang_vel
```

默认阈值：

```text
success_radius = 0.5 m
strict_success_radius = 0.2 m
stable_radius = 0.3 m
stable_lin_vel = 0.25 m/s
stable_ang_vel = 0.8 rad/s
```

## TensorBoard 查看

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games
```

稳定 v1 重点看：

```text
logs/rl_games/quadcopter_waypoint_v1
```

官方复刻基线重点看：

```text
logs/rl_games/quadcopter_direct
```

## 关键调试记录

### 1. 当前 PPO 配置下建议固定 `num_envs=4096`

官方 PPO 配置中：

```yaml
horizon_length: 24
minibatch_size: 24576
```

当 `num_envs=1024` 时：

```text
batch size = 24 * 1024 = 24576
```

此时每次 PPO update 只有 1 个 minibatch。实际测试中，这个设置训练出的策略容易出现到目标附近盘旋、不够稳的现象。

当 `num_envs=4096` 时：

```text
batch size = 24 * 4096 = 98304
```

此时每次 PPO update 可以分成 4 个 minibatch，复现了官方稳定悬停效果。

如果后续因为显存限制必须降低 `num_envs`，不能只改 `num_envs`，还需要同步调整：

```text
minibatch_size
horizon_length
mini_epochs
learning_rate
```

### 2. 不要混用 task 和 checkpoint

调试过程中出现过“训练其实没问题，但播放效果不对”的情况，原因是不同任务共享或混用了 checkpoint 目录。后续必须显式匹配：

```text
OfficialClone task  + quadcopter_direct checkpoint
WaypointV1 task     + quadcopter_waypoint_v1 checkpoint
```

### 3. 训练环境保持干净，指标单独评估

调试中尝试过在训练环境里加入 `success_rate`、`stable_hover_rate`、`final_lin_vel` 等指标统计。虽然这些指标理论上不直接修改 reward、obs 或 done，但在 GPU 并行 RL 训练中，额外 tensor 运算和状态写入仍可能改变训练轨迹。

当前策略是：

```text
训练环境：保持和 OfficialClone 一致
评估指标：后续单独写 evaluation 脚本统计
```

这样更利于复现实验结果。

## 迁移过程总结

1. 采用 Isaac Lab external project 方式，而不是直接修改 Isaac Lab 官方源码。
2. 新增 `scripts/rl_games/train.py` 和 `scripts/rl_games/play.py` wrapper，在不破坏 Isaac Sim 启动顺序的前提下注入 external task 注册。
3. 早期尝试了扩大目标范围和环境内指标统计，但训练效果出现盘旋，且日志/checkpoint 容易混乱。
4. 新建 `OfficialClone`，完全复刻官方 `Isaac-Quadcopter-Direct-v0` 环境逻辑。
5. 通过固定 checkpoint 路径和 `num_envs=4096`，确认 external project 可以复现官方稳定悬停效果。
6. 废弃旧实验任务，避免后续继续在不稳定版本上叠加修改。
7. 新建 `WaypointV1`，训练环境仍复用 `OfficialClone`，但使用独立实验名 `quadcopter_waypoint_v1`，避免 checkpoint 混用。

## 后续计划

1. 使用独立 evaluation 脚本对 OfficialClone 和 WaypointV1 的 checkpoint 做量化对比。
2. 在稳定评估基础上实现连续航点：无人机到达一个目标点后，不结束 episode，而是采样下一个目标点。
3. 在官方目标范围内稳定后，再逐步扩大目标范围，做 curriculum 训练。

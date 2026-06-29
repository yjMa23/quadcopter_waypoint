# WaypointV2 连续航点版本

## 任务定位

`quadrotor_waypoint_v2` 是低速精确穿点的连续航点任务。

任务 ID：

```text
Isaac-Quadcopter-WaypointV2-Direct-v0
```

代码位置：

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_waypoint_v2/
```

用途：

- 在官方目标工作空间内实现连续航点任务。
- 当无人机以低速精确进入当前目标后，不结束 episode，而是立即采样下一个目标点。
- 使用“距离进展 + 航点完成”奖励，避免策略停在成功半径外刷分。
- 记录航点数、真实到点距离和到点速度，为后续 landing 任务提供阶段基础。

## 当前参数

```text
waypoint_reach_radius = 0.15 m
waypoint_reach_lin_vel = 0.25 m/s
waypoint_segment_length = [0.75, 2.0] m
goal x/y range = [-2.0, 2.0]
goal z range = [0.5, 1.5]
episode_length_s = 10.0
progress_reward_scale = 10.0
waypoint_completion_reward = 10.0
```

观测维度保持 12 维：

```text
root_lin_vel_b       3
root_ang_vel_b       3
projected_gravity_b  3
desired_pos_b        3
```

## 训练

v2 修改了 reward 和航点切换语义，旧 checkpoint 不应继续使用。修改后必须重新训练。

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-WaypointV2-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=200
```

预期日志目录：

```text
logs/rl_games/quadcopter_waypoint_v2/<timestamp>/
```

## 播放

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v2/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v2.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-WaypointV2-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

## 评估

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v2/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v2.pth"

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-WaypointV2-Direct-v0 \
  --checkpoint "$CKPT" \
  --num_envs=64 \
  --episodes=256 \
  --csv "$LATEST_RUN/eval_metrics.csv" \
  --headless
```

评估脚本会读取环境产生的真实穿点事件，避免目标在 step 内切换导致漏记成功。

输出指标：

```text
waypoint_episode_success_rate
mean_waypoints_per_episode
mean_waypoint_reach_distance
mean_waypoint_reach_lin_vel
termination_rate
timeout_rate
```

## 已验证结果

当前实现已经使用 `seed=42`、`num_envs=4096` 完成 200 epochs 从头训练，并对最佳 checkpoint 使用 64 个并行环境评估了 256 个完整 episode：

| 指标 | 结果 |
| --- | ---: |
| 航点 episode 成功率 | 98.05% |
| 平均每回合航点数 | 8.96 |
| 平均到点距离 | 0.1391 m |
| 平均到点线速度 | 0.2155 m/s |
| 终止率 | 0% |

结果满足 `0.15 m` 到点半径和 `0.25 m/s` 速度约束，也没有再次出现 reward 上升而航点数下降的旧版行为。

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games/quadcopter_waypoint_v2
```

重点标签：

```text
Episode/Episode_Reward/progress_to_goal
Episode/Episode_Reward/waypoint_completion
Episode/Metrics/waypoint_count
Episode/Metrics/waypoint_success_rate
Episode/Metrics/mean_waypoint_reach_distance
Episode/Metrics/mean_waypoint_reach_lin_vel
```

## 调试注意事项

1. 不要混用 WaypointV1 / WaypointV2 checkpoint。
2. v2 的成功事件依赖环境中的 `_waypoint_reached`、`_waypoint_reach_distance`、`_waypoint_reach_lin_vel`。
3. 若 reward 上升但航点数下降，优先检查 `progress_to_goal`、`waypoint_completion` 和目标切换逻辑。
4. 后续 ShipLanding 已经独立成包，不建议继续在 WaypointV2 内加入 landing 逻辑。

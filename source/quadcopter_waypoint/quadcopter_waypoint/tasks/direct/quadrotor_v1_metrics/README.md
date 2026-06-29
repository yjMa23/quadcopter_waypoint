# WaypointV1 稳定起点版本

## 任务定位

`quadrotor_v1_metrics` 是官方复刻基线之后的稳定起点版本。该版本继续复用 OfficialClone 的环境逻辑，但使用独立任务 ID 和独立 rl_games 实验名，避免 checkpoint 与官方基线混在一起。

任务 ID：

```text
Isaac-Quadcopter-WaypointV1-Direct-v0
```

代码位置：

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_v1_metrics/
```

用途：

- 保持训练环境干净，验证 independent task / independent checkpoint 的稳定性。
- 作为 WaypointV2 之前的可回退起点。
- 避免早期旧实验中的 success_rate / stable_hover_rate 等环境内指标影响训练轨迹。

## 训练

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

## 播放

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v1/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v1.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

## 评估

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_waypoint_v1/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_waypoint_v1.pth"

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-WaypointV1-Direct-v0 \
  --checkpoint "$CKPT" \
  --num_envs=64 \
  --episodes=256 \
  --csv "$LATEST_RUN/eval_metrics.csv" \
  --headless
```

评估脚本会输出固定目标类任务的指标：

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

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games/quadcopter_waypoint_v1
```

## 调试注意事项

1. WaypointV1 的作用是稳定起点，不建议继续叠加复杂任务逻辑。
2. 若要开发连续航点，应进入 `quadrotor_waypoint_v2/`，不要在 V1 上直接改。
3. 播放和评估时必须使用 `quadcopter_waypoint_v1` 目录下的 checkpoint。

# OfficialClone 官方复刻基线

## 任务定位

`quadrotor_official_clone` 是 Isaac Lab 官方四旋翼任务 `Isaac-Quadcopter-Direct-v0` 的 external project 复刻版本。

任务 ID：

```text
Isaac-Quadcopter-OfficialClone-Direct-v0
```

代码位置：

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_official_clone/
```

用途：

- 验证 external project 能否稳定复现官方四旋翼悬停任务。
- 作为后续所有任务改动的干净基线。
- 不在训练环境中额外插入复杂指标统计，避免影响训练轨迹。

## 核心设定

```text
episode_length_s = 10.0
scene.num_envs = 4096
scene.env_spacing = 2.5
goal x/y range = [-2.0, 2.0]
goal z range = [0.5, 1.5]
height termination = z < 0.1 or z > 2.0
reward = 线速度惩罚 + 角速度惩罚 + 目标距离奖励
```

## 训练

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=200
```

推荐保持 `num_envs=4096`。该配置可以复现官方四旋翼悬停效果。

## 播放

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_direct/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_direct.pth"

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --num_envs=1 \
  --checkpoint "$CKPT"
```

## 评估

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

LATEST_RUN=$(ls -td logs/rl_games/quadcopter_direct/2026-* | head -n 1)
CKPT="$LATEST_RUN/nn/quadcopter_direct.pth"

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-OfficialClone-Direct-v0 \
  --checkpoint "$CKPT" \
  --num_envs=64 \
  --episodes=256 \
  --csv "$LATEST_RUN/eval_metrics.csv" \
  --headless
```

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games/quadcopter_direct
```

重点观察：

```text
rewards/iter
Episode_Termination/died
Episode_Termination/time_out
```

## 调试注意事项

1. 不要把 OfficialClone 的 checkpoint 与 WaypointV1 / WaypointV2 / ShipLanding 混用。
2. 不建议在该环境中加入额外指标统计。需要指标时用 `scripts/rl_games/eval_metrics.py` 独立评估。
3. 如果修改该任务后无法复现官方悬停，应先回退该包，再排查 external project 注册和 PPO 配置。

# rl_games Wrapper Scripts

该目录保存本 external project 使用的 rl_games 启动脚本。

代码位置：

```text
scripts/rl_games/
```

文件：

```text
train.py         训练 wrapper
play.py          播放 wrapper
eval_metrics.py  独立评估脚本
```

这些脚本会在 Isaac Lab 官方脚本中注入：

```python
import quadcopter_waypoint.tasks
```

从而注册 external project 中的自定义任务。这样可以避免直接修改 Isaac Lab 官方源码。

## 训练模板

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=<TASK_ID> \
  --num_envs=<N> \
  --headless \
  --max_iterations=<ITER>
```

继续训练模板：

```bash
python scripts/rl_games/train.py \
  --task=<TASK_ID> \
  --num_envs=<N> \
  --headless \
  --max_iterations=<ITER> \
  --checkpoint <CHECKPOINT_PATH>
```

注意：使用 `--checkpoint` 时，rl_games 会继承 checkpoint 内部 epoch 计数，因此 `--max_iterations` 应大于 checkpoint 已训练 epoch。

## 播放模板

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/play.py \
  --task=<TASK_ID> \
  --num_envs=1 \
  --checkpoint <CHECKPOINT_PATH>
```

播放时务必确保 task 与 checkpoint 对应，不要混用不同任务的模型。

## 评估模板

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=<TASK_ID> \
  --checkpoint <CHECKPOINT_PATH> \
  --num_envs=64 \
  --episodes=256 \
  --csv <OUTPUT_CSV> \
  --headless
```

## 支持的评估类型

### 固定目标类任务

适用于 OfficialClone 和 WaypointV1。

输出示例：

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

### 连续航点类任务

适用于 WaypointV2。

脚本会读取环境事件：

```text
_waypoint_reached
_waypoint_reach_distance
_waypoint_reach_lin_vel
```

输出示例：

```text
waypoint_episode_success_rate
mean_waypoints_per_episode
mean_waypoint_reach_distance
mean_waypoint_reach_lin_vel
termination_rate
timeout_rate
```

### 降落类任务

适用于 ShipLanding。

脚本会读取环境事件：

```text
_landing_success
_landing_touchdown_distance
_landing_touchdown_rel_vel
_crash
```

输出示例：

```text
align_success_rate
landing_success_rate
mean_final_distance
mean_min_distance
mean_touchdown_distance
mean_touchdown_rel_vel
crash_rate
termination_rate
timeout_rate
```

## 常用任务 ID

```text
Isaac-Quadcopter-OfficialClone-Direct-v0
Isaac-Quadcopter-WaypointV1-Direct-v0
Isaac-Quadcopter-WaypointV2-Direct-v0
Isaac-Quadcopter-ShipLanding-Direct-v0
```

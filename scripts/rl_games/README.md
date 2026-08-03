# rl_games Wrapper Scripts

该目录保存 external project 使用的 rl_games 启动与评估 wrapper。脚本按 Isaac Lab 官方启动顺序注入：

```python
import quadcopter_waypoint.tasks
```

不修改 Isaac Lab 官方源码。

## 文件

```text
train.py                    训练 wrapper
play.py                     播放 wrapper
eval_metrics.py             独立评估与逐 episode CSV
eval_metrics_utils.py       可脱离 Isaac Sim 测试的统计辅助逻辑
```

## 训练

```bash
python scripts/rl_games/train.py \
  --task=<TASK_ID> \
  --num_envs=<N> \
  --headless \
  --max_iterations=<ITER>
```

继续训练：

```bash
python scripts/rl_games/train.py \
  --task=<TASK_ID> \
  --num_envs=<N> \
  --headless \
  --max_iterations=<ITER> \
  --checkpoint <CHECKPOINT>
```

rl_games 会继承 checkpoint 内 epoch 计数，因此 `--max_iterations` 必须大于 checkpoint epoch。

当前 PPO 配置：

```text
horizon_length = 24
minibatch_size = 384
```

小规模 PPO 冒烟必须满足：

```text
(num_envs × 24) % 384 == 0
```

最小合法环境数为 16，不要使用 4 环境。

## 播放

```bash
python scripts/rl_games/play.py \
  --task=<TASK_ID> \
  --num_envs=1 \
  --checkpoint <CHECKPOINT>
```

任务 ID、网络结构、观测维度和 checkpoint 必须对应。

## 评估

```bash
python scripts/rl_games/eval_metrics.py \
  --task=<TASK_ID> \
  --checkpoint <CHECKPOINT> \
  --num_envs=64 \
  --episodes=256 \
  --seed=42 \
  --csv <OUTPUT_CSV> \
  --headless
```

正式结果至少运行 `seed=42,43,44`，不得只报告最好 seed。

## Terminal 状态锁存

`DirectRLEnv.step()` 在返回前自动 reset 已结束环境。底层状态 Tensor 是 live reference，reset 后会被覆盖；因此不能在 `env.step()` 返回后直接读取 robot state 作为 terminal 指标，也不能把前一帧状态命名为 terminal。

ShipLanding 环境在 `_reset_idx()` 覆盖状态前锁存：

```text
terminal robot position
terminal robot linear velocity
terminal robot angular velocity
terminal pad position
terminal pad velocity
terminal relative velocity
terminal horizontal error
terminal surface clearance
termination / timeout flags
```

`eval_metrics.py` 对 ShipLanding 强制读取该锁存；缺少有效 latch 时直接报错。

字段语义：

```text
final_vertical_speed
  terminal robot world-z velocity

terminal_vertical_relative_speed
  terminal robot world-z velocity - terminal pad world-z velocity

final_horizontal_speed
  terminal robot/pad relative xy speed

final_distance
  terminal robot/pad 3D distance

touchdown_*
  只对 success episode 汇总
```

空 success 集合返回 `NaN`，不输出误导性的 0，也不会除零。

## 支持的任务类型

### 固定目标

OfficialClone、WaypointV1：

```text
success_rate
strict_success_rate
stable_hover_rate
final_stable_hover_rate
mean_final_distance
mean_min_distance
mean_final_lin_vel
mean_final_ang_vel
termination_rate
timeout_rate
```

### 连续航点

WaypointV2：

```text
waypoint_episode_success_rate
mean_waypoints_per_episode
mean_waypoint_reach_distance
mean_waypoint_reach_lin_vel
termination_rate
timeout_rate
```

### ShipLanding marker / proxy

Phase 5D、Phase 6A：

```text
align_success_rate
landing_success_rate
touchdown distance mean/P50/P90/P95
touchdown relative speed
terminal world/relative speed
landing time
descent speed
pad speed buckets
crash_rate
timeout_rate
```

### PhysicalDeck

Phase 6B 额外输出：

```text
contact_success_rate
settled_landing_rate
hard_contact_rate
ground_crash_rate
deck_miss_rate
first_contact_xy_error_deck_frame
first_contact_normal_rel_speed
first_contact_tangential_rel_speed
first_contact_force
max_contact_force
settle_time
terminal_xy_error
minimum_surface_clearance
maximum_penetration
```

成功 first-contact P95 只在成功 episode 上统计；同时输出 all-contact P95，避免隐藏失败接触分布。

## 纯 Python 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：

```text
live Tensor 在 reset 后变化
terminal latch 优先级
pad speed bucket 边界
成功 episode 独占 touchdown 汇总
空成功集合 NaN 语义
```

## 常用任务 ID

```text
Isaac-Quadcopter-OfficialClone-Direct-v0
Isaac-Quadcopter-WaypointV1-Direct-v0
Isaac-Quadcopter-WaypointV2-Direct-v0
Isaac-Quadcopter-ShipLanding-Direct-v0
Isaac-Quadcopter-ShipLanding-Heave-Direct-v0
Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0
```

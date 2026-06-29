# ShipLanding 静态平台降落阶段

## 任务定位

`quadrotor_ship_landing` 是“面向移动船舶的无人机自主降落”的第一阶段任务。当前阶段只做 **静态 landing pad 降落**，不引入移动平台和正弦船舶运动。

任务 ID：

```text
Isaac-Quadcopter-ShipLanding-Direct-v0
```

代码位置：

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/
```

实验名：

```text
quadcopter_ship_landing
```

当前目标：

```text
先飞到 landing pad 上方 → 对准 → 缓慢下降 → 接近板面后终止
```

## 当前 observation

当前观测维度为 16：

```text
0-2   root_lin_vel_b
3-5   root_ang_vel_b
6-8   projected_gravity_b
9-11  pad_rel_pos_b
12-14 pad_rel_vel_w
15    align_success
```

其中：

- `pad_rel_pos_b`：landing pad 相对于机体系的位置。
- `pad_rel_vel_w`：landing pad 相对无人机的世界系速度差。
- `align_success`：是否已经完成对准阶段。该标志用于告诉策略是否可以进入下降阶段。

## 两阶段任务逻辑

### 1. Align 阶段

无人机需要先到达 pad 上方安全高度，并保持水平误差、水平速度和姿态满足条件。

当前参数：

```text
align_radius = 0.20 m
align_height_min = 0.55 m
align_height_max = 0.95 m
align_max_horizontal_speed = 0.25 m/s
align_upright = 0.92
align_hold_steps = 15
```

当满足 `align_candidate` 并持续 `align_hold_steps` 后：

```text
align_success = True
can_land = True
```

### 2. Landing 阶段

只有完成 `align_success` 后，才允许触发 landing success。

当前参数：

```text
landing_success_radius = 0.16 m
landing_success_height = 0.10 m
landing_success_rel_vel = 0.30 m/s
landing_success_ang_vel = 0.9 rad/s
landing_success_upright = 0.93
landing_success_hold_steps = 4

landing_target_height = 0.08 m
descent_speed_limit = 0.22 m/s
```

该参数组是在实验中折中得到的：

- `landing_success_height = 0.16` 会导致视觉上还未接触板面就提前终止。
- `landing_success_height = 0.08` 从零训练容易失败并产生 crash。
- 当前 `0.10` 搭配已有静态成功 checkpoint 微调，可以稳定接近板面并保持 100% 成功率。

## Reward 设计

当前 reward 采用“先对准、再下降”的结构：

```text
lin_vel：线速度惩罚
ang_vel：角速度惩罚
progress_to_pad：水平接近 pad 的进展奖励
height_tracking：阶段性高度目标跟踪
rel_vel：相对速度惩罚
tilt：姿态倾斜惩罚
descent_vel：下降速度超限惩罚
align_bonus：满足 align_candidate 的奖励
align_hold：完成 align_success 后的保持奖励
post_align_descent：只在 can_land 后奖励继续下降
landing_bonus：满足 landing_success 后给终止奖励
crash_penalty：飞出工作空间或高度异常时惩罚
```

其中最关键的是：

```text
post_align_descent 只在 can_land 后生效
```

这样可以避免早期策略直接砸向 pad 刷 landing bonus，同时又解决“只对准不下降直到 timeout”的问题。

## 训练

### 从头训练

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --num_envs=1024 \
  --headless \
  --max_iterations=200
```

### 基于稳定 checkpoint 微调

当前推荐从稳定静态降落 checkpoint 继续微调：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --num_envs=1024 \
  --headless \
  --max_iterations=350 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

注意：`--checkpoint` 继续训练时，rl_games 会继承 checkpoint 内部 epoch 计数。因此 `--max_iterations` 需要大于 checkpoint 已训练 epoch，否则可能只跑 1 个 epoch 就结束。

## 播放

当前稳定 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

播放命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --num_envs=1 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

视觉检查重点：

```text
1. 是否先飞到 landing pad 上方。
2. 是否完成短暂对准后再下降。
3. 是否在接近板面时终止，而不是明显悬空提前结束。
4. 是否没有重新出现直接砸向板面的行为。
```

## 评估

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/eval_metrics.csv \
  --headless
```

评估指标：

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

## 已验证结果

当前稳定 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

256 个 episode 评估结果：

| 指标 | 结果 |
| --- | ---: |
| align success rate | 100% |
| landing success rate | 100% |
| crash rate | 0% |
| timeout rate | 0% |
| final distance mean | 0.0993 m |
| touchdown distance mean | 0.0212 m |
| touchdown distance P95 | 0.0304 m |
| touchdown rel vel mean | 0.0518 m/s |
| touchdown rel vel P95 | 0.0580 m/s |

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games/quadcopter_ship_landing
```

重点标签：

```text
Episode/Metrics/align_success_rate
Episode/Metrics/landing_success_rate
Episode/Metrics/final_distance_to_pad
Episode/Metrics/mean_touchdown_distance
Episode/Metrics/mean_touchdown_rel_vel
Episode/Episode_Termination/landing_success
Episode/Episode_Termination/crash
Episode/Episode_Termination/time_out
Episode/Episode_Reward/post_align_descent
Episode/Episode_Reward/landing_bonus
```

## 调试记录

### 问题 1：直接砸向 pad

早期 reward 中 `landing_bonus` 较大，且 success 只依赖最终状态，策略学会了快速俯冲触发 success。

解决：加入 align gate，要求先在 pad 上方完成对准，再允许进入 landing success。

### 问题 2：只对准不下降

加入 align gate 后，策略能对准但不愿继续下降，导致 timeout。

解决：加入 `post_align_descent`，只在 `can_land=True` 后奖励继续下降。

### 问题 3：明显悬空提前终止

`landing_success_height = 0.16` 时，root 接近到 pad 上方约 0.16 m 就会触发 success，视觉上还未接触板面。

解决：将终止高度调为中间值：

```text
landing_success_height = 0.10
landing_target_height = 0.08
```

该设置比 0.16 更接近板面，同时比 0.08 更容易稳定训练。

## 下一步

进入 Phase 5A：低速匀速移动平台。

建议从当前静态降落 checkpoint 继续微调，先设置：

```text
pad_vel_xy ∈ [-0.05, 0.05] m/s
```

不要直接加入正弦船舶运动。推荐顺序：

```text
静态平台 → 低速匀速移动平台 → 中速移动平台 → 正弦船舶运动
```

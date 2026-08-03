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

## Phase 5A：低速匀速移动平台

当前版本已经从静态平台推进到低速匀速移动平台。

实现方式：

```text
pad_velocity_xy_range = 0.05 m/s
pad_vel_x, pad_vel_y ∈ [-0.05, 0.05] m/s
```

每个 episode reset 时随机采样 landing pad 的水平速度，并在每个 RL step 中更新：

```text
pad_pos_xy += pad_vel_xy * step_dt
```

为了避免移动平台在 10 s episode 内过早接近工作空间边界，初始 pad 位置从静态阶段的 `[-1.0, 1.0] m` 收紧为：

```text
pad initial x/y ∈ [-0.8, 0.8] m
```

当前低速移动平台稳定 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_10-34-48/nn/quadcopter_ship_landing.pth
```

该模型基于静态降落 checkpoint 微调得到：

```text
source checkpoint:
logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

训练命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --num_envs=1024 \
  --headless \
  --max_iterations=350 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

评估命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-29_10-34-48/nn/quadcopter_ship_landing.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-29_10-34-48/eval_metrics.csv \
  --headless
```

256 个 episode 评估结果：

| 指标 | 结果 |
| --- | ---: |
| align success rate | 100% |
| landing success rate | 100% |
| crash rate | 0% |
| timeout rate | 0% |
| final distance mean | 0.1102 m |
| touchdown distance mean | 0.0495 m |
| touchdown distance P95 | 0.0759 m |
| touchdown rel vel mean | 0.0560 m/s |
| touchdown rel vel P95 | 0.0679 m/s |

## Phase 5B：中速匀速移动平台

当前版本已经将平台速度提高到：

```text
pad_velocity_xy_range = 0.10 m/s
pad_vel_x, pad_vel_y ∈ [-0.10, 0.10] m/s
```

为了让高速一些的平台仍能尽量落在 pad 中心区域，当前 reward 在 Phase 5A 基础上增强了水平对准项：

```text
progress_reward_scale = 5.0
horizontal_error_reward_scale = -2.5
rel_vel_reward_scale = -0.6
landing_bonus = 40.0
```

同时保留较稳的终止条件：

```text
landing_success_radius = 0.16 m
landing_success_height = 0.10 m
landing_success_rel_vel = 0.30 m/s
```

当前 Phase 5B 稳定 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_10-59-53/nn/quadcopter_ship_landing.pth
```

训练来源 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_10-40-58/nn/quadcopter_ship_landing.pth
```

评估命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-29_10-59-53/nn/quadcopter_ship_landing.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-29_10-59-53/eval_metrics.csv \
  --headless
```

256 个 episode 评估结果：

| 指标 | 结果 |
| --- | ---: |
| align success rate | 100% |
| landing success rate | 100% |
| crash rate | 0% |
| timeout rate | 0% |
| final distance mean | 0.1243 m |
| touchdown distance mean | 0.0816 m |
| touchdown distance P95 | 0.1347 m |
| touchdown rel vel mean | 0.1984 m/s |
| touchdown rel vel P95 | 0.2483 m/s |

该版本相比 Phase 5A 落点误差变大，但能在 `±0.10 m/s` 平台速度下保持 100% 成功率和 0 crash。相比第一版 Phase 5B，落点均值从约 `0.110 m` 降到 `0.082 m`，但触地相对速度升高。因此后续继续加速前，建议先补充更细的速度质量指标。

下一步建议先做评估脚本增强，而不是直接继续加速：

```text
landing_time
max_descent_speed
mean_descent_speed
final_vertical_speed
mean_horizontal_speed
```

## Phase 5C：更快匀速移动平台

当前版本已经将平台速度继续提高到：

```text
pad_velocity_xy_range = 0.20 m/s
pad_vel_x, pad_vel_y ∈ [-0.20, 0.20] m/s
```

为了避免平台在 10 s episode 内过早接近工作空间边界，初始 pad 位置进一步收紧为：

```text
pad initial x/y ∈ [-0.5, 0.5] m
```

当前 Phase 5C 推荐 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

该 checkpoint 相比同目录下的 `quadcopter_ship_landing.pth` 更稳，评估中没有 crash 和 timeout。

评估命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/eval_metrics_phase5c_plus.csv \
  --headless
```

256 个 episode 评估结果：

| 指标 | 结果 |
| --- | ---: |
| align success rate | 100% |
| landing success rate | 100% |
| crash rate | 0% |
| timeout rate | 0% |
| final distance mean | 0.1358 m |
| touchdown distance mean | 0.1002 m |
| touchdown distance P95 | 0.1533 m |
| touchdown rel vel mean | 0.1453 m/s |
| touchdown rel vel P95 | 0.2646 m/s |

该版本能明显看到平台移动速度提升，并且仍能保持 100% 成功率。代价是落点误差相较 Phase 5B 进一步变大，P95 接近当前 landing success 半径上限。

## Phase 5C+：增强评估指标

`eval_metrics.py` 已增强 ShipLanding 的行为质量评估，新增逐 episode 指标：

```text
landing_time
max_descent_speed
mean_descent_speed
final_vertical_speed
max_horizontal_speed
mean_horizontal_speed
final_horizontal_speed
pad_speed
pad_speed_bucket
```

Phase 5C checkpoint 的增强评估结果：

| 指标 | 结果 |
| --- | ---: |
| landing success rate | 100% |
| crash rate | 0% |
| timeout rate | 0% |
| mean landing time | 3.6684 s |
| mean max descent speed | 0.8263 m/s |
| mean descent speed | 0.2769 m/s |
| mean final vertical speed | 0.0000 m/s |
| mean max horizontal speed | 0.7820 m/s |
| mean horizontal speed | 0.2427 m/s |
| mean final horizontal speed | 0.1042 m/s |
| mean pad speed | 0.1618 m/s |

按 pad 速度分桶的结果：

| pad speed bucket | episodes | success rate | touchdown distance mean | touchdown rel vel mean |
| --- | ---: | ---: | ---: | ---: |
| 0.00-0.05 m/s | 7 | 100% | 0.0544 m | 0.1280 m/s |
| 0.05-0.10 m/s | 23 | 100% | 0.0869 m | 0.1521 m/s |
| 0.10-0.15 m/s | 64 | 100% | 0.0908 m | 0.1367 m/s |
| >=0.15 m/s | 162 | 100% | 0.1079 m | 0.1484 m/s |

结论：

```text
1. 高速 pad 分桶仍保持 100% 成功率。
2. pad 速度越高，touchdown distance 整体变大。
3. 当前策略的最终触地相对速度可接受，但中途 max_descent_speed 偏高，说明下降过程还不够平滑。
```

## Phase 5C-Smooth：已尝试但不采用

在进入正弦船舶运动前，曾尝试保持 `pad_velocity_xy_range = 0.20 m/s` 不变，只优化下降过程的平滑性。

本阶段尝试前已备份 Phase 5C+ 状态：

```text
backups/phase5c_plus/README.md
```

Smooth v2 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_12-12-33/nn/quadcopter_ship_landing.pth
```

Smooth v2 评估结果：

| 指标 | Phase 5C+ | Smooth v2 |
| --- | ---: | ---: |
| landing success rate | 100% | 100% |
| crash rate | 0% | 0% |
| timeout rate | 0% | 0% |
| mean touchdown distance | 0.1002 m | 0.1229 m |
| touchdown distance P95 | 0.1533 m | 0.1557 m |
| mean touchdown rel vel | 0.1453 m/s | 0.2002 m/s |
| touchdown rel vel P95 | 0.2646 m/s | 0.2723 m/s |
| mean landing time | 3.6684 s | 4.3842 s |
| mean max descent speed | 0.8263 m/s | 0.6514 m/s |
| max descent speed P95 | 1.0726 m/s | 0.9009 m/s |
| mean descent speed | 0.2769 m/s | 0.2264 m/s |

结论：

```text
Smooth v2 虽然降低了下降速度峰值，但 GUI 观察效果不理想，并且落点误差、触地相对速度和 landing time 均变差。因此当前不采用 Smooth v2，代码和主线 checkpoint 已回退到 Phase 5C+。
```

当前主线继续使用 Phase 5C+ checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

## Phase 5C-Contact：更接近板面的终止条件

为解决 GUI 中无人机仍与平台存在轻微距离就结束的问题，本阶段不进入正弦船舶运动，而是继续优化 landing success 的高度条件。

原 Phase 5C+ 终止条件：

```text
landing_success_height = 0.10
landing_target_height = 0.08
landing_success_hold_steps = 4
```

Phase 5C-Contact v1 曾尝试：

```text
landing_success_height = 0.07
landing_target_height = 0.055
landing_success_hold_steps = 5
```

该版本终止高度更低，但成功率下降到约 `95.7%`，并出现少量 crash / timeout，因此不采用。

当前采用 Contact v4 终止条件：

```text
landing_success_height = 0.070
landing_target_height = 0.055
landing_success_hold_steps = 4
landing_success_rel_vel = 0.32
```

注意：Contact v4 的最佳组合不是重新训练得到的 checkpoint，而是：

```text
Contact v4 代码条件 + 原 Phase 5C+ checkpoint
```

推荐 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

评估命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/eval_metrics_phase5c_checkpoint_contact_v4_env.csv \
  --headless
```

256 个 episode 评估结果：

| 指标 | Phase 5C+ 原条件 | Contact v4 条件 + Phase 5C+ checkpoint |
| --- | ---: | ---: |
| landing success rate | 100% | 100% |
| crash rate | 0% | 0% |
| timeout rate | 0% | 0% |
| mean touchdown distance | 0.1002 m | 0.0927 m |
| mean touchdown rel vel | 0.1453 m/s | 0.1270 m/s |
| mean landing time | 3.6684 s | 4.0807 s |
| estimated terminal height error mean | 0.0862 m | 0.0613 m |
| estimated terminal height error P95 | 0.0981 m | 0.0693 m |

结论：

```text
1. Contact v4 能把终止高度从约 8.6 cm 压低到约 6.1 cm。
2. 成功率仍保持 100%，没有 crash 和 timeout。
3. 不建议使用重新训练的 Contact checkpoint，因为重新训练版本成功率下降明显。
4. Contact v4 继续作为当前贴近平台的高度终止条件。
```

## Phase 5C-Track：下降阶段持续横向跟踪

针对 GUI 中观察到的现象：无人机一开始能对齐移动平台，但下降过程中没有继续匹配平台横向速度，导致越降越偏。本阶段不进入正弦船舶运动，而是在 Phase 5C-Contact 基础上增加下降阶段横向跟踪约束。

当前新增参数：

```text
landing_success_horizontal_rel_vel = 0.16

descent_horizontal_rel_vel_reward_scale = -3.0
near_pad_track_height = 0.45
near_pad_horizontal_rel_vel_reward_scale = -5.0

expected_descent_speed = 0.25
max_prediction_time = 1.5
predicted_pad_error_reward_scale = -2.0
```

新增逻辑：

```text
1. landing success 中额外要求水平相对速度足够小。
2. can_land 之后惩罚无人机与 pad 的水平相对速度。
3. 越接近 pad，越强制水平速度与 pad 同步。
4. 下降过程中追踪预测 pad 位置，而不是只追当前 pad 位置。
```

注意：Track v1 的最佳组合不是重新训练出来的 checkpoint，而是：

```text
Track v1 代码条件 + 原 Phase 5C+ checkpoint
```

推荐 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth
```

评估命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/nn/last_quadcopter_ship_landing_ep_600_rew_41.586376.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-29_11-16-09/eval_metrics_phase5c_checkpoint_track_v1_env.csv \
  --headless
```

256 个 episode 评估结果：

| 指标 | Contact v4 条件 | Track v1 条件 + Phase 5C+ checkpoint |
| --- | ---: | ---: |
| landing success rate | 100% | 100% |
| crash rate | 0% | 0% |
| timeout rate | 0% | 0% |
| mean touchdown distance | 0.0927 m | 0.0813 m |
| mean touchdown rel vel | 0.1270 m/s | 0.1084 m/s |
| mean final horizontal speed | 0.1076 m/s | 0.0902 m/s |
| high-speed pad bucket touchdown distance | 0.1008 m | 0.0859 m |
| mean landing time | 4.0807 s | 4.3578 s |
| estimated terminal height error mean | 0.0613 m | 0.0553 m |
| estimated terminal height error P95 | 0.0693 m | 0.0691 m |

重新训练得到的 Track v1 checkpoint 评估结果不如原 checkpoint：成功率约 `98.8%`，且有少量 crash / timeout，因此当前不采用重新训练 checkpoint。

当前结论：

```text
1. Track v1 约束能降低下降末端的水平相对速度。
2. 高速 pad 分桶下的落点误差从 0.1008 m 降到 0.0859 m。
3. 无人机在下降过程中会更倾向于继续带着 pad 的横向速度走。
4. 当前主线：Track v1 代码条件 + 原 Phase 5C+ checkpoint。
```

## Deck proxy 尝试记录：接触式判定不能直接替换成功条件

为了给后续平台横滚和起伏做准备，尝试过将 landing success 从 root 高度阈值改为“机体底部到平台顶面”的接触代理判定：

```text
pad_surface_height = pad_z + pad_thickness / 2
robot_bottom_height = root_z - robot_landing_surface_offset
landing_surface_clearance = robot_bottom_height - pad_surface_height
```

该计算已经保留在代码中，后续可以继续用于奖励、评估或真正接触式训练。

但当前直接用它替换 landing success 会导致成功率下降：

| 版本 | 主要设置 | landing success rate | 现象 |
| --- | --- | ---: | --- |
| Deck proxy v1 | clearance < 0.012 m, lookahead 0.50 s | 86.7% | 更贴近、横向速度更低，但 timeout 明显增加 |
| Deck proxy v2 | clearance < 0.025 m, lookahead 0.50 s | 94.1% | 成功率仍不够稳定 |
| Deck proxy v3 | clearance < 0.035 m, lookahead 0.35 s | 89.5% | 不适合作为主线 |

结论：

```text
1. 仅靠旧 Phase 5C+ checkpoint 无法稳定满足真正接触代理条件。
2. Deck proxy 作为成功终止条件需要重新训练新策略，而不是直接套旧 checkpoint。
3. 当前代码主线仍使用稳定的 Track v1 判定；landing_surface_clearance 已保留，后续用于 Phase 5D/Deck-Contact 训练。
4. 如果下一阶段加入横滚/起伏，必须把 pad 从 marker 升级为具有姿态和表面法向的 deck 状态，最好进一步升级为带碰撞的实体平台。
```

## Phase 5D-DeckContact：接触代理降落

本阶段正式将 landing success 从 root 高度阈值升级为接触代理判定：

```text
pad_surface_height = pad_z + pad_thickness / 2
robot_bottom_height = root_z - robot_landing_surface_offset
landing_surface_clearance = robot_bottom_height - pad_surface_height
```

当前 DeckContact v4 参数：

```text
align_radius = 0.25
align_max_horizontal_speed = 0.30
align_hold_steps = 8

landing_contact_clearance = 0.060
max_landing_surface_penetration = 0.010
landing_contact_target_clearance = 0.005
contact_clearance_reward_scale = -8.0
```

成功判定改为：

```text
horizontal_error < landing_success_radius
landing_surface_clearance < landing_contact_clearance
landing_surface_clearance > -max_landing_surface_penetration
rel_vel < landing_success_rel_vel
horizontal_speed < landing_success_horizontal_rel_vel
ang_vel_norm < landing_success_ang_vel
upright > landing_success_upright
```

当前推荐 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

评估命令：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth \
  --num_envs=64 \
  --episodes=256 \
  --csv logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/eval_metrics_deck_contact_v4_ep650.csv \
  --headless
```

256 个 episode 评估结果：

| 指标 | DeckContact v4 |
| --- | ---: |
| landing success rate | 99.22% |
| align success rate | 99.22% |
| crash rate | 0% |
| timeout rate | 0.78% |
| mean touchdown distance | 0.0888 m |
| mean touchdown rel vel | 0.1376 m/s |
| mean final horizontal speed | 0.1295 m/s |
| mean landing time | 2.9768 s |
| high-speed pad bucket success rate | 100% |
| high-speed pad bucket touchdown distance | 0.0936 m |
| landing surface clearance mean | 0.0116 m |
| landing surface clearance median | 0.0088 m |
| landing surface clearance P95 | 0.0403 m |

对比尝试：

| 版本 | 结果 |
| --- | --- |
| DeckContact v1 best | 92.58% success，timeout 7.42%，不采用 |
| DeckContact v1 ep850 | 82.03% success，明显退化，不采用 |
| DeckContact v2 ep650 | 98.83% success，接近可用 |
| DeckContact v4 ep650 | 99.22% success，0 crash，当前采用 |

结论：

```text
1. Phase 5D 已经从“几何高度成功”升级为“机体底部接近 deck 顶面成功”。
2. 当前成功率 99.22%，0 crash，已经可作为进入平台起伏前的接触代理基线。
3. 仍有极少数 timeout，后续可通过更长训练或实体 deck/contact sensor 继续消除。
4. 进入 Phase 6 前，建议先保持该 checkpoint，播放 GUI 观察是否确实更像“落到平台上”。
```

### 冻结说明

当前 `quadrotor_ship_landing` 任务目录作为 Phase 5D-DeckContact 基线记录，不再直接在该目录继续迭代下一阶段。

后续如果继续做平台起伏、roll / pitch 或实体 deck，应新建独立任务目录和任务 ID，例如：

```text
quadrotor_ship_landing_heave
Isaac-Quadcopter-ShipLanding-Heave-Direct-v0
```

或者：

```text
quadrotor_ship_landing_deck_motion
Isaac-Quadcopter-ShipLanding-DeckMotion-Direct-v0
```

新任务可以从当前 `quadrotor_ship_landing` 拷贝初始化，但后续修改不再污染 Phase 5D 基线。

### Terminal 指标修复

Phase 5D 环境保留了通用 terminal-state latch，供 Phase 5D、Phase 6A 和后续派生任务使用。Isaac Lab `DirectRLEnv.step()` 会在返回前自动 reset 完成环境，底层 Tensor 是 live reference，reset 后的零值不能作为 terminal 指标。

环境在 reset 覆盖状态前锁存：

```text
robot position / linear velocity / angular velocity
pad position / velocity
robot-pad relative velocity
horizontal error
surface clearance
terminated / timeout
```

`eval_metrics.py` 对 ShipLanding 强制读取锁存状态。字段定义见根 README 和 `scripts/rl_games/README.md`。

### 后续阶段状态

```text
Phase 6A Heave Precision: 已完成并冻结，仍为 marker / contact proxy
Phase 6B PhysicalDeck: 已在独立任务目录完成实体 deck 和真实接触基线
```

因此本目录不再承担 heave、实体 deck 或 roll / pitch 的新增实现。

# rl_games Wrapper Scripts

该目录保存 external project 使用的 rl_games 启动与评估 wrapper。脚本按 Isaac Lab 官方启动顺序注入：

```python
import quadcopter_waypoint.tasks
```

不修改 Isaac Lab 官方源码或安装环境中的 RL-Games 源码。actor-preserving PPO custom agent 通过项目 wrapper 在 `Runner` 构造后注册。

## 文件

```text
train.py                                   训练 wrapper
play.py                                    播放 wrapper
eval_metrics.py                            独立评估与逐 episode CSV
eval_metrics_utils.py                      可脱离 Isaac Sim 测试的统计辅助逻辑
expand_checkpoint_observation.py           16→22 维 rl_games checkpoint 迁移
check_physical_deck_attitude_physics.py     甲板运动与 deck/ground 接触诊断
summarize_physical_deck_attitude.py         physical-deck-attitude task 三种子 benchmark 聚合
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

rl_games 会继承 checkpoint 内 epoch 计数，因此 `--max_iterations` 必须大于 checkpoint epoch。actor-preserving PPO checkpoint 还会恢复嵌入的冻结 BC reference actor、warm-up epoch、optimizer 和 RMS 冻结语义。

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

actor-preserving PPO 训练必须显式选择 separate actor/critic 配置：

```bash
python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --agent=rl_games_actor_preserving_cfg_entry_point \
  --num_envs=256 --seed=42 --headless --max_iterations=200 \
  --checkpoint=logs/imitation/actor_preserving_ppo/bc_init_separate_formal_lambda50.pth \
  agent.params.config.name=actor_preserving_formal_lambda50 \
  +agent.params.config.full_experiment_name=seed42
```

actor-preserving PPO 的 epoch 1–10 为 critic-only warm-up；actor、fixed sigma、observation RMS 和 adaptive LR scheduler 均冻结。epoch 11 从基础 `1e-4` learning rate 开始 actor 更新，并加入预注册的 BC mean-action L2 anchor。

## 播放

```bash
python scripts/rl_games/play.py \
  --task=<TASK_ID> \
  --num_envs=1 \
  --checkpoint <CHECKPOINT>
```

任务 ID、网络结构、观测维度和 checkpoint 必须对应。actor-preserving PPO separate checkpoint 播放也必须显式添加 `--agent=rl_games_actor_preserving_cfg_entry_point`。

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

正式结果至少运行所有冻结 training/evaluation seed，不能只报告最好 seed。actor-preserving PPO separate checkpoint 的评估命令必须添加 `--agent=rl_games_actor_preserving_cfg_entry_point`；checkpoint-selection analysis/imitation-learning benchmark shared checkpoint 继续使用默认 agent。

actor-preserving PPO 正式协议已完成：8 个去重 actor-preserving PPO/BC 物理 checkpoint 在 test seeds 245/246/247 上各运行 256 episodes，共 24 条、6144 episodes，全部 completed。metric-selected 三 training seed 聚合 settled landing 为 96.7448%，完整 manifest 和逐 episode CSV 见 `logs/imitation/actor_preserving_ppo/formal_test/`，可提交聚合见 `benchmarks/actor_preserving_ppo/`。

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

Deck-Contact Proxy Baseline、Heave-Precision Proxy：

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

Physical Deck 额外输出：

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

### PhysicalDeckAttitude

physical-deck-attitude task 在 Physical Deck 字段基础上增加：

```text
first_contact_deck_roll / pitch / tilt
first_contact_deck_angular_speed
first_contact_body_deck_normal_angle
terminal_body_deck_normal_angle
first/terminal normal relative speed
first/terminal tangential relative speed
max_contact_impulse
terminal deck roll / pitch / tilt / angular speed
deck pose/velocity consistency errors
deck tilt buckets
deck angular-speed buckets
```

法向与切向速度使用 deck 表面点速度 `v_center + omega × r`，而不是只减甲板中心线速度。姿态角以弧度写入 CSV、以度输出 P95。

## 16→22 checkpoint 迁移

```bash
PYTHONPATH=source/quadcopter_waypoint python \
  scripts/rl_games/expand_checkpoint_observation.py \
  --input <physical-deck_16D.pth> \
  --output <physical-deck-attitude_22D.pth>
```

迁移器会检查真实 state-dict key 和 shape，复制第一层前 16 列，将新增 6 列置零，扩展 observation mean/variance，并同步扩展 Adam moment。源文件和已存在的输出文件均不会被覆盖；旁路 JSON 保存输入/输出 SHA256。

## physical-deck-attitude task 物理诊断

```bash
PYTHONPATH=source/quadcopter_waypoint python \
  scripts/rl_games/check_physical_deck_attitude_physics.py \
  --num_envs=16 \
  --motion_steps=500 \
  --output benchmarks/physical_deck_attitude/physics_check_16env.json \
  --headless
```

脚本运行确定性 xy/heave/roll/pitch 完整周期，检查最低底角、位姿/速度一致性，并分别把无人机放到 deck 和 GroundSlab，要求目标 ContactSensor 非零且另一通道为零。

## 纯 Python 测试

```bash
PYTHONPATH=source/quadcopter_waypoint python -m pytest -q tests
```

覆盖：

```text
live Tensor 在 reset 后变化
terminal latch 优先级
pad/deck tilt/deck angular-speed bucket 边界
成功 episode 独占 touchdown 汇总
空成功集合 NaN 语义
quaternion 与 Euler-rate angular velocity 一致性
world/deck frame 转换、omega × r、法/切向分解
倾斜平面 clearance 与最低高度安全边界
checkpoint 第一层、normalization 与 optimizer moment 迁移
错误维度与覆盖保护
```

## 常用任务 ID

```text
Isaac-Quadcopter-OfficialClone-Direct-v0
Isaac-Quadcopter-WaypointV1-Direct-v0
Isaac-Quadcopter-WaypointV2-Direct-v0
Isaac-Quadcopter-ShipLanding-Direct-v0
Isaac-Quadcopter-ShipLanding-Heave-Direct-v0
Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

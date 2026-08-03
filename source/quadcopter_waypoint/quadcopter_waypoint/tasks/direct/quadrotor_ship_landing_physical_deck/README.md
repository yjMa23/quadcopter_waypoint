# Quadcopter Ship Landing PhysicalDeck

## 阶段定位

```text
Phase 6B PhysicalDeck
```

独立任务：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0
```

rl_games 实验名：

```text
quadcopter_ship_landing_physical_deck
```

本任务从已冻结的 Phase 6A Heave Precision 派生，但不修改 Phase 5D 或 Phase 6A 目录。目标是先证明无人机能在水平、平移且升沉的实体甲板上完成真实接触和稳定降落，再进入 roll / pitch。

## 当前运动范围

```text
水平甲板
xy 匀速范围 = [-0.20, 0.20] m/s
z(t) = base_height + amplitude * sin(phase)
amplitude = 0.08–0.12 m
frequency = 0.18–0.30 Hz
pad_base_height = 0.18 m
```

暂未实现：

```text
roll
pitch
yaw oscillation
复杂六自由度波浪
```

## 实体甲板与地面

甲板使用 Isaac Lab 5.1 标准 API：

```text
RigidObjectCfg
CuboidCfg
RigidBodyPropertiesCfg(kinematic_enabled=True)
CollisionPropertiesCfg
write_root_pose_to_sim
write_root_velocity_to_sim
```

甲板尺寸：

```text
0.50 × 0.50 × 0.04 m
```

每个并行环境有独立实体 `Deck` 和独立 `GroundSlab`：

```text
/World/envs/env_.*/Deck
/World/envs/env_.*/GroundSlab
```

`GroundSlab` 顶面比全局 plane 高 1 cm，因此无人机首先与可过滤的实体 ground 接触。最低 deck bottom 保持在 ground 之上，不会与地面发生无关碰撞。

Isaac Lab 5.1 的 `ContactSensor` rigid-body view 需要 USD 中存在每个环境的 authored clone。本任务显式设置：

```python
clone_in_fabric = False
replicate_physics = True
```

否则 Fabric-only clone 会导致 ContactSensor 的 per-environment body count 校验失败。

## 接触传感器

使用两个过滤式 ContactSensor：

```text
deck sensor:
  sensor body = Deck
  filter = Robot/body

ground sensor:
  sensor body = GroundSlab
  filter = Robot/body
```

因此能够明确区分：

```text
robot-deck contact
robot-ground contact
```

不使用 marker 或几何 clearance 代替真实接触报告。

## 状态真值与观测兼容

策略观测仍保持 Phase 5D / Phase 6A 的 16 维接口，便于迁移旧 checkpoint。

实体 deck 的：

```text
root_pos_w
root_lin_vel_w
```

是平台位置和速度的唯一真值。`_pad_pos_w` 与 `_pad_vel_w` 只是把实体状态同步到旧策略接口，不再独立积分 marker 状态。

观测包含：

```text
robot body linear velocity
robot body angular velocity
projected gravity
pad relative position in body frame
pad relative linear velocity
align state
```

## 接触分类

### contact_success

episode 中至少出现一次经过 deck filter 的真实接触报告。

### safe_contact

必须同时满足：

1. robot 与 deck 有真实接触；
2. robot 位于 deck 有效区域；
3. 当前水平误差小于 0.12 m；
4. 首次接触也发生在 0.12 m 精度区域内；
5. 法向相对速度绝对值小于 0.55 m/s；
6. 切向相对速度小于 0.30 m/s；
7. 角速度小于 1.50 rad/s；
8. upright 大于 0.90；
9. penetration 不超过 0.025 m。

### successful_settle

`safe_contact` 连续保持 3 个控制步：

```text
3 × 0.02 s = 0.06 s
```

首次接触精度是不可逆门槛。偏心首次接触不能通过后续滑动进入中心而变成有效成功。

### hard_contact

真实 deck contact 满足任一条件：

```text
contact force > 2.50 N
abs(normal relative speed) > 0.80 m/s
penetration > 0.025 m
```

### deck_miss

包括：

```text
首次接触在 0.12 m 精度区之外
或
无人机从实体 deck 有效区域外穿过 deck surface
```

### ground_crash

```text
filtered GroundSlab contact
或
robot root height < crash threshold
```

### timeout

达到 episode 最大步数且没有成功或其他终止分类。

## 评估字段

PhysicalDeck CSV 在通用 terminal 指标外增加：

```text
contact_success
settled_landing
hard_contact
ground_crash
deck_miss
first_contact_seen
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

当前甲板保持水平，因此 deck-frame xy 与 world xy 方向一致；字段命名和实现结构为后续 roll / pitch 保留 deck-frame 语义。

## Zero-shot 结果

初始化 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

第一版 64-episode zero-shot：

```text
contact_success_rate = 78.12%
settled_landing_rate = 60.94%
hard_contact_rate = 0%
ground_crash_rate = 0%
deck_miss_rate = 32.81%
timeout_rate = 6.25%
```

结论：Phase 5D checkpoint 能产生真实 deck 接触，但偏心落点和接触后滑出明显，不能直接采用。

## 微调课程

### Stage A：宽实体 deck 接触

先允许 deck 有效区域内稳定接触，学习从 proxy 过渡到真实 collision。

### Stage B：18 cm 精度区

```text
landing_success_radius = 0.18 m
```

强化 center precision 和 off-center contact 惩罚。

### Stage C：14 cm 精度区

```text
landing_success_radius = 0.14 m
```

继续收紧首次接触位置。

### Stage D：12 cm 最终精度区

```text
landing_success_radius = 0.12 m
```

增加首次接触不可逆门槛，降低学习率到 `1e-4`，并提高近甲板中心、预测落点和下降速度约束。

训练过程中评估多个 checkpoint。后期存在 policy drift，因此没有采用最后的 ep1050，而选择 ep990。

## 最佳 checkpoint

```text
logs/rl_games/quadcopter_ship_landing_physical_deck/2026-08-03_18-46-00/nn/last_quadcopter_ship_landing_physical_deck_ep_990_rew_61.680832.pth
```

SHA256：

```text
614cf3bea439883b7b2c478f0dd21641f9eb750df9f08d711d8cf122f133b3aa
```

## 正式评估

命令模板：

```bash
python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0 \
  --checkpoint logs/rl_games/quadcopter_ship_landing_physical_deck/2026-08-03_18-46-00/nn/last_quadcopter_ship_landing_physical_deck_ep_990_rew_61.680832.pth \
  --num_envs=64 \
  --episodes=256 \
  --seed=42 \
  --csv logs/rl_games/quadcopter_ship_landing_physical_deck/p6b_final_seed42.csv \
  --headless
```

三种子结果：

| seed | settled | hard contact | ground crash | deck miss | timeout | 成功 first-contact xy P95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 96.48% | 0% | 0% | 3.52% | 0% | 0.0937 m |
| 43 | 97.27% | 0% | 0% | 2.73% | 0% | 0.1032 m |
| 44 | 94.53% | 0% | 0% | 5.47% | 0% | 0.1037 m |
| aggregate | 96.09% | 0% | 0% | 3.91% | 0% | 0.1000 m |

聚合补充指标：

```text
contact_success_rate = 100.00%
touchdown_distance mean = 0.0563 m
touchdown_distance P95 = 0.1022 m
first-contact normal relative speed mean = -0.1544 m/s
first-contact tangential relative speed mean = 0.2034 m/s
max contact force mean = 0.2981 N
settle time mean = 0.2354 s
maximum observed penetration = 0.0203 m
```

完整 benchmark：

```text
benchmarks/phase6b_physical_deck/summary.json
```

## 训练命令

合法 PPO smoke：

```bash
python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0 \
  --num_envs=16 \
  --headless \
  --max_iterations=651 \
  --checkpoint logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

正式微调示例：

```bash
python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0 \
  --num_envs=256 \
  --headless \
  --max_iterations=1050 \
  --checkpoint <PREVIOUS_STAGE_CHECKPOINT>
```

`horizon_length=24`、`minibatch_size=384`，因此 `num_envs` 必须使 `num_envs × 24` 能整除 384。

## 播放

```bash
python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0 \
  --num_envs=1 \
  --checkpoint logs/rl_games/quadcopter_ship_landing_physical_deck/2026-08-03_18-46-00/nn/last_quadcopter_ship_landing_physical_deck_ep_990_rew_61.680832.pth
```

## 当前结论

Phase 6B 已达到推荐验收目标：

```text
三种子 settled landing >= 95%
ground crash <= 1%
hard contact <= 2%
timeout <= 3%
成功回合 first-contact xy P95 <= 0.12 m
```

可以进入独立的 roll / pitch 任务，但不得继续直接修改本目录作为下一阶段开发分支。后续应在 deck 局部坐标系中计算位置、速度、法向接触和姿态误差。

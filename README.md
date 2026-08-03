# Quadcopter Waypoint RL

基于 **Isaac Lab External Project** 的四旋翼强化学习项目。仓库已从官方悬停基线、连续航点和静态平台降落，推进到带 xy 平移、z 升沉及小幅 roll/pitch 的实体甲板真实接触降落。

根 README 只维护当前状态、任务索引和通用命令；任务实现、指标定义、训练记录与阶段结论写在对应任务目录和 `benchmarks/` 中。

## 当前阶段

当前主线为 **P6C-PhysicalDeckAttitude**：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

独立任务在已冻结 P6B 基础上实现：

```text
xy 匀速平移
+ z 正弦升沉
+ roll / pitch 正弦运动（当前正式验证幅值均为 ±5°）
+ 真实 Deck / GroundSlab collision
+ 独立 deck / ground ContactSensor
+ deck-frame 落点、clearance 与成功判定
+ v_center + omega × r 接触点速度
+ body-z / deck-normal 姿态成功条件
```

推荐 checkpoint 是由 P6B ep990 严格执行 16→22 维观测迁移得到的本地文件：

```text
logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/expanded_from_p6b_ep990_16to22.pth
```

SHA256：

```text
95424bb0d6b98d8dfbf2455d6fd84e99a77d52bca28489654036a25aea5a697d
```

三随机种子正式评估，`seed=42,43,44`，每个 seed 256 episodes，roll/pitch 幅值范围 `0–5°`、频率 `0.08–0.15 Hz`：

| 指标 | 聚合结果 |
| --- | ---: |
| episodes | 768 |
| contact success | 99.87% |
| settled landing | 94.66% |
| hard contact | 0.13% |
| ground crash | 0.00% |
| deck miss | 5.34% |
| timeout | 0.00% |
| 成功回合 first-contact xy P95 | 0.1023 m |
| 成功回合 first-contact normal speed P95 | 0.3851 m/s |
| 成功回合 body-z/deck-normal angle P95 | 6.42° |
| touchdown distance P95 | 0.1065 m |
| 最大观测 penetration | 0.0203 m |

逐 seed settled landing：

```text
seed 42: 241 / 256 = 94.14%
seed 43: 244 / 256 = 95.31%
seed 44: 242 / 256 = 94.53%
aggregate: 727 / 768 = 94.66%
```

Stage D 的 30 epoch 微调候选发生 policy drift，ep1000/1010/1020 的固定种子 settled 分别为 89.06%、70.31%、89.06%，因此没有默认采用最后 checkpoint。

P6C 尚未包含 yaw oscillation、随机波谱、水动力或复杂六自由度船舶运动。完整证据：

```text
benchmarks/phase6c_physical_deck_attitude/summary.json
benchmarks/phase6c_physical_deck_attitude/physics_check_1env.json
benchmarks/phase6c_physical_deck_attitude/physics_check_16env.json
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck_attitude/README.md
```

## 已冻结阶段

### Phase 6B PhysicalDeck

任务：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0
```

P6B 保留水平实体甲板、真实 deck/ground 接触区分和 16 维策略基线，不再通过修改其目录推进姿态运动。三种子 768 episodes 聚合 settled landing 为 96.09%，完整证据见：

```text
benchmarks/phase6b_physical_deck/summary.json
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck/README.md
```

### Phase 5D DeckContact

任务：

```text
Isaac-Quadcopter-ShipLanding-Direct-v0
```

该阶段使用 marker 与几何表面 clearance 代理，不是真实实体甲板。目录已冻结，不再用于后续实体接触开发。

### Phase 6A Heave Precision

任务：

```text
Isaac-Quadcopter-ShipLanding-Heave-Direct-v0
```

该阶段在 Phase 5D 基础上加入水平平台的 z 向正弦升沉，仍使用接触代理。采用 Phase 5D checkpoint 迁移评估：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth
```

三种子 768 episodes 聚合：

| 指标 | 聚合结果 |
| --- | ---: |
| landing success | 98.83% |
| crash | 0.00% |
| timeout | 1.17% |
| touchdown distance mean | 0.0661 m |
| touchdown distance P95 | 0.0970 m |
| terminal horizontal relative speed mean | 0.0908 m/s |
| terminal vertical relative speed mean | 0.0114 m/s |

证据：

```text
benchmarks/phase6a_heave_precision/summary.json
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_heave/README.md
```

## 项目结构

```text
.
├── README.md
├── backups/                         # 明确标记为历史归档的阶段记录
├── benchmarks/
│   ├── phase6a_heave_precision/
│   │   └── summary.json
│   ├── phase6b_physical_deck/
│   │   └── summary.json
│   └── phase6c_physical_deck_attitude/
│       ├── summary.json
│       ├── physics_check_1env.json
│       └── physics_check_16env.json
├── logs/                            # checkpoint、CSV、TensorBoard，本地保留且 Git 忽略
├── scripts/
│   └── rl_games/
│       ├── README.md
│       ├── train.py
│       ├── play.py
│       ├── eval_metrics.py
│       ├── eval_metrics_utils.py
│       ├── expand_checkpoint_observation.py
│       ├── check_physical_deck_attitude_physics.py
│       └── summarize_physical_deck_attitude.py
├── tests/
│   ├── test_eval_metrics_utils.py
│   ├── test_checkpoint_observation_expansion.py
│   └── test_physical_deck_attitude_math.py
└── source/quadcopter_waypoint/
    ├── setup.py
    └── quadcopter_waypoint/tasks/direct/
        ├── quadrotor_official_clone/
        ├── quadrotor_v1_metrics/
        ├── quadrotor_waypoint_v2/
        ├── quadrotor_ship_landing/                 # Phase 5D frozen
        ├── quadrotor_ship_landing_heave/           # Phase 6A frozen
        ├── quadrotor_ship_landing_physical_deck/   # Phase 6B frozen
        └── quadrotor_ship_landing_physical_deck_attitude/ # P6C current
```

## 文档索引

| 模块 | 任务 ID / 用途 | 文档 |
| --- | --- | --- |
| OfficialClone | `Isaac-Quadcopter-OfficialClone-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_official_clone/README.md` |
| WaypointV1 | `Isaac-Quadcopter-WaypointV1-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_v1_metrics/README.md` |
| WaypointV2 | `Isaac-Quadcopter-WaypointV2-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_waypoint_v2/README.md` |
| Phase 5D | `Isaac-Quadcopter-ShipLanding-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/README.md` |
| Phase 6A | `Isaac-Quadcopter-ShipLanding-Heave-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_heave/README.md` |
| Phase 6B | `Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck/README.md` |
| P6C | `Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0` | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck_attitude/README.md` |
| rl_games wrapper | 训练、播放、评估、迁移与物理诊断 | `scripts/rl_games/README.md` |

## 环境准备

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
conda activate env_isaaclab
python -m pip install -e source/quadcopter_waypoint
```

任务注册通过：

```python
import quadcopter_waypoint.tasks
```

触发。项目 wrapper 已按 Isaac Lab 官方启动顺序注入该 import，不需要修改 Isaac Lab 源码。

## 通用命令

训练：

```bash
python scripts/rl_games/train.py \
  --task=<TASK_ID> \
  --num_envs=<N> \
  --headless \
  --max_iterations=<ITER>
```

播放：

```bash
python scripts/rl_games/play.py \
  --task=<TASK_ID> \
  --num_envs=1 \
  --checkpoint <CHECKPOINT>
```

评估：

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

## PPO 小规模冒烟约束

当前 rl_games 配置：

```text
horizon_length = 24
minibatch_size = 384
```

因此：

```text
num_envs × 24
```

必须能被 384 整除。最小合法值为 16：

```text
16 × 24 = 384
```

不要用 4 环境执行 PPO 冒烟。

## Terminal 指标语义

Isaac Lab `DirectRLEnv.step()` 会在返回前自动 reset 已结束环境。ShipLanding 环境会在 reset 覆盖状态前锁存精确 terminal 状态；评估脚本只读取该锁存，不读取 reset 后 Tensor，也不把前一帧冒充 terminal 状态。

主要字段：

```text
final_vertical_speed
  terminal robot world-z velocity

terminal_vertical_relative_speed
  terminal robot world-z velocity - terminal deck world-z velocity

final_horizontal_speed
  terminal robot/deck relative xy speed

final_distance
  terminal robot/deck 3D distance

touchdown_*
  only summarized over successful episodes
```

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games
```

## 下一阶段门槛

P6C 已通过 ±5° roll/pitch 验收，但当前 **不具备直接加入复杂六自由度运动的充分条件**。进入 yaw 或波浪谱前应先：

1. 为 yaw 后 deck-frame 切向速度和有效落区增加专项回归测试；
2. 将姿态课程改为可控分层采样，保证高倾角/高角速度桶有足够样本；
3. 解决 Stage D 继续 PPO 微调导致的落点 policy drift，优先检查 reward 与 normalization 学习率；
4. 在有 DISPLAY 的会话完成真实人工 GUI 目视记录；
5. 继续保持实体 deck pose、velocity、normal 和 ContactSensor 为唯一真值，不修改已冻结的 P6B。

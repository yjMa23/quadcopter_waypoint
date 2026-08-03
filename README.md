# Quadcopter Waypoint RL

基于 **Isaac Lab External Project** 的四旋翼强化学习项目。仓库已从官方悬停基线、连续航点和静态平台降落，推进到水平移动且正弦升沉的实体甲板真实接触降落。

根 README 只维护当前状态、任务索引和通用命令；任务实现、指标定义、训练记录与阶段结论写在对应任务目录和 `benchmarks/` 中。

## 当前阶段

当前主线为 **Phase 6B PhysicalDeck**：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0
```

已实现：

```text
xy 匀速平移
+ z 正弦升沉
+ 水平实体甲板
+ 真实 collision
+ 过滤式 deck / ground 接触检测
+ 首次接触精度门槛
+ 持续安全接触判定
```

本阶段没有加入 roll / pitch / yaw oscillation。实体甲板状态是任务唯一真值；旧 landing marker 仅可作为辅助可视化，不能替代实体状态或接触报告。

最佳 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing_physical_deck/2026-08-03_18-46-00/nn/last_quadcopter_ship_landing_physical_deck_ep_990_rew_61.680832.pth
```

SHA256：

```text
614cf3bea439883b7b2c478f0dd21641f9eb750df9f08d711d8cf122f133b3aa
```

三随机种子正式评估，`seed=42,43,44`，每个 seed 256 episodes：

| 指标 | 聚合结果 |
| --- | ---: |
| episodes | 768 |
| contact success | 100.00% |
| settled landing | 96.09% |
| hard contact | 0.00% |
| ground crash | 0.00% |
| deck miss | 3.91% |
| timeout | 0.00% |
| touchdown distance mean | 0.0563 m |
| touchdown distance P95 | 0.1022 m |
| 成功回合 first-contact xy P95 | 0.1000 m |
| 最大观测 penetration | 0.0203 m |

逐 seed settled landing：

```text
seed 42: 247 / 256 = 96.48%
seed 43: 249 / 256 = 97.27%
seed 44: 242 / 256 = 94.53%
aggregate: 738 / 768 = 96.09%
```

完整证据：

```text
benchmarks/phase6b_physical_deck/summary.json
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck/README.md
```

## 已冻结阶段

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
│   └── phase6b_physical_deck/
│       └── summary.json
├── logs/                            # checkpoint、CSV、TensorBoard，本地保留且 Git 忽略
├── scripts/
│   └── rl_games/
│       ├── README.md
│       ├── train.py
│       ├── play.py
│       ├── eval_metrics.py
│       └── eval_metrics_utils.py
├── tests/
│   └── test_eval_metrics_utils.py
└── source/quadcopter_waypoint/
    ├── setup.py
    └── quadcopter_waypoint/tasks/direct/
        ├── quadrotor_official_clone/
        ├── quadrotor_v1_metrics/
        ├── quadrotor_waypoint_v2/
        ├── quadrotor_ship_landing/                 # Phase 5D frozen
        ├── quadrotor_ship_landing_heave/           # Phase 6A frozen
        └── quadrotor_ship_landing_physical_deck/   # Phase 6B current
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
| rl_games wrapper | 训练、播放、评估 | `scripts/rl_games/README.md` |

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

Phase 6B 已达到进入 roll / pitch 的基线门槛，但下一阶段必须继续使用独立任务目录和任务 ID。建议先增加小幅 deck-frame roll / pitch，并保持：

1. 实体 deck pose、velocity 和 normal 为唯一真值；
2. 成功、相对速度与落点全部在 deck 局部坐标系计算；
3. 继续区分 safe contact、hard contact、deck miss、ground crash 和 timeout；
4. 先小幅姿态课程，再扩展六自由度波浪；
5. 不修改已冻结的 Phase 5D、Phase 6A 和 Phase 6B 基线任务逻辑。

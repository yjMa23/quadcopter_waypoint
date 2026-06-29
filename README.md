# Quadcopter Waypoint RL

这是一个基于 **Isaac Lab External Project** 方式搭建的四旋翼强化学习项目，用于复现、验证并扩展 Isaac Lab 官方四旋翼任务。当前项目已经从官方悬停基线逐步扩展到连续航点任务和静态平台降落任务，后续将继续推进到低速移动平台和船舶运动降落。

本仓库根目录只保留项目总览、目录索引和通用环境准备方式。各任务的训练、播放、评估、调试记录和阶段性结果，统一写在对应任务包目录下的 `README.md` 中。

## 当前阶段

当前稳定阶段是 **ShipLanding 静态平台降落**：

```text
Isaac-Quadcopter-ShipLanding-Direct-v0
```

已完成：

```text
先飞到 landing pad 上方 → 对准 → 缓慢下降 → 接近板面后终止
```

当前稳定 checkpoint：

```text
logs/rl_games/quadcopter_ship_landing/2026-06-28_23-40-26/nn/quadcopter_ship_landing.pth
```

该阶段的详细说明、训练命令和评估结果见：

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/README.md
```

## 项目结构

```text
.
├── README.md
├── logs/                         # 本地训练日志和 checkpoint，不纳入 Git
├── scripts/
│   └── rl_games/
│       ├── README.md              # train / play / eval wrapper 使用说明
│       ├── train.py
│       ├── play.py
│       └── eval_metrics.py
└── source/
    └── quadcopter_waypoint/
        ├── setup.py
        └── quadcopter_waypoint/
            └── tasks/
                └── direct/
                    ├── quadrotor_official_clone/
                    │   ├── README.md
                    │   ├── __init__.py
                    │   └── quadrotor_official_clone_env.py
                    ├── quadrotor_v1_metrics/
                    │   ├── README.md
                    │   ├── __init__.py
                    │   ├── quadrotor_v1_metrics_env.py
                    │   └── agents/
                    │       └── rl_games_ppo_cfg.yaml
                    ├── quadrotor_waypoint_v2/
                    │   ├── README.md
                    │   ├── __init__.py
                    │   ├── quadrotor_waypoint_v2_env.py
                    │   └── agents/
                    │       └── rl_games_ppo_cfg.yaml
                    ├── quadrotor_ship_landing/
                    │   ├── README.md
                    │   ├── __init__.py
                    │   ├── quadrotor_ship_landing_env.py
                    │   └── agents/
                    │       └── rl_games_ppo_cfg.yaml
                    └── quadrotor_waypoint/
                        ├── README.md              # 已废弃旧实验说明
                        ├── __init__.py
                        └── quadrotor_waypoint_env.py
```

## 文档索引

| 模块 | 任务 ID / 用途 | 文档 |
| --- | --- | --- |
| OfficialClone | `Isaac-Quadcopter-OfficialClone-Direct-v0`，官方四旋翼任务 external project 复刻基线 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_official_clone/README.md` |
| WaypointV1 | `Isaac-Quadcopter-WaypointV1-Direct-v0`，稳定起点版本，复用官方环境并独立实验名 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_v1_metrics/README.md` |
| WaypointV2 | `Isaac-Quadcopter-WaypointV2-Direct-v0`，连续航点版本 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_waypoint_v2/README.md` |
| ShipLanding | `Isaac-Quadcopter-ShipLanding-Direct-v0`，静态 landing pad 降落阶段 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/README.md` |
| Deprecated Waypoint | 已废弃旧实验版本，不作为后续开发基础 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_waypoint/README.md` |
| rl_games scripts | 训练、播放、独立评估 wrapper | `scripts/rl_games/README.md` |

## 环境准备

先激活 Isaac Lab 环境：

```bash
conda activate env_isaaclab
```

进入项目根目录，并以 editable 模式安装 external project：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
python -m pip install -e source/quadcopter_waypoint
```

安装后，任务注册由：

```python
import quadcopter_waypoint.tasks
```

自动触发。`scripts/rl_games/train.py`、`play.py` 和 `eval_metrics.py` 已经在 wrapper 中注入该 import，不需要修改 Isaac Lab 官方源码。

## 通用命令模板

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
  --checkpoint <CHECKPOINT_PATH>
```

评估：

```bash
python scripts/rl_games/eval_metrics.py \
  --task=<TASK_ID> \
  --checkpoint <CHECKPOINT_PATH> \
  --num_envs=64 \
  --episodes=256 \
  --csv <OUTPUT_CSV> \
  --headless
```

各任务推荐参数、稳定 checkpoint 和注意事项见对应任务目录的 `README.md`。

## TensorBoard

```bash
tensorboard --logdir /home/j/Isaac_RL_Projects/quadcopter_waypoint/logs/rl_games
```

日志目录按任务实验名区分：

```text
logs/rl_games/quadcopter_direct
logs/rl_games/quadcopter_waypoint_v1
logs/rl_games/quadcopter_waypoint_v2
logs/rl_games/quadcopter_ship_landing
```

## 开发原则

1. 不直接修改 Isaac Lab 官方源码。
2. 每个任务使用独立 Gym task ID 和独立 rl_games 实验名，避免 task / checkpoint 混用。
3. 根 README 只维护项目索引；任务细节写到对应任务包目录。
4. 训练环境尽量只保留任务必需状态；复杂指标优先放到 `scripts/rl_games/eval_metrics.py` 中独立评估。
5. 新任务难度通过 curriculum 逐步增加，例如：静态平台 → 低速匀速移动平台 → 更高速度平台 → 正弦船舶运动。

## 后续计划

1. 继续确认 ShipLanding 静态降落在 GUI 中的视觉表现。
2. 扩展 `eval_metrics.py` 的行为质量指标，例如 `landing_time`、`max_descent_speed`、`mean_descent_speed`、`final_vertical_speed`。
3. 进入 Phase 5A：低速匀速移动平台，先设置 `pad_vel_xy ∈ [-0.05, 0.05] m/s`，并从当前静态降落 checkpoint 微调。
4. 低速移动平台稳定后，再进入更高平台速度和正弦船舶运动。

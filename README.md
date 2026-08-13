# Quadcopter Waypoint RL

基于 Isaac Lab External Project 的四旋翼运动甲板自主降落项目。当前主线是 22 维状态策略与 actor-preserving PPO；任务环境、训练资产、评估结果和理论文档均保留在仓库中。

## 当前实现

任务 ID：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

策略接口：

```text
state observation: 22
continuous action: 4
actor: 22 -> 64 -> 64 -> 4, ELU
```

当前策略是 state-based policy，输入包含无人机与甲板的相对状态，不包含相机图像或真实视觉投影。

actor-preserving PPO 使用独立 actor/critic：前 10 个 epoch 只更新 critic，之后联合训练；observation RMS 全程冻结，BC actor 通过 `bc_anchor_coefficient=50` 约束。训练 seed 为 42/43/44，validation seed 为 145/146/147，最终 test seed 为 245/246/247。

冻结 teacher：

```text
logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/
expanded_from_physical_deck_ep990_16to22.pth
```

SHA-256：

```text
95424bb0d6b98d8dfbf2455d6fd84e99a77d52bca28489654036a25aea5a697d
```

## 当前结果

独立 test 使用 3 个评估 seed，每个 checkpoint 每个 seed 256 episodes。actor-preserving PPO 的三个 validation-selected checkpoint 共评估 2304 episodes。

| 方法 | settled landing | deck miss | hard contact | ground crash | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen PPO teacher | 94.66% | 5.34% | 0.13% | 0.00% | 0.00% |
| BC epoch 0 | 86.20% | 13.80% | 0.00% | 0.00% | 0.00% |
| ordinary BC+PPO | 76.69% | 23.05% | 0.91% | 0.00% | 0.00% |
| checkpoint-selected BC+PPO | 91.67% | 8.33% | 0.30% | 0.00% | 0.00% |
| **actor-preserving PPO** | **96.74%** | **3.17%** | **0.09%** | **0.00%** | **0.04%** |

actor-preserving PPO 获得 2229/2304 次 settled landing，Wilson 95% CI 为 95.94%–97.40%。critic-only warm-up 结束时，三个训练 seed 的 actor 参数、确定性动作和 observation RMS 漂移均为 0。hard contact 相比 BC 从 0 增至 0.09%，因此不能声称所有安全指标都严格改善。

完整结果与复现实验包：

```text
benchmarks/actor_preserving_ppo/
docs/actor_preserving_ppo.md
```

## 使用方式

安装项目：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
conda activate env_isaaclab
python -m pip install -e source/quadcopter_waypoint
export PYTHONPATH=source/quadcopter_waypoint
```

运行测试：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

图形终端播放冻结 teacher：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=1 \
  --checkpoint=logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/expanded_from_physical_deck_ep990_16to22.pth
```

该播放命令应在已登录 Ubuntu 桌面的图形终端中执行，不要添加 `--headless`。SSH、tmux、Docker 和离屏渲染说明见 `docs/runtime_display_troubleshooting.md`。

训练与评估入口：

```bash
python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=256 --seed=42 --headless --max_iterations=200

python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --checkpoint=<CHECKPOINT> \
  --num_envs=64 --episodes=256 --seed=245 \
  --csv=<OUTPUT.csv> --headless
```

可直接复现的完整命令分别位于：

```text
benchmarks/imitation_hybrid/commands.txt
benchmarks/checkpoint_selection/commands.txt
benchmarks/actor_preserving_ppo/commands.txt
```

## 文档与资产

| 内容 | 入口 |
| --- | --- |
| 理论文档索引 | `docs/README.md` |
| 升沉甲板精确降落 | `docs/heave_precision_theory.md` |
| 水平实体甲板接触 | `docs/physical_deck_theory.md` |
| 倾斜实体运动甲板 | `docs/physical_deck_attitude_theory.md` |
| 模仿学习与普通 BC+PPO | `docs/imitation_hybrid_paper.md` |
| checkpoint 选模与 policy drift | `docs/checkpoint_selection_and_policy_drift.md` |
| actor-preserving PPO | `docs/actor_preserving_ppo.md` |
| 文献综述与研究路线 | `docs/literature_review_ship_landing_rl.md`、`docs/literature_comparison_matrix.md` |
| benchmark 数据 | `benchmarks/` |

文献路线用于研究建议，不代表项目交付承诺。

## 当前客观限制

- 甲板运动目前主要是确定性正弦 roll/pitch，尚未覆盖随机波谱、yaw、水动力和完整船舶六自由度运动。
- 策略使用仿真中的精确相对状态，尚未加入视觉估计误差、延迟、丢帧和传感器噪声。
- demonstration 只保留 teacher 成功回合，对分布外扰动与失败恢复的覆盖有限。
- 接触、旋翼气动和船体运动仍是仿真模型，尚未完成系统辨识或实机闭环验证。
- 当前视频由 headless 离屏渲染生成，manifest 中仍明确记录 `human_review_completed=false`。

## 后续待完成

- 引入随机海况与 JONSWAP 波谱，扩展甲板运动分布。
- 加入动力学随机化，并用实测数据完成系统辨识。
- 接入 ArUco 相对状态估计，建模噪声、延迟、丢帧和状态历史。
- 建立 PID 与 NMPC baseline，统一成功定义和评估预算后比较。
- 推进 Sim-to-Real；完成状态策略实机验证后，再研究后续视觉策略。

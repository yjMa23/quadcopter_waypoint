# Quadcopter Waypoint RL

基于 **Isaac Lab External Project** 的四旋翼强化学习项目。当前主线已完成 **P7：专家轨迹、Behavior Cloning 与 BC 初始化 PPO 对比**；底层任务仍为冻结的 P6C 实体运动甲板降落环境。

## 当前主线

任务 ID：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

策略输入/输出：

```text
state observation: 22
continuous action: 4
actor: 22 -> 64 -> 64 -> 4, ELU
```

当前策略是 **state-based policy**，不包含相机图像，也不包含真实视觉投影输入。

P6C 冻结 teacher：

```text
logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/
expanded_from_p6b_ep990_16to22.pth
```

SHA256：

```text
95424bb0d6b98d8dfbf2455d6fd84e99a77d52bca28489654036a25aea5a697d
```

冻结标签：

```text
p6c-physical-deck-attitude-v1
```

## P7 正式结果

正式评估统一使用 `seed=42,43,44`，每个 seed 256 episodes，roll/pitch 幅值 `0–5°`、频率 `0.08–0.15 Hz`。

| 方法 | settled landing | contact success | hard contact | deck miss | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen PPO teacher | 94.66% ± 0.49% | 99.87% | 0.13% | 5.34% | 0.00% |
| PPO from scratch | 0.91% ± 0.66% | 10.55% | 5.73% | 21.61% | 73.57% |
| BC only | 88.28% ± 0.55% | 99.74% | 0.26% | 11.72% | 0.00% |
| BC initialized PPO | 76.69% ± 9.16% | 95.05% | 0.91% | 23.05% | 0.00% |

结论：

- 专家数据集和 BC-only 验收通过；BC-only 闭环 settled landing 达到 88.28%，超过 80% 目标。
- 在相同 1,228,800 online environment steps/seed 预算内，PPO-from-scratch 未达到 80%。
- BC+PPO 在 0 online steps 时继承 BC 的 88.28%，但 PPO 更新后发生 policy drift；未达到 90% 或 92%。
- 已做一次依据明确的 `1e-5` 保守学习率诊断，聚合 settled 进一步降至 63.67%，因此保留为负结果，不作为主结果。
- 不能声称“BC+PPO 优于 BC”或“达到 90% 的样本效率优于 PPO-from-scratch”。

完整证据：

```text
benchmarks/phase7_imitation_hybrid/README.md
benchmarks/phase7_imitation_hybrid/summary.json
benchmarks/phase7_imitation_hybrid/training_runs.json
benchmarks/phase7_imitation_hybrid/formal_evaluations/
docs/interview_p7_evidence.md
```

## P7 专家数据集

```text
successful episodes: 3976
transitions: 540321
collection seeds: 42 / 43 / 44
split: 3180 / 397 / 399 episodes
split policy: whole episode, 80% / 10% / 10%
```

本地大文件与哈希记录在 benchmark 中；原始 shard、checkpoint、TensorBoard 和视频不提交 Git。

## 项目索引

```text
.
├── benchmarks/
│   ├── phase6a_heave_precision/
│   ├── phase6b_physical_deck/
│   ├── phase6c_physical_deck_attitude/
│   └── phase7_imitation_hybrid/
├── docs/
│   └── interview_p7_evidence.md
├── scripts/
│   ├── imitation/
│   └── rl_games/
├── source/quadcopter_waypoint/quadcopter_waypoint/
│   ├── imitation/
│   └── tasks/direct/
└── tests/
```

| 阶段 | 任务/用途 | 文档 |
| --- | --- | --- |
| P6A | 升沉平台代理接触 | `benchmarks/phase6a_heave_precision/summary.json` |
| P6B | 水平实体甲板 | `benchmarks/phase6b_physical_deck/summary.json` |
| P6C | roll/pitch 实体运动甲板 teacher | `benchmarks/phase6c_physical_deck_attitude/summary.json` |
| P7 | 专家数据、BC、BC+PPO | `benchmarks/phase7_imitation_hybrid/README.md` |
| imitation scripts | 采集、训练、迁移、汇总 | `scripts/imitation/README.md` |
| rl_games scripts | PPO 训练、播放、闭环评估 | `scripts/rl_games/README.md` |

## 环境

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
conda activate env_isaaclab
python -m pip install -e source/quadcopter_waypoint
export PYTHONPATH=source/quadcopter_waypoint
```

推荐解释器：

```text
/home/j/anaconda3/envs/env_isaaclab/bin/python
```

## 常用命令

单元测试：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

PPO 训练：

```bash
python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=256 --seed=42 --headless --max_iterations=200
```

正式闭环评估：

```bash
python scripts/rl_games/eval_metrics.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --checkpoint=<CHECKPOINT> \
  --num_envs=64 --episodes=256 --seed=42 \
  --csv=<OUTPUT.csv> --headless
```

P7 完整命令：

```text
benchmarks/phase7_imitation_hybrid/commands.txt
```

## 当前客观局限

- demonstration 仅保留 teacher 成功回合，off-distribution recovery 覆盖不足。
- PPO critic 从随机值头开始，在线更新容易破坏已经较强的 BC actor。
- 当前 checkpoint 选择依据仍是公共训练循环的 mean episode reward，并不完全等价于 settled landing。
- 当前运动分布不包含 yaw、随机波谱、水动力或完整船舶六自由度运动。
- 当前机器没有可用 DISPLAY；已保存 teacher、BC、BC+PPO 成功轨迹和代表性 timeout 失败轨迹，但未声称完成人工 GUI/视频验收。

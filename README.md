# Quadcopter Waypoint RL

基于 **Isaac Lab External Project** 的四旋翼强化学习项目。当前主线已完成 **P8B：actor-preserving PPO 正式 validation、独立 test 与可复现实验包**；底层任务仍为冻结的 P6C 实体运动甲板降落环境。

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

完整证据与理论说明：

```text
docs/p7_imitation_hybrid_paper.md
benchmarks/phase7_imitation_hybrid/README.md
benchmarks/phase7_imitation_hybrid/summary.json
benchmarks/phase7_imitation_hybrid/training_runs.json
benchmarks/phase7_imitation_hybrid/formal_evaluations/
docs/interview_p7_evidence.md
```

其中 `docs/p7_imitation_hybrid_paper.md` 按论文结构给出问题建模、甲板/接触运动学、观测与动作、BC 加权目标、BC→PPO 迁移、PPO 微调公式、实验协议、负结果分析和代码可追溯矩阵。文档顶部的 `CODE_SYNC` 参数块由单元测试与实际源码进行一致性检查。

## P7 专家数据集

```text
successful episodes: 3976
transitions: 540321
collection seeds: 42 / 43 / 44
split: 3180 / 397 / 399 episodes
split policy: whole episode, 80% / 10% / 10%
```

本地大文件与哈希记录在 benchmark 中；原始 shard、checkpoint、TensorBoard 和视频不提交 Git。

## P8A 周期 checkpoint 选模与 policy drift

P8A 不重新训练，也不修改冻结的 P6C 环境语义。它扫描现有 BC+PPO 周期 checkpoint，使用 validation settled landing 选模，并在独立 test seeds 上比较 teacher、BC、metric-selected BC+PPO 和原 reward-selected BC+PPO。

| 方法 | settled landing | deck miss | hard contact | touchdown distance |
| --- | ---: | ---: | ---: | ---: |
| frozen PPO teacher | 94.66% ± 2.58% | 5.34% | 0.13% | 0.0585 m |
| BC epoch 0 | 86.20% ± 1.29% | 13.80% | 0.00% | 0.0588 m |
| metric-selected BC+PPO | 91.67% ± 3.03% | 8.33% | 0.30% | 0.0569 m |
| reward-selected BC+PPO | 78.08% ± 8.66% | 21.57% | 1.00% | 0.0684 m |

三个训练 seed 的 validation 最优 checkpoint 均为 epoch 10；原 reward-selected checkpoint 分别为 epoch 130、21、75。metric-selected BC+PPO 在独立 test 上比 BC 高 5.47 个百分点，属于 P8A 情况 A，但仍未达到 92%。action drift 与 settled landing/deck miss 呈中等统计关联，不作严格因果解释。进一步提升建议采用 actor-preserving PPO，而不是修改冻结环境 reward。

完整结果：

```text
benchmarks/phase8a_checkpoint_selection/README.md
benchmarks/phase8a_checkpoint_selection/summary.json
benchmarks/phase8a_checkpoint_selection/comparison.md
benchmarks/phase8a_checkpoint_selection/checkpoint_drift.json
```

## P8B Actor-Preserving PPO 正式结果

P8B 不修改冻结环境的 observation、action、reward、termination 或接触语义。正式方法固定为 separate actor/critic、epoch 1–10 critic-only warm-up、全程冻结 observation RMS、`bc_anchor_coefficient=50`，并严格使用 validation seeds 145/146/147 选模后，才在独立 test seeds 245/246/247 上评估。

| 方法 | settled landing | deck miss | hard contact | timeout |
| --- | ---: | ---: | ---: | ---: |
| frozen PPO teacher | 94.66% | 5.34% | 0.13% | 0.00% |
| BC epoch 0 | 86.20% | 13.80% | 0.00% | 0.00% |
| P7 ordinary BC+PPO | 76.69% | 23.05% | 0.91% | 0.00% |
| P8A metric-selected | 91.67% | 8.33% | 0.30% | 0.00% |
| **P8B metric-selected** | **96.74%** | **3.17%** | **0.09%** | **0.04%** |
| P8B reward-selected | 95.40% | 4.51% | 0.13% | 0.04% |
| P8B epoch-200 last | 92.40% | 7.55% | 0.04% | 0.04% |

三个 training seed 的 validation-selected checkpoint 分别为 epoch 91、30、51；独立 formal test settled landing 分别为 97.79%、96.48%、95.96%。P8B metric-selected 共 2229/2304 成功，Wilson 95% CI 为 95.94%–97.40%，同时达到 90% 和 92% 参考目标，并超过 BC、P8A 与 frozen teacher。相对 BC，hard contact 从 0 增至 0.09%，因此不能声称所有安全指标都严格改善；绝对值仍低于 2% 目标。

critic-only warm-up 后，三个 seed 的 epoch10 actor parameter drift、deterministic action MSE/max error、observation mean/variance/count drift均为 0；critic relative L2 分别为 0.1954、0.1907、0.1836。七项预注册预测均按原文验证为 `supported`，其中 drift 与闭环指标仅作统计/观测关联，不作严格因果解释。

完整证据：

```text
docs/p8b_actor_preserving_ppo.md
benchmarks/phase8b_actor_preserving_ppo/README.md
benchmarks/phase8b_actor_preserving_ppo/summary.json
benchmarks/phase8b_actor_preserving_ppo/comparison.md
benchmarks/phase8b_actor_preserving_ppo/validation_selection.json
benchmarks/phase8b_actor_preserving_ppo/prediction_verification.json
benchmarks/phase8b_actor_preserving_ppo/policy_drift.csv
benchmarks/phase8b_actor_preserving_ppo/videos/video_manifest.json
```

## 项目索引

```text
.
├── benchmarks/
│   ├── phase6a_heave_precision/
│   ├── phase6b_physical_deck/
│   ├── phase6c_physical_deck_attitude/
│   ├── phase7_imitation_hybrid/
│   ├── phase8a_checkpoint_selection/
│   └── phase8b_actor_preserving_ppo/
├── docs/
│   ├── README.md
│   ├── p7_imitation_hybrid_paper.md
│   ├── interview_p7_evidence.md
│   └── runtime_display_troubleshooting.md
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
| P7 理论 | 论文式建模、算法、公式、实验与代码映射 | `docs/p7_imitation_hybrid_paper.md` |
| P7 证据 | 专家数据、BC、BC+PPO benchmark | `benchmarks/phase7_imitation_hybrid/README.md` |
| P7 面试 | 可使用表述、证据和失败归因 | `docs/interview_p7_evidence.md` |
| P8A | 周期 checkpoint 指标选模与 policy drift 诊断 | `benchmarks/phase8a_checkpoint_selection/README.md` |
| P8B | actor-preserving PPO、validation 选模、独立 test、drift 与视频 | `benchmarks/phase8b_actor_preserving_ppo/README.md` |
| Display/GUI | 默认 display、headless、SSH、tmux、Docker | `docs/runtime_display_troubleshooting.md` |
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

本地图形终端查看 frozen PPO teacher：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint && \
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=1 \
  --checkpoint=logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/expanded_from_p6b_ep990_16to22.pth
```

该命令必须在已登录 Ubuntu 桌面的图形终端中执行，不要添加 `--headless`。BC-only、BC+PPO、scratch 的可直接复制命令见 `docs/runtime_display_troubleshooting.md`。

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

P8A 扫描、选模、正式测试和 drift 复现命令：

```text
benchmarks/phase8a_checkpoint_selection/commands.txt
```

## 文档同步检查

单独检查论文参数与代码一致性：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_p7_documentation_sync.py
```

## Display 与 GUI

P8B 收尾会话中已通过 Isaac Sim `--headless --enable_cameras` 的 `rgb_array` 离屏渲染生成真实成功和真实 `deck_miss` MP4，并用 terminal latch、sidecar SHA256 与 `ffprobe` 做自动结构校验。自动生成不等于人工目视验收，因此所有视频 manifest 仍明确记录 `human_review_completed=false`。完整诊断与 GUI 启动方式见：

```text
docs/runtime_display_troubleshooting.md
```

## 当前客观局限

- demonstration 仅保留 teacher 成功回合，off-distribution recovery 覆盖不足。
- PPO critic 从随机值头开始，在线更新容易破坏已经较强的 BC actor。
- P7 原 checkpoint 选择依据是公共训练循环的 mean episode reward；P8A 已证明它与 settled landing 最优 checkpoint 不一致。
- 当前运动分布不包含 yaw、随机波谱、水动力或完整船舶六自由度运动。
- P8B 已生成 targeted headless 成功和 `deck_miss` 视频，但未声称完成人工 GUI 目视验收。

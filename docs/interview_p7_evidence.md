# P7 专家数据、Behavior Cloning 与 BC+PPO 面试证据

完整论文式建模、公式和代码映射见：

```text
docs/p7_imitation_hybrid_paper.md
```

本文件侧重面试表述和可核验数字；理论文档侧重问题建模、算法推导、实验设计及实现可追溯性。

## 1. 问题定义

P7 在冻结的 P6C 实体运动甲板降落任务上比较四种策略：

1. frozen PPO teacher/reference；
2. PPO from scratch；
3. Behavior Cloning only；
4. BC initialized PPO。

任务 ID：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

策略输入为 22 维状态观测，输出为 4 维连续控制动作。当前策略为 **state-based policy，不包含相机图像或真实视觉投影输入**。

## 2. Expert 来源

Expert 使用冻结的 P6C PPO checkpoint：

```text
logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/
expanded_from_p6b_ep990_16to22.pth
```

SHA256：

```text
95424bb0d6b98d8dfbf2455d6fd84e99a77d52bca28489654036a25aea5a697d
```

其正式三随机种子评估为 768 episodes，settled landing 94.66%，contact success 99.87%。采集时使用 deterministic mean action，并在送入环境前明确执行 `[-1, 1]` clamp。

## 3. Expert 数据集

数据集真实规模：

```text
successful episodes: 3976
transitions: 540321
collection seeds: 42, 43, 44
rejected/non-success episodes during collection: 201
```

按完整 episode 划分，禁止 transition 随机划分：

| split | episodes | transitions |
| --- | ---: | ---: |
| train | 3180 | 434288 |
| validation | 397 | 54132 |
| test | 399 | 51901 |

阶段覆盖：

| phase | transitions | proportion |
| --- | ---: | ---: |
| approach | 79119 | 14.64% |
| align | 187752 | 34.75% |
| descent | 251353 | 46.52% |
| contact/settle | 22097 | 4.09% |

训练时采用 inverse-frequency phase weighting，避免 contact/settle 样本被其他阶段淹没；原始未抽样 shard 保留不变。

每条 transition 至少保存：episode/step/seed、raw observation、实际执行 action、reward、terminated/time-out、flight phase、deck 运动参数，以及 touchdown/contact 结果。数据使用压缩 NPZ 分片、manifest、逐 shard SHA256、可恢复采集和严格 schema 校验。

Manifest：

```text
logs/imitation/p7_expert_dataset/manifest.json
SHA256: 72847f6c9bb6f6c2c10d2f9862dd894ccd91c5ecb69bbbe948120a0ab8f64744
```

## 4. BC 网络与训练

网络与 PPO actor 保持一致：

```text
22 -> Linear(64) -> ELU -> Linear(64) -> ELU -> Linear(4)
```

关键语义：

- 输入是未归一化的 22 维环境观测；
- 使用 teacher checkpoint 中冻结的 RL-Games running mean/variance；
- normalization epsilon 和 clamp 与 RL-Games 推理一致；
- 标签是 teacher 最终执行的 deterministic action；
- 输出按环境约束 clamp 到 `[-1, 1]`；
- 损失为 action MSE，并使用阶段权重；
- 按 validation loss 选择 checkpoint。

训练配置：50 epochs、batch size 4096、Adam、learning rate `1e-3`、seed 42。最佳 epoch 为 50，训练耗时约 234.33 s。

离线 test split：

```text
weighted action MSE: 1.66546e-4
per-action MSE:
  action 0: 4.47829e-4
  action 1: 6.99266e-5
  action 2: 5.38782e-5
  action 3: 9.45484e-5
```

BC checkpoint：

```text
logs/imitation/p7_bc/best_bc.pth
SHA256: 44d49d53cabffd14e6e1a0f8639b422ae4f13881c9b5f24d36b801f4a9d93c67
```

离线 MSE 只用于检查拟合，不作为成功结论；最终结论来自闭环环境评估。

## 5. BC 到 PPO 的迁移

迁移工具执行：

- actor MLP 和 mu head复制 BC 权重；
- observation running mean/variance/count 完整复制；
- value head 使用固定 seed 的 `torch.nn.Linear` 默认初始化；
- value normalization 重置；
- PPO optimizer state 清空；
- epoch、frame、历史 reward 清零；
- env state 清空；
- fixed sigma 保持 P6C PPO 配置；
- 写入来源 BC、dataset hash、template checkpoint 和迁移元数据。

生成 checkpoint：

```text
logs/imitation/p7_bc/bc_init_rlgames.pth
SHA256: 0669022b8b88af22a3ef3a3baebdd1d3c32c26ff0095ef9e2a2498b3e3aa7517
```

固定 observation batch 上，standalone BC 与恢复后的 RL-Games deterministic actor 最大动作误差为 0，满足 `<1e-5` parity 要求。

## 6. 公平对比设计

PPO-from-scratch 与主 BC+PPO 实验统一：

```text
task: same frozen P6C task
training seeds: 42, 43, 44
num_envs: 256
horizon_length: 24
steps per epoch: 6144
maximum epochs: 200
maximum online environment steps per seed: 1,228,800
PPO learning rate/config: same default configuration
checkpoint rule: highest RL-Games rolling mean episode reward
formal evaluation: same evaluator, 64 envs, 256 episodes/seed
```

BC+PPO 与 scratch 的唯一主实验差异是 actor/normalization 初始化；critic value head、optimizer 和训练进度均为 fresh state。

固定学习曲线检查点使用 epoch 20/50/100/150/200，每个点评估 3 seeds × 128 episodes。正式最终结果使用公共 checkpoint 选择规则保存的 checkpoint，再执行 3 seeds × 256 episodes。

## 7. 关键闭环结果

| method | settled landing | contact success | hard contact | deck miss | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen PPO teacher | 94.66% ± 0.49% | 99.87% | 0.13% | 5.34% | 0.00% |
| PPO from scratch | 0.91% ± 0.66% | 10.55% | 5.73% | 21.61% | 73.57% |
| BC only | 88.28% ± 0.55% | 99.74% | 0.26% | 11.72% | 0.00% |
| BC initialized PPO | 76.69% ± 9.16% | 95.05% | 0.91% | 23.05% | 0.00% |

BC-only 达到预设 `>=80%` 目标，证明纯监督策略能够闭环完成大多数运动甲板降落，但仍落后于 teacher 约 6.38 个百分点。

BC+PPO 未达到 `>=92%`。其初始策略在 0 online environment steps 时就是 88.28% BC policy，因此达到 80% 的 online step 数为 0；但固定评估点和最终 checkpoint 均未达到 90% 或 92%。PPO-from-scratch 在 1,228,800 steps 内未达到 80%、90% 或 92%。

因此可以准确地说“BC 显著改善有限预算下的初始策略”，但不能说“BC+PPO 最终超过 BC”或“BC+PPO 达到 90% 的样本效率优于 scratch”。

## 8. Failure analysis

### 8.1 BC-only

BC-only contact success 为 99.74%，但 settled 仅 88.28%，主要差距体现为 deck miss 增加到 11.72%。这说明离线 action MSE 很低并不等于完全消除闭环 covariate shift；成功 demonstrations 对偏离 teacher state distribution 后的恢复动作覆盖不足。

### 8.2 PPO-from-scratch

在当前 1.23M steps/seed 预算内，scratch 主要 failure 为 timeout，聚合达到 73.57%。该任务包含移动/升沉/倾斜实体甲板、真实接触和严格 settle 条件，当前预算不足以从随机策略学习完整 approach-align-descent-contact 链路。

### 8.3 BC+PPO

主 BC+PPO 实验在更新后从 88.28% 降至 76.69%，同时 seed std 扩大到 9.16%。表现为 contact 仍高，但 deck miss 增加，说明策略能够接近/接触甲板，却破坏了 BC 已学到的精细对准和稳定触地行为。

最可能原因：

- critic cold start 使早期 advantage 估计噪声较大；
- PPO reward 与 settled landing 指标并非完全同构；
- 公共 mean-reward checkpoint 规则不能保证选择 settled 最优 checkpoint；
- demonstration 只包含成功轨迹，缺少 recovery data；
- online policy drift 比继续优化带来的收益更大。

基于该诊断仅执行一次针对性修正：将 BC+PPO 配置初始学习率从 `1e-4` 降为 `1e-5`，其他设置不变。修正后三种子正式 settled landing 为 63.67% ± 18.02%，更差且方差更高，因此保留为负结果并停止继续调参。

## 9. 可视化与轨迹证据

已保存：

```text
benchmarks/phase7_imitation_hybrid/rollout_cases/teacher_success.npz
benchmarks/phase7_imitation_hybrid/rollout_cases/bc_success.npz
benchmarks/phase7_imitation_hybrid/rollout_cases/bc_ppo_success.npz
benchmarks/phase7_imitation_hybrid/rollout_cases/scratch_timeout_failure.npz
```

每个 NPZ 包含逐步 raw observation、action、reward、flight phase、机器人/甲板世界坐标位置、姿态和速度；JSON sidecar 保存 checkpoint hash、seed、终止结果和接触指标。

当前自动执行 shell 中 `DISPLAY`、`WAYLAND_DISPLAY` 和 `XDG_SESSION_TYPE` 均为空，且不存在本地 X11 socket，因此无法打开交互式 GUI。客观状态轨迹和复现实验命令已保存。这里不能把“无交互 display”误写为“headless 永远无法录制 MP4”；当前未生成视频的直接原因还包括 rollout 脚本只实现了数值轨迹记录，没有实现 render-enabled recorder。完整说明见 `docs/runtime_display_troubleshooting.md`。

## 10. 局限

- 任务是 state-based，不是视觉端到端策略。
- demonstrations 仅保留成功回合，失败恢复覆盖不足。
- 当前 motion distribution 为 xy 匀速、正弦升沉及最大 5° roll/pitch，不包含 yaw、随机波谱、水动力或完整六自由度船舶运动。
- scratch 结论只适用于当前 1.23M online steps/seed 预算，不能外推为“PPO 无法学会”。
- BC+PPO 使用公共 reward checkpoint 规则；未来更合理的方案应加入独立 validation rollout checkpointing、critic warm-up 或 actor update protection，但这些不属于本次已验证结果。

## 11. 简历中允许使用的准确表述

```text
基于冻结 PPO teacher 构建无人机运动甲板降落 imitation-learning 流程，采集 3976 条成功轨迹、54.0 万状态-动作样本，并按 episode 完成 80/10/10 数据划分与分片校验；实现与 PPO actor 对齐的 22-64-64-4 Behavior Cloning、RL-Games observation normalization 复用和 BC→PPO checkpoint 迁移，迁移前后确定性动作误差小于 1e-5。BC-only 在 3 个随机种子、共 768 回合闭环评估中取得 88.28% settled landing；进一步发现直接 PPO fine-tuning 会产生 policy drift，并通过对照实验定位 critic cold start、reward/settled 指标不一致和 demonstration recovery coverage 不足等问题。
```

更短版本：

```text
完成 PPO expert trajectory → Behavior Cloning → BC-initialized PPO 全流程；构建 54.0 万 transition 数据集，BC-only 三种子闭环稳定降落率 88.28%，并通过公平对照与失败归因验证直接 PPO 微调会破坏 BC 策略。
```

## 12. 不允许声称的内容

不得声称：

- 使用了相机图像、真实视觉投影特征或视觉端到端输入；
- BC+PPO 达到 92% settled landing；
- BC+PPO 最终优于 BC-only；
- BC+PPO 达到 90% 的交互样本效率优于 scratch；
- 已完成人工 GUI/视频验收；
- 当前结果覆盖 yaw、随机海浪、水动力或真实船舶六自由度运动；
- 离线 action MSE 代表闭环成功。

所有可复核数字、原始 CSV、图表、训练配置和 checksum 位于：

```text
benchmarks/phase7_imitation_hybrid/
```

# P8B：Actor-Preserving PPO（保守策略微调）

> 文档状态：**设计与实验预注册版本**。本版本在任何 P8B 核心训练代码、pilot 或正式实验之前写入。实验后只追加真实实现差异、结果和预测验证，不删除未被支持的预测。

## 1. 问题定义与冻结边界

P7 的 BC actor 在 P6C 实体倾斜运动甲板上已具有较高稳定降落率，但普通在线 PPO 会明显破坏策略；P8A 进一步发现三个训练 seed 的任务指标最优 checkpoint 都出现在最早保存的 epoch 10，而 reward-selected checkpoint 只有 78.08% settled landing。P8A metric-selected 达到 91.67%，仍未稳定达到 92%。

P8B 的问题定义为：在固定 P6C 任务语义、既有 BC 初始化和每 seed 200 epochs/1,228,800 environment steps 预算下，减少 early actor drift，使 PPO 改善或至少不快速破坏 BC 策略。

以下内容冻结：任务 ID、动力学、甲板模型、22 维 observation、4 维 action、动作缩放、reward、termination、接触/稳定/失败判定、训练预算和正式评估指标。P8B 不修改 reward，不放宽 success 或 safety，不更换 expert dataset，不减少正式 episodes，不隐藏失败 seed。当前策略是 state-based，不包含相机图像。

## 2. 历史证据

P7 三 seed、每 seed 256 episodes：teacher 94.66% ±0.49%，PPO scratch 0.91% ±0.66%，BC-only 88.28% ±0.55%，普通 BC+PPO 76.69% ±9.16%，learning-rate=1e-5 诊断 63.67% ±18.02%。

P8A 使用 validation seeds 145/146/147 选模、formal test seeds 245/246/247 独立测试。train seeds 42/43/44 均选择 epoch10；metric-selected 91.67% ±3.03%，reward-selected 78.08% ±8.66%。epoch10 action MSE 相对 BC 约为 `6.1e-4`、`6.5e-4`、`9.2e-4`。这些是统计证据，不足以单独证明 actor drift 是唯一因果机制。

## 3. 候选方案比较

| 候选 | 优点 | 当前不单独采用的原因 |
|---|---|---|
| A 仅降低全局 learning rate | 实现简单 | P7 的 1e-5 诊断更差且方差更大；不能隔离 critic 梯度和 RMS drift |
| B 仅按 validation settled landing 早停 | 避免后期退化 | P8A 已采用，仍未稳定达到 92%；不能阻止 epoch10 前漂移 |
| C actor/critic 分离 | 直接消除 value loss 对 actor feature 的梯度 | 必须处理 checkpoint migration 与 optimizer schema |
| D critic-only warm-up | 先改善 cold critic | 需精确定义 epoch、resume 和 sigma/RMS 冻结 |
| E 冻结 observation RMS | actor 权重不变时保持策略输入函数 | 只冻结 RMS 不能限制 PPO actor update |
| F BC action L2 anchor | 与 P8A action MSE 对齐，梯度直接 | coefficient 需受控 pilot，不保证闭环必然改善 |
| G KL anchor | 概率分布解释明确 | fixed sigma 下与加权 L2 等价，但实现和数值诊断更复杂 |
| H 离线 replay BC anchor | 保持 demonstration 分布动作 | 容易与 on-policy 状态分布脱节，额外混合比例 |
| I DAgger/recovery demonstrations | 可改善 covariate shift | 需要重新采集和标注，超出本阶段范围 |
| J residual policy | 强保护 BC baseline | 改变策略参数化和比较语义，工程复杂度更高 |

本阶段采用：**separate actor/critic + 10 epochs critic-only warm-up + frozen observation RMS + on-policy BC mean-action L2 anchor + validation settled landing 选模**。它直接针对 P7/P8A 的共享梯度、cold critic、RMS 变化和 early action drift，不改变环境、不需要新 demonstration，且可通过 hash、parity、gradient isolation 和闭环评估证伪。

暂不采用完整 DAgger、大规模 recovery 数据、residual policy、二阶 trust-region 或大规模超参数搜索。

## 4. Actor/Critic 分离

共享网络为

\[
h=f_\phi(o),\quad \mu=g_\theta(h),\quad V=v_\psi(h).
\]

P7 配置 `network.separate: false`，RL-Games 的 Adam optimizer 包含 model 全部参数，真实总损失为

\[
L=L_{actor}+0.5c_vL_{value}-c_eH+c_bL_{bounds}.
\]

因此共享参数梯度为

\[
\nabla_\phi L=\nabla_\phi L_{actor}+0.5c_v\nabla_\phi L_{value}-c_e\nabla_\phi H+c_b\nabla_\phi L_{bounds}.
\]

P8B 改为

\[
\mu_\theta(o)=g_\theta(f_\theta(o)),\qquad V_\psi(o)=v_\psi(f_\psi(o)),
\]

目标是 \(\partial L_{value}/\partial\theta=0\)。

参数集合预定义为：

- actor：`model.a2c_network.actor_mlp.*`、`model.a2c_network.mu.*`；
- critic：`model.a2c_network.critic_mlp.*`、`model.a2c_network.value.*`；
- fixed sigma：`model.a2c_network.sigma`，属于策略分布但 warm-up 冻结；
- observation RMS buffers：`model.running_mean_std.running_mean/running_var/count`，全程冻结；
- value RMS buffers：`model.value_mean_std.*`，按 RL-Games value normalization 继续更新；
- BC reference actor：独立、eval、`requires_grad=False` 的 actor MLP 与 mu head；
- optimizer：Adam 仍包含 actor、critic、sigma 的可训练参数；warm-up 用 `requires_grad` 显式冻结 actor/sigma，critic optimizer state 保留。

单元测试必须证明 actor/critic 无共享 storage、value backward 无 actor gradient、actor backward 无 critic gradient。

## 5. Shared→Separate Checkpoint Migration

输入为 P7 `bc_init_rlgames.pth` 和 P8B separate template/schema。迁移规则：

- `a2c_network.actor_mlp.*` → 同名 actor keys，逐 tensor 完全复制；
- `a2c_network.mu.*` → 同名 mu keys，完全复制；
- shared MLP → `a2c_network.critic_mlp.*` 使用固定 `critic_seed=2026` 独立初始化，不复制 BC actor；
- value head使用同一固定 seed 可复现初始化；
- observation RMS 完全复制；
- value RMS 重置为 mean 0、variance 1、count 1；
- fixed sigma 从统一配置/template 复制；
- optimizer state 清空并按新参数结构重建；
- epoch/frame/history 置零，env_state 清空；
- 保存 source checkpoint 路径与 SHA256、dataset manifest SHA256、schema version、network shape 和初始化 seed；
- 输出路径存在时拒绝覆盖。

parity 在同一 raw observation、同一 RMS、eval 模式和 action clamp 下比较 standalone BC 与 separate RL-Games actor：

\[
\epsilon_{parity}=\max|\operatorname{clip}(\mu_{BC},-1,1)-\operatorname{clip}(\mu_{P8B},-1,1)|\le10^{-5}.
\]

parity 不通过时禁止训练。

## 6. Critic-Only Warm-Up

RL-Games 在每轮优化前先执行 `update_epoch()`，epoch 从 0 自增到 1，再进入 `train_epoch()`。因此定义：

\[
\text{warmup\_active}\iff 1\le epoch\_num\le10.
\]

checkpoint `epoch=10` 是完成第 10 次 critic-only 更新后的边界快照；epoch11 是首次允许 actor/sigma 更新的 epoch。warm-up 期间 actor 仍用于 rollout 和前向概率计算，但 actor、bounds、entropy 和 anchor 项不进入反向更新；只最小化 `0.5*critic_coef*value_loss`。actor 与 sigma `requires_grad=False`，BC reference 永远冻结。observation RMS 全程 eval；value RMS 保持 RL-Games 设计。

warm-up 结束时不重建 optimizer：critic Adam moments 保留；actor/sigma 因没有 gradient 不产生 Adam state，epoch11 首次梯度时自然初始化其 optimizer state。**warm-up 期间冻结 adaptive KL scheduler 和 optimizer learning rate 为基础值 `1e-4`**；否则 policy KL≈0 会被 RL-Games scheduler 解释为“更新过小”，连续放大学习率，造成 epoch11 首次 actor update 突跳。epoch11 起才恢复 adaptive scheduler，并从基础 learning rate 启动。resume 根据 checkpoint 的 `epoch` 和 P8B metadata 恢复，下一 epoch 自动决定 warm-up 与 scheduler 状态。

checkpoint metadata 和日志至少记录 epoch、warmup_active、actor_trainable、critic_trainable、actor/critic hash、gradient norm、parameter delta、RMS hash、anchor loss。

## 7. Frozen Observation RMS

BC 统计量 \(m_{BC},v_{BC},n_{BC}\) 定义

\[
\hat o=\operatorname{clip}\left(\frac{o-m_{BC}}{\sqrt{v_{BC}+10^{-5}}},-5,5\right).
\]

P8B 全训练和 resume 后保持

\[
m_t=m_{BC},\quad v_t=v_{BC},\quad n_t=n_{BC}.
\]

实现上每次 RL-Games `set_train()` 后重新把 `model.running_mean_std` 置于 eval，并禁止任何隐式统计更新；不冻结 `value_mean_std`。需要区分 observation RMS、value RMS、reward scale 0.01 和 batch advantage normalization。actor hash 不变不保证动作不变，故同时检查 actor weight hash、RMS hash 和固定 raw observation 上的 deterministic action。

## 8. BC Mean-Action L2 Anchor

rollout observations 来自当前 on-policy 分布。current actor 与 reference actor接收同一个冻结 RMS 归一化结果：

\[
L_{BC-anchor}=\mathbb E_{o\sim d_{\pi_\theta}}
\left[\|\mu_\theta(\hat o)-\operatorname{stopgrad}(\mu_{BC}(\hat o))\|_2^2\right].
\]

源码使用所有 batch/action 元素的 mean。tensor shape：`raw_obs [B,22]`、`normalized_obs [B,22]`、`current_mu [B,4]`、`bc_mu [B,4]`、`anchor_per_element [B,4]`、`anchor_loss []`。

anchor 使用 deterministic pre-clamp、pre-sampling mean；当前 mu activation 为 None。reference actor eval、无 gradient；anchor 只对 current actor MLP/mu 产生梯度，不对 critic、sigma 或 RMS 产生梯度。PPO 真正最小化：

\[
L_{total}=L_{PPO-actor}+0.5c_vL_{value}-c_eH+c_bL_{bounds}+\lambda_{BC}L_{BC-anchor}.
\]

固定相同协方差时

\[
D_{KL}(\mathcal N(\mu_{BC},\Sigma)\|\mathcal N(\mu_\theta,\Sigma))
=\frac12(\mu_\theta-\mu_{BC})^\top\Sigma^{-1}(\mu_\theta-\mu_{BC}).
\]

因此 L2 与 KL 有明确关系；L2 更直接对应 P8A action MSE，且 \(\lambda=0\) 可验证退化为无 anchor 控制组。

P7 首轮 actor loss 约 0.03，epoch10 action MSE 约 `6e-4`–`9e-4`。预注册 pilot 系数为 0、10、50：系数 10 对应约 0.006–0.009 的约束量级，系数 50 对应约 0.03–0.045，与早期 actor loss 同量级。禁止再增加候选或按 seed 使用不同系数。

reference actor state_dict 写入每个 P8B checkpoint，同时记录源 BC SHA256；resume 时加载并校验，不依赖可被移动的外部路径。

## 9. Checkpoint Selection 与 Policy Drift

每个正式训练 seed 保存 epoch0、epoch10 warm-up 边界、每 10 epochs、last 和 reward-selected checkpoint。validation 前先按 actor+RMS SHA 去重，但保留 checkpoint SHA。选择规则沿用 P8A 真实代码：最大 settled landing、最小 deck miss、最小 hard contact、最小 touchdown distance、最早 epoch。

固定 drift observation batch 沿用 P8A dataset test split，报告 checkpoint/actor/critic SHA、obs RMS hashes/count、action MSE/max error、actor/critic relative L2、RMS drift、sigma drift。action drift 与 settled landing 只作统计关联，不作因果结论。

## 10. Pilot 与正式实验预注册

Pilot 使用同一 BC migration、training seed 42、30 epochs、256 envs、checkpoint interval 10、validation seeds 145/146/147、每 seed 128 episodes。候选只为 \(\lambda\in\{0,10,50\}\)，其余配置相同。选择顺序：最大 validation settled landing；最小 deck miss；最小 hard contact；最小 touchdown distance；更低 action drift 仅作后续 tie-break；仍相同时选择更小系数。

最多允许一次有明确失败诊断的统一修正；必须先更新本文，不使用 formal test 结果。

正式训练 seeds 42/43/44，每 seed 200 epochs、256 envs、horizon 24，即 1,228,800 environment steps。validation seeds 145/146/147，每 checkpoint 每 seed 128 episodes。formal test seeds 245/246/247，每选中 checkpoint 每 seed 256 episodes。formal test 绝不参与 coefficient 或 checkpoint 选择。

正式比较至少包含：BC-only、P8A metric-selected、P8B metric-selected、P8B reward-selected、P8B last。安全目标沿用 P6C：ground crash ≤1%、hard contact ≤2%、timeout ≤3%；性能参考目标 settled landing ≥92%。未达到时保留负结果并完成诊断。

## 11. 机器可解析唯一参数块

```yaml
p8b_preregistered_config:
  task_id: Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
  observation_dim: 22
  action_dim: 4
  network:
    separate: true
    units: [64, 64]
    activation: elu
    fixed_sigma: true
  migration:
    schema_version: p8b-separate-v1
    critic_seed: 2026
    parity_max_abs_error: 1.0e-5
  warmup_epochs: 10
  warmup_active_epoch_max: 10
  freeze_lr_scheduler_during_warmup: true
  freeze_observation_rms: true
  bc_anchor:
    type: mse_mean_action
    coefficient: 10.0
    pilot_candidates: [0.0, 10.0, 50.0]
    reduction: mean_all_elements
    action_representation: deterministic_pre_clamp_mean
  ppo:
    learning_rate: 1.0e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    critic_coef: 2.0
    entropy_coef: 0.0
    bounds_loss_coef: 1.0e-4
    reward_scale: 0.01
    normalize_input: true
    normalize_value: true
    normalize_advantage: true
    horizon_length: 24
    minibatch_size: 384
    mini_epochs: 5
    max_epochs: 200
  checkpoint:
    frequency_epochs: 10
    save_epoch_zero: true
    save_warmup_boundary: true
  pilot:
    training_seed: 42
    max_epochs: 30
    episodes_per_validation_seed: 128
  seeds:
    training: [42, 43, 44]
    pilot_validation: [145, 146, 147]
    formal_validation: [145, 146, 147]
    formal_test: [245, 246, 247]
  evaluation:
    validation_episodes_per_seed: 128
    episodes_per_seed: 256
    target_settled_landing: 0.92
    target_ground_crash_max: 0.01
    target_hard_contact_max: 0.02
    target_timeout_max: 0.03
```

该块是 P8B 参数唯一文档源。pilot 后如系数 0 或 50 被选中，必须以独立 Git 历史更新 `coefficient` 和正式 YAML，并说明选择证据；不得维护第二个冲突参数块。

## 12. 可证伪预测

1. epoch1–10 actor SHA256 不变。
2. warm-up 中 BC 与 current deterministic action max error 接近 0，并满足 `≤1e-5`。
3. warm-up 中 critic 参数/hash 发生变化。
4. 冻结 RMS 后 running mean、variance、count drift 为 0。
5. 正 anchor 在相同 epoch 的 action MSE 低于无 anchor 控制组。
6. 更低 action drift 不保证 settled landing 必然提高；若 early drift 是主要机制，应至少减缓快速退化。
7. validation settled landing 选模优于只按 training reward 选模。

实验后用“预先预测—实际结果—是否支持—证据”表逐条回填，不删除不支持项。

## 13. 分阶段实现前置说明

### A. Separate checkpoint migration

输入为 P7 BC-init checkpoint、dataset manifest 与 P8B config；输出为不覆盖旧文件的 separate checkpoint。验证 shape/key、来源 SHA、actor/RMS parity、actor/critic storage isolation、optimizer/epoch reset。

### B. Critic-only warm-up

输入为 on-policy rollout batch；epoch1–10 仅 value loss 反向，actor/sigma 冻结，同时 adaptive scheduler 不更新且 optimizer LR 固定为 `1e-4`。epoch11 从基础 LR 恢复 adaptive 调度。单测覆盖 epoch9/10/11、hash、resume、optimizer state 和 scheduler/LR 边界。

### C. Frozen observation RMS

输入 raw obs，输出固定 BC normalization；train/eval/resume 均不更新 mean/var/count。单测在若干 update 后检查 RMS hash与 action parity。

### D. BC action anchor

输入 `[B,22]` raw obs，输出 scalar MSE；梯度仅流入 current actor。测试零系数退化、同权重零 loss、扰动增大、NaN/Inf 拒绝。

### E. Checkpoint selection

输入 validation CSV/manifest，输出每 train seed 唯一 checkpoint；复用 P8A tie-break，formal seeds 不进入输入。

### F. Policy drift evaluation

输入固定版本化 observation batch和 checkpoint inventory，输出 CSV/JSON/plots；hash 与 drift 分开，相关性只作描述。

### G. Headless video 与 GUI 验收

优先用 Isaac Sim offscreen/video recorder 生成一成功一失败案例与 manifest。无真实 DISPLAY 时不声称完成人眼 GUI 验收，只记录 headless 产物和用户唯一必要播放命令。

## 14. 理论—代码—配置—测试—证据映射（预注册）

| 理论内容 | 代码位置/函数 | 配置键 | 单元测试 | 实验证据 |
|---|---|---|---|---|
| separate migration | 预定 `imitation/p8b_checkpoint.py::build_p8b_separate_checkpoint` | `migration.*`, `network.*` | `test_p8b_checkpoint_migration.py` | `checkpoint_hashes.json` |
| gradient isolation | RL-Games separate network + 预定 helper | `network.separate` | `test_p8b_actor_critic_separation.py` | warm-up logs/hashes |
| warm-up | 预定 `imitation/p8b_agent.py::ActorPreservingA2CAgent` | `warmup_epochs` | `test_p8b_critic_warmup.py` | training metadata |
| frozen RMS | 同 agent `_enforce_frozen_obs_rms` | `freeze_observation_rms` | `test_p8b_frozen_obs_rms.py` | RMS hashes |
| BC anchor | 同 agent `_bc_anchor_loss` | `bc_anchor.*` | `test_p8b_bc_anchor.py` | TensorBoard/policy drift |
| checkpoint selection | `imitation/checkpoint_sweep.py::select_validation_best` | seeds/evaluation | `test_p8b_checkpoint_selection.py` | validation CSV/selection JSON |
| drift | 预定 `imitation/p8b_drift.py` | fixed batch/schema | `test_p8b_policy_drift.py` | `policy_drift.csv/json` |
| 文档同步 | 本文参数块 + P8B YAML/defaults | 全部预注册键 | `test_p8b_documentation_sync.py` | pytest report |
| 正式指标 | `scripts/rl_games/eval_metrics.py` | episodes/seeds/thresholds | eval smoke | `formal_test_results.csv`, `formal_aggregate.json` |

## 15. 当前理论前置结论

没有未解决的任务语义歧义。已明确的工程验证点是：RL-Games custom agent 的注册入口、checkpoint save/restore metadata、warm-up epoch 边界、adaptive scheduler 边界和 observation RMS train-mode 强制恢复。只有文档同步、migration parity、gradient isolation、warm-up hash、LR freeze 和 RMS freeze 测试全部通过后才允许启动 pilot。

## 16. 冒烟前设计修正记录

2026-08-05 的首个 16-env、12-epoch Isaac Sim smoke 验证了 actor/RMS 在 epoch1–10 不变、critic 持续更新、epoch11 actor 开始更新；但还观察到 epoch11 actor parameter delta 约 1.13。根因是原预注册允许 adaptive KL scheduler 在 warm-up 中运行，零 policy KL 使 RL-Games 连续提高 learning rate。该现象发生在 pilot 和 formal experiment 之前，不能作为配置优劣结果。

因此先修改理论再修改代码：新增 `freeze_lr_scheduler_during_warmup: true`，epoch1–10 的 optimizer LR 固定为基础 `1e-4`，epoch11 从基础值恢复 adaptive scheduler。该修正对所有候选和 seed 一致，不修改环境、训练预算、anchor 候选、validation 或 formal test 协议。首个 smoke 作为失败诊断证据保留在 `logs/rl_games/p8b_smoke/seed942`，不作为正式结果。

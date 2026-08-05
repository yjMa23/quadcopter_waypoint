# P8B：Actor-Preserving PPO（保守策略微调）

> 文档状态：**设计预注册、pilot、正式训练、validation 选模、独立 formal test、drift、targeted headless 视频和可复现实验包均已完成**。初始设计在任何 P8B 核心训练代码、pilot 或正式实验之前写入；实验后只追加真实结果，不删除或改写原预测。

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

Pilot 已按上述协议完成，33 个 checkpoint×seed 评估全部成功，formal test seeds 未使用。每个 coefficient 按相同规则选出的最佳 checkpoint 为：

| coefficient | selected epoch | settled landing | deck miss | hard contact | touchdown distance | action MSE vs BC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 88.0208% | 11.9792% | 0.2604% | 0.05908 m | 0 |
| 10 | 20 | 79.4271% | 20.3125% | 1.3021% | 0.06609 m | 2.8872e-4 |
| 50 | 30 | **94.5312%** | **5.4688%** | 0.7812% | **0.05098 m** | 1.5900e-4 |

因此唯一选择 \(\lambda_{BC}=50\)。相同 epoch30 下，action MSE 为 λ=0 `8.8090e-4`、λ=10 `6.5034e-4`、λ=50 `1.5900e-4`，支持“正 anchor 降低 drift”的 pilot 预测；闭环优势仍只能由受控 pilot 结果表述，不能把 action MSE 与 settled landing 的关系推广为普遍因果。pilot 后未发生额外设计修正。

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
    coefficient: 50.0
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

该块是 P8B 参数唯一文档源。pilot 已选择系数 50，并同步更新正式 YAML、文档同步测试和 `benchmarks/phase8b_actor_preserving_ppo/preregistered_config.yaml`；不得维护第二个冲突参数块。

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

## 17. 正式 validation 与 checkpoint selection

formal validation 使用冻结的 training seeds 42/43/44、validation seeds 145/146/147、每 checkpoint/seed 128 episodes。每个 training seed 的最终统一网格为 22 个 checkpoint × 3 validation seeds = 66 条评估；三个 manifest 共 198 条、25,344 episodes。由于全局 BC epoch0 在三个 manifest 中重复，按 checkpoint SHA 去重后的 selection 输入为 192 条、24,576 episodes。seed42 曾有一条 `num_envs=64` 中断记录；原记录未删除，已标记 `superseded`，最终选模只使用完整一致的 `num_envs=48` 网格。

固定 tie-break 得到：

| training seed | metric-selected epoch | 类型 | validation settled | deck miss | hard contact | touchdown distance |
|---:|---:|---|---:|---:|---:|---:|
| 42 | 91 | reward-selected | 98.4375% | 1.5625% | 0.0000% | 0.05426 m |
| 43 | 30 | periodic | 96.6146% | 3.3854% | 0.0000% | 0.05660 m |
| 44 | 51 | reward-selected | 94.7917% | 5.2083% | 0.2604% | 0.05646 m |

reward-selected epochs 分别为 91、128、51；last 均为 epoch200。seed42 和 seed44 的 metric-selected 与 reward-selected 是同一物理 checkpoint，seed43 不同。selection 文件在 formal test 启动前写入，且明确记录 `formal_test_seeds_used_for_selection=false`。

正式 migration checkpoint 哈希重新计算为：

```text
checkpoint: ff87150ad17c726494cd2dd6bb9d8666e3849658707cc921b68cc61eae111249
actor weights: 4a917d9f5ff41bd59f10ce964cb8838ef529600c117fff94e9c2d576cefe1b77
critic weights: a23f8c5f6c10ea7af7f4a11b79c279b6a1a055af05ac4bd3f7b9201bdb0f30e1
observation RMS: 7aa08af9f501e50f41452eae8eb99ad818c4bad7508a2cc864d4cfff2a05d4f3
```

旧 sweep inventory 的 `actor_sha256` 历史口径同时包含 actor 与 observation RMS；P8B 正式产物额外给出无歧义的 `actor_weights_sha256`、`critic_sha256` 和 `observation_rms_sha256`，不覆盖历史字段。

## 18. 独立 formal test 结果

formal test 仅在 selection 冻结后运行，固定 test seeds 245/246/247、每 checkpoint/seed 256 episodes。P8B/BC 共 8 个去重物理 checkpoint × 3 seeds = 24 条评估、6,144 episodes，全部 `completed`，无 failed/running 条目。frozen teacher 复用相同正式协议下已有的 3 条、768 episodes 原始评估。

| 方法 | episodes | settled landing | deck miss | hard contact | ground crash | timeout |
|---|---:|---:|---:|---:|---:|---:|
| frozen PPO teacher | 768 | 94.6615% | 5.3385% | 0.1302% | 0.0000% | 0.0000% |
| BC epoch0 | 768 | 86.1979% | 13.8021% | 0.0000% | 0.0000% | 0.0000% |
| P7 ordinary BC+PPO | 768 | 76.6927% | 23.0469% | 0.9115% | 0.0000% | 0.0000% |
| P8A metric-selected | 2304 | 91.6667% | 8.3333% | 0.3038% | 0.0000% | 0.0000% |
| **P8B metric-selected** | **2304** | **96.7448%** | **3.1684%** | **0.0868%** | **0.0000%** | **0.0434%** |
| P8B reward-selected | 2304 | 95.3993% | 4.5139% | 0.1302% | 0.0000% | 0.0434% |
| P8B epoch200 last | 2304 | 92.4045% | 7.5521% | 0.0434% | 0.0000% | 0.0434% |

P8B metric-selected 为 2229/2304 settled landing，Wilson 95% CI `[95.9388%, 97.3952%]`。三个 training seed 的独立 test settled 分别为 97.7865%、96.4844%、95.9635%，均达到 90% 和 92%。聚合结果比 BC 高 10.5469 个百分点，比 P8A 高 5.0781 个百分点，比 frozen teacher 高 2.0833 个百分点。

P8B metric-selected 的其他核心指标：contact success 99.4792%，touchdown distance mean 0.05443 m、p95 0.10073 m，first-contact XY error mean 0.05032 m，normal relative speed mean -0.14705 m/s，tangential relative speed mean 0.16119 m/s，body/deck normal angle mean 0.02921 rad，maximum penetration mean 0.01988 m。

必须保留的负面信息：BC formal test 的 hard contact 恰为 0，而 P8B metric-selected 为 2/2304（0.0868%），因此 hard contact 相对 BC 在数值上恶化，虽然仍远低于 2% 安全目标。P8B metric-selected 另有 73 deck miss、1 timeout、0 ground crash；不能声称所有安全维度都严格优于 BC。last checkpoint 低于 validation-selected，说明 anchor 限制 drift 但不保证持续更新单调改善。

## 19. 七项预注册预测验证

| # | 原预测 | verdict | 真实证据 |
|---:|---|---|---|
| 1 | epoch1–10 actor SHA 不变 | supported | 三 seed epoch10 actor relative L2 均为 0，actor weights SHA 均保持 `4a917d...` |
| 2 | warm-up action max error ≤1e-5 | supported | 三 seed epoch10 action MSE 和 max absolute error 均为 0 |
| 3 | warm-up critic 发生变化 | supported | epoch10 critic relative L2 为 0.1954、0.1907、0.1836 |
| 4 | observation RMS drift 为 0 | supported | 所有 P8B checkpoint 的 mean/variance/count drift 均为 0，RMS SHA 保持 `7aa08a...` |
| 5 | 正 anchor 降低同 epoch action MSE | supported | pilot epoch30：λ=0 为 `8.8090e-4`，λ=50 为 `1.5900e-4` |
| 6 | 较低 drift 可减缓退化但不保证成功 | supported | selected 96.74% 高于 BC 86.20%，但 epoch200 降至 92.40%；仅作观测关联 |
| 7 | validation metric 选模优于 reward 选模 | supported | 独立 test：96.74% > 95.40% |

七项 verdict 均保持原预测表述。第 6 项及 action drift/settled correlation 不能解释为严格因果；formal test 只能支持本协议、本环境和本训练预算下的观测结论。

## 20. Targeted headless 视频与人工验收状态

使用最终 seed42 metric-selected policy、单环境、Isaac Sim `rgb_array` 离屏渲染生成：

```text
benchmarks/phase8b_actor_preserving_ppo/videos/selected_success.mp4
benchmarks/phase8b_actor_preserving_ppo/videos/selected_success.npz
benchmarks/phase8b_actor_preserving_ppo/videos/selected_success.json
benchmarks/phase8b_actor_preserving_ppo/videos/selected_failure.mp4
benchmarks/phase8b_actor_preserving_ppo/videos/selected_failure.npz
benchmarks/phase8b_actor_preserving_ppo/videos/selected_failure.json
benchmarks/phase8b_actor_preserving_ppo/videos/video_manifest.json
```

成功视频的 terminal outcome 为真实 `settled_landing`。失败回合先在同一 checkpoint、同一 evaluation seed245 的无渲染单环境序列中定位到第 148 回合，再只缓存该回合帧；terminal outcome 必须为真实 `deck_miss`，不能用非目标回合替代。MP4、NPZ、checkpoint 均记录 SHA256，并用 `ffprobe` 验证 H.264、分辨率、帧数和时长。

自动 terminal/结构校验不能代替人类目视检查，因此无论视频是否成功生成，正式状态均保持：

```text
human_review_completed: false
```

## 21. 结论与不能声称的内容

P8B 在冻结方法和独立 test 下达到 96.74% settled landing，超过 90%、92%、BC、P8A 和 frozen teacher，并显著降低 deck miss。它支持 separate actor/critic、critic warm-up、冻结 RMS 和 BC anchor 组合在当前任务中有效保护强 BC actor，但不能单独证明某一个组件是提升的唯一原因，也不能把 policy drift 的相关性解释为严格因果。

不能声称：策略使用视觉图像；结果覆盖 yaw、随机波谱、完整六自由度船舶或真实水动力；所有安全指标均比 BC 更好；epoch200 优于 validation-selected；自动 headless 视频已经完成人工 GUI 验收。完整机器可解析证据位于 `benchmarks/phase8b_actor_preserving_ppo/`。

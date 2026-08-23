# RL 研究长期路线图：运动船舶甲板自主降落

> 项目：`/home/j/Isaac_RL_Projects/quadcopter_waypoint`  
> 更新时间：2026-08-23  
> 文档用途：只描述 RL 研究线尚未完成的长期任务、统一指标、阶段门和停止条件。  
> 当前事实与历史结果仍以根 `README.md`、各理论文档及 `benchmarks/` 中冻结实验包为准；本文件不得改写历史 benchmark 语义。

---

## 1. 论文侧最终目标

RL 研究线不再以“证明 PPO 能在运动甲板上降落”作为目标。当前 deterministic physical-deck-attitude benchmark 上，actor-preserving PPO 已取得：

```text
settled landing = 96.74%
frozen teacher  = 94.66%
```

因此后续研究问题收敛为：

> **在保留现有安全降落技能的前提下，使策略从理想仿真状态输入和有限规则甲板运动，逐步泛化到随机海况、感知不确定性、动力学偏差，并形成可部署到 PX4/实际计算平台上的分层 RL 控制接口。**

最终 RL 线至少应回答四个问题：

1. **R1：冻结策略在随机海况 distribution shift 下的鲁棒边界在哪里？**
2. **R2：发生可重复退化后，actor-preserving adaptation 是否能恢复新分布性能，同时保留原始任务能力？**
3. **R3：从 perfect simulator state 切换到 noisy / delayed / dropout / estimated state 后，策略性能如何退化，是否需要状态历史或 recurrent policy？**
4. **R4：相比直接输出总推力与三轴力矩的 Direct RL，分层 RL 是否能在可部署性、安全性和 Sim-to-Real 鲁棒性上取得更好的综合折中？**

---

## 2. 永久冻结的基线与实验合同

以下内容作为后续全部实验的 reference contract，禁止为了新结果原地修改历史语义。

### 2.1 Deterministic benchmark

冻结任务：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

冻结：

```text
22-D observation
4-D action = collective thrust + xyz moments
reward
contact semantics
termination
settled landing definition
```

当前正式 reference：

```text
actor-preserving PPO settled landing = 96.74%
Wilson 95% CI = 95.94% .. 97.40%
hard contact = 0.09%
ground crash = 0.00%
timeout = 0.04%
```

### 2.2 Sea-State v1 core

冻结：

```text
JONSWAP formula
finite spectral synthesis
surrogate second-order vessel response
analytic pose / velocity
8 deg roll/pitch safety envelope
0.12 m heave safety envelope
spectral coefficient scaling logic
```

后续 robustness profile 只能改变允许的输入分布 / response 参数，不得通过修改核心公式、成功定义或安全 envelope 人工制造性能退化。

### 2.3 每次新策略必须回归的指标

每个正式 candidate 都必须同时报告：

```text
settled landing rate + Wilson 95% CI
deck miss rate
hard contact rate
ground crash rate
timeout rate
touchdown XY distance mean / P95
normal relative speed mean / P95
tangential relative speed mean / P95
body-deck normal angle mean / P95
contact impulse mean / P95
observation/action dimensions
checkpoint hash
seed set
```

### 2.4 原始能力保留门

后续 adaptation / hierarchical / perception-aware 策略不得只报告新分布性能。

正式 retention target：

```text
PhysicalDeckAttitude settled landing >= 95%
或相对当前 96.74% reference 下降不超过 2 percentage points
```

并要求：

```text
ground crash = 0
hard contact 不得出现明显统计性恶化
```

若新方法提升 shifted-domain 性能却显著破坏 deterministic retention，不作为默认主方法。

---

## 3. 长期阶段路线

后续严格按“先诊断、再训练、再扩展”的顺序推进。没有通过前一阶段 gate，不提前开启下一阶段大规模训练。

---

# 阶段 A：Controlled Sea-State Robustness Boundary Closure

## 目标

完成当前已经开始的 stochastic Sea-State 主研究线，回答 frozen policy 的可重复鲁棒边界，而不是继续扩大海况参数直到策略失败。

当前已知：

```text
nominal stochastic success ≈ 98%
多数已接受 shift profile success >= 95%
尚未找到稳定 75..90% robustness boundary
```

当前最强退化线索集中在 realized motion：

```text
deck angular speed ≈ 0.08..0.12 rad/s
heave velocity     ≈ 0.08..0.12 m/s
```

但现有 bucket 非单调，且混合多个 profile，不能写成因果结论。

## 必须完成

1. 保持 Sea-State v1 核心数学冻结。
2. 围绕 realized angular speed `0.08..0.12 rad/s` 设计窄分布 controlled profiles。
3. 围绕 realized heave velocity `0.08..0.12 m/s` 设计窄分布 controlled profiles。
4. 只保留少量 interaction probe，验证 angular-rate × tilt / heave-rate 是否存在联合失效。
5. 每个 profile 先做离线 realization audit，再运行 frozen teacher。
6. 使用多个固定 seed 验证退化是否可重复。
7. 失败必须分解为 deck miss / hard contact / ground crash / timeout，并关联 touchdown kinematics。
8. 若在安全 envelope 内仍找不到可重复 boundary，正式冻结“当前策略 zero-shot robustness 较强”的结论，停止继续向极端海况扩张，转入感知 distribution shift。

## Profile 预筛门

正式 candidate 不允许由 spectral scaling 主导。优先要求：

```text
scaling fraction <= 10%
P05(min scale) >= 0.95
physics consistency = PASS
compatibility regression = PASS
NaN/Inf = 0
```

若 profile 超过上述范围，只能作为 exploratory probe，不进入正式 boundary 结论。

## Boundary discovery protocol

建议第一轮：

```text
3 fixed eval seeds
128 completed episodes / seed / profile
```

若出现 candidate，再正式验证：

```text
3 fixed eval seeds
256 completed episodes / seed / profile
```

## Candidate 判定

满足以下任一情况才允许进入 adaptation：

```text
A. aggregate settled landing 进入 75..90%

或

B. 相对 deterministic / nominal reference 可重复下降 >= 5 percentage points，
   且 aggregate settled landing < 95%
```

同时要求：

```text
退化至少在多个 seed 上方向一致
不是 physics bug
不是 spectral scaling artifact
成功定义未改变
主导 failure mode 可解释
```

## 停止条件

最多进行两轮窄分布 boundary redesign。

如果仍无法在冻结安全 envelope 内得到稳定退化：

> 不继续人为放大 Hs、姿态或海况到不合理范围；把“未发现明显 Sea-State robustness boundary”作为正式结果，进入阶段 C 的 perception shift。

---

# 阶段 B：Distribution-Shift Adaptation

## 进入条件

只有阶段 A 产生正式 repeatable candidate 才执行。

## 必须比较

```text
Frozen policy
PPO scratch
Ordinary PPO fine-tuning
Actor-preserving PPO fine-tuning
```

同一训练预算、同一 train/validation/test seed 角色、同一 evaluator。

## 主要研究假设

> actor-preserving fine-tuning 能在恢复 shifted-domain 性能的同时，比 ordinary fine-tuning 更好地保留原 deterministic landing skill。

## 主要指标

### Shifted-domain

```text
settled landing
failure distribution
touchdown XY
relative normal/tangential speed
contact impulse
learning curve / environment steps
```

### Retention-domain

重新运行冻结 `PhysicalDeckAttitude` 正式测试：

```text
settled landing >= 95%
ground crash = 0
hard contact 无明显恶化
```

### Policy drift

继续保留：

```text
action drift
actor parameter drift
observation RMS drift
critic drift / value adaptation
```

## 退出条件

只有在 shifted-domain 取得明确恢复，同时通过 retention gate，才将 adaptation 作为论文正结果。

若 frozen policy 本身已经 >=95%，不为了制造“微调收益”强制训练。

---

# 阶段 C：Perception-Aware RL

## 目标

把策略从 perfect simulator state 推进到真实视觉可实现的 estimated relative state。

第一版不做 RGB end-to-end RL，优先保持：

```text
Camera / ArUco
→ relative pose estimator
→ relative velocity / angular-rate estimator
→ state vector
→ policy
```

## C1. Synthetic sensor degradation

先在当前 22-D state interface 上加入可控退化：

```text
measurement noise
latency
dropout
outlier / stale observation
```

正式实验至少覆盖：

### Latency

```text
0 ms
50 ms
100 ms
150 ms
```

### Dropout

```text
0%
5%
10%
20%
```

Noise 数值不预先拍脑袋固定，最终范围应由传统视觉项目 / 实际 ArUco 链路测得的 RMSE、P95 和 jitter 反推。

## C2. Estimated-state adapter

建立显式 observation adapter，使训练和部署接口保持一致：

```text
GT deck state                ┐
noisy/delayed synthetic state├→ canonical policy observation
real estimator state         ┘
```

不得在部署接口偷偷加入 simulator-only Ground Truth。

## C3. Memory / POMDP 判断

先比较：

```text
single-frame state
frame stack / finite history
```

只有在 dropout/latency 下存在稳定性能瓶颈时，再增加 recurrent policy（LSTM/GRU）。不默认引入 recurrent complexity。

## 主要指标

```text
settled landing vs noise
settled landing vs latency
settled landing vs dropout
failure distribution
state estimation error vs policy failure
recovery after observation restoration
policy inference latency
```

## 论文目标

至少证明：

> 策略对真实可测范围内的视觉误差有明确鲁棒区间；超过该范围后能解释主要失效模式，而不是只展示理想状态结果。

---

# 阶段 D：PX4-Compatible Deployable Hierarchical RL

## 目标

保留当前 `22D → 4D thrust/moment` Direct RL 为永久冻结的重要 baseline，同时新增一个**独立方法**，使 RL action 可以不改变物理语义地通过 PX4 标准 Offboard reference interface 部署。

2026-08-23 的架构决策将原先“先只输出 `vz_ref`、不足时再扩展 xyz”的候选方案更新为第一版直接使用 deck-relative 3D velocity reference：

```text
deployable relative state
        ↓
Hierarchical RL
        ↓
[v_t1_rel_ref, v_t2_rel_ref, v_n_rel_ref]
        ↓
PX4-compatible Reference Adapter
        ↓
world/ENU velocity reference
        ↓
training: Vectorized PX4-like controller
SITL/HIL/real: PX4 Offboard velocity controller
```

其中 `t1/t2/n` 属于 deck frame；policy 不学习 yaw，yaw 由 deterministic rule 提供。完整理论、坐标系、contact-point velocity、action bounds、安全门、训练/部署 backend 区分和测试协议以 `docs/px4_compatible_hierarchical_rl_theory.md` 为唯一实现前置文档。

明确：

```text
Direct RL != deprecated
Hierarchical RL is a new independent method
```

不得把已有 4D thrust/moment checkpoint 直接解释成 3D velocity policy，也不从第一版扩展成 motor-level policy。

## 与 Direct RL 的正式比较

```text
Direct RL: 22D → thrust + Mx/My/Mz
Hierarchical RL: estimated/predicted state → landing decision + velocity reference
```

比较：

```text
nominal success
Sea-State robustness
perception robustness
dynamics mismatch robustness
safety violations
control smoothness
inference cost
PX4 integration effort
```

## 基本验收

Hierarchical RL 在 deterministic benchmark 上不要求机械超过 96.74%，但应满足：

```text
settled landing >= 95% 或与 Direct RL 差距 <= 2 pp
ground crash = 0
```

且至少在一个 distribution-shift 维度上表现出更好的鲁棒性、安全性或部署可解释性，否则不把它作为论文主方法。

---

# 阶段 E：Dynamics Randomization 与 Sim-to-Real Preparation

## 原则

不做“所有参数同时大范围随机化”。采用：

```text
system identification / measured range
→ sensitivity analysis
→ selective randomization
```

## 候选随机化项

按优先级：

```text
1. mass / inertia
2. thrust coefficient / actuator gain
3. actuator delay / first-order response
4. wind disturbance
5. contact friction / restitution
6. sensor timing jitter
```

只有测量或 sensitivity 证明重要的项才进入正式 domain randomization。

## 指标

```text
success vs parameter shift
retention on nominal dynamics
control saturation
hard contact / crash
policy variance across seeds
```

最终保存每个随机化范围的来源：

```text
measured
identified
manufacturer spec
engineering assumption
```

不得把工程假设写成真实系统辨识结果。

---

# 阶段 F：PX4 / HIL / 实机接口验证

## 最低目标

RL 线最终必须输出一个能进入实际飞控链路的策略接口，而不是只在 Isaac Lab 内运行。

最低验证路径：

```text
Isaac Lab policy
→ exported inference checkpoint
→ Jetson / target compute
→ PX4 SITL/HIL compatible reference interface
```

推荐继续：

```text
actual camera / estimator
+ Jetson Orin NX
+ Pixhawk / PX4
+ controlled moving platform
```

## 工程指标

RL policy 本身至少报告：

```text
inference latency mean / P95 / max
policy frequency
missed deadline count
CPU/GPU/memory
checkpoint load time
NaN/Inf action count
saturation ratio
```

对于当前 50 Hz Direct policy，部署侧目标为：

```text
policy inference P95 <= 10 ms
```

最终完整闭环频率与延迟以真实接口实测为准，不在仿真阶段虚构。

---

# 阶段 G：毕业论文正式实验矩阵

## 方法维度

RL 仓库至少保留：

```text
M0 Frozen Direct PPO teacher
M1 Actor-preserving Direct PPO
M2 PX4-Compatible Hierarchical RL
```

若阶段 A 后续形成正式 robustness boundary，再把 shift-adapted actor-preserving PPO 作为额外 adaptation 方法单独编号，不占用 M2，也不得改变 M0/M1/M2 的 action semantics。

传统方法由传统项目继续推进，本仓库只需要预留统一输入/输出和评测字段，最终论文跨项目比较：

```text
Traditional reactive / predictive baseline
vs
Direct RL
vs
Hierarchical RL
```

不在 RL 仓库重复实现另一套传统系统。

## 场景维度

```text
Deterministic PhysicalDeckAttitude
Nominal stochastic Sea-State
Controlled Sea-State shift
Perception noise
Latency
Dropout
Dynamics shift
Combined representative shift
```

## 正式统计规则

- smoke：每条件至少 `3 seeds`。
- 正式主要对比：至少 `3 fixed seeds × 256 episodes`；核心方法优先扩大样本。
- 固定 seed，不根据结果更换。
- 成功率报告 Wilson 95% CI。
- 连续指标至少报告 mean / P95，核心指标增加 bootstrap 95% CI。
- 失败样本全部保留并分类。
- 每个正式 checkpoint 保存 SHA-256。
- 保存环境 manifest、代码 commit、命令、配置、结果 CSV/JSON、图表脚本。

---

## 4. 最终毕业级 RL 完成指标

以下项目全部形成可复现证据后，RL 研究线视为论文级完整：

1. 冻结 deterministic Direct RL 基线与 96.74% reference 可复现。
2. 完成 stochastic Sea-State zero-shot 鲁棒性研究，并给出 boundary 或“安全 envelope 内未发现明显 boundary”的严格结论。
3. 只有存在真实 boundary 时，完成 frozen / scratch / ordinary FT / actor-preserving FT 对比和 retention 实验。
4. 完成 noise / latency / dropout 的 perception robustness 曲线。
5. 建立 simulator-only GT 与 deployable estimated-state 的统一 observation adapter。
6. 完成 Direct RL 与 Hierarchical RL 的统一比较。
7. 完成至少一组 selective dynamics randomization，并记录范围来源。
8. 至少完成一次 target-compute inference / PX4 SITL 或 HIL 接口验证。
9. 所有正式方法同时报告 success、touchdown kinematics、安全、失败模式和推理实时性，不只报告 reward。
10. 形成可直接用于论文的 baseline 表、robustness 图、ablation 表、failure case 和完整复现实验包。

---

## 5. 明确不作为默认任务的内容

除非前序实验明确证明需要，否则不默认执行：

- 不继续为了 96.74% → 97%/98% 在同一 deterministic 分布上反复调 PPO。
- 不为了制造退化突破冻结 Sea-State v1 安全 envelope。
- 不默认做 RGB end-to-end visual RL。
- 不默认使用 recurrent policy。
- 不默认重新设计 reward 或 success contract。
- 不在 RL 仓库重新实现传统 ArUco/MPC 完整系统。
- 不进行 CFD 或完整船舶水动力研究。
- 不在没有 repeatable boundary 时启动大规模 adaptation training。
- 不用 shifted-domain 提升交换 deterministic retention 或安全性。

---

## 6. 当前唯一下一任务

2026-08-23 已完成 PX4-Compatible Hierarchical RL 从 action-interface 到 PPO 前置证据的 Stage 0~2：

```text
theory gate                          = PASS
Reference Adapter unit tests         = PASS
Vectorized PX4-like controller tests = PASS
independent 3D-action task           = IMPLEMENTED
1-env deterministic smoke            = PASS
16-env GPU-vectorized smoke           = PASS
post-evaluator full regression        = 116 tests + 21 subtests PASS
M2 evaluator terminal diagnostics     = IMPLEMENTED / TESTED
16-env zero-relative-action baseline  = PASS
```

Zero-action 四场景均表现为 `timeout=100%`、`contact=0`、`settled=0`、`hard_contact=0`、`ground_crash=0`、reference/controller saturation=0；因此该 baseline 只是 deck contact-point velocity following，不会自己完成下降与落地。

证据入口：

```text
docs/px4_compatible_hierarchical_rl_theory.md
benchmarks/px4_hierarchical_smoke/
benchmarks/px4_hierarchical_training/
```

当前唯一允许的下一步仍属于：

> **A. train PX4-compatible hierarchical RL**

但现在进一步收紧为：先执行 `seed=42 / num_envs=64 / max_iterations=30` PPO sanity。只有 sanity 证明 reward 或 landing intermediate metrics 出现稳定、可解释改善，且 NaN/Inf=0、controller 无 explosion、saturation 不长期接近 100%、ground crash 不持续恶化，才允许进入 256-env / 100~200-iteration C0 candidate training。

选择 A 而不是先做 B（PX4 SITL）的原因不变：当前接口、controller surrogate、25 Hz/100 Hz 分层频率、实体接触和 evaluator 已被验证，但仍没有经过学习门禁与 deterministic benchmark 选择的 `22D → 3D velocity reference` checkpoint。只有 M2 nominal benchmark PASS 后，才进入 exported policy → PX4 SITL 并量化 surrogate controller mismatch。

仍禁止：

```text
修改 M0/M1 checkpoint 或 action semantics
改写 Sea-State 历史 benchmark
把 smoke 写成 PPO success result
4096× PX4 SITL training
第一版学习 yaw / motor action
在没有小规模 sanity 的情况下直接启动正式大规模 PPO
```

Sea-State Controlled Robustness Boundary Closure 仍保留为后续研究任务；本次顺序调整不否定、删除或重写已有 Sea-State 结论与资产。

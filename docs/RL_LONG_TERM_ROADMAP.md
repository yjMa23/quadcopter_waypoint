# RL 研究长期路线图：运动船舶甲板自主降落

> 项目：`/home/j/Isaac_RL_Projects/quadcopter_waypoint`  
> 更新时间：2026-08-30
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

# 阶段 D：论文第一创新点 — Continuous-Stage PX4-Compatible Hierarchical RL

## 目标

保留 `22D → 4D thrust/moment` Direct RL 为永久 baseline，并把当前已经实现、但 D1 sanity 失败的 3D-velocity M2 冻结为：

```text
Fixed-Stage PX4-Compatible Hierarchical RL baseline
```

毕业论文第一创新点正式升级为：

> **面向移动旋转甲板的三维相对速度分层强化学习着陆参考规划方法**

最终 architecture：

```text
deployable relative state
        ↓
Hierarchical RL
   ↙             ↘
3-D deck-relative  continuous landing stage s∈[0,1]
velocity reference
   \             /
      Reference Planner
       ↙         ↘
contact-point   terminal attitude
compensation    guidance
       \         /
            PX4
 velocity / attitude / rate / allocation
```

系统边界固定：

```text
RL learns:
- 3-D deck-relative motion
- continuous landing commitment

Analytical guidance handles:
- rigid-body contact-point compensation
- deterministic terminal deck-normal/deck-heading alignment

PX4 handles:
- low-level stabilization
- attitude/rate control
- allocation/motor control
```

完整长期研究合同：

```text
docs/first_innovation_hierarchical_landing_plan.md
```

S1 数学 Theory Gate：

```text
docs/continuous_stage_terminal_attitude_theory.md
```

S3 independent task implementation contract：

```text
docs/continuous_stage_px4_task_contract.md
```

当前 `docs/px4_compatible_hierarchical_rl_theory.md` 继续作为 Fixed-Stage 3-D velocity action/controller 的历史理论与实现依据，不删除、不改写 benchmark 语义。

## 关键方法变化

新 policy action：

```text
[a_t1, a_t2, a_n, a_stage]
```

前三维仍映射为 deck-frame relative velocity；第四维只映射为内部 filtered stage：

```text
s ∈ [0,1]
```

stage 连续塑形：

```text
normal descent envelope
tangential velocity envelope
reference slew envelope
terminal attitude alignment weight
```

不再以 hard `can_land` 作为下降 shaping 的总开关。

Observation dimension 第一版仍为 22，但：

```text
old index 15 = align_success
new index 15 = previous filtered stage s_{t-1}
```

旧 M2 checkpoint 禁止重解释为新 task checkpoint。

## Terminal attitude 与 PX4 boundary

姿态不由 RL 学习。正常 flight 使用 velocity-controller attitude `R_vel`；临近接触时根据：

```text
alpha = f(stage, surface_clearance)
```

连续趋向 deck attitude / deck heading，并显式限制 tilt 与 attitude-reference rate。

部署优先：

```text
Route A:
velocity-level PX4 deployment
+ attitude-shaping acceleration/feedforward guidance
```

如果 PX4 source/SITL 证明 Route A 不能实现所需 terminal attitude shaping，才使用 fallback：

```text
Route B:
approach = PX4 velocity mode
terminal = PX4 attitude-setpoint mode
```

仓库当前没有 Route A 的 PX4 source/SITL 实证，因此任何具体内部行为必须等 S12 source/interface gate 验证，禁止虚构。

## Landing safety migration

连续 stage 属于 learned decision；landing success 仍是 deterministic physical contract。

旋转甲板的 rotational touchdown metric 从旧 baseline 的 absolute UAV angular velocity，升级为：

```text
omega_rel = omega_uav - omega_deck
```

并与 deck-frame position、normal/tangential relative velocity、body-deck attitude、physical contact、hard-contact/penetration/ground-contact 一起判定。

## 固定 S0–S15 顺序

```text
S0  Freeze current Fixed-Stage M2 baseline
S1  Continuous-Stage + Terminal-Attitude Theory Gate
S2  Pure mathematical stage/attitude utilities + unit tests
S3  Independent Continuous-Stage task
S4  1-env deterministic smoke
S5  16-env GPU smoke
S6  Off-center contact-point / rotating-deck deterministic benchmark
S7  64-env / seed42 / 30-iteration PPO sanity
S8  256-env / 100-200 iteration candidate + >=3 seeds
S9  Fixed-stage vs continuous-stage ablation
S10 deck-center vs contact-point compensation ablation
S11 yaw / rotating-deck benchmark
S12 PX4 SITL source/interface validation
S13 surrogate vs PX4 SITL comparison
S14 perception / dynamics uncertainty
S15 thesis formal benchmark
```

每阶段 entry gate、deliverable、PASS/FAIL、stop condition 已在第一创新点长期文档中冻结；禁止跨 gate 直接进入长训或 SITL。

## 与 Direct RL 的正式比较

```text
Direct RL: 22D → thrust + Mx/My/Mz
Fixed-Stage M2: 22D → 3D relative velocity + hard landing phase
Continuous-Stage method: 22D → 3D relative velocity + learned continuous stage
```

正式比较至少报告：

```text
nominal settled landing
post-latch/stage recovery
timeout
reference smoothness
controller tracking
contact kinematics / safety
Sea-State / perception / dynamics robustness
inference cost
PX4 integration boundary
```

Nominal candidate 仍应达到：

```text
settled landing >= 95%
或与 Direct 96.74% reference 差距 <= 2 pp
ground crash = 0
hard contact 无明显统计性恶化
```

只有 S9/S10/S11 等正式 ablation 解锁对应论文 claim。

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

截至 2026-08-30，PX4-compatible 分层 RL 已形成完整的 Fixed-Stage evidence chain：

```text
3-D relative velocity action                     = IMPLEMENTED
Reference Adapter                                = PASS
Vectorized PX4-like controller                   = PASS
1-env / 16-env smoke                             = PASS
zero-relative-action baseline                    = PASS
M2 evaluator / terminal diagnostics              = IMPLEMENTED
D0 reward audit                                  = D0-B supported
D1 reward-only recovery gate                     = implemented / tested
D1 full regression                               = 128 passed + 21 subtests
D1 sanity                                        = FAIL
```

D1 ep30 关键固定评估：

```text
align                    = 48.44%
crash                    = 1.56%
deck miss                = 0%
ground crash             = 0%
hard contact             = 0%
settled landing          = 0%
timeout                  = 98.44%
controller tracking mean = 0.4425 m/s
```

D1 已显著修复原先 `predicted_pad_error/contact_clearance` 主导的 reward compatibility 问题，但没有解决：

```text
within-bound high-variation reference
controller tracking degradation
post-latch horizontal drift
reward-gate crossing among aligned episodes
descent completion failure
```

因此当前 3D-action task 正式冻结为：

```text
Fixed-Stage PX4-Compatible Hierarchical RL baseline
```

不再继续通过 D2/D3 人工 `align_success/can_land` 阈值 patch 构造论文主方法。

毕业论文第一创新点已经转向：

```text
3-D deck-relative velocity
+
continuous learned landing stage
+
contact-point rigid-body compensation
+
terminal deck-attitude guidance
+
PX4 low-level reuse
```

第一创新点文档入口：

```text
docs/first_innovation_hierarchical_landing_plan.md
docs/continuous_stage_terminal_attitude_theory.md
```

截至 2026-08-30，当前 gate：

```text
S0 Fixed-Stage baseline freeze        = PASS
S1 Theory Gate                        = PASS
S2 Pure mathematical guidance        = PASS
S3 Independent Continuous-Stage task  = PASS
```

S2 pure guidance 位于：

```text
source/quadcopter_waypoint/quadcopter_waypoint/utils/continuous_landing_stage.py
```

S3 已新增独立 task：

```text
Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
22-D observation
4-D action = 3-D deck-relative velocity + continuous stage
```

并完成：caller-owned stage/filter state、stage-conditioned velocity/slew、现有 contact-point adapter 复用、terminal `q_vel -> alpha -> SLERP -> tilt -> rate-limit -> q_ref`、controller additive external-attitude path、relative-angular safe-contact migration、continuous reward boundary和 scratch PPO config。Frozen Direct / PhysicalDeckAttitude / Fixed-Stage M2 task source未修改。

S3 验证：

```text
targeted regression = 82 passed
full regression     = 167 passed + 21 subtests
pre-S3 baseline     = 155 passed + 21 subtests
added S3 tests      = 12
git diff --check    = PASS
frozen task source diff = 0
```

S4 1-env deterministic Continuous-Stage smoke 已于 2026-08-30 PASS：

```text
task ID        = Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
num_envs       = 1
seed           = 42
physics/policy = 100 / 25 Hz
scripted cases = 9/9 PASS
NaN/Inf        = 0
ground crash   = 0
controller saturation ratio = 0 in all cases
max |delta_stage| = 0.08 <= 2.0 * 0.04
reward path    = finite
```

关键 S4 interface 证据：low-stage 负 normal action 被 `V_down=0` 阻断；high-stage normal reference 达 `-0.2370 m/s`；recovery stage 可从 `0.9880` 连续降至 `0.00158` 且 positive normal reference 达 `+0.2100 m/s`；terminal alpha 最大 `0.8693`，q_ref tilt 最大 `5.276 deg`，attitude-reference rate 最大约 `[0.2255, 0.1803, 0.0089] rad/s`；static yaw deck/q_vel/q_ref heading 均约 `+15 deg`；off-center contact-point correction 最大 `0.00536 m/s`。这些不替代 S6/S11 formal benchmark。

```text
targeted S4 regression = 71 passed
full regression        = 171 passed + 21 subtests
pre-S4 baseline        = 167 passed + 21 subtests
added S4 tests         = 4
git diff --check       = PASS
frozen historical source diff = 0
```

S4 之后新增一条 supplementary deterministic full-landing evidence，但它不新增 formal gate：固定 `static level deck + 1 env + seed=42 + initial clearance=0.25 m`，使用 deterministic scripted high-level action。两次相同场景均在 step 104 首次 safe contact、step 119 达成 `settle_hold=3/3` 与 `landing_success=true`，核心 first-contact/final metrics 完全一致；无 hard contact / ground crash。demo 的 `first-contact latch + contact_settle seating reference` 仅属于 demonstration-side scripted action logic，production policy / Reference Adapter / PX4-like controller / success thresholds 均未修改。

复现说明：历史 `v6` JSON 的真实 `initial_clearance_m=0.25`；一次 `0.45 m` 初始间隙运行未在 episode 结束前检测到 deck contact，因此不视为同场景 repeat。demo 默认值已冻结到实际验证的 `0.25 m`。补充证据后的 full regression 为 `176 passed + 21 subtests`，S4 no-video regression 仍为 `9/9 PASS`，`git diff --check=PASS`。这些 evidence 不替代 S5/S6/S11、PPO 或 real-world evidence。

**当前唯一允许的下一阶段是 S5：**

```text
16-env GPU Continuous-Stage smoke
```

S5 之前仍禁止 PPO training、64/256-env candidate 和 PX4 SITL。

仍禁止：

```text
D2/D3 hard-gate patch
64/256-env training before pure-math/task/smoke gates
100-200 iteration candidate before S7 sanity PASS
PX4 SITL before S8 nominal candidate PASS
修改 M0/M1/Fixed-Stage M2 历史 benchmark
把 old M2 checkpoint 重解释为 Continuous-Stage checkpoint
让 RL 直接学习 yaw/attitude/torque/motor action
```

Sea-State、Perception-Aware、Dynamics Randomization 等长期研究线仍保留；第一创新点的 S0–S15 顺序只约束 PX4-compatible hierarchical 主方法的内部推进，不删除其他已冻结路线。

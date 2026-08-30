# 毕业论文第一创新点长期设计：连续着陆阶段的 PX4-Compatible 分层强化学习参考规划

> 项目：`/home/j/Isaac_RL_Projects/quadcopter_waypoint`
> 冻结日期：2026-08-30
> 文档性质：论文第一创新点的长期研究合同。它定义方法边界、阶段门、可证伪实验与允许/禁止的论文表述；历史 M0/M1/M2 benchmark 不因本文被改写。
> 当前执行边界：S0–S4 已完成；下一唯一 gate 为 S5 16-env GPU smoke。S4 仅形成 1-env deterministic smoke 证据，不训练 PPO、不进入 PX4 SITL。

---

## 1. Thesis innovation statement

### 中文暂定名称

**面向移动旋转甲板的三维相对速度分层强化学习着陆参考规划方法**

### 英文暂定名称

`3-D Deck-Relative Velocity Hierarchical Reinforcement Learning for Landing Reference Planning on Moving and Rotating Decks`

### 一句话方法定义

本方法把强化学习限制在**高层可部署参考规划**：策略输出三维 deck-relative velocity 与连续 landing-stage state；已知刚体几何由解析 contact-point compensation 处理，临近接触的 deck-attitude matching 由 deterministic terminal guidance 处理，PX4 继续负责低层稳定、body-rate control 与 actuator allocation。

```text
RL != replacement of PX4
RL = high-level landing reference planner
```

RL 学习：

```text
3-D deck-relative motion
+
continuous landing commitment
```

解析模块负责：

```text
known rigid-body contact-point kinematics
+
terminal deck-attitude guidance
```

PX4 负责：

```text
low-level stabilization
attitude/rate control
control allocation
motor control
```

---

## 2. Problem motivation

### 2.1 问题 A：Direct RL deployment gap

冻结 Direct RL 链路为：

```text
22-D observation
→ RL
→ collective thrust + body moments
→ simulator rigid-body wrench
```

它与真实 PX4 部署之间至少存在：

```text
controller implementation gap
actuator allocation gap
action semantic gap
```

第一创新点采用：

```text
RL
→ deployable high-level reference
→ PX4
```

使真实部署尽可能继续复用 PX4 的 velocity / attitude / rate / allocation / motor control 层。论文允许声称该架构**降低 deployment gap**，不得声称其“消除 Sim-to-Real gap”。

### 2.2 问题 B：人工 landing window 自适应不足

当前 M2 Fixed-Stage baseline 仍依赖：

```text
align_radius
align_height
horizontal_speed threshold
attitude threshold
hold steps
→ align_success
→ can_land
```

这是一个人工离散阶段机。对移动、旋转、时变甲板，它可能产生：

```text
stage switching / post-latch drift
reward-gate crossing/chattering
recovery/descent conflict
timeout
```

最终主方法不再让一个 hard `can_land` gate 决定“能否下降”，而使用：

```text
continuous learned landing stage s ∈ [0,1]
```

连续调节下降权限、切向速度 envelope、reference slew 与 terminal attitude alignment。

---

## 3. Existing evidence and D1 failure motivation

截至起始 commit `3404cfc`，已经有如下实证：

```text
3-D deck-relative velocity action               = IMPLEMENTED
PX4-compatible Reference Adapter                = IMPLEMENTED
contact-point rigid-body compensation           = IMPLEMENTED
Vectorized PX4-like controller                  = IMPLEMENTED
1-env smoke                                     = PASS
16-env GPU smoke                                = PASS
zero-relative-action baseline                   = PASS
M2 evaluator / terminal diagnostics             = IMPLEMENTED
full regression                                 = 128 passed + 21 subtests
```

D1 `seed=42 / 64 env / 30 iterations` 的固定 seed145 ep30 关键结果：

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

并存在：

```text
within-bound high-variation action/reference
post-latch horizontal drift
reward-gate crossing/chattering among aligned episodes
controller tracking degradation
descent completion failure
```

D1 同时证明 reward-only gate 对原 D0 reward attribution 问题有效，因此不能把失败简单归因于“奖励完全错误”。更关键的结构问题是：**人工离散 landing phase 与参考 envelope 没有形成连续、可学习、可跟踪的阶段演化。**

---

## 4. Frozen Fixed-Stage baseline

从本文开始，当前：

```text
Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0
```

及其 D0/D1 reward-gating history 正式冻结为：

```text
Fixed-Stage PX4-Compatible Hierarchical RL baseline
```

冻结原则：

- 不删除 `align_success / can_land` 旧实现；
- 不改写 D0/D1 benchmark；
- 不把旧 22D→3D checkpoint 重解释为新 22D→4D continuous-stage checkpoint；
- 不再通过 D2/D3 式阈值堆叠把 Fixed-Stage 方法伪装为最终主方法；
- 若未来需要对比，只运行冻结语义或显式复现实验。

---

## 5. Final proposed architecture

```text
Deployable relative state
          │
          ▼
   Hierarchical RL
      │        │
      │        └───────────────┐
      ▼                        ▼
3-D deck-relative       continuous landing stage
velocity action               s ∈ [0,1]
      │                        │
      └──────────┬─────────────┘
                 ▼
          Reference Planner
          │              │
          ▼              ▼
 translational ref   terminal attitude guidance
          │              │
          ▼              ▼
contact-point rigid-  deck-normal / deck-heading
body compensation       alignment shaping
          │              │
          └──────┬───────┘
                 ▼
               PX4
                 │
 velocity / attitude / rate / allocation
                 │
               motors
```

方法职责固定为：

```text
policy:       where/how to move relative to deck + how strongly to commit
analytical:   rigid-body geometry + terminal attitude reference shaping
PX4:          low-level stabilization and actuation
```

任何设计若重新让 RL 直接输出 `roll/pitch/yaw/thrust/torque/motor command`，必须停止并重新论证，不能默认实现。

---

## 6. RL/action responsibility

新方法高层 action：

```text
[a_t1, a_t2, a_n, a_stage] ∈ [-1,1]^4
```

其中：

```text
[a_t1, a_t2, a_n]
→ stage-conditioned deck-frame relative velocity
→ [v_t1_rel, v_t2_rel, v_n_rel] [m/s]
```

第四维：

```text
a_stage
→ raw stage
→ filtered stage s ∈ [0,1]
```

`stage` 只属于高层 reference planner 内部状态，不是发送给 PX4 的第四维飞控输入。

语义仅用于解释，不形成 hard bins：

```text
s ≈ 0   : approach / tracking / recovery
s ≈ 0.5 : landing transition
s ≈ 1   : terminal landing / touchdown commitment
```

---

## 7. Continuous landing stage

Continuous stage 必须连续影响至少三类 reference contract：

1. **normal descent envelope**：stage 低时下降权限小；stage 增大时连续允许 terminal descent；禁止 `if can_land` 式硬开关。
2. **tangential velocity envelope**：stage 越高，允许切向相对速度越低，使 approach 允许快速追踪而 touchdown 强调精度。
3. **reference slew envelope**：stage 越高，reference acceleration/slew limit 越严格，直接针对 D1 high-variation reference 与 tracking degradation。

精确定义、工程初值和纯函数接口见 `continuous_stage_terminal_attitude_theory.md`。

---

## 8. Contact-point compensation

当前正确刚体公式永久保留：

\[
\mathbf v_c^W
=
\mathbf v_D^W
+
\boldsymbol\omega_D^W\times(\mathbf p_c^W-\mathbf p_D^W).
\]

最终世界速度 reference：

\[
\boxed{
\mathbf v_{uav,ref}^W
=
\mathbf v_c^W+R_{WD}\mathbf v_{rel,ref}^D
}
\]

Continuous stage 只塑形 `v_rel,ref`，不得重定义 `v_contact`。实现必须继续复用 `px4_reference_adapter.py` / `physical_deck_attitude_math.rigid_surface_point_velocity()`，禁止复制第二套 `omega × r`。

---

## 9. Terminal attitude guidance

姿态不是 RL 第五/第六自由度输出。

必须定义：

```text
R_vel  = normal velocity-controller flight attitude
R_deck = deck attitude
alpha  = f(filtered stage, surface clearance) ∈ [0,1]
```

远离甲板：

```text
alpha ≈ 0
R_ref ≈ R_vel
```

近接触且 landing commitment 高：

```text
alpha → 1
R_ref → feasible deck-aligned attitude
```

这只是一项 terminal / near-contact maneuver。由于多旋翼 underactuation，整个 approach 阶段要求：

```text
exact deck-attitude matching
+
zero relative translational acceleration
```

通常不能同时满足。论文不得把 terminal attitude guidance 写成“全程刚性贴合甲板姿态”。

Yaw 采用 deterministic deck/landing-board heading；不由 RL 学习。

---

## 10. PX4 low-level reuse

### Route A — 首选部署路线

```text
high-level velocity reference
+
attitude-shaping acceleration/feedforward guidance
→ PX4 velocity-level deployment
```

目标是尽量保留：

```text
PX4 velocity controller
PX4 attitude controller
PX4 rate controller
PX4 control allocation
```

但仓库目前没有 PX4 source/SITL 证据证明 arbitrary terminal attitude shaping 能在 velocity mode 内按本方法所需方式实现。涉及具体 PX4 setpoint priority、feedforward coupling 和 controller internals 的结论必须标记为：

```text
to be validated in PX4 SITL/source audit
```

### Route B — fallback

如果 Route A 经 source/SITL 证明无法满足 terminal deck-attitude tracking：

```text
approach: PX4 velocity mode
terminal: PX4 attitude-setpoint mode
```

此时仍保留 PX4 attitude/rate/allocation，但 terminal 段绕过 velocity controller。Route B 是 fallback，不是第一版默认训练架构。

---

## 11. Observation migration

新 independent Continuous-Stage task 保持 observation dimension 22，避免无必要扩大网络输入。

```text
old index 15 = align_success
new index 15 = previous filtered landing stage s_{t-1}
```

目的：

```text
remove simulator-only discrete FSM state
preserve Markov information for stage filter
keep observation dimension stable
support deployment
```

其余 21 维第一版保持现有物理语义。尤其 19:22 继续使用可部署的相对角速度信息；真实部署必须由估计器而不是 simulator GT 构造。

旧 M2 checkpoint 与新 task 不兼容，禁止 checkpoint semantic reinterpretation。

---

## 12. Reward migration

最终 reward 结构按物理语义分组，而不是继续围绕 hard `can_land` gate：

### Always-active

```text
horizontal/deck tracking
relative velocity matching
flight attitude quality
safety margins
progress / time efficiency
```

### Stage-weighted

```text
descent progress
terminal low relative speed
contact precision
terminal attitude alignment
relative angular-velocity alignment
```

### Smoothness

```text
Δstage
Δrelative-velocity-reference
```

### Terminal

```text
safe settled landing bonus
hard-contact penalty
ground-crash penalty
deck-miss penalty
```

本 Theory Gate 不根据未来训练结果预调 reward coefficient；系数必须在新 task 编码后以 preregistered sanity protocol 校准。

---

## 13. Landing success / safety migration

必须严格区分：

```text
landing decision / commitment = learned continuous stage
landing success / safety       = deterministic physical contract
```

旋转甲板上的角速度安全量改为同 frame 的 relative angular velocity：

\[
\boldsymbol\omega_{rel}^W
=
\boldsymbol\omega_{uav}^W-
\boldsymbol\omega_{deck}^W.
\]

新 task 的 touchdown contract 主要包含：

```text
physical deck contact
inside landing region
deck-frame position error
normal relative velocity
tangential relative velocity
body-deck attitude error
relative angular velocity
hard-contact force/impulse/penetration
ground contact
settle hold
```

当前 Fixed-Stage baseline 使用 absolute UAV angular velocity 的历史定义保持冻结，不回写。

---

## 14. Training/deployment split

Training：

```text
policy + pure reference math
→ Vectorized PX4-like controller
→ Isaac dynamics
```

Deployment：

```text
estimator
→ canonical observation
→ same policy
→ same pure reference math
→ PX4 interface
```

必须继续明确：

```text
Vectorized PX4-like controller != real PX4
```

Surrogate 只保留控制层级与主要饱和机制，不是 source-level PX4 replica。

---

## 15. Ablation matrix

论文第一创新点至少预留：

| Ablation | A | B | 主要指标 |
|---|---|---|---|
| landing stage | frozen hard Fixed-Stage | continuous learned stage | settled, timeout, drift, tracking, smoothness |
| contact velocity | deck-center velocity | contact-point rigid-body velocity | touchdown relative velocity, XY, hard contact, success |
| terminal attitude | velocity attitude only | stage+clearance terminal guidance | body-deck angle, relative angular speed, hard contact |
| stage smoothness | raw stage | filtered/rate-limited stage | stage variation, reference variation, tracking |
| reference slew | fixed limit | stage-conditioned limit | velocity tracking, controller saturation, touchdown |
| yaw | fixed world yaw | deck-heading guidance | heading error, body-deck attitude, rotating-deck success |

不得通过给某个 ablation 使用不同 success contract 获得表面优势。

---

## 16. Benchmark matrix

长期正式 benchmark 至少覆盖：

```text
static deck
constant XY translation
heave
roll
pitch
combined roll/pitch/heave/XY
off-center rotating contact point
yaw oscillation / yaw-rate
representative stochastic Sea-State
perception noise / latency / dropout
dynamics mismatch
```

### Off-center rotating-deck benchmark

将 landing point 从旋转中心显式偏置一个非零 `(x,y)`，使：

\[
\omega\times r\neq0.
\]

比较：

```text
A deck-center velocity compensation
B contact-point rigid-body compensation
```

在 roll / pitch / yaw motion 下报告：

```text
touchdown relative velocity
contact position error
hard contact
settled success
```

### Yaw rotation benchmark

第一版可用 kinematically prescribed rigid-deck yaw，不要求完整 6-DOF hydrodynamic ship model；必须保证：

```text
pose
deck angular velocity
contact-point velocity
deck heading
```

数学一致。

---

## 17. Stage-by-stage implementation roadmap

以下 S0–S15 顺序固定。禁止跨 gate 直接进入长训或 SITL。

### S0 — Freeze Fixed-Stage M2 baseline

**Entry gate**：D1 evidence complete。
**Deliverable**：将当前 M2/D1 明确标记为 Fixed-Stage baseline，历史 benchmark 保持不变。
**PASS**：起始 commit、D1 FAIL、action/controller/success contract 均可追溯。
**FAIL**：历史证据缺失或被重写。
**Stop condition**：baseline contract 被破坏时停止后续新方法实现。

### S1 — Continuous-Stage + Terminal-Attitude Theory Gate

**Entry gate**：S0 PASS。
**Deliverable**：本长期设计文档 + `continuous_stage_terminal_attitude_theory.md`，完整定义 frames/action/stage/envelope/attitude/safety/observation/reward/deployment/tests。
**PASS**：关键数学无阻塞 TBD；full regression PASS；PX4 未验证内容被明确隔离。
**FAIL**：RL/PX4/解析 guidance 权限边界不清，或存在未定义 frame/sign/unit。
**Stop condition**：不得编码 pure utility。

### S2 — Pure mathematical stage/attitude guidance utilities

**Entry gate**：S1 PASS。
**Deliverable**：`continuous_landing_stage.py` 或等价功能文件 + pure unit tests；复用 `px4_reference_adapter.py`。
**PASS**：stage mapping/filter/envelopes/slew/alpha/attitude math/relative angular velocity 全部 pure-math tests PASS，CPU/GPU batch/device/dtype contract 正确。
**FAIL**：引入 Isaac/Gym/ROS2/PX4 runtime 或复制 contact-point math。
**Stop condition**：unit tests 未闭合，不创建新 task。

### S3 — Independent Continuous-Stage PX4-compatible task

**Entry gate**：S2 PASS。
**Deliverable**：独立 22D→4D task；index15=`s_prev`；旧 M2 task 不修改。
**PASS**：注册、shape、reset/filter state、reward/success contracts 明确；旧 M0/M1/M2 regression 不变。
**FAIL**：旧 checkpoint 被重解释或旧 task 数值行为改变。
**Stop condition**：不得进入 simulator smoke。

### S4 — 1-env deterministic smoke

**Entry gate**：S3 PASS。
**Deliverable**：static/translation/heave/attitude 的 scripted action/reference smoke。
**PASS**：NaN/Inf=0、shape/runtime 正常、stage/envelope/attitude reference 与解析预期一致、ground crash=0。
**FAIL**：frame/sign/attitude discontinuity/controller explosion。
**Stop condition**：修数学/接口，不训练 PPO。

#### S4 validation evidence — 2026-08-30

```text
task ID      = Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
num_envs     = 1
seed         = 42
physics rate = 100 Hz
policy rate  = 25 Hz
script       = scripts/rl_games/check_px4_continuous_stage_smoke.py
```

9 个 scripted cases 全部 PASS：static hover、stage ramp、constant XY deck、heave、normal descent stage ramp、roll/pitch terminal-attitude blend、static yaw heading、off-center contact-point interface、recovery。全局 gate：NaN/Inf=0、ground crash=0、reward finite、controller saturation ratio 在全部 case 中为 0、stage 单步最大变化不超过冻结的 `2.0 * 0.04 = 0.08`。

关键接口证据：low-stage `V_down=0` 且负 normal action 不能绕过 stage；high-stage normal reference 达 `-0.2370 m/s`；recovery stage `0.9880 -> 0.00158` 且正 normal reference 达 `+0.2100 m/s`；terminal alpha 最大 `0.8693`，`q_ref` 最大 tilt `5.276 deg`，attitude-reference rate 最大约 `[0.2255, 0.1803, 0.0089] rad/s`；static yaw 的 deck/q_vel/q_ref heading 均约 `+15 deg`；off-center contact-point velocity correction 最大 `0.00536 m/s`。这些只构成 S4 interface smoke，不构成 S6 contact-point superiority 或 S11 rotating-yaw benchmark 证据。

```text
targeted S4 regression = 71 passed
full regression        = 171 passed + 21 subtests
pre-S4 baseline        = 167 passed + 21 subtests
added S4 tests         = 4
git diff --check       = PASS
frozen historical source diff = 0
```

#### Supplementary deterministic full-landing evidence

该补充证据不新增 gate。冻结场景是 `static level deck + 1 env + seed=42 + initial clearance=0.25 m`，高层输入为 deterministic scripted action。demo 的 `first-contact latch + contact_settle seating reference` 仅属于 demonstration-side scripted logic，不进入 production policy、Reference Adapter、PX4-like controller 或 success detector，也没有改动任何 contact/success threshold。

同一场景重复两次得到完全一致的核心 terminal metrics：step 104 首次 safe contact，horizontal error `5.07e-6 m`，normal/tangential relative speed `0.00812 / 0.00501 m/s`，relative angular speed `0.12173 rad/s`；step 119 `settle_hold=3/3` 并 `landing_success=true`，horizontal error `5.14e-5 m`，normal/tangential relative speed `0.000641 / 0.000791 m/s`，relative angular speed `0.03062 rad/s`，无 hard contact / ground crash。

复现时还验证到：原 `v6` 证据实际使用 `initial_clearance_m=0.25`；一次 `0.45 m` 初始间隙运行未在 episode 结束前形成可检测 deck contact，因此不属于相同 deterministic case。demo 默认初始间隙已冻结为实际验证的 `0.25 m`。

```text
full-landing contract tests = PASS
deterministic repeatability = PASS
S4 no-video regression      = 9/9 PASS
full regression             = 176 passed + 21 subtests
git diff --check            = PASS
```

该证据只说明当前单环境 deterministic chain 能完成 touchdown + settle，不替代 S5/S6/S11，不是 moving/rotating-deck superiority、PPO 或 real-world evidence。

### S5 — 16-env GPU smoke

**Entry gate**：S4 PASS。
**Deliverable**：batch GPU smoke 与 controller/reference diagnostics。
**PASS**：与 1-env contract 一致，无 per-env Python math loop，finite，controller 无异常饱和。
**FAIL**：vectorization/device/dtype/reset race。
**Stop condition**：不得进入 rotating benchmark/训练。

### S6 — Off-center contact-point / rotating-deck deterministic benchmark

**Entry gate**：S5 PASS。
**Deliverable**：非零 landing-point offset，roll/pitch/yaw prescribed motion，center-vs-contact compensation deterministic evidence。
**PASS**：`omega×r` 数学/仿真一致，contact-point compensation 在预注册运动条件下显著降低 contact-point velocity reconstruction error；无 physics inconsistency。
**FAIL**：offset 太小导致不可分辨，或 pose/omega/heading 不一致。
**Stop condition**：修 benchmark，不进入 PPO。

### S7 — 64-env / seed42 / 30-iteration PPO sanity

**Entry gate**：S6 PASS。
**Deliverable**：单 seed 小预算 sanity + ep10/20/30 deterministic evaluator。
**PASS**：finite、controller/reference 不爆炸、stage/ref 不持续饱和/抖动，landing intermediate metrics 与 normalized return 至少出现清晰学习信号。
**FAIL**：无学习信号、saturation/chattering/tracking 明显恶化。
**Stop condition**：先 diagnosis，不加长训练。

### S8 — 256-env / 100–200 iteration candidate + >=3 seeds

**Entry gate**：S7 PASS。
**Deliverable**：至少 3 个固定 seed candidate、checkpoint hash、统一 evaluator。
**PASS**：nominal settled target 达到 `>=95%` 或与 Direct 96.74% reference 差距 `<=2 pp`，ground crash=0，hard contact 无统计性恶化。
**FAIL**：仅 reward 提升而 physical landing 不达标。
**Stop condition**：不进入论文 ablation/SITL。

### S9 — Fixed-stage vs continuous-stage ablation

**Entry gate**：S8 PASS。
**Deliverable**：同场景/seed/预算/物理 success contract 的 frozen Fixed-Stage vs Continuous-Stage。
**PASS**：至少在 timeout/post-latch drift/reference smoothness/controller tracking/settled 中有预注册主指标的明确改善，且安全不回退。
**FAIL**：只在 reward 上改善或依赖不同 contract。
**Stop condition**：不把 continuous stage 作为论文主贡献。

### S10 — Deck-center vs contact-point compensation ablation

**Entry gate**：S8 PASS + S6 benchmark valid。
**Deliverable**：off-center roll/pitch/yaw 旋转条件下严格 ablation。
**PASS**：contact-point compensation 在 touchdown relative velocity / contact error / hard contact 或 success 上产生可解释优势。
**FAIL**：`omega×r` 贡献不可测或实现不一致。
**Stop condition**：论文不得把该项写成已验证优势。

### S11 — Yaw / rotating-deck benchmark

**Entry gate**：S6 mathematical consistency PASS。
**Deliverable**：prescribed yaw/yaw-rate + deck-heading terminal guidance。
**PASS**：heading、angular velocity、contact-point kinematics 一致；方法保持稳定并形成可解释 touchdown 指标。
**FAIL**：yaw 仅视觉旋转但 omega/contact velocity 未同步。
**Stop condition**：不得使用“moving and rotating deck”完整结论。

### S12 — PX4 SITL source/interface validation

**Entry gate**：S8 PASS；Route A/Route B interface theory complete。
**Deliverable**：source audit + single/few-env PX4 SITL，验证 setpoint semantics、mode transition、failsafe、frame/timing。
**PASS**：确认 Route A 可实现所需 terminal shaping，或有证据切换到 Route B；禁止凭猜测选 backend。
**FAIL**：reference semantics/priority 与设计不兼容。
**Stop condition**：不进入 surrogate-vs-SITL 对比。

### S13 — Surrogate vs PX4 SITL comparison

**Entry gate**：S12 PASS。
**Deliverable**：同 exported policy/reference sequence 下 surrogate 与 SITL controller response 对比。
**PASS**：velocity/attitude/rate tracking mismatch 在预注册可接受范围，失败模式可解释。
**FAIL**：surrogate success 不能迁移到 SITL。
**Stop condition**：先校准 controller/backend，不做实机声称。

### S14 — Perception / dynamics uncertainty

**Entry gate**：S8 与 S13 PASS。
**Deliverable**：estimated-state adapter、noise/latency/dropout、selective dynamics mismatch。
**PASS**：报告 robustness curve、failure taxonomy 与 retention；随机范围有来源。
**FAIL**：偷偷使用 simulator-only GT 或无来源大范围随机化。
**Stop condition**：不进入论文 formal benchmark。

### S15 — Thesis formal benchmark

**Entry gate**：核心 ablation 与 deployment evidence 完整。
**Deliverable**：固定方法/场景/seed/episodes/checkpoint hash/统计协议，生成论文表图与 failure cases。
**PASS**：主 claim 均有对应可复现实验证据，success/safety/kinematics/runtime/deployment 全部报告。
**FAIL**：claim 强于证据或跨项目 metric contract 不一致。
**Stop condition**：收缩论文 claim，不补“结果导向”临时实验。

---

## 18. Explicit stop/go gates

全程使用：

```text
GO only if previous gate PASS
STOP on math/frame/contract inconsistency
STOP on failed smoke before PPO
STOP on failed sanity before candidate training
STOP on failed nominal candidate before SITL
STOP on failed SITL semantics before real deployment claims
```

禁止：

```text
failed sanity → simply train longer
failed ablation → change success definition
failed Route A → silently claim PX4 velocity+attitude coexistence
weak omega×r effect → enlarge claim without off-center benchmark
```

---

## 19. Thesis claims allowed / forbidden

### 当前允许

```text
- Direct RL 与真实 PX4 低层控制链存在 action/controller deployment gap。
- 3-D deck-relative velocity interface 与 contact-point rigid-body compensation 已有实现和 smoke 证据。
- 当前 Fixed-Stage M2 的 D1 reward compatibility 改善，但仍以 timeout/tracking/post-latch drift 失败。
- Continuous-stage + terminal-attitude architecture 已形成数学与工程 Theory Gate（仅在对应文档完成并 regression PASS 后）。
```

### 当前禁止

```text
- continuous-stage policy 已经优于 Fixed-Stage
- terminal attitude guidance 已经提高成功率
- Route A 已经被真实 PX4 支持/验证
- PX4 interface 消除了 Sim-to-Real gap
- contact-point compensation 已在 off-center yaw benchmark 中显著提高 landing success
- 新方法已经完成实机验证
```

对应 claim 只有在 S9–S15 的相应证据 gate PASS 后才解锁。

---

## 20. 当前 gate 状态与唯一推荐下一步

截至 2026-08-30：

```text
S0 Fixed-Stage baseline freeze       = PASS
S1 Theory Gate                       = PASS
S2 Pure mathematical guidance       = PASS
S3 Independent Continuous-Stage task = PASS
```

S3 已新增独立：

```text
Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
22-D observation -> 4-D high-level action
```

并完成：continuous stage caller-owned filter、stage-conditioned relative velocity/slew、contact-point adapter reuse、25 Hz terminal attitude reference、100 Hz PX4-like attitude/rate reuse、relative-angular touchdown success migration、continuous reward boundary和独立 scratch PPO config。Frozen Direct / PhysicalDeckAttitude / Fixed-Stage M2 task source未修改。

S3 验证：

```text
targeted regression = 82 passed
full regression     = 167 passed + 21 subtests
pre-S3 baseline     = 155 passed + 21 subtests
added S3 tests      = 12
git diff --check    = PASS
```

当前唯一允许的下一阶段：

```text
S4
1-env deterministic Continuous-Stage smoke
```

S4 之前仍禁止：

```text
16-env GPU smoke
PPO training / checkpoint tuning
64/256-env candidate training
PX4 SITL
ROS2/HIL/real vehicle
```

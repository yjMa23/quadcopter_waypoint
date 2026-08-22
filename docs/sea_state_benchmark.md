# Stochastic Sea-State Benchmark

## 1. 目标与冻结边界

本阶段新增独立任务：

```text
Isaac-Quadcopter-ShipLanding-SeaState-Direct-v0
```

目的不是重新定义降落任务，而是在冻结 `PhysicalDeckAttitude` 的 22-D observation、4-D action、reward、接触判据、termination 和 settled-landing success contract 后，只改变甲板运动分布，从而测量 maritime distribution shift。

旧任务：

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

仍是正式 deterministic benchmark，不原地加入 JONSWAP，也不重写旧 benchmark 结论或 checkpoint。

本轮同时冻结 **Sea-State Benchmark v1** 的核心数学：

```text
JONSWAP formula
finite spectral synthesis
surrogate second-order vessel response
analytic pose / velocity
8 deg roll/pitch safety envelope
0.12 m heave safety envelope
conservative spectral coefficient scaling
```

factor-isolated robustness study 只通过外部 profile 改变允许的输入分布/response 参数，不改上述实现公式。

## 2. JONSWAP 有限谱

SeaState 使用一侧角频率谱的有限离散。令峰值角频率：

```text
omega_p = 2*pi/Tp
```

JONSWAP 形状写为：

```text
S(omega) ∝ g^2 * omega^-5
           * exp[-1.25 * (omega_p / omega)^4]
           * gamma^r

r = exp[-0.5 * ((omega - omega_p)/(sigma*omega_p))^2]
sigma = 0.07, omega <= omega_p
sigma = 0.09, omega >  omega_p
```

实现不依赖无限积分中的闭式 Phillips 常数，而是对 simulator 使用的有限 bins 做离散归一化：

```text
sum_k S_k * Delta_omega = (Hs/4)^2
```

对应 cosine component：

```text
A_k = sqrt(2 * S_k * Delta_omega)
phi_k ~ Uniform(0, 2*pi)
eta(t) = sum_k A_k cos(omega_k*t + phi_k)
eta_dot(t) = -sum_k A_k*omega_k*sin(omega_k*t + phi_k)
```

runtime 不使用 finite difference。same seed 可以复现 spectral phases / trajectory；不同 seed 得到不同 realization。

## 3. Surrogate vessel response

海面 elevation 不直接作为船体 roll/pitch。当前链路为：

```text
JONSWAP incident wave
        ↓
frequency-dependent second-order response
        ↓
heading projection
        ↓
heave / roll / pitch
```

每个自由度使用可配置二阶 transfer function：

```text
H(j*omega) = K * omega_n^2 /
             (omega_n^2 - omega^2 + j*2*zeta*omega_n*omega)
```

heave 使用完整 excitation；第一版方向性采用：

```text
roll  ∝ sin(heading)
pitch ∝ cos(heading)
```

这只是用于 benchmark 的 surrogate vessel response，**不是经过真实船舶系统辨识的 RAO，也不是水动力求解器**。后续可替换 measured RAO / lookup table，但不能把当前 profile 对应到真实特定海况等级。

## 4. Pose / velocity 一致性

SeaState 采用绝对 episode time 生成 pose 和 velocity：

```text
x(t) = x0 + vx*t
y(t) = y0 + vy*t
z(t) = z0 + heave(t)
R(t) = Rz(0) Ry(pitch(t)) Rx(roll(t))
```

`heave_dot / roll_dot / pitch_dot` 均由同一 spectral realization 解析求导；roll/pitch Euler-rate 使用 `PhysicalDeckAttitude` 已验证的 XYZ Euler-rate → world angular velocity 映射。

因此 pose 与 velocity 来自同一 analytic motion model，physics diagnostic 继续检查 simulator root state 与 command state 的 position、orientation、linear velocity、angular velocity consistency。

## 5. Safety envelope 与 spectral scaling

多频率叠加时，对每个 response DOF 使用 phase-independent conservative bound：

```text
B = sum_k abs(A_response,k)
scale = min(1, envelope / B)
A_k <- scale * A_k
```

runtime 禁止：

```text
torch.clamp(heave/roll/pitch)
```

因为 runtime clamp 会破坏频谱形状、解析速度和 pose/velocity consistency。

冻结 envelope：

```text
heave <= 0.12 m
|roll| <= 8 deg
|pitch| <= 8 deg
```

所有 episode 记录 heave/roll/pitch scale。robustness profile 如果出现大量 `scale << 1`，必须重新设计，而不能把 scaling 人工制造的分布当作正式难度。

## 6. Task 继承关系

`QuadcopterShipLandingSeaStateEnv` 继承 `QuadcopterShipLandingPhysicalDeckAttitudeEnv`，只覆盖 sea-state motion generation 和附加诊断 metric。以下 contract 继续继承：

```text
22-D observation
4-D action = collective thrust + xyz moments
reward
alignment logic
deck-frame kinematics
surface-point velocity
safe contact / hard contact
deck miss / ground crash
settled landing
```

`benchmarks/physical_deck_attitude/` 与 `benchmarks/actor_preserving_ppo/` 不属于本轮可修改路径。

## 7. Compatibility 与 physics regression

`env.sea_state_mode=compatibility` 直接走冻结的 `PhysicalDeckAttitude` deterministic motion path。

2026-08-15 复核：

```text
observation = 22-D
action      = 4-D
pose max abs error     = 0
velocity max abs error = 0
roll max abs error     = 0
pitch max abs error    = 0
status = PASS
```

默认 Sea-State v1 的 1-env / 16-env physics diagnostic 均 PASS：无 NaN/Inf、无异常 teleport、deck-ground clearance PASS、pose/linear/angular velocity consistency PASS。

为排除高 severity profile 的环境 bug，还额外复核：

```text
frequency_shift_tp1p6_2p0, 16 env:
  deck angular speed max ≈ 0.188 rad/s
  heave velocity max     ≈ 0.198 m/s
  heave scale min        ≈ 0.982
  pose/velocity consistency PASS

combined_high_tilt_very_high_rate, 16 env:
  deck angular speed max ≈ 0.205 rad/s
  pitch scale min        ≈ 0.977
  position/linear/angular-velocity consistency error = 0
  max orientation consistency error ≈ 0.040 deg
  status = PASS
```

## 8. Factor-isolated profile architecture

所有工程 profile 存放在：

```text
benchmarks/sea_state/profiles.yaml
```

并由 `sea_state_profiles.py` 的白名单加载。profile 不允许修改 `sea_state_num_components`、频率网格、8°/0.12 m envelope 等 Sea-State v1 core definition。

正式 family：

```text
compatibility
nominal_stochastic
frequency_shift_*
tilt_shift_*
heave_rate_shift_*
combined_shift_*
```

这些都是 **engineering benchmark profiles**，不得称为真实 Sea State 3/4/5。

### 8.1 Frequency shift

冻结频率网格：

```text
0.05 .. 0.80 Hz
24 components
```

扫描 `Tp`：

```text
6.0..7.0 s
5.0..6.0 s
4.0..5.0 s
3.2..4.0 s
2.5..3.2 s
2.0..2.5 s
1.6..2.0 s
```

直接缩短 `Tp` 会被冻结二阶 low-pass response 同时衰减 motion amplitude，因此 profile 的 response gain 仅作为**幅值归一化因子**，使 realized tilt 保持约 3° P95 附近、主要让 realized angular rate 随频率提高。

512-realization 离线 audit 对全部频率剖面复用同一基础种子和随机数流；因此七档的 `Hs` 与 `gamma` 样本均值分别完全相同。前六档 P95 tilt 约 `2.86..3.13 deg`，P95 angular speed 从约 `0.051` 严格递增至 `0.145 rad/s`，均无 spectral scaling；最终 `Tp=1.6..2.0 s` probe 为：

```text
tilt P95             ≈ 3.38 deg
deck angular P50/P95 ≈ 0.131 / 0.194 rad/s
heave velocity P95   ≈ 0.213 m/s
scaling fraction     ≈ 4.9%
scale P50            = 1.0
```

所以 frequency family 的解释链是：

```text
Tp distribution
    ↓
realized deck angular rate
    ↓
policy outcome
```

而不是把 `Tp` 本身当作策略 observation。

频率族还显式记录目标倾角 P95 `3.0±0.5 deg`。离线实现统计用共同基础种子 `20260815`、每剖面 512 个实现、10 s 时长和 0.05 s 采样间隔计算当前 P95，并按 `next_gain=current_gain*target_P95/realized_P95` 给出下一次响应增益；只有实际 P95 在容差内且缩放质量门槛合格才冻结。当前七个频率剖面均为 `PASS`，目标、容差、冻结增益和推荐增益均写入 profile/realization 结果，不依赖未记录的经验调节。

### 8.2 Tilt shift

固定 spectrum 与 heave response，仅逐级增加 roll/pitch response gain。离线 P95 tilt 约：

```text
1.96°
2.97°
3.96°
5.19°
5.92°
```

前四档无 scaling；最高档 scaling fraction 约 9.6%，P05 min-scale 约 0.961，仍不是 `scale << 1` 主导。更高的 7° tentative profile 因 scaling 明显增多而被拒绝，没有进入正式 pilot。

### 8.3 Heave-rate shift

为了隔离 vertical velocity，roll/pitch gain 固定为低值，使用较快 `Tp=2.0..2.6 s` 与 heave response family。离线结果：

```text
P95 tilt          ≈ 0.12..0.13 deg
P95 angular speed ≈ 0.006 rad/s
P95 heave velocity≈ 0.045, 0.083, 0.120, 0.159, 0.183 m/s
```

最高档 scaling fraction 约 15%，P05 min-scale 约 0.944，因此仅保留为 pilot high probe，不将其视为无条件 formal profile。

当前实际样本主要覆盖：

```text
0.00..0.04 m/s
0.04..0.08 m/s
0.08..0.12 m/s
0.12..0.16 m/s
0.16..0.20 m/s
```

这比预先硬设 `>=0.20 m/s` 更符合当前 frozen envelope 下的 realization。

### 8.4 Combined shift

只保留少量 interaction probes，不做大规模 Cartesian product。重点组合中 medium/high tilt 与 medium/high angular rate 交叉，最终增加一个：

```text
combined_high_tilt_very_high_rate
```

离线约：

```text
tilt P95             ≈ 4.49 deg
angular speed P95    ≈ 0.214 rad/s
scaling fraction     ≈ 10.4%
min-scale P05        ≈ 0.954
```

用于检查单因素仍稳健时是否会出现 tilt × angular-rate interaction failure。

## 9. 为什么 Hs 不是唯一 severity metric

`Hs/Tp/gamma/heading` 是 wave/excitation 参数；frozen policy 的 22-D observation 实际感受到的是 deck-relative pose/velocity/attitude kinematics。因此研究优先级是：

```text
environment parameter
        ↓
realized vessel/deck motion
        ↓
policy success / failure
```

主要 realized robustness axes：

```text
max deck angular speed
max deck tilt
max |heave velocity|
```

`Hs` 只能作为解释变量之一，不能代替 realized motion severity。

## 10. Zero-shot pilot protocol

第一轮只使用 frozen teacher：

```text
logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/
expanded_from_physical_deck_ep990_16to22.pth
```

actor 第一层输入已验证为 22-D。统一使用：

```text
num_envs = 32
episodes = 64 / profile
primary eval seed = 245
```

最接近 transition 的 `frequency_shift_tp1p6_2p0` 再使用 seed=246 重复 64 episodes。

Evaluator 始终复用：

```text
scripts/rl_games/eval_metrics.py
```

success definition 未改变，并继续记录 touchdown XY、normal/tangential relative speed、body-deck normal angle、max contact impulse，以及所有 Sea-State realized-motion metadata。

本轮 teacher raw 数据共：

```text
1536 completed episodes
23 aggregated profile rows
```

其中 frequency 最严重档有两个 eval seeds，共 128 episodes。

## 11. Frozen teacher pilot 结果

代表性 profile：

| profile | episodes | settled | deck miss | hard contact | realized severity |
|---|---:|---:|---:|---:|---|
| nominal_stochastic | 64 | 98.44% | 1.56% | 0.00% | angular P95 ≈ 0.069 rad/s |
| frequency `Tp=2.0..2.5` | 64 | 95.31% | 4.69% | 0.00% | frequency-only transition signal |
| frequency `Tp=1.6..2.0` | 128 | 96.88% | 3.12% | 0.00% | angular mean/P95 ≈ 0.100/0.176 rad/s |
| tilt target 5° | 64 | 96.88% | 3.12% | 0.00% | realized tilt P95 ≈ 4.61° |
| tilt target 6° | 64 | 96.88% | 3.12% | 0.00% | realized tilt P95 ≈ 4.91° |
| heave medium-high | 64 | 95.31% | 4.69% | 0.00% | heave-velocity P95 ≈ 0.132 m/s |
| heave high | 64 | 96.88% | 3.12% | 0.00% | heave-velocity P95 ≈ 0.165 m/s |
| combined high tilt + very high rate | 64 | 98.44% | 1.56% | 1.56% | angular P95 ≈ 0.206 rad/s |

最重要的复现结果是：

```text
frequency_shift_tp1p6_2p0:
seed245 = 95.31%
seed246 = 98.44%
aggregate = 124/128 = 96.88%
```

因此不能把单 seed 的 95.31% 当作稳定 boundary。

## 12. Realized-motion robustness curves

`analyze_sea_state_robustness.py` 生成 `robustness_curves.csv`。聚合所有 teacher factor-isolated pilot 后，样本数至少 20 的关键 bucket 为：

### Deck angular speed

```text
[0.00,0.04): n=756, settled=98.28%
[0.04,0.08): n=481, settled=99.38%
[0.08,0.12): n=196, settled=93.88%, deck miss=6.12%, hard contact=0.51%
[0.12,0.16): n=74,  settled=97.30%
[0.16,0.20): n=22,  settled=100.00%
```

`0.08..0.12 rad/s` 是当前最明显的局部 degradation signal，但 bucket 来自多个不同 profile，后续需要用更窄的 controlled distribution 复现，不能据此宣称成功率随 angular speed 单调下降。

### Deck tilt

```text
[0,2): n=1094, settled=98.35%
[2,3): n=301, settled=97.34%
[3,4): n=100, settled=97.00%
[4,5): n=32,  settled=96.88%
```

存在温和下降趋势，但没有进入 75..90% adaptation target。

### Heave velocity

```text
[0.00,0.04): n=661, settled=98.94%
[0.04,0.08): n=607, settled=98.02%
[0.08,0.12): n=172, settled=94.77%
[0.12,0.16): n=70,  settled=97.14%
[0.16,0.20): n=20,  settled=100.00%
```

同样存在局部信号，但不具有单调性，因此还不足以定义 heave-rate boundary。

## 13. Failure-mode diagnosis

pilot 的主要失败类型是：

```text
deck miss
```

hard contact 只零星出现，ground crash 与 timeout 在关键 profile 中未形成主导失效。

当前最值得继续验证的机制假设是：

```text
higher realized deck angular / vertical rate
        ↓
touchdown window 相对运动更快
        ↓
horizontal/relative-velocity correction margin 变小
        ↓
deck miss 概率增加
```

但当前 bucket 非单调，且不同 profile 混合后仍有 confounding，因此只能称为**最强线索**，不能写成因果定论。`failure_analysis.csv` 保留 success/failure outcome 对应的 tilt、angular speed、heave speed、first-contact normal/tangential relative speed、body-deck normal angle 与 impulse 基本统计，供下一轮窄分布验证。

## 14. Automatic robustness-boundary result

自动逻辑按每个 family 的 `severity_rank` 排序，并寻找从 `>=95%` 进入 `75..90%` 或至少 `<95%` 的 profile-level transition。候选剖面还必须同时满足：至少两个独立评价种子、每个种子稳定降落率的 Wilson 95% 置信区间上界均低于 95%、发生系数缩放的回合比例不超过 20%，且逐回合最小缩放因子的 5 分位不低于 0.90。这样不会把单种子或小样本波动、主要由安全包络缩放形成的分布误判为控制策略鲁棒性边界。

本轮输出：

```text
combined_shift:  no robustness boundary found
frequency_shift: no robustness boundary found
heave_rate_shift:no robustness boundary found
tilt_shift:      no robustness boundary found
```

因此 `boundary_candidates.json` 的 `candidates` 为空、`adaptation_training_allowed=false`，并记录阻断原因为 `no eligible robustness boundary candidate`。

最接近的 `frequency_shift_tp1p6_2p0` 虽有两个评价种子，但其稳定降落率分别为 95.31% 和 98.44%，每种子 Wilson 95% 置信区间上界的最大值为 99.72%，故不满足跨种子置信门槛。2026-08-22 为增益校准、自动门控、共同随机数及缩放—截断对照新增独立测试后，Sea-State 运动、剖面、任务契约和边界门控合计 18 项测试全部通过。

这不是分析失败，而是当前 frozen policy 在已接受、安全且非 scaling-dominated 的 Sea-State v1 controlled shifts 下仍保持较强 zero-shot robustness。

## 15. Actor-preserving checkpoint gate

从 `benchmarks/actor_preserving_ppo/validation_selection.json` 读取的 metric-selected seed42/43/44 checkpoint 均已验证：

```text
actor input = 22
```

但本轮**没有执行它们的 candidate-condition evaluation**，原因是 teacher 没有产生满足当前冻结门槛的 profile-level candidate。按协议，先用 actor-preserving checkpoints 去追一个未成立的 boundary 会引入不必要的多重比较。

后续一旦 teacher profile-level candidate 成立，再执行：

```text
3 metric-selected checkpoints
× 3 eval seeds
× 256 episodes
= 2304 episodes / condition
```

详见 `benchmarks/sea_state/formal_protocol.json`。

## 16. Adaptation gate

本轮 gate 状态：

```text
compatibility regression       PASS
Sea-State default physics      PASS
severe-profile physics         PASS
frozen success contract        PASS
factor-isolated profiles       PASS
realized-motion metadata       PASS
scaling diagnostics            PASS / flagged where non-zero
repeatable 75..95 profile boundary  NOT FOUND
actor-preserving candidate validation BLOCKED
PPO training                   NOT STARTED
```

所以当前结论是：

> **不进入 PPO scratch、ordinary fine-tuning 或 actor-preserving fine-tuning。继续设计更窄、更可控的 target distribution，优先围绕 realized angular speed 0.08..0.12 rad/s 与 heave velocity 0.08..0.12 m/s 的局部退化区间做复现实验，而不是继续提高 Hs 或放宽安全 envelope。**

## 17. 下一阶段 protocol（只冻结计划，不执行）

只有新的 controlled distribution 在多个 teacher eval seeds 下形成可重复 boundary 后，才进入：

```text
Frozen policy
vs PPO scratch
vs ordinary PPO fine-tuning
vs actor-preserving PPO fine-tuning
```

核心评价必须同时包含：

```text
shifted Sea-State performance
+
original PhysicalDeckAttitude deterministic retention
```

尤其不能用 stochastic improvement 换取对现有正式 `96.74%` actor-preserving deterministic benchmark 的破坏。若后续需要 adaptation，actor-preserving 仍是主候选，但这一选择必须由真实 boundary 与 retention 结果驱动，而不是预设结论。

## 18. 与文献调研的关系

当前路线延续 `docs/literature_comparison_matrix.md` 中对 random-wave/JONSWAP maritime landing work 的吸收，但保持本项目 22-D / 4-D contact-aware landing contract；WaveLander 等工作更适合后续 hierarchical landing-timing baseline，而不是替代当前 Sea-State robustness benchmark。

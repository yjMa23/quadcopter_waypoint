# 运动/船舶平台无人机自主降落文献对比矩阵

> 核对日期：2026-08-11  
> 目标：为当前 `quadcopter_waypoint` 项目的论文写作、算法选型、baseline 设计和后续 Sim-to-Real 路线提供可追溯的横向比较。  
> 配套综述：`docs/literature_review_ship_landing_rl.md`

## 0. 使用说明与证据规则

这份文档不是只比较论文摘要，而是尽量回到论文正文、正式出版页和作者官方代码仓库核对以下字段：

- observation / state；
- action / controller output；
- reward / optimization objective；
- landing platform / deck motion model；
- policy / controller frequency；
- learning / control algorithm；
- success definition；
- Sim-to-Real 方法；
- real-world validation；
- source-code availability。

为避免把不同论文中并不等价的“成功率”直接放在一起比较，本文采用以下规则：

1. **论文明确给出数值的才写数值**；没有明确给出的字段标记为“未明确报告”，不推测。
2. **RL policy frequency、physics frequency、camera frequency、optimizer update frequency 分开理解**。例如某论文写 `learning frequency=20`，不能自动解释成 `20 Hz` 控制频率。
3. **touchdown、进入 landing region、真实接触并稳定保持不是同一成功语义**。矩阵会保留原论文定义。
4. `Sim-to-Real-oriented` 不等于已经完成真实部署。仅做 domain randomization + numerical simulation 的工作不会标记为“实机 Sim-to-Real 已验证”。
5. 2026 年预印本的发表状态以 2026-08-11 可公开核对的信息为准。

证据等级：

- **A**：已核对论文全文/正式出版页，关键字段可直接定位；
- **A+**：论文全文 + 作者官方完整代码/实验数据均可核对；
- **B**：论文公开文本可核对主要设计，但部分关键字段尚未公开或未明确报告；
- **C**：仅有摘要/仓库占位内容，不能可靠复原关键实现细节。

---

# 1. 一眼看懂：核心方法总矩阵

> 表格较宽，建议在 Markdown Preview 中横向查看。下方第 2 节给出逐篇展开说明。

| 工作 | 控制粒度 | Observation / State | Action | Reward / Objective | 平台运动模型 | 控制/策略频率 | 算法 | Success 定义 | Sim-to-Real | 实机结果 | 开源 | 证据 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **本项目 actor-preserving PPO (2026)** | 直接连续飞行控制 | **22D state**：机体速度/角速度、投影重力、deck 相对位置、deck 表面点相对速度、align 状态、deck normal、相对角速度 | **4D**：总推力 + 3 轴 moment | 冻结 physical-deck-attitude task reward：位置/预测落点/中心精度、下降与相对速度、接触、landing bonus、crash penalty 等 | XY 平移 + heave；roll/pitch 独立正弦，正式范围 0–5°、0.08–0.15 Hz；真实实体碰撞甲板 | physics **100 Hz**；decimation=2 → policy/action **50 Hz** | PPO；imitation-learning benchmark BC；actor-preserving PPO separate actor/critic + critic warm-up + frozen obs RMS + BC action anchor | 真实 deck contact；落点半径 <0.12 m；安全法向/切向速度、角速度、body-deck angle 等；连续 safe contact 3 control steps | **尚未做** | 无 | 项目本身 | **A+** |
| Rodriguez-Ramos et al. 2019 | RL 控制横向飞行 | 相对平台的连续状态；原工作用于 longitudinal/lateral relative motion | 连续 attitude-level control，核心为 roll/pitch 横向控制 | DDPG landing reward；公开论文可确认其以连续 landing task 为目标，但本次未把每个 reward term 重新誊写 | 移动地面平台；主要横向移动，非 6-DoF 海况 | 原方法被 Goldschmid 2024 作为 baseline，报告 baseline agent **20 Hz** | **DDPG** actor-critic | landing-on-moving-platform；Goldschmid 对其复现实验报告 RPM 0.4 / 1.2 success 为 91% / 73% | 训练仿真 → 真实飞行 | 有真实飞行验证，但原论文未给出像 Goldschmid 那样的大规模实机统计 | `alejodosr/drl-landing`；旧 ROS/Gazebo 栈，README 不完整 | **A/B** |
| Saj et al. 2022 preprint / 2025 journal | 视觉相对状态 → 横向 RL 控制 | 当前 + **过去 5 帧** `p,v`：`(p_t,v_t,...,p_{t-5},v_{t-5})`；p 来自单目 horizon reference bar PnP | RL 只控制 **roll / pitch**；yaw/heave 不作为 RL action | 分区 reward：0.1 m 内主要惩罚 action 与 action change；0.1–0.4 m 加距离惩罚；0.4–2 m 固定惩罚；>2 m 终止惩罚 | **6-DoF ship deck** + wind gusts；真实缩比 motion deck | 未明确报告单一 policy Hz | robust **TD3** + problem-specific domain randomization | 控制目标先进入并保持 0.4 m safe zone，0.1 m 为优选 hover 区；论文没有给出一个与本项目 settled-contact 完全等价的统一二值阈值 | **是**：仿真 robust RL → Parrot ANAFI | Parrot ANAFI + 5×5×10 ft 缩比船台 + 6-DoF deck；论文展示优于 nonlinear PID，但未报告统一 consolidated landing success rate | 未找到作者官方公开控制代码 | **A** |
| Goldschmid & Ahmad 2024 | 分轴 RL + 低层 PID | 1D agent：离散化后的相对位置 `p`、速度 `v`、加速度 `a`、pitch/roll index；连续原始量来自 relative kinematics | 3 个离散 action：**增加角度 / 减少角度 / 不变**；x→pitch，y→roll；z/yaw 用 PID | `r_p+r_v+r_theta+r_dur+r_term`；奖励相对位置/速度/角度改善，惩罚耗时，terminal success/fail | **horizontal rectilinear periodic motion (RPM)**；无平台旋转；另测 8-shape | 场景相关：simulation 11.46/17.19/22.92 Hz；hardware case **22.92 Hz** | **Double Q-Learning** + multiresolution discretization + sequential curriculum + transfer learning | 论文定义：UAV **touches down on moving-platform surface**；训练子任务另有 goal-state 保持条件 | **是** | 实机：static **96%** (23 trials)；RPM0.2 **68%** (25)；RPM0.4 **79%** (28) | **完整官方仓库**，含 sim/UAV/GCS/rosbag/video/实验处理 | **A+** |
| Ali et al. 2024 Offshore Docking | 分层；RL 只做最终垂向 docking | landing phase state：`[z-z_w, zdot]^T`（2D） | PPO：标量连续 virtual vertical input `U∈[Umin,Umax]`；DQN 为离散 U+/U0/U- | `R=-k1 e_p-k2 e_v`；高度误差 + 高度速率误差，目标是降低 impact | 每 episode 用 **JONSWAP spectrum + random phase** 生成随机 vertical wave displacement `z_w(t)` | **未报告 control Hz**；PPO `learning frequency=20` 是算法设置；PPO inference 6.788 ms | PPO + DQN / Double DQN / Dueling DQN | 没有给出与接触阈值绑定的统一 binary success；主要评价 impact velocity、time-to-land、final height | **面向 Sim-to-Real 设计，但未做实机 transfer** | **无真实飞行**；numerical experiments only | 官方 `phoenixrider12/drone_docking`；PPO/DQN 分支，规模较小 | **A** |
| Aikins et al. 2024 | 端到端 rotor-thrust RL | **13D**：relative distance 3D + linear velocity 3D + angular velocity 3D + quaternion 4D；POMDP 中加入噪声/flicker | **4D individual rotor thrusts** | `r_dist=1/(1+distance²)` + yaw/pitch upright shaping，距离越近姿态越重要 | Clearpath Husky UGV 随机移动轨迹；**不是船舶 roll/heave** | 未给单一固定 policy Hz；论文专门改变 sensor sampling / flicker 条件 | **RPO-LSTM**；recurrent robust on-policy，asymmetric actor-critic | UAV-platform distance < **0.20 m** 且发生接触 | 只做鲁棒仿真，不是实机 transfer | 无 | 未找到配套官方仓库 | **A** |
| WaveLander 2026 | **hierarchical RL**，只学 landing timing | compact relative obs：relative height、vertical velocity、platform tilt、tilt-rate | **1D scalar vertical velocity reference** | 截至 2026-08-11，公开摘要/README 未披露可复核的完整 reward 公式 | randomized wave-induced marine-platform motion；具体公开参数尚不足以复核 | 未公开 | RL high-level + conventional low-level attitude/lateral/velocity controller | 公开摘要/README 未给可复核的精确 touchdown thresholds | MuJoCo → Isaac Sim SIL；官方 README 还写明有 representative real-world deployment test | 有代表性实机部署测试，但公开仓库尚无完整实验数据 | 官方 repo **code coming soon**；ICARCV 2026 under review | **B/C** |
| Angelis et al. 2026 | 视觉 sensorimotor RL + conventional low-level attitude controller | 当前/上一帧 sparse keypoints+descriptors；当前/上一帧 `[v,q,omega]`；previous action；**部署时不显式估计 platform state** | **4D** `[roll,pitch,yaw,collective thrust]` command | approach progress + camera perception alignment + action smoothness + near-platform velocity/orientation alignment + collision penalty + terminal reward | training：heave/roll/pitch/yaw sinusoid；roll/pitch ±30°，yaw ±11.25°，heave 0–5 m，shared f 0–0.5 Hz；eval 用 5 个 sinusoid superposition | 论文只写 camera/state fixed frequency，**未给具体 Hz** | **PPO**；separate 2×512 actor/critic；joint visual encoder training；curriculum | `d_xy<1 m`、vertical error <0.2 m、orientation error <10°、relative speed <1.5 m/s；crash if distance>10 m or contact force>10 N | **是，zero-shot visual transfer** | >400 real trials；Stewart 358：touchdown 58.38%、general 96.93%；pool 53：touchdown 90.57%、general 98.11% | 截至核对日未发现论文配套完整公开控制仓库 | **A** |

---

# 2. 核心 RL 工作逐篇拆解

## 2.1 Rodriguez-Ramos et al. — DDPG moving-platform landing

**论文**：A Deep Reinforcement Learning Strategy for UAV Autonomous Landing on a Moving Platform, JIRS 2019.  
**定位**：较早把 continuous-action actor-critic 真正用于 moving-platform UAV landing，并做了 sim → real flight。

### 关键结构

- 使用 **DDPG**，不是 PPO。
- 学习重点是无人机相对移动平台的横向控制。
- Goldschmid 2024 将其作为直接 RL baseline，并明确说明该 baseline 用 continuous states/actions 控制 longitudinal/lateral movement。
- 其最大价值是历史基线意义：证明“仿真学习连续动作 → 真实飞行”在 moving-platform landing 上可行。

### 与当前项目的区别

当前 actor-preserving PPO 已经直接输出总推力和 3 轴 moment，并使用实体碰撞和严格 settled landing。因此不能把 Rodriguez-Ramos 的成功率直接与 actor-preserving PPO 的 96.74% 横比；其接触安全语义远弱于当前项目。

### 值得复用

- related-work 早期 DRL baseline；
- sim-to-real workflow 的历史依据；
- 与 Goldschmid 2024 做“连续深度 RL vs 结构化 tabular curriculum”对照。

### 开源状态

`https://github.com/alejodosr/drl-landing`

仓库公开，但属于 Ubuntu 16.04 / ROS Kinetic / Gazebo 时代的旧栈，直接迁移价值低于论文方法价值。

---

## 2.2 Saj et al. — Robust vision-based ship landing

**论文**：Robust Reinforcement Learning Algorithm for Vision-based Ship Landing of UAVs, arXiv 2022；后续 journal version 为 *Robust Reinforcement Learning Control for Vision-Based Ship Landing of VTOL-UAVs*, Journal of the American Helicopter Society 70(2), 2025, DOI `10.4050/JAHS.70.022004`。

### Observation

策略不直接吃 RGB，而是先通过单目视觉 + horizon reference bar 得到 relative position，再构造时间历史：

```text
s_t = (p_t, v_t, p_{t-1}, v_{t-1}, ..., p_{t-5}, v_{t-5})
```

这点对本项目后续 ArUco 阶段非常重要：**第一版视觉迁移完全可以保持 state-based PPO，只把 ground-truth relative state 换成视觉 estimator 输出，同时加入有限历史。**

### Action

Parrot ANAFI 有 roll / pitch / yaw / heave 接口，但论文 RL 主要学习 roll 和 pitch；两个横向轴可独立处理。底层 autopilot 执行实际电机闭环。

### Reward

对单轴 deviation `d` 分四区：

- `|d| <= 0.1 m`：主要惩罚 action magnitude 和 action change，强调稳定 hover；
- `0.1 <= |d| <= 0.4 m`：同时惩罚 position deviation 和动作；
- `0.4 <= |d| <= 2 m`：固定强惩罚，鼓励尽快进入 safe zone；
- `|d| > 2 m`：episode terminal penalty。

核心思想不是复杂 reward，而是把 **“远处赶紧回来—近处变平滑”** 明确分区。

### Sim-to-Real

这是比 Ali 2024 更强的真实 Sim-to-Real 证据：

```text
Gazebo robust RL + domain randomization
        ↓
Parrot ANAFI
        ↓
monocular horizon-bar localization
        ↓
6-DoF sub-scale ship motion deck + wind
```

### 本项目最值得复用

1. `relative-state estimator + history`，而不是直接 RGB end-to-end；
2. vision latency / state history；
3. domain randomization 不只随机 deck motion，还要随机 wind / perception mismatch；
4. 非线性 PID baseline。

---

## 2.3 Goldschmid & Ahmad — curriculum + interpretable RL

**论文**：Autonomous Robots 2024, DOI `10.1007/s10514-024-10162-8`。

这是目前对本项目最有工程复用价值的公开项目之一。

### Observation

连续基础量为相对：

```text
relative position     p_c
relative velocity     v_c
relative acceleration a_c
relative orientation  phi_c
```

对单轴 agent 离散成：

```text
s = (p_d, v_d, a_d, i_theta)
```

### Action

纵向：

```text
increase pitch
reduce pitch
do nothing
```

横向复用同一个 agent，改为 roll。

vertical 与 yaw 由 PID 管理；论文实验下降速度使用 `v_z=-0.1 m/s`。

### Platform

训练平台采用 rectilinear periodic movement：

```text
omega_mp = v_mp / r_mp
```

无 roll/pitch/yaw rotation，因此比当前 physical-deck-attitude task 简单很多。

### Agent frequency

它是少数**明确从平台运动能力推导 agent frequency**的论文：

```text
f_ag = 2 n_theta k_man omega_mp / pi
```

论文实际配置：

| 场景 | f_ag |
|---|---:|
| simulation, v=0.8 m/s | 11.46 Hz |
| simulation, v=1.2 m/s | 17.19 Hz |
| simulation, v=1.6 m/s | 22.92 Hz |
| hardware case | 22.92 Hz |

### Reward

```text
r = r_p + r_v + r_theta + r_dur + r_term
```

- `r_p`：relative-position improvement；
- `r_v`：relative-velocity improvement；
- `r_theta`：减少过大倾角；
- `r_dur`：时间惩罚；
- `r_term`：goal / fail terminal reward。

### Success

论文最终 landing-trial 定义非常简单：**无人机 touch down 在 moving-platform surface 上即成功**。

它没有像本项目一样额外要求：

- touchdown normal/tangential relative speed；
- hard-contact impulse；
- body-deck angle；
- penetration；
- 连续 contact settle hold。

所以其 99% simulation success 不能与 actor-preserving PPO settled landing 直接等价。

### 实机

| 实机场景 | 成功率 | trials |
|---|---:|---:|
| static | 96% | 23 |
| RPM 0.2 | 68% | 25 |
| RPM 0.4 | 79% | 28 |

论文还很诚实地分析了 sim-to-real gap：vehicle inertia、低层 controller mismatch、state-estimation glitch、未建模 ground effect。

### 本项目最值得复用

**curriculum 的实验思想，而不是其 tabular Q-learning 本身。**

本项目可以形成：

```text
static / translation
→ heave
→ physical contact
→ roll/pitch
→ randomized sea state
→ perception noise
```

而且应该像它一样做 curriculum ablation，而不是只报告最终策略。

---

## 2.4 Ali et al. — PPO + JONSWAP offshore docking

**论文**：Applied Soft Computing 162 (2024) 111843, DOI `10.1016/j.asoc.2024.111843`。

### 任务分解

```text
approach phase: model-based controller
landing phase : DRL
```

它不是完整 6-DoF end-to-end landing。最终 landing phase 被压缩为垂向 docking 问题。

### Observation

```text
S = [z - z_w, zdot]^T
```

其中 `z_w` 是波浪导致的 docking-station vertical displacement。

### Action

PPO 使用连续标量 virtual vertical control：

```text
U ∈ [U_min, U_max]
```

DQN 系列则采用离散：

```text
U+ / U0 / U-
```

### Reward

定义：

```text
e_p = z - z_w
e_v = zdot - zdot_d
R   = -k1 e_p - k2 e_v
```

`zdot_d` 可以设计为衰减下降速度，以降低 impact force。

### Wave model

这是本文对当前项目最有价值的地方：

```text
JONSWAP spectrum
   ↓
random phase for each episode
   ↓
inverse frequency-domain construction
   ↓
z_w(t)
```

每回合重新生成随机 wave，使策略不会只记住单一 sinusoid phase。

### PPO 配置与结果

- actor/critic learning rate：`3e-4`；
- likelihood clip：`0.2`；
- 论文写 `learning frequency=20`，**不能据此声称 policy 控制是 20 Hz**；
- PPO inference time：6.788 ms。

主要结果：

| Agent | Impact velocity | Time to land | Inference |
|---|---:|---:|---:|
| PPO | 0.327 m/s | 4.9 s | 6.788 ms |
| DQN | 0.820 m/s | 5.4 s | 9.854 ms |
| Double DQN | 0.223 m/s | 7.7 s | 7.220 ms |
| Dueling DQN | 2.419 m/s | 6.3 s | 11.296 ms |

### 必须避免的误读

论文标题包含 `Sim-to-Real Policy Transfer`，但正式出版页明确说明结果是 **numerical experiments**。因此目前最稳妥的表述是：

> 该工作设计了面向 Sim-to-Real 的 domain-randomized policy，但没有报告真实无人机 offshore docking 实验。

### 本项目最值得复用

**JONSWAP，不是它的 2D state / 1D action。**

本项目已经有更完整的 22D / 4D contact-aware controller，不应降维回它的垂向动力学；更合理的是把当前 deterministic sinusoidal heave 替换/扩展成海况驱动的 stochastic deck excitation。

---

## 2.5 Aikins et al. — partial observability / RPO-LSTM

**论文**：Drones 8(6):232, 2024。

### Observation

```text
[d(3), v(3), omega(3), q(4)] = 13D
```

其中 `d` 是 UAV 到 moving platform 的 3D relative distance。

### Action

```text
[T1, T2, T3, T4]
```

直接给四个 rotor thrust，因此从控制粒度看甚至比当前项目的 total-thrust + moments 更底层。

### POMDP 设计

论文重点不是 ship motion，而是：

- sensor noise；
- flicker：整个 observation 暂时变成零/不可用；
- noise + flicker；
- 不同 sampling rate。

RPO-LSTM 用 recurrent memory 处理 observation corruption。

### Success

```text
relative distance < 0.20 m
AND contact with platform
```

### 结果中最值得记的部分

在严重 flicker 下，普通 MDP policy 很快崩溃，而 recurrent policy 仍保持明显更高成功率。例如论文报告 flicker probability 0.4 时：

```text
MDP policy: 41%
RPO-LSTM : 100%
```

这对本项目视觉阶段的启示非常直接：**如果 ArUco 会短暂丢失，不能只训练“每帧都有完美 pose”的 Markov state policy。**

---

## 2.6 WaveLander — hierarchical landing timing

**论文**：arXiv:2607.01281，2026。  
**当前状态**：官方仓库截至 2026-08-11 写明已投 ICARCV 2026、under review、code release coming soon。

### Observation

官方 README 明确列出 compact relative observation：

```text
relative height
vertical velocity
platform tilt
tilt-rate information
```

### Action

只有：

```text
vertical velocity reference
```

低层传统控制器负责：

```text
attitude stabilization
lateral tracking
velocity tracking
```

因此 RL 学的是：

```text
when to descend
when to hold
when to retreat
```

而不是“如何产生每个姿态/推力动作”。

### 为什么对本项目重要

它可以成为非常强的 **architecture ablation**：

```text
本项目：22D state → 4D thrust/moments
WaveLander-style：relative deck state → vz_ref → conventional controller
```

如果以后答辩被问“为什么需要让 RL 连低层控制也学”，这个 baseline 会非常有价值。

### 不能现在写死的字段

截至核对日公开资料不足以严谨恢复：

- 完整 reward equation；
- exact platform-wave parameter ranges；
- policy Hz；
- exact touchdown-success thresholds；
- 完整 real-world statistics。

这些字段在官方代码/完整 manuscript 公开后应再次补充。

---

## 2.7 Angelis et al. — Vision-Based Agile Landing on Turbulent Waters

**论文**：arXiv:2605.23717v2, 2026-07-31。

这是当前最值得作为“最终目标形态”研究的工作之一。

### Observation

策略部署时**没有 explicit platform state**。

原始 observation：

```text
M_t, M_{t-1}       sparse local features
s_t, s_{t-1}       UAV state
previous action
```

其中 UAV state：

```text
s_t = [linear velocity, quaternion, angular velocity]
```

视觉不是 raw RGB，而是：

```text
keypoint coordinates + normalized descriptors
```

经 SRPose encoder 得到 feature `F_t` 后，policy 输入为：

```text
x_t = [F_t, s_t, s_{t-1}, a_{t-1}]
```

### Action

```text
[roll, pitch, yaw, collective thrust]
```

低层 attitude controller 跟踪这些 command。

### Platform training motion

```text
eta_i(t) = A_i sin(2 pi f t + phi_i)
```

训练范围：

| DoF | range |
|---|---|
| roll | `[-pi/6, pi/6]` ≈ ±30° |
| pitch | `[-pi/6, pi/6]` ≈ ±30° |
| yaw | `[-pi/16, pi/16]` ≈ ±11.25° |
| heave | 0–5 m |
| shared frequency | 0–0.5 Hz |

评估时 roll/pitch/heave 使用 **5 个 sinusoidal components superposition**，最大 tilt 30°、vertical displacement 5 m。

### Reward

总 reward：

```text
approach progress
+ velocity-direction reward
+ perception/FOV alignment
+ action smoothness
+ proximity-gated relative velocity alignment
+ orientation alignment
+ unsafe-contact configuration penalty
+ terminal success/fail
```

与当前项目很值得对照的一点：它直接给 **action difference penalty**，而当前 actor-preserving PPO 的主要策略保护来自 BC action anchor，目标不同但都在限制 aggressive control change / drift。

### Curriculum

```text
static platform
→ success > 90% 后开启 roll/pitch/yaw
→ 每 4096 episodes 调整 heave max
   success >60% : +0.5 m
   success <20% : -0.25 m
```

同时训练时每 episode 随机 25 个平台点，每点 64D descriptor；每 step 随机 drop 20% descriptor 并加噪声。

### Success / Failure

成功：

```text
d_xy < 1 m
vertical error < 0.2 m
orientation error < 10 deg
relative speed < 1.5 m/s
```

失败：

```text
total distance > 10 m
OR contact force > 10 N
```

如果低冲击接触但还没满足 success，episode 可以继续，让无人机 recovery。

**注意：这个 success definition 与当前 actor-preserving PPO 的 “physical safe contact 连续保持 3 control steps” 并不等价。**

### PPO

- Gaussian actor-critic；
- actor / critic separate；
- each MLP: 2×512, tanh；
- PPO total steps `1e8`；
- rollout 256；
- 16 mini-batches；
- update epochs 1；
- lr `3e-4`；
- gamma `0.98`；
- GAE `0.95`；
- clip `0.2`。

这和 actor-preserving PPO 的 separate actor/critic 结构有直接可比性，但它没有当前项目的 BC initialization / actor-preserving anchor。

### Simulation results

每种方法 500 trials：

| Method | Success | Crash | Maneuver time |
|---|---:|---:|---:|
| MPC-NE | 48.60% | 51.40% | 9.06 ± 9.14 s |
| Ours (PD) | 74.21 ± 3.93% | 25.79 ± 3.93% | 5.02 ± 2.28 s |
| Ours (RD) | 63.08 ± 11.26% | 36.92 ± 11.26% | 4.28 ± 1.62 s |

### Real-world results

论文 v2 明确写 **>400 real-world trials**：

| 实验 | N | Tilt violation | Velocity violation | Touchdown success | General success | MT |
|---|---:|---:|---:|---:|---:|---:|
| Stewart platform | 358 | 40.5% | 4.19% | 58.38% | 96.93% | 4.24 ± 5.68 s |
| Pool | 53 | 3.77% | 7.54% | 90.57% | 98.11% | 2.90 ± 1.56 s |

这里也说明为什么只报一个“success rate”容易误导：论文同时区分 touchdown success 与 general success。

### 本项目最值得复用

1. 不直接 sim-render photorealistic RGB，而是训练 **synthetic sparse feature interface**；
2. two-frame visual history；
3. descriptor dropout/noise；
4. static→attitude→heave curriculum；
5. MPC baseline；
6. success / crash / maneuver time 分开统计。

---

# 3. 非学习 baseline 对比矩阵

这些工作不应该硬塞进“reward”列，而应看其 state estimator、optimization objective、trajectory generator 与 low-level controller。

| 工作 | State / perception | Controller output | Objective | 平台 | 频率 | Success / metric | 实机 | 对当前项目价值 |
|---|---|---|---|---|---|---|---|---|
| **Paris et al. 2020, ICRA** | EKF：远处 simulated GPS，近处 visual fiducial；估计 platform position/orientation/velocity；UAV 本体状态 | online landing trajectory + boundary-layer sliding control | receding-horizon fast/direct landing；显式处理 bounded turbulent disturbance | moving ground platform + strong turbulent wind；非 marine 6-DoF | 本次核对未找到明确 single controller Hz | 论文用 tracking / landing experiments 验证，未给与 actor-preserving PPO 相同的 settled-contact threshold | **有**，leaf-blower turbulent-wind hardware | **最适合做“RL 为什么必要”的 robust classical baseline** |
| **Wang et al. 2022** | **ArUco relative pose** + restricted localization | gradient-based local planner 生成 collision-free reference trajectory；FSM 执行 takeoff/tracking/landing | 轨迹可行、避障、移动目标追踪 | horizontally moving platform；不是波浪船舶 | 未明确报告 | sim + indoor + outdoor successful landing demonstrations；未给统一 binary threshold | **有** | **ArUco 第一版系统架构直接参考** |
| **NMPC-Lander 2025** | UAV 全 12 states + landing-platform position；VICON 实验反馈 | NMPC 发布 angular velocities + normalized thrust | NMPC trajectory tracking + platform-position cost + CBF obstacle constraint | static / dynamic ground platform；动态实机平台约 1 m/s，人工不规则移动 | **10 Hz** publish | Final Point Error (FPE)，不是 binary success | **有**：dynamic no obstacle 6.4 cm；with obstacle 11 cm FPE | **高质量 MPC/NMPC baseline；还能顺带做 obstacle-aware landing** |

## 3.1 为什么 NMPC-Lander 很适合做本项目 baseline

它与早期 PID baseline 相比更有说服力：

- 12-state nonlinear prediction；
- moving target tracking；
- real-time 10 Hz；
- CBF safety constraint；
- real hardware；
- 有明确 FPE 数值。

但它的平台没有 wave-induced roll/pitch/heave，因此如果后面实现 baseline，应当统一给它输入当前项目相同的 deck state，避免比较不公平。

---

# 4. actor-preserving PPO 相关的方法学论文

## Kernbach et al. 2026 — Actor-Critic Pretraining for PPO

这篇不是 landing 论文，不应放在 landing-success 表中硬比，但它与 imitation-learning benchmark/actor-preserving PPO 的算法问题高度相关。

核心问题：

```text
expert demonstrations
→ actor BC pretraining
→ PPO fine-tuning
```

如果只预训练 actor，而 critic 是 cold-start，PPO 初期可能产生不稳定更新。论文进一步预训练 critic，并在多个 manipulation / locomotion tasks 中展示 sample-efficiency 改善。

与当前 actor-preserving PPO 的关系：

| 问题 | Kernbach 2026 | 当前 actor-preserving PPO |
|---|---|---|
| actor 初始策略 | expert pretraining | BC actor |
| critic cold start | 用 critic pretraining 缓解 | **critic-only warm-up 10 epochs** |
| actor 保护 | 预训练后 PPO | **warm-up 冻 actor + frozen obs RMS + BC mean-action anchor** |
| actor/critic 参数 | actor-critic pretraining | **separate actor/critic** |
| 目标 | generic PPO sample efficiency | moving-deck landing policy preservation |

因此它适合作为 actor-preserving PPO 方法动机的外部文献支持，但不能被写成“已有工作已经做了本项目同样的 actor-preserving landing 方法”。

---

# 5. 开源项目可复用矩阵

| 项目 | 技术栈/状态 | 完整度 | 直接可复用内容 | 不建议照搬的部分 | 当前优先级 |
|---|---|---:|---|---|---:|
| `robot-perception-group/rl_multi_rotor_landing` | ROS Noetic + Gazebo 11 + RotorS + Python3 | **很高**；35 commits；sim/UAV/GCS/real data 全 | curriculum、relative-state pipeline、noise evaluation、实机统计流程 | tabular Q-learning、旧 ROS1/Gazebo 运行栈 | **S** |
| `phoenixrider12/drone_docking` | Python；PPO/DQN branches | 中 | **JONSWAP wave generation**、offshore docking RL framing | 2D state + 1D vertical action，过度简化当前 actor-preserving PPO | **S** |
| `JackyLi-HKUST/WaveLander` | repo placeholder；MIT；code coming soon | 当前低 | 等代码公开后重点看 high-level vertical policy interface | 现在还没有可迁代码 | **S / watch** |
| `alejodosr/drl-landing` | ROS Kinetic + Gazebo + Bebop2 + DDPG | 低到中；旧项目 | early DDPG workflow、历史 baseline | 直接迁旧依赖 | **B** |
| `btx0424/OmniDrones` | Isaac Sim drone RL research platform | 高 | sensor/control abstraction、多无人机 RL task 设计 | 不应为了它把当前 Isaac Lab 5.1 项目迁平台 | **A** |
| `ntnu-arl/aerial_gym_simulator` | Isaac Gym / GPU aerial robot simulation | 高 | Angelis 论文训练基础；GPU parallelism、controller、visual task 设计 | 与当前 Isaac Lab 版本栈不同 | **A** |
| `thu-uav/SimpleFlight` | quadrotor Sim-to-Real RL | 高 | dynamics randomization、zero-shot Sim-to-Real 设计 | 不是 moving-deck landing 专项 | **A，实机前** |

---

# 6. 与当前 actor-preserving PPO 的严格横向对比

这一节最适合直接转成毕业论文中的 related-work discussion。

| 维度 | 当前 actor-preserving PPO | 最接近工作 | 当前优势 | 当前缺口 |
|---|---|---|---|---|
| 物理接触语义 | **真实 deck ContactSensor + hard/safe contact + settle hold** | Angelis 2026 有 contact-force fail；Goldschmid touchdown surface | 成功定义严格，可区分 deck miss / hard contact / timeout | 尚无实机 contact validation |
| 平台姿态 | roll/pitch + heave + translation | Angelis 2026；Saj 2022/2025 | 已有实体倾斜甲板和刚体表面点运动学 | 没有 yaw / full marine 6-DoF；波浪仍规则 |
| 波浪真实性 | 正弦参数化 | Ali 2024 JONSWAP | 可控、可重复 benchmark | 缺 stochastic sea spectrum / vessel response |
| policy action | thrust + 3 moments | Angelis 为 attitude+thrust；Aikins 为 rotor thrust | 控制权限高，可直接学习动态补偿 | Sim-to-Real 难度高于 high-level velocity policy |
| observation | 22D perfect state | Saj historical relative state；Angelis sparse visual features | MDP 明确、训练稳定 | 依赖 simulator ground truth，未含 perception noise/delay |
| partial observability | 无显式 memory | Aikins RPO-LSTM；Saj 6-step history；Angelis two-frame vision | 当前无需额外 RNN 复杂度 | ArUco dropout 时可能脆弱 |
| curriculum | 已经历 heave-precision task→physical-deck task→physical-deck-attitude task，但不是单次自动 curriculum | Goldschmid；Angelis | 工程阶段递进非常完整 | 缺 formal curriculum ablation |
| imitation / initialization | BC + actor-preserving PPO | Kernbach 2026 generic PPO pretraining | **这是本项目最鲜明的方法特点之一** | 还没有对 marine/perception shift 验证 |
| model-based baseline | 当前未形成正式强 baseline | Paris 2020；NMPC-Lander；Angelis MPC-NE | RL benchmark 已完整 | 毕业论文最好补 PID/NMPC 至少一个 |
| Sim-to-Real | 未做 | Saj、Goldschmid、Angelis | 已有严格 formal sim benchmark | 这是后续最大缺口 |

---

# 7. 从矩阵直接得到的后续研究顺序

## Priority 1：不要动 actor-preserving PPO frozen benchmark

actor-preserving PPO 已经回答一个独立问题：

> 如何在已有高质量 landing actor 的基础上，用 PPO 继续优化而不过度破坏原策略。

后续海况、视觉和 Sim-to-Real 应作为新阶段，不应回头修改 actor-preserving PPO 的 observation/reward/success，使当前 96.74% 失去可追溯性。

## Priority 2：引入 JONSWAP / stochastic sea-state benchmark

优先借鉴 Ali 2024 的 **wave generation**，而不是它的控制器：

```text
sea-state parameters
      ↓
JONSWAP wave spectrum
      ↓
random wave realization per episode
      ↓
deck heave / attitude excitation
      ↓
current 22D → 4D actor-preserving PPO policy
```

理想情况下进一步加：

```text
wave elevation
→ simplified vessel response / RAO
→ heave + roll + pitch
```

这样才比“给 roll/pitch 再多加几个随机正弦”更像真正 maritime benchmark。

## Priority 3：先做 state-estimation Sim-to-Real，不做 raw-image end-to-end

第一版最稳：

```text
multi-ArUco coplanar board
      ↓
relative pose estimator
      ↓
finite-difference / filter relative velocity
      ↓
与现有 22D contract 对齐
      ↓
actor-preserving PPO actor
```

训练阶段加入：

- pose noise；
- velocity noise；
- latency；
- random frame drop；
- short history。

这里直接吸收 Saj + Aikins，而不是一步跳到 Angelis 的 visual-feature policy。

## Priority 4：加一个强 classical baseline

推荐顺序：

1. relative-state PID / state machine：快速得到最低基线；
2. NMPC：论文主 baseline；
3. WaveLander-style hierarchical RL：架构 ablation。

最终比较：

```text
PID / NMPC
vs
high-level RL (vz timing)
vs
current end-to-end state PPO
vs
actor-preserving PPO
```

## Priority 5：最后再做 Angelis-style visual feature policy

如果毕业设计时间允许，最终研究扩展才是：

```text
raw camera
→ keypoints/descriptors
→ temporal feature encoder
→ policy
```

而且更推荐它的 synthetic-feature sim-to-real 思路，而不是追求 photorealistic RGB domain randomization。

---

# 8. 最关键的五个结论

1. **当前项目的 settled-landing 成功语义比多数同类 RL 论文严格。** 很多论文把 touchdown、进入 landing region 或距离阈值视作成功；本项目还要求真实接触、安全相对速度/姿态和连续保持。因此以后论文中不要只用“成功率更高”做宣传，要明确 success contract。

2. **最值得马上工程复用的是 Ali 2024 的 JONSWAP，而不是换成它的 1D PPO。** 当前 22D/4D 策略已经解决了更复杂的控制问题，下一步应该增强 disturbance realism。

3. **视觉第一版最应该学 Saj 2022/2025，不是直接学 Angelis 2026。** 即“视觉负责 relative-state estimation，策略仍是 state policy”；这样能最大程度复用 actor-preserving PPO。

4. **partial observability 必须提前设计。** Saj 用 6-step state history，Aikins 用 LSTM，Angelis 用 two-frame sparse features；三条路线都说明真实视觉落船不能长期假设每一帧都有完美平台状态。

5. **actor-preserving PPO 的 actor-preserving PPO 是当前项目区别于这些 landing 工作的一个真实方法贡献点。** 现有高质量 moving/maritime landing 论文主要研究平台扰动、视觉、curriculum 或 landing timing；它们并未解决“高性能 imitation-initialized actor 在 PPO fine-tuning 中如何避免 policy drift”这一与你 imitation-learning benchmark→actor-preserving PPO 直接对应的问题。

---

# 9. 主要来源

## Landing / control papers

1. Rodriguez-Ramos, A., et al. *A Deep Reinforcement Learning Strategy for UAV Autonomous Landing on a Moving Platform.* Journal of Intelligent & Robotic Systems, 2019. DOI: `10.1007/s10846-018-0891-8`  
   OA: https://oa.upm.es/67140/  
   Code: https://github.com/alejodosr/drl-landing

2. Paris, A., Lopez, B. T., How, J. P. *Dynamic Landing of an Autonomous Quadrotor on a Moving Platform in Turbulent Wind Conditions.* ICRA 2020.  
   https://arxiv.org/abs/1909.11071

3. Wang, P., et al. *Quadrotor Autonomous Landing on Moving Platform.* 2022.  
   https://arxiv.org/abs/2208.05201

4. Saj, V., Lee, B., Kalathil, D., Benedict, M. *Robust Reinforcement Learning Algorithm for Vision-based Ship Landing of UAVs.* 2022.  
   https://arxiv.org/abs/2209.08381  
   Journal version DOI: `10.4050/JAHS.70.022004`

5. Goldschmid, P., Ahmad, A. *Reinforcement Learning based Autonomous Multi-Rotor Landing on Moving Platforms.* Autonomous Robots, 2024.  
   https://arxiv.org/abs/2302.13192  
   Code: https://github.com/robot-perception-group/rl_multi_rotor_landing

6. Ali, A. M., Gupta, A., Hashim, H. A. *Deep Reinforcement Learning for Sim-to-Real Policy Transfer of VTOL-UAVs Offshore Docking Operations.* Applied Soft Computing, 2024.  
   https://arxiv.org/abs/2406.00887  
   DOI: `10.1016/j.asoc.2024.111843`  
   Code: https://github.com/phoenixrider12/drone_docking

7. Aikins, G., Jagtap, S., Nguyen, K.-D. *A Robust Strategy for UAV Autonomous Landing on a Moving Platform under Partial Observability.* Drones, 2024.  
   https://www.mdpi.com/2504-446X/8/6/232

8. Batool, A., et al. *NMPC-Lander: Nonlinear MPC with Barrier Function for UAV Landing on a Mobile Platform.* 2025.  
   https://arxiv.org/abs/2505.03931

9. Li, C.-K., et al. *WaveLander: A Generalizable Hierarchical Control Framework for UAV Landing on Wave-Disturbed Platforms via Reinforcement Learning.* 2026.  
   https://arxiv.org/abs/2607.01281  
   Official repo: https://github.com/JackyLi-HKUST/WaveLander

10. Angelis, D., Bauersfeld, L., Scaramuzza, D., Boukas, E. *Vision-Based Agile Landing on Turbulent Waters.* arXiv v2, 2026-07-31.  
    https://arxiv.org/abs/2605.23717

11. Kernbach, A., et al. *Actor-Critic Pretraining for Proximal Policy Optimization.* 2026.  
    https://arxiv.org/abs/2602.23804

## Frameworks

- OmniDrones: https://github.com/btx0424/OmniDrones
- Aerial Gym Simulator: https://github.com/ntnu-arl/aerial_gym_simulator
- SimpleFlight: https://github.com/thu-uav/SimpleFlight

## Current-project evidence

- `README.md`
- `docs/physical_deck_attitude_theory.md`
- `docs/imitation_hybrid_paper.md`
- `docs/checkpoint_selection_and_policy_drift.md`
- `docs/actor_preserving_ppo.md`
- `benchmarks/actor_preserving_ppo/summary.json`

---

# 10. 后续维护约定

当以下任一事件发生时应更新本矩阵：

- WaveLander 正式论文/代码公开；
- Angelis 等工作开放代码；
- 本项目增加 JONSWAP / vessel-response 环境；
- 本项目接入 multi-ArUco perception；
- 出现正式 NMPC/PID baseline；
- 出现实机 landing 数据。

更新时必须继续区分：

```text
simulation success
physical-contact success
settled success
real-world touchdown success
real-world general success
```

不要把这些指标压成一个没有定义的“landing success rate”。

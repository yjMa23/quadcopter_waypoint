# PX4-Compatible Hierarchical RL / Deployable Action Interface

> 状态：实现前理论与架构门禁文档。  
> 项目：`/home/j/Isaac_RL_Projects/quadcopter_waypoint`  
> 日期：2026-08-23  
> 目的：在**不修改任何冻结 Direct RL 方法语义**的前提下，新增可通过 PX4 Offboard velocity reference 部署的独立 Hierarchical RL 方法。  
> 关键原则：`Direct RL != deprecated`；`PX4-Compatible Hierarchical RL` 是新的独立方法，不是旧 4D checkpoint 的重解释。

---

## 0. Scope、事实基线与非目标

当前冻结 Direct RL 控制链为：

```text
22-D observation
→ PPO / actor-preserving PPO
→ normalized 4-D action
→ collective thrust + body moments
→ Isaac Lab rigid-body wrench
```

冻结任务包括：

```text
quadrotor_ship_landing
quadrotor_ship_landing_heave
quadrotor_ship_landing_physical_deck
quadrotor_ship_landing_physical_deck_attitude
quadrotor_ship_landing_sea_state
```

本设计不得改变上述任务的 action semantics、reward、success/contact contract、历史 benchmark 或 checkpoint 语义。

传统项目 `/home/j/ws_aruco_landing` 当前实际接口经代码核对为：

```text
ROS2
→ px4_msgs::msg::OffboardControlMode
   position=true
→ px4_msgs::msg::TrajectorySetpoint
   position + velocity feedforward + acceleration feedforward + yaw
→ PX4 Offboard
```

`control_rate_hz` 默认 20 Hz。传统项目只作为只读部署接口参考。

本任务新增：

```text
deployable relative observation
→ RL policy
→ deck-relative velocity reference
→ PX4-compatible reference adapter
→ training: vectorized PX4-like controller
   deployment: real PX4 velocity Offboard
```

本阶段明确不做：

- 不把已有 4D thrust/moment action 原地替换为 velocity action；
- 不把旧 22D→4D checkpoint 当成 22D→3D velocity policy；
- 不学习 yaw；
- 不学习 motor action；
- 不在大规模训练中为每个环境启动 PX4 SITL；
- 不引入 ROS2/PX4 运行时依赖到训练数学模块；
- 不因为接口变化而默认重做 reward 或 success contract；
- 不宣称 PX4-like controller 等价于 PX4。

---

# 1. Motivation：为什么 Direct RL 存在较大的部署 gap

当前 `PhysicalDeckAttitude` 的动作是：

\[
\mathbf a_{direct}=[a_T,a_{M_x},a_{M_y},a_{M_z}]\in[-1,1]^4
\]

映射为：

\[
F_z^B=1.9mg\frac{a_T+1}{2},\qquad
\boldsymbol\tau^B=0.01[a_{M_x},a_{M_y},a_{M_z}]^T.
\]

随后直接通过 Isaac Lab rigid-body wrench 施加于机体。该接口对训练非常直接，但部署到真实 PX4 时，策略的动作语义位于飞控内部控制层级的低层位置，且训练时没有经历实际 PX4 控制器、control allocation、motor dynamics、估计器延迟和执行器限制。

必须区分以下控制权限：

| 控制接口 | policy / 外部控制器直接决定 | 仍由飞控或低层控制器负责 | 典型部署 gap |
|---|---|---|---|
| motor-level | 每个电机/执行器命令 | 几乎无 | 最大；依赖电机、ESC、allocation、机体参数 |
| thrust/torque-level | 总推力与三轴力矩 | allocation / actuator | 很大；绕过 position/velocity/attitude/rate 控制 |
| body-rate + thrust | 体轴角速度与推力 | rate loop 以下 | 较大；需要策略学习平移到姿态/推力映射 |
| attitude + thrust | 姿态与推力 | attitude/rate/allocation | 中等 |
| acceleration-level | 世界加速度参考 | acceleration→attitude/thrust 以下 | 中等；仍要求策略直接决定动力学级命令 |
| velocity-level | 世界速度参考 | velocity→acceleration→attitude→rate→allocation | 较小且具有良好动态响应 |
| position-level | 世界位置参考 | position loop 及以下 | 最小，但对高速运动甲板会增加外环滞后 |

本项目当前主要 gap 不是“PX4 能否接受 thrust/torque 消息”，而是**训练策略绕过了真实部署时希望复用的稳定控制层级**。因此，仅把 4D thrust/moment 改为 PX4 `VehicleThrustSetpoint + VehicleTorqueSetpoint`，虽然消息类型变成 PX4 标准接口，却仍不能复用 PX4 的 position/velocity/attitude/rate 控制器，不能从根本上缩小当前主要 controller-implementation gap。

---

# 2. PX4 control hierarchy 与 Offboard 语义

## 2.1 多旋翼控制层级

本文采用以下功能层级描述 PX4 多旋翼控制：

```text
position reference
→ position control
→ velocity reference
→ velocity control
→ acceleration / thrust-vector target
→ attitude + collective thrust
→ attitude control
→ body-rate reference
→ rate control
→ torque demand
→ control allocation
→ motor / actuator command
→ vehicle dynamics
```

真实实现内部存在前馈、限幅、anti-windup、状态估计耦合等细节，上图是本项目训练 surrogate 所需的最小功能抽象，不是 PX4 源码逐函数复制。

## 2.2 OffboardControlMode 的层级含义

PX4 当前 Offboard contract 中，`OffboardControlMode` 的字段具有优先级。对于多旋翼：

```text
position=true
→ TrajectorySetpoint.position 有效
→ position + velocity + lower loops 保持

velocity=true, position=false
→ TrajectorySetpoint.position = NaN
→ TrajectorySetpoint.velocity 有效
→ velocity + lower loops 保持

acceleration=true, position=false, velocity=false
→ TrajectorySetpoint.acceleration 有效
→ position/velocity 外环不作为主要 reference loop

attitude=true
→ VehicleAttitudeSetpoint

body_rate=true
→ VehicleRatesSetpoint

thrust_and_torque=true
→ VehicleThrustSetpoint + VehicleTorqueSetpoint
→ PX4 内部运动控制环被绕过

direct_actuator=true
→ ActuatorMotors / ActuatorServos
→ 更低层直接执行器控制
```

对 velocity Offboard，关键要求是：

```text
OffboardControlMode.position     = false
OffboardControlMode.velocity     = true
OffboardControlMode.acceleration = false
OffboardControlMode.attitude     = false
OffboardControlMode.body_rate    = false
OffboardControlMode.thrust_and_torque = false
OffboardControlMode.direct_actuator   = false

TrajectorySetpoint.position     = [NaN, NaN, NaN]
TrajectorySetpoint.velocity     = [vn, ve, vd]
TrajectorySetpoint.acceleration = [NaN, NaN, NaN]
TrajectorySetpoint.yaw          = deterministic yaw reference
```

所有 position / velocity / acceleration setpoint 均按 PX4 local NED 解释。

## 2.3 为什么 thrust + torque PX4 mode 不是本项目答案

`VehicleThrustSetpoint + VehicleTorqueSetpoint` 的优点是消息标准化、可以让 PX4 继续负责更低层 actuator allocation；但它仍让 RL 对平移稳定、姿态稳定和角速率稳定承担主要责任。当前 Direct RL 最大的 Sim-to-Real gap 正来自训练时直接 rigid-body wrench 与实机低层控制/执行器链的差异。因此 thrust+torque Offboard 只能减小“消息/API gap”，不能充分减小“controller implementation gap”和“动作语义 gap”。

---

# 3. Architecture decision：控制接口正式比较

评分定义：`5 = 对本项目最有利`，`1 = 最不利`。`parallel-training cost` 分数越高表示训练成本越低。

| Candidate | Sim2Real gap | PX4 reuse | moving-deck responsiveness | policy learning difficulty | safety | parallel-training cost | traditional-baseline comparability | deployment complexity | 总体判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A. position reference | 5 | 5 | 2 | 5 | 5 | 5 | 5 | 5 | 部署最简单，但运动甲板最终跟踪外环滞后偏大 |
| **B. velocity reference** | **4** | **5** | **5** | **4** | **4** | **5** | **5** | **5** | **首选** |
| C. acceleration reference | 3 | 4 | 5 | 3 | 3 | 5 | 4 | 4 | 响应快，但动作更接近动力学层，约束/迁移更难 |
| D. attitude + thrust | 2 | 3 | 5 | 2 | 2 | 5 | 2 | 3 | 仍需策略学习大部分平移动力学映射 |
| E. body rate + thrust | 2 | 2 | 5 | 2 | 2 | 5 | 2 | 3 | 低层权限过高 |
| F. thrust + torque | 1 | 1 | 5 | 1 | 1 | 5 | 1 | 2 | 与当前 Direct RL gap 本质接近 |

选择 velocity-level 的原因：

1. **直接可部署**：输出可映射为 `TrajectorySetpoint.velocity`，无需重新解释 action semantics。
2. **复用 PX4 关键控制层**：保留速度、姿态、角速度、allocation 等实际飞控链。
3. **运动甲板响应足够快**：比 position reference 少一层位置外环，适合平台速度/角速度引起的瞬时 surface-point velocity 补偿。
4. **policy 学习难度适中**：policy 学的是“相对甲板应如何运动”，而不是机体姿态或 wrench。
5. **安全约束天然可解释**：m/s 级 velocity clamp、slew limit、normal descent limit 可直接与 landing contact 指标对应。
6. **传统 baseline 可比**：传统项目已经通过 `TrajectorySetpoint` 接入 PX4；Hierarchical RL 只是把高层 reference 生成器换成 policy。
7. **GPU 并行友好**：训练只需 vectorized controller surrogate，不需要 N 套 SITL。

---

# 4. Mathematical formulation

## 4.1 Frames、方向、单位

本文固定以下 frame contract：

### Simulation / policy side

- \(\mathcal F_W\)：Isaac world，部署边界**显式约定为 ENU-compatible**：`+x = East, +y = North, +z = Up`。
- \(\mathcal F_B\)：Isaac robot body，FLU-compatible：`+x forward, +y left, +z up`。
- \(\mathcal F_D\)：deck frame：`+x=t1, +y=t2, +z=n`，其中 `n` 为甲板上表面外法向。
- Isaac / 项目数学 quaternion：`(w,x,y,z)`，表示 local frame → parent/world 的主动旋转。

### PX4 deployment side

- \(\mathcal F_N\)：PX4 local NED：`+x North, +y East, +z Down`。
- PX4 body：FRD：`+x forward, +y right, +z down`。
- SI 单位：位置 m，速度 m/s，加速度 m/s²，角速度 rad/s，力 N，力矩 N·m。

重要：Isaac 并不全局强制“世界坐标就是 ENU”。**本方法的 deployment adapter 显式把训练 world 规定为 ENU-aligned contract**。若未来场景资产采用其他世界轴定义，必须在 adapter 边界改 frame mapping，禁止在业务代码中散落符号翻转。

## 4.2 ENU ↔ NED

对向量：

\[
\begin{bmatrix}v_N\\v_E\\v_D\end{bmatrix}
=
\begin{bmatrix}
0&1&0\\
1&0&0\\
0&0&-1
\end{bmatrix}
\begin{bmatrix}v_E\\v_N\\v_U\end{bmatrix}.
\]

即：

```text
v_ned = [v_enu_y, v_enu_x, -v_enu_z]
```

逆变换相同：

```text
v_enu = [v_ned_y, v_ned_x, -v_ned_z]
```

ENU yaw（从 +East 朝 +North 正向旋转）到 PX4 NED yaw（从 +North 朝 +East）采用：

\[
\psi_{NED}=wrap(\pi/2-\psi_{ENU}).
\]

第一版 policy **不输出 yaw**。simulation yaw contract 默认固定为 `yaw_enu = 0`；deployment adapter 由配置显式转换。若未来要沿用传统任务已有 yaw contract，只修改 deterministic yaw source，不改变 3D action。

## 4.3 FLU ↔ FRD

体轴向量转换：

\[
[x,y,z]_{FRD}=[x,-y,-z]_{FLU}.
\]

第一版 policy action 在 deck frame 中，最终部署输出为 NED velocity，因此 Reference Adapter 的主链不需要将 action 先转 FRD。FLU/FRD 只在低层 controller state/torque 对照和未来 PX4 attitude/rate validation 中使用。

## 4.4 Deck frame 与 contact-point velocity

设 deck reference point 世界位置、线速度、角速度为：

\[
\mathbf p_d^W,\quad \mathbf v_d^W,\quad \boldsymbol\omega_d^W.
\]

目标 landing contact point 世界位置为 \(\mathbf p_c^W\)，lever arm：

\[
\mathbf r_c^W=\mathbf p_c^W-\mathbf p_d^W.
\]

刚体表面点速度：

\[
\boxed{\mathbf v_c^W=\mathbf v_d^W+\boldsymbol\omega_d^W\times\mathbf r_c^W}
\]

这必须复用 `physical_deck_attitude_math.rigid_surface_point_velocity()`，不能复制另一套符号不同的公式。

令 deck-to-world rotation：

\[
R_{WD}=[\mathbf e_{t1}^W,\mathbf e_{t2}^W,\mathbf e_n^W].
\]

policy 输出 normalized action：

\[
\mathbf a=[a_{t1},a_{t2},a_n]^T\in[-1,1]^3.
\]

经物理尺度映射得到 deck-relative velocity reference：

\[
\mathbf v_{rel,ref}^D=[v_{t1},v_{t2},v_n]^T.
\]

其中：

- `v_t1/v_t2`：沿甲板切平面运动；
- `v_n < 0`：沿 `-n` 朝甲板下降；
- `v_n = 0`：匹配 landing contact point 法向速度；
- `v_n > 0`：沿 `+n` 离开甲板，适用于 hold/recover/abort。

转换到 world：

\[
\mathbf v_{rel,ref}^W=R_{WD}\mathbf v_{rel,ref}^D.
\]

最终 UAV world velocity reference：

\[
\boxed{
\mathbf v_{uav,ref}^W
=
\mathbf v_c^W+R_{WD}\mathbf v_{rel,ref}^D
}
\]

随后唯一地执行 ENU→NED：

\[
\boxed{
\mathbf v_{uav,ref}^{NED}=T_{NED\leftarrow ENU}\mathbf v_{uav,ref}^{W}
}
\]

## 4.5 为什么使用 landing contact point 而不是 deck center velocity

倾斜/旋转甲板上，当 \(\boldsymbol\omega_d\neq0\) 且接触点离参考中心不为零时：

\[
\boldsymbol\omega_d\times\mathbf r_c\neq0.
\]

若仅加 `deck center velocity`，policy 即使输出 `v_rel_ref=0`，UAV 也不能真正与目标接触点速度一致。当前 PhysicalDeckAttitude 已在接触指标中使用相同刚体点速度，因此新 action adapter 必须与该定义完全一致。

---

# 5. Observation contract 与 deployability audit

当前 22D observation：

```text
0:3    robot root linear velocity in body frame
3:6    robot root angular velocity in body frame
6:9    projected gravity in body frame
9:12   deck reference position relative to robot, expressed in body frame
12:15  deck surface-point velocity in world - robot root velocity in world
15     align_success
16:19  deck normal expressed in body frame
19:22  deck angular velocity in world - robot angular velocity in world,
       expressed in body frame
```

审计如下：

| observation item | 当前 source | 实机可用 source | deployable? | 主要 noise | 主要 latency | required adapter |
|---|---|---|---|---|---|---|
| body linear velocity | Isaac root GT | PX4 local/vehicle odometry + attitude frame transform | 是 | EKF velocity noise | estimator + bridge | ENU/NED 与 world/body transform |
| body angular velocity | Isaac root GT | gyro / PX4 angular velocity estimate | 是 | gyro noise/bias | 很低但有滤波 | FRD↔FLU |
| projected gravity | Isaac root GT | PX4 attitude estimate / IMU gravity direction | 是 | attitude estimate error | estimator | quaternion/frame adapter |
| deck relative position in body | Isaac exact rigid state | ArUco relative pose / fused deck estimator | 是 | PnP/marker geometry | camera + estimator | camera→body→local transform，time alignment |
| deck surface relative linear velocity | exact GT + `omega×r` | deck position/velocity estimator + deck angular-rate estimator + UAV velocity | 条件可部署 | finite-difference/KF noise，contact-point lever arm error | camera/estimator | rigid contact-point velocity reconstruction |
| `align_success` | simulator task FSM | deployment landing supervisor 状态 | 可替代，不应直接依赖 GT | threshold chattering | supervisor period | 由 deployable measurements 计算；未来可做 obs ablation |
| deck normal in body | exact deck quaternion | ArUco board attitude / plane/deck attitude estimator | 是 | normal bias、marker pose ambiguity | vision | normalized normal + freshness gate |
| deck relative angular velocity in body | exact GT | deck attitude motion estimator - UAV angular velocity | 条件可部署 | differentiation noise、bias | camera history | filtered angular-rate reconstruction |

结论：当前 22D 的**物理量本身大多可测/可估计**，但当前 source 是 simulator GT，因此不能把“22D 数值定义可部署”误写成“当前 observation 已经可直接实机部署”。Hierarchical RL 第一阶段为了隔离 action-interface 影响，可以保留 22D shape；正式 Sim-to-Real 前必须通过 canonical observation adapter 将 simulator GT source 替换为 noisy/delayed/estimated source，并做噪声、延迟、dropout audit。

特别约束：

```text
simulator-only GT
≠
real-world estimator output
```

任何部署代码不得订阅/注入仿真 deck ground truth。

---

# 6. Action contract

## 6.1 Normalized → physical action

policy 输出：

```text
a_norm ∈ [-1, 1]^3
```

第一版 physical bounds 作为**工程初值而非实机系统辨识结果**：

```text
t1 velocity: [-0.80, +0.80] m/s
t2 velocity: [-0.80, +0.80] m/s
normal velocity: [-0.40, +0.30] m/s
```

来源：

- 当前任务安全接触要求 `|v_n| < 0.55 m/s`，hard-contact 阈值 `|v_n| > 0.80 m/s`；首版最大下降 reference 取 `0.40 m/s`，保留 controller tracking/error margin；
- 当前安全切向 contact 要求 `v_t < 0.30 m/s`，但远离接触阶段需要更大的横向追踪能力，因此 horizontal reference bound 允许到 `0.80 m/s`，touchdown 最终仍由 reward/success/safety gate 约束到更低相对速度；
- 这些不是 PX4 或真实机体性能极限，后续必须根据实机飞行 envelope 和 traditional controller limit 校准。

映射采用**以零动作对应零相对速度**的分段线性缩放：

\[
v_i=
\begin{cases}
a_i\,v_{i,max}, & a_i\ge 0,\\
a_i\,|v_{i,min}|, & a_i<0.
\end{cases}
\]

因此 `a=0` 严格对应 `v_rel=0`，不会因为 normal 上升/下降范围非对称而产生隐含下降偏置；`a=-1/+1` 分别对应配置的负/正物理边界。`t1/t2` 对称时退化为普通线性缩放。

## 6.2 Safety clamps

Reference Adapter 必须定义：

```text
1. normalized action clamp to [-1, 1]
2. physical per-axis clamp
3. optional horizontal vector-norm clamp
4. normal descent clamp
5. acceleration / slew-rate clamp between consecutive references
6. finite-value validation
```

第一版建议：

```text
max horizontal relative speed norm = 0.80 m/s
max relative-reference slew       = 2.0 m/s² per axis (engineering initial value)
```

若后续在 touchdown neighborhood 使用更严格 normal descent clamp，应由**deployable clearance/relative-height**触发，例如把下行最小值从 `-0.40` 收紧到 `-0.25 m/s`；该阈值必须进入 cfg 与测试，不能写死在环境代码。

## 6.3 NaN / Inf

- policy action 包含任意 NaN/Inf：不得送入 controller；training 默认 raise/fail-fast 以暴露数值问题；deployment 默认输出 safe hold/abort path，不复用非法历史值无限飞行。
- state/reference 包含 NaN/Inf：controller training backend fail-fast；deployment supervisor 走 stale/invalid observation 规则。

## 6.4 stale observation / stale action

定义两级 freshness：

```text
observation_fresh_timeout
policy_action_fresh_timeout
```

训练 benchmark 可以显式注入 stale behavior；真实 deployment：

- 短时 missed deadline：保持上一个**已限幅** velocity reference，且必须受 stale timeout 限制；
- 超过 timeout：reference 回到 hover/zero-relative-velocity 或进入 abort/recover，由 supervisor 配置决定；
- Offboard heartbeat loss 由 PX4 自身 failsafe 参数继续兜底。

## 6.5 touchdown 附近 `v_n` 的物理意义

`v_n` 始终是**UAV 相对目标 contact point 的 deck-normal velocity reference**，不是 world-z speed。

```text
v_n = 0
→ 理想上 UAV 与目标 contact point 法向速度相同

v_n < 0
→ UAV 沿 deck -normal 接近表面

v_n > 0
→ UAV 沿 deck +normal 离开表面
```

这使得 tilted/heaving/rolling deck 的下降动作不需要 policy 自己学习 ENU z 与 deck normal 的几何关系。

---

# 7. PX4-compatible training backend：VectorizedPx4LikeController

## 7.1 定位

`VectorizedPx4LikeController` 只用于 Isaac Lab 大规模训练/验证：

```text
velocity setpoint
→ desired acceleration
→ desired thrust vector + attitude
→ attitude error
→ body-rate reference
→ rate error
→ body torque
→ Isaac rigid-body wrench
```

它的目标是保留真实飞控的**控制层级与主要饱和机制**，而不是逐行复刻 PX4。正式文档、实验和论文中必须写：

> `PX4-like controller != real PX4`.

## 7.2 Velocity loop

第一版最简实现：

\[
\mathbf e_v^W=\mathbf v_{ref}^W-\mathbf v^W
\]

\[
\mathbf a_{cmd}^W=K_v\mathbf e_v^W
+K_{vd}(\mathbf a_{ref}^W-\mathbf a_{est}^W)
+K_{vi}\int\mathbf e_vdt.
\]

默认：

```text
Kvi = 0
Kvd = 0
```

即首版只启用 P velocity loop，避免在没有可靠 acceleration estimate / anti-windup 理论前引入积分或微分复杂度。cfg 仍保留相应参数入口；只有 smoke/validation 证明需要时再启用。

首先逐轴/向量限制 `a_cmd` 到 `max_acceleration`。

## 7.3 Acceleration → desired thrust direction

ENU world gravity：

\[
\mathbf g^W=[0,0,-g]^T.
\]

实现所需总外力目标：

\[
\mathbf f_{des}^W=m(\mathbf a_{cmd}^W-\mathbf g^W)
=m(\mathbf a_{cmd}^W+g\mathbf e_z).
\]

期望 body +z：

\[
\mathbf b_{3,d}^W=\frac{\mathbf f_{des}^W}{\|\mathbf f_{des}^W\|}.
\]

通过 `max_tilt` 将 \(\mathbf b_{3,d}\) 与 world +z 的夹角限制在允许范围。期望 collective thrust magnitude：

\[
T_{des}=\|\mathbf f_{des}^W\|
\]

并限制在：

```text
min_thrust <= T_des <= max_thrust
```

第一版不建模 per-motor allocation，只输出与当前 Direct environment 相同形式的 body +z total thrust 和 xyz moment，以便把接口变化隔离在新 task 内。

## 7.4 Deterministic yaw 与 desired attitude

第一版 `yaw_ref` 为配置提供的 deterministic world yaw，不属于 RL action。

给定 desired body +z 与 yaw heading，构造正交 desired rotation \(R_{WB,d}\)。当 heading 与 body-z 接近平行导致横叉积退化时，必须有确定的 fallback heading，并通过测试覆盖。

## 7.5 Attitude loop

使用 quaternion / SO(3) shortest-path orientation error，生成 body-rate reference：

\[
\boldsymbol\omega_{ref}^B=K_R\,\mathrm{Log}(R_{BW}R_{WB,d})
\]

并限制：

```text
|omega_ref_i| <= max_body_rate_i
```

项目已经有 `(w,x,y,z)` quaternion multiply/conjugate/axis-angle helper，应复用，不复制另一套 quaternion convention。

## 7.6 Rate loop → moment

第一版按角加速度命令实现：

\[
\boldsymbol\alpha_{cmd}^B
=K_\omega(\boldsymbol\omega_{ref}^B-\boldsymbol\omega^B).
\]

利用 simulator 当前 inertia：

\[
\boldsymbol\tau^B
=I\boldsymbol\alpha_{cmd}^B
+\boldsymbol\omega^B\times(I\boldsymbol\omega^B).
\]

最终：

```text
|max moment_i| <= max_moment_i
```

这样 controller parameter 可随不同仿真机体 mass/inertia 工作，而不是把 Crazyflie inertia hardcode 进算法。

## 7.7 必须进入 cfg 的参数

至少：

```text
velocity_gain
velocity_integral_gain
velocity_derivative_gain
max_acceleration
max_tilt
min_thrust
max_thrust
attitude_gain
max_body_rate
rate_gain
max_moment
```

额外允许：

```text
yaw_ref_enu
controller_frequency
```

参数来源分三类记录：

1. `simulator property`：mass、inertia、gravity；
2. `existing task safety contract`：contact normal/tangential limits；
3. `engineering initial tuning`：Kv、attitude/rate gains、acceleration/tilt/reference slew limit。

第三类必须在 smoke 后回填验证结果，不能伪装成 PX4 默认参数或实机辨识参数。

---

# 8. PX4 deployment backend

未来部署链：

```text
Deployable Observation Adapter
→ same exported 3-D policy
→ same deck-relative ReferenceAdapter
→ ENU velocity reference
→ ENU→NED
→ ROS2 bridge
→ OffboardControlMode.velocity=true
→ TrajectorySetpoint.velocity=[vn,ve,vd]
→ PX4
```

训练包的 `px4_reference_adapter.py` 必须纯 PyTorch / Python 数学实现，不依赖 ROS2 或 `px4_msgs`。

部署层以后新增 ROS2 node，职责仅为：

- 收集 deployable estimator state；
- 构造 canonical observation；
- policy inference；
- 调用相同数学 reference adapter；
- frame conversion；
- freshness/safety/supervisor；
- 发布标准 PX4 messages。

禁止把 ROS executor、DDS、PX4 topic 依赖放入 GPU vectorized training controller。

---

# 9. Training frequency decision

当前：

```text
physics = 100 Hz
Direct policy = 50 Hz (decimation=2)
traditional high-level baseline ≈ 20 Hz
```

比较：

| policy Hz | 优点 | 缺点 | 判断 |
|---:|---|---|---|
| 20 | 与传统 high-level loop 最接近；100Hz/20Hz=5 整数 decimation；部署压力小 | 对快速相对运动 correction 较粗；每步 reference 变化更大 | 可作为 ablation / 部署保守档 |
| **25** | 100Hz/25Hz=4；仍明显低于 Direct 50Hz；比 20Hz 增加 25% 更新带宽；易与 100Hz controller 分层 | 与当前传统 20Hz 不完全相同 | **首版选择** |
| 50 | 最大 responsiveness；与 Direct RL 同频 | 更依赖 inference/estimator latency；不能体现高层 action 的低带宽优势；部署 burden 更高 | 后续 ablation，不默认 |

首版：

```text
physics / low-level surrogate controller = 100 Hz
policy / velocity reference update         = 25 Hz
```

关键实现要求：policy 只在 25 Hz 更新 reference；`VectorizedPx4LikeController` 在每个 100 Hz physics step 用保持的 reference 重新计算 wrench。这比“25 Hz 算一次 wrench 并保持 4 个 physics step”更符合真实 PX4 层级。

选择 25 Hz 不是因为 Direct RL 现有频率，而是因为：

1. 高层 reference 与低层 control 的带宽应分离；
2. 25 Hz 能整除 100 Hz；
3. 比传统 20 Hz 增加一定 moving-deck correction margin；
4. 未来必须用 20/25/50 Hz ablation 验证，而不是把 25 Hz 当普适最优值。

---

# 10. Safety contract

## 10.1 Reference safety

- normalized action clamp；
- horizontal speed clamp；
- normal descent/ascent clamp；
- velocity-reference slew/acceleration clamp；
- finite check；
- stale action timeout；
- stale observation timeout。

## 10.2 Controller safety

- `max_acceleration`；
- `max_tilt`；
- `min_thrust / max_thrust`；
- `max_body_rate`；
- `max_moment`；
- quaternion normalization / degenerate vector handling；
- controller output finite check。

## 10.3 Deployment safety

PX4/ROS deployment 必须显式处理：

```text
policy deadline miss
observation stale
estimator invalid
NaN/Inf
Offboard heartbeat loss
PX4 Offboard rejection/loss
manual takeover
abort / recover
```

建议 supervisor 行为：

```text
short missed deadline
→ hold last bounded reference

stale beyond configured timeout
→ zero-relative-velocity / safe climb depending current clearance

estimator invalid near deck
→ stop descent and recover/abort

PX4 Offboard loss
→ obey PX4 configured failsafe; external node must not fight mode transition

manual takeover
→ stop producing authoritative motion command and record transition
```

真实机上的具体 abort altitude、climb speed 和 mode transition 必须由 HIL/SITL 先验证，本理论文档不虚构实机安全值。

---

# 11. Sim-to-Real assumptions 与剩余 gap

## 11.1 本架构降低的 gap

与当前 Direct RL 相比，velocity-reference hierarchy 主要降低：

- **controller implementation gap**：deployment 直接复用 PX4 velocity/attitude/rate controllers；
- **actuator allocation gap**：deployment 复用 PX4 control allocation；
- **motor command semantics gap**：policy 不直接输出 motor/wrench；
- **action-interface gap**：policy 输出 m/s reference，可在 simulator 和 PX4 使用同一物理语义；
- **deployment API gap**：目标直接映射到标准 Offboard `TrajectorySetpoint.velocity`。

## 11.2 仍然存在的 gap

仍存在且必须后续验证/随机化：

```text
mass mismatch
inertia mismatch
thrust model / thrust coefficient mismatch
battery voltage / motor response
actuator delay
velocity controller surrogate vs real PX4 mismatch
state-estimator noise/bias
vision noise / latency / dropout
ROS2 / DDS timing
wind / deck airwake
contact friction / restitution / landing-gear compliance
deck state estimation error
reference frame calibration error
```

因此禁止使用“接 PX4 后 Sim2Real gap 消失”的表述。

---

# 12. Formal comparison protocol

保留：

```text
M0 Frozen Direct PPO teacher
M1 Actor-preserving Direct PPO
```

新增：

```text
M2 PX4-Compatible Hierarchical RL
```

M2 不继承 M0/M1 action checkpoint。若未来使用 Direct teacher，只允许独立实验：

```text
teacher rollout
→ state/trajectory dataset
→ reconstruct velocity-reference teacher
→ behavior cloning
```

正式比较：

| Dimension | M0/M1 Direct | M2 Hierarchical |
|---|---|---|
| nominal settled landing | 必报 | 必报 |
| Sea-State robustness | 必报 | 必报 |
| dynamics mismatch | 必报 | 必报 |
| observation noise | 必报 | 必报 |
| latency | 必报 | 必报 |
| hard contact | 必报 | 必报 |
| ground crash | 必报 | 必报 |
| touchdown normal/tangential velocity | 必报 | 必报 |
| action/reference smoothness | Direct action Δ | velocity reference Δ / accel |
| saturation | thrust/moment | ref + controller limits |
| inference latency | 必报 | 必报 |
| controller runtime | N/A/Direct apply | surrogate runtime 必报 |
| PX4 integration effort | 高 | 低/直接 Offboard velocity |

公平性原则：

- success/contact/failure taxonomy 复用 PhysicalDeckAttitude contract；
- static/XY/heave/attitude/Sea-State scenario distribution 尽量一致；
- 不因 M2 action 改变而修改旧 M0/M1 benchmark；
- 如 reward 必须因 action smoothness 语义调整，只新增 M2 专属项并在实验表中披露，不能改 success contract。

---

# 13. Failure modes

| Failure mode | 机理 | 可观测证据 | 缓解/实验 |
|---|---|---|---|
| velocity controller lag | reference 变化快于低层跟踪 | velocity error P95、touchdown relative velocity | frequency/gain/slew ablation |
| excessive deck acceleration | `v_contact` 高频变化，reference acceleration 过大 | ref slew saturation、deck angular/heave rate | acceleration clamp、prediction 后续研究 |
| normal-vector error | descent direction偏离真实 deck normal | body/deck angle、touchdown tangential speed | noise/bias injection |
| deck angular-rate error | `omega×r` 补偿错误 | reconstructed contact velocity error | estimator latency/noise sweep |
| PX4 saturation | tilt/thrust/rate/moment 达限 | saturation ratio | envelope calibration |
| policy latency | reference stale | missed deadlines、age | 20/25Hz budget、target compute benchmark |
| velocity clipping | policy 常在边界 | ref saturation ratio | action range/reward/normalization review |
| touchdown overshoot | descent reference + controller lag | normal impact speed、hard contact | near-touchdown clamp / timing policy |
| controller surrogate mismatch | M2 在 surrogate 稳定但 SITL 不稳定 | same policy surrogate vs SITL delta | few-env PX4 SITL validation |
| frame conversion error | ENU/NED 或 deck frame 符号错误 | deterministic transform tests | centralized adapter only |
| stale estimator | deck reference继续运动但 observation冻结 | freshness counters | hold/recover/abort |

---

# 14. Implementation architecture

## 14.1 New files

按当前 package 结构，优先最少新增：

```text
source/quadcopter_waypoint/quadcopter_waypoint/utils/px4_reference_adapter.py
source/quadcopter_waypoint/quadcopter_waypoint/utils/vectorized_px4_like_controller.py

source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/
  quadrotor_ship_landing_px4_hierarchical/
    __init__.py
    quadrotor_ship_landing_px4_hierarchical_env.py
    agents/rl_games_ppo_cfg.yaml
    agents/__init__.py
    README.md

tests/test_px4_reference_adapter.py
tests/test_vectorized_px4_like_controller.py
tests/test_px4_hierarchical_task_contract.py
```

如果新 task 能通过继承 `PhysicalDeckAttitude` 且只覆盖 action/controller path 保持 reward/contact/observation 逻辑，则优先继承，禁止复制整个环境文件。

## 14.2 Modified files

理论 gate PASS 后预计仅修改：

```text
README.md
docs/README.md
docs/RL_LONG_TERM_ROADMAP.md
docs/px4_compatible_hierarchical_rl_theory.md
```

以及为 task auto-registration 所需的最小 `__init__.py`（若项目现有 import 机制要求）。冻结 Direct task 源文件原则上不修改；若 registration 只需在父级 package 导入新 task，则只做该最小增量。

## 14.3 Class responsibilities

### `Px4ReferenceAdapterConfig` / pure functions

负责：

```text
normalized action → deck-relative physical velocity
deck contact-point rigid velocity
deck frame → world velocity
world ENU → PX4 local NED
velocity-reference slew limiting
finite/saturation checks
```

不负责：RL network、Isaac scene、ROS2、PX4 messages。

### `VectorizedPx4LikeController`

负责：

```text
world velocity reference + robot state + mass/inertia
→ body thrust + moment
```

纯 torch、batch-first、GPU vectorized、无 per-env Python loop。

### `QuadcopterShipLandingPx4HierarchicalEnv`

负责：

```text
inherit frozen PhysicalDeckAttitude observation/reward/contact
3D policy action preprocessing at 25 Hz
compute deck target contact-point velocity
hold velocity reference
call low-level controller at every 100 Hz physics step
apply resulting thrust/moment only inside new task
log saturation/runtime/controller metrics
```

### future ROS2 deployment node

本阶段不实现。未来负责：

```text
estimator → canonical obs → exported policy → ReferenceAdapter
→ OffboardControlMode + TrajectorySetpoint
```

## 14.4 Data flow

Training：

```text
PhysicalDeckAttitude-compatible observation
→ 3D policy action at 25 Hz
→ normalized_action_to_relative_velocity
→ target deck contact point + rigid_surface_point_velocity
→ deck-relative-to-world reference
→ held world velocity reference
→ VectorizedPx4LikeController at 100 Hz
→ thrust/moment
→ Isaac dynamics
→ unchanged contact/reward/success logic
```

Deployment：

```text
real estimator state
→ canonical 22D-compatible observation adapter
→ same exported 3D policy
→ same ReferenceAdapter math
→ world ENU velocity ref
→ NED velocity ref
→ ROS2 PX4 Offboard velocity
→ real PX4 control stack
```

## 14.5 Configuration

所有 action/controller 参数必须在 new task cfg：

```text
relative_velocity_min/max
max_horizontal_relative_speed
max_reference_acceleration
policy frequency / decimation
velocity gains
max acceleration
max tilt
thrust min/max
attitude gains
max body rate
rate gains
max moment
yaw reference
```

质量、惯量和重力优先从 simulator 当前 robot property 读取，不 hardcode。

## 14.6 Unit tests

Reference Adapter 必须覆盖：

```text
stationary level deck
constant translating deck
heaving deck
roll-rate deck
pitch-rate deck
combined angular + translation
zero relative velocity
pure normal descent
pure tangential correction
action saturation
NaN/Inf rejection
ENU ↔ NED
deck-frame transform
rigid contact velocity parity with physical_deck_attitude_math
```

Controller 必须覆盖：

```text
hover equilibrium finite/stable command
horizontal velocity error tilts toward correction
vertical climb/descent changes thrust
max acceleration clamp
max tilt clamp
thrust clamp
body-rate clamp
moment clamp
batch shape/device/dtype
NaN/Inf rejection
no Python per-env loop in algorithm path
```

Task contract 必须验证：

```text
new action_space = 3
new policy frequency = 25 Hz
frozen Direct task action_space remains 4
observation remains 22 in first action-interface-isolation experiment
old success/contact thresholds unchanged by inheritance
```

## 14.7 Smoke benchmark

不直接启动正式 PPO。第一轮 deterministic/controller smoke：

```text
static deck
constant XY deck
heave deck
physical-deck-attitude
```

先测试 scripted velocity references / deterministic actions：

```text
hover
horizontal tracking
vertical descent
moving-deck velocity feedforward
tilted-deck normal descent
contact
```

必须记录：

```text
settled_landing
ground_crash
hard_contact
timeout
relative velocity at touchdown
normal impact velocity
tangential velocity
action saturation ratio
velocity reference saturation
max tilt
max body rate
max moment
policy/controller frequency
controller runtime
```

smoke gate：

```text
NaN/Inf = 0
controller explosion = 0
ground crash = 0 for basic deterministic smoke
velocity saturation not continuously active
basic hover stable
basic moving-deck tracking stable
```

只有 smoke PASS 后才允许进入 PPO 调参。

## 14.8 SITL validation plan

训练后不直接实机。先 single/few-env：

```text
exported policy
→ deployable observation source
→ same ReferenceAdapter
→ ROS2
→ PX4 SITL
```

比较 surrogate vs SITL：

```text
velocity tracking error
attitude/rate saturation
touchdown kinematics
failure taxonomy
reference age / deadline
```

如果 surrogate smoke PASS 但 SITL 明显失败，优先诊断 controller surrogate mismatch / frame / timing，而不是立即重训大规模 PPO。

---

# 15. Theory gate self-check

- [x] 问题定义完整：明确冻结 Direct baseline、新独立 M2、非目标和部署目标。
- [x] 坐标系完整：W/ENU、B/FLU、D、PX4 NED/FRD、quaternion convention 已定义。
- [x] 单位完整：m、m/s、m/s²、rad/s、N、N·m 已定义。
- [x] action 定义完整：3D deck-relative velocity、normal sign、bounds、scaling、slew/stale/finite contract 已定义。
- [x] PX4 interface 完整：velocity Offboard flags、TrajectorySetpoint NaN/velocity/yaw 规则已定义。
- [x] contact-point velocity 推导完整：`v_contact=v_deck+omega×r`，并明确复用现有 rigid-body math。
- [x] ENU/NED 转换完整：向量与 yaw 规则已给出并要求集中实现。
- [x] observation deployability 审计完成：22D 每项 current/real source/noise/latency/adapter 已列出。
- [x] safety boundary 完整：reference/controller/deployment 三层 safety contract 已定义。
- [x] training backend 与 deployment backend 区分清楚：surrogate 不等于 PX4，训练无 ROS2/SITL dependency。
- [x] 参数来源明确：simulator property / frozen safety contract / engineering initial tuning 三类已区分。
- [x] test plan 明确：数学、controller、task contract、SITL parity 测试已列出。
- [x] benchmark 明确：static/XY/heave/attitude smoke gate 与正式 M0/M1/M2 matrix 已定义。

**Gate result: PASS.**

因此后续实现允许开始，但必须遵循以下顺序：

```text
1. 更新长期路线图，保留历史 Sea-State 与 Direct RL 结果
2. 实现纯数学 Reference Adapter
3. Reference Adapter unit tests 全 PASS
4. 实现 VectorizedPx4LikeController
5. controller unit tests 全 PASS
6. 创建独立 3D-action task
7. regression tests
8. minimal Isaac Lab smoke
9. smoke PASS 后才讨论 PPO training
```

---

# 16. 当前理论决策摘要

```text
为什么不用 thrust/torque PX4 mode：
因为它仍绕过 PX4 position/velocity/attitude/rate 控制，主要 controller gap 仍在。

为什么选择 velocity reference：
它在 moving-deck responsiveness、PX4 low-level reuse、action 可解释性、传统 baseline 可比性和 GPU 训练成本之间最均衡。

为什么选择 deck-relative action：
policy 直接表达“相对甲板怎么运动”，并通过 v_contact + R_WD v_rel 自动包含平移、heave、roll/pitch angular velocity 对接触点速度的影响。

为什么训练不用 full PX4 SITL：
大规模环境需要 GPU vectorization；per-env SITL 会破坏训练吞吐和复杂度。训练用最小 PX4-like surrogate，SITL/HIL/实机只做后续少量高保真验证。
```

---

# 17. M2 PPO evidence gate

2026-08-23 在 action-interface baseline `ca974ee5118f8742af69a698a1c47a96aa7d0a9f` 之后，M2 正式进入 PPO 前证据门禁。顺序固定为：

```text
regression
→ evaluator completeness
→ deterministic zero-relative-action baseline
→ 64-env / seed-42 / 30-iteration PPO sanity
→ diagnosis
→ only-if-pass small candidate training
```

## 17.1 Episode diagnostics contract

M2 环境必须在自动 reset 之前累计并冻结：

```text
relative_velocity_reference_norm mean/P95/max
reference_saturation_ratio
controller_velocity_tracking_error mean/max
controller_acceleration_saturation_ratio
controller_tilt_saturation_ratio
controller_thrust_saturation_ratio
controller_body_rate_saturation_ratio
controller_moment_saturation_ratio
max_desired_tilt
max_body_rate
max_moment
controller_runtime mean/P95/max
normalized action mean/std/abs-max per axis
```

`eval_metrics.py` 只有在 task 暴露完整 optional M2 latch contract 时才追加这些字段；M0/M1 不进入该分支，因此历史 CSV 字段和成功定义保持不变。正式 evaluator 可开启同步 CUDA wall-time 测量；PPO training 默认不执行每个 100 Hz substep 的 CUDA synchronize，避免 diagnostics 人为串行化训练。

## 17.2 Zero-relative-action baseline

定义：

```text
normalized action = [0, 0, 0]
→ v_rel_ref^D = 0
→ v_uav_ref^W = v_contact^W
```

因此它是 **deck contact-point velocity following baseline**，不是零推力，也不是 RL 方法。

16-env deterministic baseline 在四个场景：

```text
static deck
constant XY deck
heave deck
physical-deck-attitude
```

均得到：

```text
timeout rate = 1.0
contact rate = 0
settled landing rate = 0
hard contact rate = 0
ground crash rate = 0
deck miss rate = 0
relative velocity reference norm = 0
reference saturation = 0
all controller saturation ratios = 0
NaN/Inf = 0
```

其中 controller velocity tracking error mean 分别约为 `0.00394 / 0.00394 / 0.01077 / 0.00898 m/s`。该结果证明：零相对速度 reference 能稳定跟随甲板运动，但不会主动完成 normal descent；后续 policy 的 horizontal correction、下降与 touchdown timing 必须由 PPO 学得。

证据：

```text
benchmarks/px4_hierarchical_training/zero_action_16env.json
```

## 17.3 PPO sanity pass/fail rule

30-iteration sanity 不要求达到 95% settled landing，只要求证明 policy is learning。至少检查：

```text
NaN/Inf = 0
controller no explosion
reference/controller saturation not persistently near 100%
reward trend or landing intermediate metrics clearly improve
ground crash does not keep worsening
```

若不满足，必须按 reference → controller tracking → action scaling → reward gradient 的顺序诊断，并停止扩大训练迭代数。

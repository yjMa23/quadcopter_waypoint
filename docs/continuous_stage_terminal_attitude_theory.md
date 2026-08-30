# Continuous Landing Stage + Terminal Attitude Guidance Theory Gate

> 项目：`/home/j/Isaac_RL_Projects/quadcopter_waypoint`
> 日期：2026-08-30
> 状态：第一创新点 S1 Theory Gate。
> 目的：把 Continuous-Stage + Terminal-Attitude 从概念收敛为下一阶段可直接编码的纯数学 contract。
> 重要边界：本文件不证明新方法已训练成功，不证明真实 PX4 已支持 Route A，不进入 SITL/HIL/实机。

---

## 0. Scope and frozen evidence

当前冻结 Fixed-Stage M2 已完成：

```text
3-D deck-relative velocity action               = implemented
PX4-compatible Reference Adapter                = implemented
contact-point rigid-body compensation           = implemented
Vectorized PX4-like controller                  = implemented
1-env / 16-env smoke                            = PASS
zero-relative-action baseline                   = PASS
M2 evaluator / terminal diagnostics             = implemented
D1 full regression                              = 128 passed + 21 subtests
```

D1 ep30 固定评估：

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

D1 的主要剩余失败信号：

```text
high within-bound action/reference variation
post-latch horizontal drift
repeated recovery-gate crossings
controller tracking degradation
descent completion failure
```

因此本 Theory Gate 不再设计新的 hard landing window，而定义一个连续 stage state，使 phase、reference envelope 与 terminal attitude guidance 在数学上连续。

---

# 1. Frames, units and conventions

## 1.1 Frames

沿用已冻结项目 contract：

- \(\mathcal F_W\)：训练/数学 world，显式按 ENU-compatible 使用，`+x East, +y North, +z Up`。
- \(\mathcal F_B\)：UAV body，FLU-compatible，`+x forward, +y left, +z up`。
- \(\mathcal F_D\)：deck frame，`+x=t1, +y=t2, +z=n`；`+n` 指向甲板上表面外法向。
- \(\mathcal F_N\)：PX4 local NED，`+x North, +y East, +z Down`。

\(R_{WB}\) 表示 body→world，\(R_{WD}\) 表示 deck→world。项目 quaternion 固定为 `(w,x,y,z)`，表示 local→parent/world 主动旋转。

## 1.2 Units

```text
position       m
linear velocity m/s
acceleration   m/s^2
angle          rad internally; deg only for readable cfg/docs
angular rate   rad/s
force          N
moment         N·m
stage          dimensionless [0,1]
normalized action dimensionless [-1,1]
```

## 1.3 Relative-velocity sign convention

Deck-frame policy velocity：

\[
\mathbf v_{rel}^D=[v_{t1},v_{t2},v_n]^T.
\]

```text
v_n < 0 : approach deck along -normal
v_n = 0 : match target contact-point normal velocity
v_n > 0 : move away/recover along +normal
```

禁止把 `v_n` 写成 world-z descent speed。

---

# 2. Policy state/action contract

## 2.1 Action

新 policy 输出：

\[
\mathbf a_t=
[a_{t1},a_{t2},a_n,a_s]^T\in[-1,1]^4.
\]

前三维决定 stage-conditioned deck-relative velocity，第四维只决定内部 continuous stage。

不得从该 action 直接生成：

```text
roll/pitch/yaw
collective thrust
body torque
motor command
```

## 2.2 Continuous stage raw mapping

定义：

\[
s_{raw,t}=\operatorname{clip}\left(\frac{a_{s,t}+1}{2},0,1\right).
\]

因此：

```text
a_s=-1 → s_raw=0
a_s= 0 → s_raw=0.5
a_s=+1 → s_raw=1
```

## 2.3 Stage state and Markov contract

Stage 不是纯瞬时 action，而是有过滤状态的 reference-planner state。新 observation 第 15 维必须提供前一控制步的 filtered stage：

\[
o_{15,t}=s_{t-1}.
\]

这样 policy 可以感知 filter memory，避免 simulator 内存在未观测隐藏状态。

---

# 3. Stage filtering and rate limiting

Stage filter 必须同时满足：

```text
bounded [0,1]
continuous
no hard threshold latch
finite
reset deterministic
rate limited
```

## 3.1 Low-pass target

Policy period沿用第一版 25 Hz：

\[
\Delta t_p=0.04\;s.
\]

工程初值：

```text
stage_filter_time_constant τ_s = 0.20 s
stage_rate_limit r_s           = 2.0 1/s
```

离散 low-pass 系数：

\[
\beta_s=1-\exp(-\Delta t_p/\tau_s).
\]

Low-pass target：

\[
\tilde s_t=s_{t-1}+\beta_s(s_{raw,t}-s_{t-1}).
\]

## 3.2 Explicit stage-rate limit

\[
\Delta s_t=
\operatorname{clip}
(\tilde s_t-s_{t-1},-r_s\Delta t_p,+r_s\Delta t_p).
\]

最终：

\[
\boxed{s_t=\operatorname{clip}(s_{t-1}+\Delta s_t,0,1)}.
\]

重置：

```text
s_prev = 0
```

代表新 episode 从 approach/recovery state 开始，不继承上一 episode landing commitment。

工程初值可在未来 sanity 中校准，但上述数学形式不再 TBD。

---

# 4. Smooth stage shaping function

多处 envelope 共用 cubic smoothstep：

\[
C(s)=3s^2-2s^3,\qquad s\in[0,1].
\]

性质：

\[
C(0)=0,\;C(1)=1,\;C'(0)=C'(1)=0.
\]

使用同一个 `smoothstep01()` pure helper，禁止在 reward/reference/attitude 中复制略有不同的 stage curve。

---

# 5. Stage-conditioned velocity envelope

目标是同时解决：

```text
approach needs responsiveness
terminal needs precision
D1 high-variation reference degrades tracking
```

所有以下数值都是第一版 engineering initial values，不是 PX4 默认或真实机系统辨识值。

## 5.1 Tangential envelope

定义最大 tangential relative speed：

\[
V_t(s)=V_{t,app}-(V_{t,app}-V_{t,term})C(s)
\]

第一版：

```text
V_t_app  = 0.80 m/s
V_t_term = 0.25 m/s
```

因此：

```text
s=0 → 0.80 m/s
s=1 → 0.25 m/s
```

`0.25 m/s` 低于冻结 physical success 的 `safe_contact_tangential_speed=0.30 m/s`，为 terminal tracking error 留出 margin。

前三维 normalized action 的 tangential provisional command：

\[
\hat v_{t1}=a_{t1}V_t(s),\qquad
\hat v_{t2}=a_{t2}V_t(s).
\]

再执行向量 norm clamp：

\[
\mathbf v_t=
\hat{\mathbf v}_t
\min\left(1,\frac{V_t(s)}{\|\hat{\mathbf v}_t\|+\epsilon}\right).
\]

这样 diagonal action 不会超过 stage-conditioned tangential envelope。

## 5.2 Normal descent envelope

Continuous stage 本身决定“允许多强下降”，不再使用 hard `can_land`。

定义最大下降速率幅值：

\[
V_{down}(s)=V_{down,max}C(s)
\]

第一版：

```text
V_down_max = 0.30 m/s
```

因此：

```text
s=0 → no commanded downward relative velocity
s=1 → up to -0.30 m/s
```

Recovery/ascent authority 保留，但随着 commitment 增强适度收紧：

\[
V_{up}(s)=V_{up,app}-(V_{up,app}-V_{up,term})C(s)
\]

第一版：

```text
V_up_app  = 0.30 m/s
V_up_term = 0.15 m/s
```

Normal normalized mapping 采用 zero-preserving asymmetric rule：

\[
v_n=
\begin{cases}
a_nV_{up}(s), & a_n\ge0,\\
a_nV_{down}(s), & a_n<0.
\end{cases}
\]

因为 \(a_n<0\)，第二式自然为负下降速度。

这意味着 policy 不能在 `s≈0` 时通过 `a_n=-1` 绕过 stage 直接下降；但仍可通过正 `a_n` 执行 recovery。

---

# 6. Stage-conditioned reference slew limits

D1 ep30 几乎无 reference bound saturation，但 action std 明显升高并伴随 tracking mean `0.4425 m/s`。因此新方法不仅限制速度幅值，还要让 terminal stage 的 reference acceleration 更保守。

定义每轴最大相对 reference acceleration：

\[
\mathbf A_{ref}(s)
=
\mathbf A_{app}
-(\mathbf A_{app}-\mathbf A_{term})C(s).
\]

第一版：

```text
A_app  = [2.0, 2.0, 1.5] m/s^2
A_term = [0.8, 0.8, 0.5] m/s^2
```

对未限 slew 的 stage-conditioned target \(\hat{\mathbf v}_{rel,t}^D\)：

\[
\Delta\mathbf v_{max}=\mathbf A_{ref}(s_t)\Delta t_p
\]

\[
\boxed{
\mathbf v_{rel,t}^D
=
\mathbf v_{rel,t-1}^D
+
\operatorname{clip}
(\hat{\mathbf v}_{rel,t}^D-\mathbf v_{rel,t-1}^D,
-\Delta\mathbf v_{max},+\Delta\mathbf v_{max})
}
\]

注意：只对 policy-relative component 做 slew limit。已知 deck rigid-body feedforward：

\[
\mathbf v_c^W
=
\mathbf v_D^W+\omega_D^W\times r_c^W
\]

不进入该 relative-action slew limiter，否则会人为延迟真实 target contact-point 速度补偿。

---

# 7. Contact-point rigid-body kinematics

设 deck reference center：

\[
\mathbf p_D^W,\;\mathbf v_D^W,\;\boldsymbol\omega_D^W.
\]

Landing target contact point：

\[
\mathbf p_c^W=\mathbf p_D^W+R_{WD}\mathbf r_c^D.
\]

其刚体速度：

\[
\boxed{
\mathbf v_c^W
=
\mathbf v_D^W
+
\boldsymbol\omega_D^W\times
(\mathbf p_c^W-\mathbf p_D^W)
}
\]

最终 world/ENU velocity reference：

\[
\boxed{
\mathbf v_{uav,ref}^W
=
\mathbf v_c^W+R_{WD}\mathbf v_{rel,ref}^D
}
\]

部署 NED：

\[
[v_N,v_E,v_D]^T=[v_y^W,v_x^W,-v_z^W]^T.
\]

实现约束：继续调用现有 `px4_reference_adapter.py` 与 `rigid_surface_point_velocity()`；Continuous-Stage utility 不允许复制 contact-point kinematics。

---

# 8. Surface clearance definition

Terminal attitude guidance 使用 UAV landing surface 与 deck top plane 的 signed clearance：

\[
h_c>0: \text{above surface},\qquad
h_c=0: \text{on surface},\qquad
h_c<0: \text{penetration}.
\]

必须复用当前 `signed_deck_surface_clearance()` 的 deck-frame几何定义，而不是 world-z 高差。

用于 terminal attitude blending 的 clearance 先裁剪：

\[
h=\max(h_c,0).
\]

---

# 9. Deterministic yaw guidance

Yaw 不由 RL 学习。

定义 deck local +x 轴在 world 的 heading vector：

\[
\mathbf d_x^W=R_{WD}[1,0,0]^T.
\]

投影到世界水平面：

\[
\mathbf h_D^W=
\operatorname{normalize}([d_{x,x}^W,d_{x,y}^W,0]^T).
\]

当水平投影 norm 小于 `1e-6`（理论退化保护）时，使用上一有效 heading；episode 首次退化时使用 world +x `[1,0,0]`。

ENU deck heading：

\[
\psi_D^{ENU}=\operatorname{atan2}(h_{D,y}^W,h_{D,x}^W).
\]

对应 PX4 NED yaw：

\[
\psi_D^{NED}=wrap(\pi/2-\psi_D^{ENU}).
\]

在当前 yaw=0 的历史任务中退化为 deterministic zero heading；未来 yaw benchmark 则随 deck heading 连续变化。

---

# 10. Normal velocity-control attitude R_vel

`R_vel` 表示正常 velocity controller 在不做 deck-attitude matching 时的期望飞行姿态。

训练 surrogate 当前由：

```text
velocity error
→ acceleration command
→ gravity compensation
→ desired thrust direction
+ deterministic yaw
→ desired attitude
```

生成。

新方法的第一版 `R_vel` 仍使用同一物理逻辑，但 deterministic yaw 从固定 world yaw 升级为 `deck heading`。

如果 desired specific force：

\[
\mathbf f_{sp}^W=\mathbf a_{cmd}^W+g\mathbf e_z,
\]

则正常 flight body-z：

\[
\mathbf b_{3,vel}^W=\frac{\mathbf f_{sp}^W}{\|\mathbf f_{sp}^W\|}.
\]

施加当前 `max_tilt=35°` 可行性限制后，结合 deck heading 构造 \(R_{vel}\)。

---

# 11. Deck attitude R_deck

\[
R_{deck}=R_{WD}.
\]

其 body/deck-aligned +z 即 deck normal：

\[
\mathbf b_{3,deck}^W=R_{WD}\mathbf e_3=\mathbf n_D^W.
\]

Terminal target 同时希望：

```text
body z → deck normal
body heading → deck heading
```

但由于多旋翼 underactuation，这一要求只在 near-contact 逐渐增强。

---

# 12. Terminal attitude alignment weight

## 12.1 Clearance weight

第一版工程初值：

```text
attitude_blend_start_clearance h_start = 0.50 m
attitude_blend_full_clearance  h_full  = 0.12 m
```

要求：

```text
h_start > h_full >= 0
```

定义 proximity：

\[
p_h=
\operatorname{clip}
\left(
\frac{h_{start}-h}{h_{start}-h_{full}},0,1
\right).
\]

clearance weight：

\[
w_h=C(p_h).
\]

## 12.2 Stage weight

\[
w_s=C(s).
\]

## 12.3 Combined terminal weight

\[
\boxed{\alpha=w_s w_h\in[0,1]}.
\]

因此：

```text
far from deck → w_h≈0 → alpha≈0 regardless of stage
near deck but low commitment → w_s≈0 → alpha≈0
near deck + high commitment → alpha→1
```

这避免 policy 仅通过把 stage 拉高，就在远离甲板时强迫飞行器倾斜贴合甲板。

---

# 13. SO(3)/quaternion attitude blending law

第一版采用 shortest-path quaternion SLERP，因为：

```text
continuous
shortest-path sign handling explicit
full roll/pitch/yaw alignment expressible
batch-friendly pure math
```

设 normalized quaternions：

\[
q_{vel},q_{deck}.
\]

先处理双覆盖：

\[
\text{if }q_{vel}^Tq_{deck}<0,
\quad q_{deck}\leftarrow-q_{deck}.
\]

定义 \(d=\operatorname{clip}(q_{vel}^Tq_{deck},-1,1)\)，\(\theta=\arccos(d)\)。

若 \(\theta>10^{-5}\)：

\[
q_{blend}
=
\frac{\sin((1-\alpha)\theta)}{\sin\theta}q_{vel}
+
\frac{\sin(\alpha\theta)}{\sin\theta}q_{deck}.
\]

若 \(\theta\le10^{-5}\)，使用 normalized lerp：

\[
q_{blend}=normalize((1-\alpha)q_{vel}+\alpha q_{deck}).
\]

最后统一 normalize。

---

# 14. Thrust-direction feasibility and tilt constraint

不能只做 SLERP 后直接要求任何姿态。必须检查 blended body-z 与 world +z 的 tilt：

\[
\theta_{tilt}=\arccos(\mathbf b_3^W\cdot\mathbf e_z).
\]

第一版：

```text
max_terminal_attitude_tilt = 35 deg
```

与当前 training surrogate `controller_max_tilt_deg=35` 一致。

若 `q_blend` 对应 tilt 超过限制：

1. 保留 deck heading；
2. 将 blended body-z 沿 world-up 与原 body-z 之间的最短球面方向投影到 `35°` cone 边界；
3. 用 projected body-z + deck heading 重构正交 `R_ref`；
4. 输出 `attitude_tilt_saturated=True` diagnostic。

当前 deck safety envelope 仅 ±8° roll/pitch，正常情况下 pure deck alignment 自身远低于 35°；该 clamp 主要用于高横向 acceleration 与 terminal blending 冲突时保证可行性。

---

# 15. Velocity-tracking conflict contract

Terminal attitude alignment 改变 thrust direction，可能与 velocity loop 当前所需 \(R_{vel}\) 冲突。

必须记录：

\[
\theta_{conflict}
=
\arccos(
\mathbf b_{3,vel}^W\cdot\mathbf b_{3,ref}^W
).
\]

第一版 diagnostic gate：

```text
terminal_attitude_conflict_angle
terminal_attitude_tilt_saturated
```

Theory 约束：

- 不在远离 deck 时增大该 conflict；
- `alpha` 仅 near-contact 增强；
- reference slew 同时随 stage 收紧，减少横向 acceleration demand；
- 如果未来实验证明 frequent tilt saturation/conflict 导致 velocity tracking 崩溃，应先调 terminal blend envelope/Route，而不是让 RL 直接输出 attitude。

---

# 16. Attitude reference rate limiting

为了避免 q_ref 在 deck motion、stage 或 clearance 改变时产生不可跟踪突变，姿态 reference 本身必须 rate limited。

定义相邻 reference shortest relative quaternion：

\[
q_\Delta=q_{ref,t}\otimes q_{ref,t-1}^{-1}.
\]

转换 shortest axis-angle \(\boldsymbol\phi\)，则 implied reference angular rate：

\[
\boldsymbol\omega_{ref}=\boldsymbol\phi/\Delta t_p.
\]

第一版 rate limit：

```text
max_terminal_reference_rate = [2.0, 2.0, 1.5] rad/s
```

若任一轴超过限值，对 axis-angle increment 按每轴 clamp 后重新 Exp-map 得到 limited q_ref。

说明：该值是 reference-planner engineering initial limit；低于当前 surrogate body-rate hard bounds `(6,6,4) rad/s`，为 tracking 留出 margin。

实现必须做 shortest-path quaternion sign handling。

---

# 17. Relative angular velocity

旋转甲板上正确 touchdown rotational metric 是 UAV 与 deck 的相对角速度，而不是要求 UAV absolute angular velocity → 0。

World frame：

\[
\boxed{
\boldsymbol\omega_{rel}^W
=
\boldsymbol\omega_{uav}^W-
\boldsymbol\omega_{deck}^W
}
\]

Deck frame：

\[
\boldsymbol\omega_{rel}^D
=R_{DW}\boldsymbol\omega_{rel}^W.
\]

Body frame：

\[
\boldsymbol\omega_{rel}^B
=R_{BW}\boldsymbol\omega_{rel}^W.
\]

Norm 在旋转下 frame invariant：

\[
\|\omega_{rel}^W\|
=\|\omega_{rel}^D\|
=\|\omega_{rel}^B\|.
\]

但逐轴阈值比较必须先统一 frame。

第一版 touchdown scalar threshold：

```text
safe_contact_relative_ang_vel = 1.50 rad/s
```

它暂沿用旧 physical task absolute body-rate 数值作为 backward-comparable engineering initial threshold，但语义改为 relative angular speed。未来可通过 motion benchmark 校准，不得写成真实机认证安全值。

---

# 18. Landing decision vs success/safety contract

## 18.1 Learned landing decision

```text
continuous filtered stage s
```

决定 reference envelope 和 terminal guidance 强度。

它**不直接决定 landing success**，也不允许把 `s>某阈值` 当成物理成功。

## 18.2 Deterministic physical success contract for new task

新 Continuous-Stage task 的 safe contact 第一版定义：

```text
deck_contact = true
ground_contact = false
inside_effective_deck = true
horizontal_error < 0.12 m
hard_contact = false
|normal_relative_speed| < 0.55 m/s
tangential_relative_speed < 0.30 m/s
body_deck_normal_angle < 12 deg
||omega_rel|| < 1.50 rad/s
world_upright > 0.90
penetration <= 0.025 m
```

`settled_landing`：safe contact 连续 3 个 policy steps，并保留首次接触精度要求。

与旧 Fixed-Stage baseline 的差异必须显式记录：

```text
old rotational success metric = absolute UAV angular velocity
new rotational success metric = UAV-deck relative angular velocity
```

因此新 task 要使用独立 ID/benchmark，不能回写历史 M2 settled rate。

## 18.3 Hard-contact/failure contract

第一版保持已验证 physical values：

```text
contact force > 2.50 N
or impulse > 0.025 N·s
or |normal relative speed| > 0.80 m/s
or penetration > 0.030 m
→ hard contact
```

并继续保留：

```text
deck miss
ground crash
workspace crash
timeout
```

Stage 不参与这些 deterministic failure predicates。

---

# 19. Observation contract

新 independent task 保持 22D：

```text
0:3    UAV root linear velocity in body frame
3:6    UAV root angular velocity in body frame
6:9    projected gravity in body frame
9:12   deck reference position relative to UAV, body frame
12:15  deck surface-point velocity world - UAV root velocity world
15     previous filtered landing stage s_{t-1}
16:19  deck normal in body frame
19:22  deck angular velocity world - UAV angular velocity world, expressed in body frame
```

说明：19:22 保持已有 observation 的 `deck - UAV` 符号，安全 contract 的 \(\omega_{rel}=UAV-deck\) 只是其相反数；不要在代码中无说明地混用。

部署：index 15 由 planner 本地 filter state 获得，其余由 UAV estimator + deck estimator + frame transform 构造。禁止使用 simulator-only `align_success`。

---

# 20. Reward term categories

本轮只冻结语义，不冻结最终 coefficient。

## 20.1 Always-active

必须始终存在、不能被 hard stage gate 完全关闭：

```text
horizontal/deck tracking quality
surface/contact-point relative velocity matching
flight attitude quality / controller feasibility
safety margin
progress / time efficiency
```

## 20.2 Stage-weighted

使用连续权重 `C(s)` 或 `alpha`：

```text
descent progress               weight C(s)
terminal low tangential speed  weight C(s)*w_h
contact precision              weight C(s)*w_h
terminal attitude alignment    weight alpha
relative angular alignment     weight alpha
```

禁止：

```text
if can_land: reward_term_on
else: reward_term_off
```

## 20.3 Smoothness

至少包含：

\[
r_{\Delta s}\propto-(s_t-s_{t-1})^2
\]

\[
r_{\Delta v}\propto-\|\mathbf v_{rel,t}^D-\mathbf v_{rel,t-1}^D\|^2.
\]

是否对 acceleration-normalized 后再罚由下一阶段 reward preregistration 决定，但不能删除这两类物理语义。

## 20.4 Terminal

```text
safe settled landing bonus
hard contact penalty
ground crash penalty
deck miss penalty
```

Timeout 是否单独赋 terminal penalty 必须在新 task reward preregistration 中根据 ranking budget 决定；本 Theory Gate 不根据 D1 结果提前拍具体数值。

---

# 21. Training controller requirements

训练继续使用 GPU-vectorized surrogate，不为每个 env 启动 PX4 SITL。

必须保持 hierarchy：

```text
velocity reference
→ acceleration
→ R_vel / terminal attitude shaping
→ attitude error
→ body-rate reference
→ rate controller
→ moment
→ simulator wrench
```

新增最小能力：

1. deterministic deck-heading yaw source；
2. 接受 pure terminal-attitude guidance 输出或等价 shaping target；
3. diagnostics：`alpha`, attitude conflict angle, attitude-reference rate, tilt saturation；
4. 不改变 M0/M1/Fixed-Stage M2 controller behavior。

禁止在 S2 pure utility 阶段修改 controller；controller integration 属于 S3 新 task。

---

# 22. PX4 deployment Route A — preferred

目标：

```text
velocity-level PX4 deployment
+
attitude-shaping acceleration/feedforward guidance
```

希望继续保留：

```text
PX4 velocity controller
PX4 attitude controller
PX4 rate controller
PX4 allocation
```

当前仓库没有 PX4 source/SITL 证据证明“在 velocity Offboard 中可以独立指定任意 roll/pitch attitude setpoint 且同时保留 velocity loop”。因此 Theory Gate 明确禁止该表述。

Route A 的 source/SITL gate 必须验证：

```text
TrajectorySetpoint velocity/acceleration semantics
feedforward contribution and limits
position/velocity/acceleration priority
how desired thrust/attitude is formed internally
whether required terminal shaping can be expressed without bypassing velocity loop
frame/yaw semantics
setpoint freshness and failsafe
```

这些均标记：

```text
to be validated in PX4 SITL/source audit
```

不是本轮阻塞 pure math 实现的 TBD。

---

# 23. PX4 deployment Route B — fallback

如果 Route A source/SITL gate 证明 terminal attitude target 无法满足：

```text
approach / tracking:
  PX4 velocity mode

terminal near-contact:
  PX4 attitude setpoint mode
```

Route B 保留：

```text
PX4 attitude controller
PX4 rate controller
PX4 allocation
```

但 terminal 时绕过 PX4 velocity controller，因此其 deployment gap 大于 Route A。

Mode switch 条件未来必须由同一 deployable `alpha/stage/clearance` supervisor 和 SITL 证据定义；本 Theory Gate 不虚构切换阈值或 PX4 mode-transition delay。

---

# 24. Failure modes

| Failure mode | Mechanism | Required evidence | First response |
|---|---|---|---|
| stage saturation | policy stays near 0/1 independent of state | stage histogram/time history | review reward/normalization, not long-train blindly |
| stage chatter | raw action flips, filter/rate limit still oscillates | Δstage, transitions, PSD/variation | filter/rate/smoothness audit |
| terminal descent avoidance | stage remains low to avoid terminal costs | stage vs clearance/time | reward ranking audit |
| premature commitment | stage high while far/misaligned | stage vs clearance/XY | always-active tracking/progress audit |
| reference high variation | velocity ref changes faster than controller | Δv, tracking error | stage-conditioned slew diagnosis |
| velocity/attitude conflict | deck alignment fights translational acceleration | conflict angle, tracking | alpha/clearance envelope audit |
| tilt saturation | blended deck/flight attitude infeasible | tilt saturation ratio | feasibility projection / route audit |
| attitude rate spike | deck motion or alpha changes too quickly | q_ref rate | rate limiter / deck omega audit |
| yaw discontinuity | heading unwrap/degenerate projection | yaw delta | shortest heading/previous heading handling |
| wrong angular safety | compares different frames or absolute omega | deterministic rotating test | frame-consistent omega_rel tests |
| center compensation error | ignores omega×r | off-center rotating benchmark | use existing rigid-point math |
| Route A invalid assumption | PX4 velocity mode cannot express required shaping | source/SITL | switch to evidenced Route B, not invent behavior |

---

# 25. Pure utility API for S2

下一阶段预计新增：

```text
source/quadcopter_waypoint/quadcopter_waypoint/utils/continuous_landing_stage.py
```

建议只包含纯 PyTorch/Python 数学：

```text
smoothstep01(stage)
normalized_stage_action(action_stage)
filter_landing_stage(raw_stage, previous_stage, dt, tau, rate_limit)
stage_conditioned_velocity_limits(stage, cfg)
map_stage_conditioned_relative_velocity(action_xyz, stage, cfg)
limit_stage_conditioned_reference_slew(previous_v, target_v, stage, dt, cfg)
terminal_alignment_weight(stage, clearance, cfg)
deck_heading_from_quaternion(deck_quat, previous_heading?)
shortest_quaternion_slerp(q_vel, q_deck, alpha)
limit_attitude_tilt(...)
limit_attitude_reference_rate(...)
relative_angular_velocity(omega_uav_w, omega_deck_w)
```

允许把明显属于通用 quaternion 的部分补充到现有 `physical_deck_attitude_math.py`，但必须遵循单一 quaternion convention，不复制第二套 multiply/apply/log helper。

该 pure utility 禁止依赖：

```text
Isaac Sim
Gym
ROS2
PX4 runtime
px4_msgs
RL-Games
environment reward
```

Contact-point compensation 继续由：

```text
px4_reference_adapter.py
physical_deck_attitude_math.rigid_surface_point_velocity
```

负责。

---

# 26. Unit-test plan for S2

## 26.1 Stage mapping/filter

```text
a_s=-1/0/+1 -> 0/0.5/1
clamp outside [-1,1]
reset s=0
monotonic response to constant target
rate limit exact
low-pass finite
NaN/Inf rejection
batch/device/dtype parity
```

## 26.2 Velocity envelope

```text
s=0: V_t=0.80, V_down=0, V_up=0.30
s=1: V_t=0.25, V_down=0.30, V_up=0.15
mid-stage continuity
zero action -> zero relative velocity
normal negative sign correct
diagonal tangential norm <= V_t(s)
terminal slew stricter than approach
policy-relative slew only
```

## 26.3 Terminal alignment weight

```text
far clearance => alpha=0
near+stage0 => alpha=0
near+stage1 => alpha≈1
alpha continuous at h_start/h_full
alpha bounded [0,1]
```

## 26.4 Quaternion / attitude

```text
q and -q shortest-path parity
alpha=0 -> q_vel
alpha=1 -> q_deck when feasible
near-equal quaternion stable
unit norm preserved
tilt clamp <=35 deg
deck heading retained after tilt projection
reference-rate limit exact
no NaN for near-degenerate heading
```

## 26.5 Relative angular velocity

```text
omega_uav == omega_deck -> zero
known world vector subtraction
frame rotation preserves norm
sign matches omega_uav - omega_deck
```

---

# 27. S3 task-contract tests planned after pure math PASS

只在 S2 PASS 后创建独立 task，并测试：

```text
new action_space = 4
new observation_space = 22
index15 = previous filtered stage
Fixed-Stage M2 remains action_space = 3
Direct M0/M1 remain action_space = 4 but old semantics unchanged
old M2 checkpoint is rejected/not silently loaded
stage reset state = 0
contact-point adapter reused
relative angular velocity used by new touchdown contract
```

---

# 28. Deterministic smoke plan

S4 已于 2026-08-30 执行并 PASS；S5 仍未执行。

至少 scripted cases：

```text
static level hover
stage ramp with zero XYZ action
constant XY deck tracking
heave tracking
pure normal descent with stage ramp
roll/pitch terminal attitude blend
yaw heading tracking
off-center omega×r contact-point tracking
recovery: stage decrease + positive normal action
```

记录：

```text
stage raw/filtered
V_t/V_down/V_up
relative velocity ref
reference slew
alpha
R_vel vs R_ref vs R_deck error
attitude conflict angle
attitude rate
velocity tracking error
controller saturation
omega_rel
contact metrics
```

Smoke gate：finite、frame/sign correct、无不连续、无 basic ground crash、controller 无异常 saturation。

S4 实测证据：

```text
task ID      = Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
num_envs     = 1
seed         = 42
physics rate = 100 Hz
policy rate  = 25 Hz
cases        = 9/9 PASS
NaN/Inf      = 0
ground crash = 0
reward path  = finite
controller saturation ratio = 0 in all cases
max |delta_stage| = 0.08
```

关键数值：stage ramp 的 `V_t` 单调下降、`V_down` 单调上升、`V_up` 单调下降；low-stage normal descent 被完全阻断，high-stage relative normal reference 达 `-0.2370 m/s`；recovery stage 从 `0.9880` 连续降至 `0.00158`，positive normal reference 达 `+0.2100 m/s`；terminal alpha 最大 `0.8693`，q_ref tilt 最大 `5.276 deg`，attitude-reference rate 最大约 `[0.2255, 0.1803, 0.0089] rad/s`；static yaw heading 为 `+15 deg` 且 q_vel/q_ref 同号一致；off-center contact-point feedforward correction 最大 `0.00536 m/s`。该证据只验证 S4 interface correctness，不宣称 S6/S11 formal benchmark 已完成。

---

# 29. PPO sanity plan

仅 S6 deterministic rotating benchmark PASS 后允许：

```text
64 env
seed 42
30 iterations
```

不预设 95% settled landing，但必须出现学习信号并通过：

```text
NaN/Inf = 0
stage/reference not persistently saturated
stage/ref variation finite and not pathological
controller tracking does not progressively explode
terminal attitude conflict/saturation not persistent
intermediate alignment/descent/contact metrics or normalized return improve
```

失败时先按：

```text
stage distribution
→ relative velocity envelope/slew
→ terminal attitude conflict
→ controller tracking
→ reward ranking
```

诊断，不直接延长到 200/1000 iterations。

---

# 30. Off-center rotating-deck ablation plan

必须选择非零 landing point：

```text
r_c^D = [x_offset, y_offset, z_top]
with at least one of x_offset/y_offset != 0
```

具体 offset 在 S6 benchmark preregistration 中根据 deck size 选择，要求：

```text
inside effective landing region
large enough that omega×r exceeds numerical/noise floor
same geometry for A/B ablation
```

比较：

```text
A v_ref uses deck-center velocity
B v_ref uses contact-point velocity v_D + omega×r
```

运动：

```text
roll-only
pitch-only
yaw-only
combined representative rotation
```

指标：

```text
contact-point velocity reconstruction error
touchdown normal/tangential relative speed
XY contact error
hard contact
settled landing
```

---

# 31. Yaw rotation benchmark plan

第一版 yaw motion 可定义：

\[
\psi_D(t)=A_\psi\sin(\omega_\psi t+\phi_\psi)
\]

并使用现有完整 `world_angular_velocity_from_xyz_rates()` 推导 world angular velocity，禁止直接把 Euler rates 当 world omega。

必须 consistency-check：

```text
pose quaternion derivative
world omega
deck heading
omega×r point velocity
```

Yaw amplitude/frequency 在 S11 preregistration 再按安全/可测性选择，属于 empirical benchmark parameter，不阻塞当前纯数学接口。

---

# 32. PX4 SITL gate

S12 才允许进入。至少需要：

```text
PX4 source/interface audit
velocity/acceleration/yaw setpoint semantics
Route A terminal-shaping feasibility
frame conversion
setpoint frequency/freshness
mode loss/failsafe
single/few-env reproducible SITL test
```

Gate 输出必须是：

```text
Route A validated
or
Route A rejected with evidence -> Route B selected
```

禁止输出“理论上应该可以”作为 PASS。

---

# 33. Allowed TBDs

仅允许以下不阻塞第一版 pure-math 实现的 TBD：

```text
empirical parameter calibration after smoke/sanity
PX4 SITL verified numerical gains / feedforward details
real vehicle identified limits
formal yaw benchmark amplitude/frequency
off-center benchmark exact offset within validated geometry
```

以下不是 TBD，已经在本文冻结：

```text
frames/sign/units
4-D high-level action semantics
stage raw mapping
stage filter mathematical form
first-version filter/rate values
stage-conditioned velocity formulas and initial bounds
stage-conditioned slew formulas and initial limits
contact-point compensation formula
alpha(stage, clearance)
attitude blend method
shortest quaternion handling
yaw source
max tilt
attitude reference-rate limit
relative angular velocity definition
new touchdown rotational metric
22D observation migration
Route A preferred / Route B fallback boundary
```

---

# 34. Theory Gate checklist

- [x] frames / units / sign convention 明确。
- [x] normalized 4-D action 与 physical semantics 明确。
- [x] stage raw mapping、filter、rate limit、reset state 明确。
- [x] normal/tangential velocity envelope 数学和第一版数值明确。
- [x] stage-conditioned slew 数学和第一版数值明确。
- [x] contact-point rigid-body compensation 复用边界明确。
- [x] clearance 使用 deck-frame physical surface 定义。
- [x] deterministic deck-heading yaw guidance 明确。
- [x] R_vel / R_deck / alpha 定义明确。
- [x] quaternion shortest-path blend、tilt constraint、rate constraint 明确。
- [x] velocity-tracking conflict diagnostic 明确。
- [x] relative angular velocity 与 frame contract 明确。
- [x] landing decision 与 deterministic success/safety 分离。
- [x] 22D observation index15 migration 明确。
- [x] reward categories 与 smoothness semantics 明确。
- [x] training controller 与 deployment backend 边界明确。
- [x] Route A 未验证内容未被虚构；Route B 明确为 fallback。
- [x] pure utility API / unit tests / smoke / PPO / ablation / SITL gate 明确。
- [x] 阻塞第一版实现的关键数学无“后续再决定”。

**Theory result: PASS.**

S1 documentation synchronization 后完整回归：

```text
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
128 passed, 1 warning, 21 subtests passed
```

测试数量与 D1 后基线一致；本轮没有删除、跳过或改写旧测试。

---

# 35. S2 implementation evidence and next gate

2026-08-30 已将本 Theory Gate 转化为 pure PyTorch/Python 数学实现：

```text
source/quadcopter_waypoint/quadcopter_waypoint/utils/continuous_landing_stage.py
```

Theory → implementation → unit-test 映射：

| Theory contract | Implementation | Unit-test evidence |
|---|---|---|
| stage mapping / filter / shared smoothstep | `normalized_stage_action`, `filter_landing_stage`, `smoothstep01` | `tests/test_continuous_landing_stage.py` stage tests |
| stage-conditioned velocity envelope | `stage_conditioned_velocity_limits`, `map_stage_conditioned_relative_velocity` | velocity endpoint/sign/norm/continuity tests |
| policy-relative reference slew | `limit_stage_conditioned_reference_slew` | exact approach/terminal per-axis clamp tests |
| terminal alignment weight | `terminal_alignment_weight` | far/near/stage/penetration/continuity tests |
| deterministic deck heading | `deck_heading_world` | yaw projection and degenerate previous/world-x fallback tests |
| shortest quaternion blend | `shortest_quaternion_slerp` | endpoint, `q/-q`, near-equal, norm, dtype/batch tests |
| terminal tilt feasibility | `limit_attitude_tilt` | feasible parity, 35 deg cone clamp, heading/fallback/finite tests |
| attitude-reference rate limit | `limit_attitude_reference_rate` | exact world-axis x/y/z rate clamp and sign tests |
| relative angular velocity | `relative_angular_velocity` | subtraction/sign/zero/frame-norm tests |
| quaternion Exp / matrix conversion reuse | `physical_deck_attitude_math.quat_from_axis_angle`, `quat_from_rotation_matrix` | `tests/test_physical_deck_attitude_math.py` helper tests |

S2 validation evidence：

```text
targeted regression = 65 passed
full regression     = 155 passed + 21 subtests
pre-S2 baseline     = 128 passed + 21 subtests
added tests         = 27
CUDA parity         = PASS (executed, not skipped)
forbidden dependency grep = 0
```

S2 实现继续复用现有 `px4_reference_adapter.py` 的 contact-point compensation；Fixed-Stage M2 action/reference semantics 未修改。

---

# 36. S3 task implementation evidence and next gate

2026-08-30 已实现独立 Continuous-Stage task：

```text
source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/
quadrotor_ship_landing_px4_continuous_stage/
```

冻结 task ID：

```text
Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0
```

S3 将 22-D observation 保持不变，但把 index 15 从 historical `align_success` 迁移为 caller-owned filtered stage，并将 action 从 3-D relative velocity 扩展为：

```text
[a_t1, a_t2, a_n, a_stage]
```

Reference path 已落实为：

```text
stage filter
-> stage-conditioned relative-velocity envelope
-> relative-reference slew
-> existing contact-point rigid-body compensation
-> world/NED velocity reference
```

Terminal attitude path 已落实为：

```text
controller velocity math -> q_vel
stage + signed deck-surface clearance -> alpha
shortest quaternion SLERP toward q_deck
35 deg tilt feasibility
25 Hz attitude-reference rate limit
-> q_ref
-> existing 100 Hz attitude/rate/moment loops
```

`VectorizedPx4LikeController` 仅增加 backward-compatible additive API：默认 `compute(...)` 不传 external attitude reference 时继续原路径；新增测试验证默认路径与显式 `q_vel` 路径的 thrust/moment/body-rate numerical parity。

新 task 的 safe-contact rotational metric 使用：

```text
omega_rel = omega_uav - omega_deck
```

Frozen PhysicalDeckAttitude / Fixed-Stage M2 success semantics 保持不变。新 reward 不使用 hard `can_land` / `align_success` 作为 landing-decision gate；尚未 preregister 的 stage/terminal-attitude/relative-angular smoothness coefficients 明确保持 0，只记录 raw metrics。

S3 validation evidence：

```text
targeted regression = 82 passed
full regression     = 167 passed + 21 subtests
pre-S3 baseline     = 155 passed + 21 subtests
added S3 tests      = 12
git diff --check    = PASS
frozen task source diff = 0
```

```text
S0 Fixed-Stage baseline freeze        = PASS
S1 Theory Gate                        = PASS
S2 Pure mathematical guidance        = PASS
S3 Independent Continuous-Stage task = PASS
S4 1-env deterministic smoke         = PASS
```

S4 validation：

```text
targeted regression = 71 passed
full regression     = 171 passed + 21 subtests
pre-S4 baseline     = 167 passed + 21 subtests
added S4 tests      = 4
git diff --check    = PASS
frozen task source diff = 0
```

下一阶段唯一允许工作：

```text
S5
16-env GPU Continuous-Stage smoke
```

本轮不自动进入 S5、PPO training 或 PX4 SITL。

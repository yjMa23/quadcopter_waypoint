# P6B：实体水平运动甲板接触理论与证据说明

> 状态：历史阶段冻结说明。P6B 首次把成功判定从几何接触代理切换为真实实体甲板与过滤 ContactSensor。

## 1. 阶段目标与上一阶段问题

P6A 的 marker/间隙代理无法证明真实碰撞发生，也不能区分安全接触、硬碰撞、滑出甲板和撞地。P6B 保持水平匀速与竖直升沉，新增 kinematic rigid deck、ground slab、摩擦材料、过滤接触传感器和稳定保持语义。任务 ID 为 `Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0`。

## 2. 状态、观测、动作与坐标系

世界系 \(\mathcal F_W\)、机体系 \(\mathcal F_B\)。甲板姿态恒为单位四元数，因此甲板系轴与世界系平行。策略继续使用 P6A 的 16 维观测和 4 维动作，保持 checkpoint 可迁移：

\[
\mathbf o=[\mathbf v_B^B,\boldsymbol\omega_B^B,\mathbf g_{proj}^B,{}^B\mathbf p_{D/B},\mathbf v_D^W-\mathbf v_B^W,s_{align}],
\]
\[
F_z^B=1.9mg(a_T+1)/2,\qquad \boldsymbol\tau^B=0.01\mathbf a_{1:4}.
\]

网络仍为共享 `[64,64]` ELU PPO actor/critic，固定 sigma。

## 3. 甲板运动和实体模型

甲板中心运动为

\[
\mathbf p_D^W(t)=[x_0+v_xt,y_0+v_yt,z_0+A_z\sin(\omega_zt+\phi_z)]^\top,
\]
\[
\mathbf v_D^W(t)=[v_x,v_y,A_z\omega_z\cos(\omega_zt+\phi_z)]^\top.
\]

实体甲板尺寸 0.50 m × 0.50 m × 0.04 m，kinematic、关闭重力，静/动摩擦均为 1、恢复系数 0。地面 slab 提供独立过滤接触。甲板 `ContactSensor` 与机器人 `body` 过滤匹配，接触力定义为所有过滤接触向量范数的最大值。

## 4. 接触、成功和失败公式

机器人根节点相对甲板线速度

\[
\mathbf v_{rel}^W=\mathbf v_B^W-\mathbf v_D^W,
\quad v_n=v_{rel,z},\quad v_t=\|\mathbf v_{rel,xy}\|_2.
\]

有效甲板内部区域为

\[
|x_B-x_D|<L/2-m_e,\qquad |y_B-y_D|<W/2-m_e,
\]

其中 \(L=W=0.50\) m，边缘裕量 \(m_e=0.025\) m。

`contact_success`：回合内至少一次过滤 robot-deck 接触；不代表安全或稳定。

硬接触：

\[
H=C_D\land(F_D>2.50\;\mathrm N\;\lor |v_n|>0.80\;\mathrm{m/s}\;\lor d_{pen}>0.025\;\mathrm m).
\]

安全接触要求：deck contact、落点在有效甲板内、\(e_{xy}<0.12\) m、非 hard contact、\(|v_n|<0.55\) m/s、\(v_t<0.30\) m/s、\(\|\omega_B\|<1.50\) rad/s、upright > 0.90、penetration ≤ 0.025 m。

稳定降落定义为安全接触连续 3 个控制步，且首次接触已在 0.12 m 精度区内：

\[
S=(N_{safe}\ge3)\land I_{first\_contact\_precision}.
\]

`deck_miss` 包括穿过有效甲板外边界，或首次接触在精度区外；后者立即终止，不能通过后续滑动补成成功。`ground_crash` 是 ground slab 接触或低于 crash height。`timeout` 是回合耗尽。`settled_landing`、`hard_contact`、`deck_miss`、`ground_crash` 和 `timeout` 是互斥终止解释；`contact_success` 可与失败共存。

## 5. Reward 公式

P6B 沿用 P6A 逐步 reward，并加入实体接触落点惩罚：

\[
r_{off}=w_{off}C_D\max(e_{xy}-r_s,0)\Delta t,
\quad w_{off}=-25,\quad r_s=0.12\text{ m}.
\]

冻结阶段关键覆盖为：`predicted_pad_error=-8`、`center_precision=-30`、`center_precision_square=-80`、`descent_vel=-6`、`rel_vel=-1`、`near_pad_horizontal_rel_vel=-7`、`landing_bonus=80`、`crash_penalty=-30`。其余基础项与 P6A 同源。

Reward 是优化信号；正式成功由 ContactSensor 和稳定保持定义。训练 reward 高不能替代 settled landing、hard contact 或 touchdown distance。

## 6. PPO 与 checkpoint 选择

PPO ratio、clipped objective 和 GAE 与 P6A 相同。当前同族配置：gamma 0.99、GAE lambda 0.95、clip 0.2、learning rate 1e-4、horizon 24、minibatch 384、mini epochs 5、critic coefficient 2、entropy coefficient 0、bounds coefficient 1e-4、fixed sigma。

课程从宽接触到 18 cm、14 cm、12 cm 精度逐步推进。正式 checkpoint 为 ep990，选择依据是固定三 seed 闭环物理指标，不是只看文件名中的 rolling reward。

## 7. 正式评估指标

每 seed 256 回合，seeds 42/43/44。报告：contact success、settled landing、hard contact、ground crash、deck miss、timeout、首次接触 xy、首次接触法/切向相对速度、最大接触力、penetration、settle time 和 touchdown distance。touchdown 统计仅在成功回合上计算。

## 8. 真实结果

checkpoint：`logs/rl_games/quadcopter_ship_landing_physical_deck/2026-08-03_18-46-00/nn/last_quadcopter_ship_landing_physical_deck_ep_990_rew_61.680832.pth`；SHA256 `614cf3bea439883b7b2c478f0dd21641f9eb750df9f08d711d8cf122f133b3aa`。

768 回合：contact 768，settled 738，hard 0，ground crash 0，deck miss 30，timeout 0。settled landing 96.0938%，contact 100%，hard/ground/timeout 0%，deck miss 3.9063%；touchdown distance mean 0.05627 m、P95 0.10221 m。

## 9. 方案原因、候选方案和局限

选择实体 ContactSensor 是因为几何 clearance 不能证明真实接触；选择首次接触精度锁存是为了禁止“先撞偏、再滑入”被计为成功；选择三步 settle hold 是在抑制瞬时接触误判与保持任务可训练之间折中。

未选择只用接触力阈值、只用 clearance、只用 episode reward 或将甲板边缘放宽。局限包括：甲板无 roll/pitch，法向固定世界 z，未包含角速度诱导的表面点速度，state-based 而非视觉策略。

## 10. 理论—代码—配置—测试—证据映射

| 理论内容 | 代码位置/函数 | 配置键 | 单元测试 | 实验证据 |
|---|---|---|---|---|
| 实体甲板、ground slab、ContactSensor | `.../quadrotor_ship_landing_physical_deck_env.py::QuadcopterShipLandingPhysicalDeckEnvCfg/_setup_scene` | `deck`, `ground_slab`, `*_contact_sensor` | task smoke；评估工具测试 | `benchmarks/phase6b_physical_deck/summary.json` |
| hard/safe/miss/crash | 同文件 `_compute_landing_terms` | `contact_force_threshold`, `safe_*`, `hard_*`, `deck_edge_margin` | P8B 文档冻结语义检查 | P6B 原始 CSV |
| settle hold 与首次接触锁存 | 同文件 `_get_dones` | `settle_hold_steps`, `landing_success_radius` | 评估回归测试 | summary 的分类计数 |
| reward | 同文件 `_get_rewards` 与 heave 基类 | `*_reward_scale` | 文档同步测试 | TensorBoard、checkpoint |
| PPO | `.../agents/rl_games_ppo_cfg.yaml` | `gamma,tau,e_clip,critic_coef,...` | YAML/document sync | 训练参数副本 |
| 正式聚合 | `scripts/rl_games/eval_metrics.py` | seeds、episodes | `tests/test_eval_metrics_utils.py` | `benchmarks/phase6b_physical_deck/summary.json` |

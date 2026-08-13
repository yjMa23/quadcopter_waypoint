# heave-precision task：升沉甲板精确降落理论与证据说明

> 状态：历史冻结说明。本文以当前源码、heave-precision task benchmark、冻结 checkpoint 和提交历史为依据；不把 physical-deck task/physical-deck-attitude task 的实体接触语义倒推到 heave-precision task。

## 1. 设计目标与基线问题

heave-precision task 在 deck-contact proxy baseline 的水平移动甲板策略上增加可见的竖直升沉，并将允许的水平落点半径从 0.16 m 收紧到 0.10 m。原基线只处理平面内匀速移动，无法检验相对竖直速度匹配；0.16 m 成功半径也不足以代表精确落点。

任务 ID 为 `Isaac-Quadcopter-ShipLanding-Heave-Direct-v0`。该任务仍使用接触代理条件，不具备 physical-deck task 的实体甲板 ContactSensor，因此这里的 `landing_success` 不等价于实体甲板任务的 `settled_landing`。

## 2. 状态、坐标系、观测与动作

世界系、机体系分别记为 \(\mathcal F_W\)、\(\mathcal F_B\)。四旋翼位置、姿态、线速度、角速度为 \(\mathbf p_B^W,\mathbf q_B^W,\mathbf v_B^W,\boldsymbol\omega_B^W\)，甲板中心位置和线速度为 \(\mathbf p_D^W,\mathbf v_D^W\)。甲板无 roll/pitch/yaw。

16 维观测为

\[
\mathbf o=[\mathbf v_B^B,\boldsymbol\omega_B^B,\mathbf g_{proj}^B,{}^B\mathbf p_{D/B},\mathbf v_D^W-\mathbf v_B^W,s_{align}].
\]

动作 \(\mathbf a=[a_T,a_{\tau_x},a_{\tau_y},a_{\tau_z}]\in[-1,1]^4\)。执行前裁剪，映射为

\[
F_z^B=1.9mg\frac{a_T+1}{2},\qquad
\boldsymbol\tau^B=0.01[a_{\tau_x},a_{\tau_y},a_{\tau_z}]^\top.
\]

策略网络为 RL-Games 共享 actor/critic MLP，隐藏层 `[64,64]`、ELU、固定方差高斯策略；输入和值均归一化。

## 3. 甲板运动模型

水平运动在单回合内为匀速：

\[
x_D(t)=x_0+v_xt,\qquad y_D(t)=y_0+v_yt.
\]

升沉为

\[
z_D(t)=z_0+A_z\sin(\omega_z t+\phi_z),
\qquad
\dot z_D(t)=A_z\omega_z\cos(\omega_z t+\phi_z).
\]

冻结范围为：\(z_0=0.16\) m，\(A_z\in[0.08,0.12]\) m，\(f_z\in[0.18,0.30]\) Hz。较早版本因底座高度过低可能进入地面；当前参数在保留可见升沉的同时避免甲板下穿。

## 4. 相对量、接触代理和判定

水平误差、相对速度和表面间隙为

\[
e_{xy}=\|\mathbf p_{D,xy}^W-\mathbf p_{B,xy}^W\|_2,
\quad
\mathbf v_{rel}=\mathbf v_B^W-\mathbf v_D^W,
\]

\[
c=(p_{B,z}^W-d_{feet})-(p_{D,z}^W+h_D/2).
\]

对准候选要求水平误差、相对水平速度、高度区间和直立度满足阈值并持续 8 个控制步。完成对准后，接触代理 landing candidate 要求：

- \(e_{xy}<0.10\) m；
- \(-0.01<c<0.060\) m；
- 总相对速度、水平相对速度、机体角速度和直立度满足源码阈值；
- 连续保持 `landing_success_hold_steps=4`。

`crash` 为高度或水平工作区越界；`timeout` 为 10 s 回合耗尽。heave-precision task 没有真实 `contact_success`、`hard_contact`、`deck_miss` 分类，不能用后续术语重命名其代理判定。

## 5. Reward 真实结构

每步 reward 为以下项之和：

\[
r_t=r_{lin}+r_{ang}+r_{progress}+r_{descent}+r_{xy}+r_h+r_{rel}+r_{tilt}
+r_{v_z}+r_{v_{xy}}+r_{near}+r_{pred}+r_c+r_{center}+r_{center^2}+r_{align}+r_{hold}+r_{land}+r_{crash}.
\]

其中主要定义为

\[
r_{lin}=w_{lin}\|\mathbf v_B^B\|^2\Delta t,
\quad r_{ang}=w_{ang}\|\boldsymbol\omega_B^B\|^2\Delta t,
\]
\[
r_{progress}=w_p(e_{xy,t-1}-e_{xy,t}),
\quad r_{xy}=w_{xy}e_{xy}\Delta t,
\]
\[
r_{rel}=w_{rel}\|\mathbf v_{rel}\|\Delta t,
\quad r_{v_z}=w_z\max(0,-v_z-v_{limit})^2\Delta t,
\]
\[
r_{center}=w_c\alpha_h e_{xy}\Delta t,
\quad r_{center^2}=w_{c2}\alpha_h e_{xy}^2\Delta t.
\]

`alpha_h` 仅在完成对准且接近甲板时增大。heave-precision task 的新增权重为 `near_center_height=0.35`、`center_precision_reward_scale=-8`、`center_precision_square_reward_scale=-20`。正式采用 checkpoint 来自 deck-contact proxy baseline，而中心 reward 微调候选因成功率下降和 crash 非零被拒绝。因此“实现了 reward hook”与“正式 checkpoint 由该 reward 训练得到”必须区分。

## 6. PPO、checkpoint 与评估

PPO 使用 clipped surrogate：

\[
r_t(\theta)=\frac{\pi_\theta(a_t|o_t)}{\pi_{old}(a_t|o_t)},
\quad
L_{clip}=\mathbb E[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)].
\]

GAE 为

\[
\delta_t=r_t+\gamma V(o_{t+1})-V(o_t),\qquad
A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}.
\]

当前同族 YAML 参数为 \(\gamma=0.99\)、\(\lambda=0.95\)、clip 0.2、horizon 24、mini epochs 5；历史正式 checkpoint 已冻结，不因本文补写而重训。checkpoint 选择基于真实闭环结果：采用 deck-contact proxy baseline ep650，而不是 reward 更高但闭环指标退化的 heave-precision task 微调候选。

评估使用 seeds 42/43/44，每 seed 256 回合。`training reward` 只表示训练目标，不等价于 `landing_success_rate`。

## 7. 真实结果

冻结 checkpoint：`logs/rl_games/quadcopter_ship_landing/2026-06-30_15-21-37/nn/last_quadcopter_ship_landing_ep_650_rew_34.6081.pth`，SHA256 `cb64364dea3d44ebaa0231b54633e8cb6e14c169d1682b66b81d8de28d92bb92`。

三 seed、768 回合：landing success 759，crash 0，timeout 9；成功率 98.8281%，crash 0%，timeout 1.1719%，成功回合 touchdown distance mean 0.06609 m、P95 0.09701 m。这里报告的是代理 landing success，不是实体接触稳定降落。

## 8. 候选方案、失败模式与局限

- 0.16 m 半径：成功率高但落点不够集中；未选。
- 0.08 m 半径：精度改善，但成功率降至 93.75%，出现 crash；未选。
- 中心 reward 微调：方向合理，但已评估 checkpoint 均不优于冻结基线；未选。
- 当前限制：无实体甲板碰撞、无 roll/pitch、无接触冲量与刚体表面点速度，state-based 而非视觉策略。

## 9. 理论—代码—配置—测试—证据映射

| 理论内容 | 代码位置/函数 | 配置键 | 单元测试 | 实验证据 |
|---|---|---|---|---|
| 16 维观测与动作映射 | `.../quadrotor_ship_landing_env.py::QuadcopterShipLandingEnv._get_observations/_pre_physics_step` | `observation_space`, `action_space`, `thrust_to_weight`, `moment_scale` | `tests/test_actor_preserving_documentation_sync.py` 的冻结任务检查 | `benchmarks/heave_precision/summary.json` |
| 正弦升沉 | `.../quadrotor_ship_landing_heave_env.py::_update_pad_motion` | `pad_heave_*` | 同上 | 同上 |
| reward 与代理成功 | `.../quadrotor_ship_landing_heave_env.py::_get_rewards`；基类 `_compute_landing_terms/_get_dones` | reward scales、success thresholds | 现有环境/评估测试 | heave-precision task CSV 与 summary |
| PPO 配置 | `.../agents/rl_games_ppo_cfg.yaml` | `gamma,tau,e_clip,horizon_length,mini_epochs` | 文档同步测试 | checkpoint metadata、训练日志 |
| 正式结果 | `scripts/rl_games/eval_metrics.py` | seeds 42/43/44、256 episodes | `tests/test_eval_metrics_utils.py` | `benchmarks/heave_precision/summary.json` |

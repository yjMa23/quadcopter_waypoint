# P6C：带 Roll/Pitch 的实体运动甲板理论与证据说明

> 状态：历史冻结说明，也是 P7、P8A、P8B 的任务语义基线。P8B 不修改本文对应的环境。

## 1. 阶段目标

P6B 假定甲板法向固定为世界 z，不能表达 roll/pitch、角速度导致的表面点速度、甲板坐标系落点以及机体—甲板法向夹角。P6C 保留实体碰撞并加入独立正弦 roll/pitch。任务 ID 为 `Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0`。

## 2. 状态、坐标系和甲板运动

世界、机体、甲板系为 \(\mathcal F_W,\mathcal F_B,\mathcal F_D\)。状态为四旋翼和甲板各自的位置、姿态、线速度、角速度。

\[
\begin{aligned}
x_D(t)&=x_0+v_xt, & y_D(t)&=y_0+v_yt,\\
z_D(t)&=z_0+A_z\sin(\omega_zt+\phi_z),\\
\varphi_D(t)&=A_\varphi\sin(\omega_\varphi t+\phi_\varphi),\\
\theta_D(t)&=A_\theta\sin(\omega_\theta t+\phi_\theta), & \psi_D(t)&=0.
\end{aligned}
\]

正式范围为 roll/pitch 振幅 0–5°、频率 0.08–0.15 Hz。姿态由绝对 episode time 计算。yaw 为零时，XYZ 欧拉角速度到世界角速度的映射为

\[
\boldsymbol\omega_D^W=[\dot\varphi\cos\theta,\dot\theta,-\dot\varphi\sin\theta]^\top.
\]

直接写入 \([\dot\varphi,\dot\theta,0]\) 在 pitch 非零时不正确。

## 3. 观测、动作与策略网络

22 维观测为

\[
\mathbf o=[\mathbf v_B^B,\boldsymbol\omega_B^B,\mathbf g_{proj}^B,{}^B\mathbf p_{D/B},
\mathbf v_S^W-\mathbf v_B^W,s_{align},{}^B\mathbf n_D,{}^B(\boldsymbol\omega_D^W-\boldsymbol\omega_B^W)].
\]

第 12:15 维保持世界系相对线速度语义。动作仍为 4 维总推力和三轴力矩：

\[
F_z^B=1.9mg(a_T+1)/2,\qquad \boldsymbol\tau^B=0.01\mathbf a_{1:4},
\]

并裁剪到 \([-1,1]\)。策略为共享 actor/critic 的 `[64,64]` ELU MLP，fixed sigma，输入和值归一化。

## 4. 刚体表面点运动学

甲板表面点与机器人底部点速度分别为

\[
\mathbf v_S^W=\mathbf v_D^W+\boldsymbol\omega_D^W\times(\mathbf p_S^W-\mathbf p_D^W),
\]
\[
\mathbf v_{B,bottom}^W=\mathbf v_B^W+\boldsymbol\omega_B^W\times(\mathbf p_{B,bottom}^W-\mathbf p_B^W).
\]

相对速度及分解：

\[
\mathbf v_{rel}^W=\mathbf v_{B,bottom}^W-\mathbf v_S^W,
\quad v_n=\mathbf v_{rel}^{W\top}\mathbf n_D^W,
\]
\[
v_t=\|\mathbf v_{rel}^W-v_n\mathbf n_D^W\|_2.
\]

落点和 signed clearance 均在甲板局部几何中计算，不使用简单世界 xy/z 替代。

## 5. 接触、成功与失败

接触冲量近似 \(J=F_D\Delta t_{sim}\)。hard contact 条件为 deck contact 且满足任一项：接触力 >2.50 N、冲量 >0.025 N·s、法向相对速度绝对值 >0.80 m/s、penetration >0.030 m。

safe contact 同时要求：有甲板接触、无地面接触、落点在有效甲板内、deck-frame 水平误差 <0.12 m、非 hard contact、\(|v_n|<0.55\) m/s、\(v_t<0.30\) m/s、机体角速度 <1.50 rad/s、机体 z 与甲板法向夹角 <12°、世界直立度 >0.90、penetration ≤0.025 m。

`settled_landing` 要求 safe contact 连续 3 个控制步且首次接触精度合格。`deck_miss` 包括甲板外穿越或首次接触超出精度区；`ground_crash` 为 ground slab 接触或低于 crash height；另有 workspace crash 和 timeout。`contact_success` 只表示发生过甲板接触，不等于稳定降落。

## 6. Reward、termination 与 PPO

P6C 冻结 P6B reward 结构，但所有接触相关量改用 deck-frame/rigid-point 运动学。逐步 reward 仍由速度、进展、位置、高度、相对速度、倾角、下降速度、预测落点、clearance、中心精度、对准、落地奖励和失败惩罚组成。正式任务指标由上述实体接触判定给出，training reward 不等于 settled landing。

PPO 参数：gamma 0.99、GAE lambda 0.95、clip 0.2、learning rate 1e-4、horizon 24、minibatch 384、mini epochs 5、critic coefficient 2、entropy 0、bounds 1e-4、fixed sigma、`separate: false`。RL-Games 实际最小化

\[
L=L_{actor}+0.5c_vL_{value}-c_eH+c_bL_{bounds}.
\]

## 7. Checkpoint 扩展与选择

P6B 的 16 维 checkpoint 扩展到 22 维时，actor 第一层前 16 列原样复制，新增 6 列置零；因此 epoch-0 actor 在相同原始 16 维信息下保持确定性输出 parity。新增 observation RMS、metadata 和结构均显式记录。

短程 fine-tune 的 ep1000、1010、1020 闭环指标均退化，所以正式选择扩展后的 P6B ep990，而不是训练 reward 更高的候选。

## 8. 实验设置与真实结果

正式协议为 seeds 42/43/44，每 seed 256 回合，最大 ±5° roll/pitch。指标包含 contact、settled、hard、ground、deck miss、timeout、首次接触 deck-frame xy、法/切向相对速度、body/deck-normal angle、penetration、接触力/冲量、settle time、touchdown distance 和运动一致性。

checkpoint：`logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/expanded_from_p6b_ep990_16to22.pth`；SHA256 `95424bb0d6b98d8dfbf2455d6fd84e99a77d52bca28489654036a25aea5a697d`。

768 回合：contact 99.8698%，settled 94.6615%，hard 0.1302%，ground crash 0%，deck miss 5.3385%，timeout 0%；成功首次接触 xy P95 0.10225 m，法向相对速度绝对值 P95 0.38513 m/s，body/deck-normal angle P95 6.42°，touchdown distance P95 0.10652 m。

冻结验收线：settled ≥92%，ground crash ≤1%，hard ≤2%，timeout ≤3%，first-contact xy P95 ≤0.12 m，normal-speed P95 ≤0.45 m/s。正式 checkpoint 通过。

## 9. 方案原因、候选方案与局限

采用刚体表面点速度是因为甲板中心速度遗漏 \(\omega\times r\)；采用 deck-frame 落点和法向是因为世界 xy/z 在倾斜甲板上语义错误；采用新增输入零权重初始化是为了保持迁移 parity。未采用后续 fine-tune，因为真实闭环指标下降。

局限：无 yaw、随机波谱、水动力、传感噪声或相机输入。当前策略是 state-based，不能称为视觉端到端。

## 10. 理论—代码—配置—测试—证据映射

| 理论内容 | 代码位置/函数 | 配置键 | 单元测试 | 实验证据 |
|---|---|---|---|---|
| 绝对时间姿态和世界角速度 | `.../quadrotor_ship_landing_physical_deck_attitude_env.py::_compute_absolute_deck_state` | `deck_*_amplitude_*`, `deck_*_frequency_*` | `tests/test_physical_deck_attitude_math.py` | P6C motion diagnostics |
| 表面点速度和 deck-frame 几何 | `.../utils/physical_deck_attitude_math.py`、环境 `_contact_kinematics` | deck dimensions、offset | 同上 | P6C 接触指标 |
| 22 维观测 | 环境 `_get_observations` | `observation_space=22` | P7/P8B documentation sync | checkpoint shape、summary |
| safe/hard/settled | 环境 `_compute_landing_terms/_get_dones` | `safe_*`, `hard_*`, `settle_hold_steps` | evaluator/math tests | `benchmarks/phase6c_physical_deck_attitude/summary.json` |
| 16→22 parity | `scripts/rl_games/expand_checkpoint_observation.py` | expansion metadata | `tests/test_checkpoint_observation_expansion.py` | checkpoint SHA |
| PPO | `.../agents/rl_games_ppo_cfg.yaml` | PPO/network keys | documentation sync | params YAML、TensorBoard |
| 正式聚合 | `scripts/rl_games/summarize_physical_deck_attitude.py` | seeds/episodes/thresholds | eval utils tests | P6C summary、CSV |

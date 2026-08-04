# 基于专家轨迹模仿学习的四旋翼运动甲板自主降落方法

> 工程论文式理论与实现说明，适用于当前仓库 P7。本文所有任务定义、公式参数、训练配置和实验数字均以仓库当前代码及 `benchmarks/phase7_imitation_hybrid/` 为准，不引入尚未实现的视觉输入、水动力或六自由度船舶运动。

<!-- CODE_SYNC
observation_dim=22
action_dim=4
hidden_units=64,64
activation=elu
observation_epsilon=1e-05
observation_clip=5.0
action_clip=1.0
thrust_to_weight=1.9
moment_scale=0.01
deck_roll_max_deg=5.0
deck_pitch_max_deg=5.0
deck_roll_frequency_min=0.08
deck_roll_frequency_max=0.15
deck_pitch_frequency_min=0.08
deck_pitch_frequency_max=0.15
landing_success_radius=0.12
settle_hold_steps=3
phase_weight_cap=8.0
ppo_gamma=0.99
ppo_tau=0.95
ppo_learning_rate=0.0001
ppo_clip=0.2
horizon_length=24
minibatch_size=384
mini_epochs=5
critic_coef=2
fixed_sigma=true
-->

## 摘要

针对四旋翼在平移、升沉并具有小幅横滚和俯仰运动的实体甲板上自主降落问题，本文构建了一套“冻结 PPO 专家策略—成功轨迹采集—行为克隆—PPO 初始化与闭环评估”的模仿学习流程。底层任务采用 Isaac Lab 实体碰撞环境，策略接收 22 维状态观测并输出总推力与三轴力矩对应的 4 维连续动作。专家数据仅保留完整稳定降落回合，并按完整 episode 进行训练、验证和测试划分，以避免同一轨迹泄漏到不同数据子集。行为克隆网络与 PPO actor 保持相同的 22-64-64-4 ELU 结构，并复用专家 checkpoint 中冻结的 RL-Games 输入归一化统计。针对接触稳定阶段样本占比较低的问题，训练采用按飞行阶段逆频率加权的均方误差目标。实验共采集 3976 个成功回合、540321 条状态—动作 transition。三随机种子、每个种子 256 回合的闭环评估表明，冻结专家稳定降落率为 94.66%，纯行为克隆达到 88.28%，随机初始化 PPO 在相同在线预算下仅为 0.91%，而直接进行 BC 初始化 PPO 微调后下降至 76.69%。结果说明专家数据显著改善了有限交互预算下的初始策略，但当前 PPO 微调会破坏已学到的精细对准和稳定触地行为。

**关键词：** 四旋翼；运动甲板降落；模仿学习；行为克隆；PPO；实体接触；策略迁移

## 1 引言

运动平台降落同时包含相对位置跟踪、相对速度匹配、姿态对准和接触稳定四类要求。与静态悬停不同，策略不仅需要靠近目标，还必须在首次接触时满足落点、法向速度、切向速度、机体角速度和姿态误差约束，并在真实碰撞后持续保持安全接触。

P6C 已提供一个冻结的 PPO teacher，在最大 ±5° roll/pitch 的实体运动甲板上取得 94.66% 的稳定降落率。P7 的目标不是修改环境或重新设计奖励，而是在保持任务定义完全不变的条件下回答三个问题：

1. 成功专家轨迹能否训练出可闭环执行的纯监督策略；
2. 行为克隆能否提高有限在线交互预算下的样本效率；
3. 直接使用 BC actor 初始化 PPO 是否能进一步提升性能。

本文围绕这三个问题给出与当前代码逐项对应的建模、算法和实验说明。

## 2 问题定义

### 2.1 坐标系与状态

记世界坐标系为 \(\mathcal{F}_W\)，四旋翼机体系为 \(\mathcal{F}_B\)，甲板坐标系为 \(\mathcal{F}_D\)。四旋翼根节点位置、姿态、线速度和角速度分别记为

\[
\mathbf{p}_B^W,\quad \mathbf{q}_B^W,\quad \mathbf{v}_B^W,\quad \boldsymbol{\omega}_B^W.
\]

甲板中心状态分别记为

\[
\mathbf{p}_D^W,\quad \mathbf{q}_D^W,\quad \mathbf{v}_D^W,\quad \boldsymbol{\omega}_D^W.
\]

当前任务为状态策略，不包含相机图像、深度图、目标检测结果或真实视觉投影特征。

### 2.2 甲板运动模型

在单个 episode 内，甲板平面位置采用匀速运动，竖直方向、横滚和俯仰采用独立正弦运动：

\[
\begin{aligned}
x_D(t) &= x_0 + v_x t,\\
y_D(t) &= y_0 + v_y t,\\
z_D(t) &= z_0 + A_z\sin(\omega_z t + \phi_z),\\
\varphi_D(t) &= A_\varphi\sin(\omega_\varphi t + \phi_\varphi),\\
\theta_D(t) &= A_\theta\sin(\omega_\theta t + \phi_\theta),\\
\psi_D(t) &= 0.
\end{aligned}
\]

正式 P7 评估中：

\[
A_\varphi,A_\theta\in[0,5^\circ],\qquad
f_\varphi,f_\theta\in[0.08,0.15]\ \text{Hz}.
\]

由于 yaw 固定为零，XYZ 欧拉角速度映射到世界角速度后为

\[
\boldsymbol{\omega}_D^W=
\begin{bmatrix}
\dot\varphi_D\cos\theta_D\\
\dot\theta_D\\
-\dot\varphi_D\sin\theta_D
\end{bmatrix}.
\]

这与直接使用 \([\dot\varphi_D,\dot\theta_D,0]^\top\) 不同；当 pitch 非零时，后者不能正确表示世界系角速度。

### 2.3 接触点运动学

设甲板接触表面点为 \(\mathbf{p}_S^W\)。刚体表面点速度为

\[
\mathbf{v}_S^W=\mathbf{v}_D^W+
\boldsymbol{\omega}_D^W\times
\left(\mathbf{p}_S^W-\mathbf{p}_D^W\right).
\]

四旋翼底部点速度同样由根节点线速度和角速度计算。两者的相对速度为

\[
\mathbf{v}_{\mathrm{rel}}^W=\mathbf{v}_{B,\mathrm{bottom}}^W-\mathbf{v}_S^W.
\]

记甲板法向为 \(\mathbf{n}_D^W\)，则法向和切向相对速度分别为

\[
v_n=\mathbf{v}_{\mathrm{rel}}^{W\top}\mathbf{n}_D^W,
\]

\[
v_t=\left\|\mathbf{v}_{\mathrm{rel}}^W-v_n\mathbf{n}_D^W\right\|_2.
\]

稳定降落判定使用真实 deck ContactSensor、ground ContactSensor、deck-frame 落点、法向/切向相对速度、机体角速度、机体 z 轴与甲板法向夹角、世界系直立度和 penetration，而不是使用可视化 marker 代替碰撞真值。

### 2.4 马尔可夫决策过程

任务可写为有限时域马尔可夫决策过程

\[
\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,r,\gamma\rangle,
\]

其中 \(\mathcal{S}\) 为环境状态，\(\mathcal{A}\subset[-1,1]^4\) 为连续动作空间，\(P\) 由 Isaac Sim 刚体动力学、运动甲板和接触模型共同决定，\(r\) 为冻结的 P6C 奖励函数，折扣因子为 \(\gamma=0.99\)。策略目标为最大化

\[
J(\pi)=\mathbb{E}_{\pi,P}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

需要强调的是，P7 不改变 P6C 的观测、动作、奖励、终止和接触判定，只改变策略的初始化与训练方式。

## 3 观测与动作空间

### 3.1 22 维观测

策略观测向量为

\[
\mathbf{o}_t=
\left[
\mathbf{v}_B^B,
\boldsymbol{\omega}_B^B,
\mathbf{g}_{\mathrm{proj}}^B,
{}^B\mathbf{p}_{D/B},
\mathbf{v}_{S/B}^W,
s_{\mathrm{align}},
{}^B\mathbf{n}_D,
{}^B\boldsymbol{\omega}_{D/B}
\right],
\]

共 22 维。各分量如下：

| 索引 | 物理量 | 坐标系/单位 |
|---|---|---|
| `0:3` | 四旋翼根节点线速度 | 机体系，m/s |
| `3:6` | 四旋翼根节点角速度 | 机体系，rad/s |
| `6:9` | 投影重力 | 机体系，单位向量 |
| `9:12` | 甲板中心相对四旋翼位置 | 机体系，m |
| `12:15` | 甲板表面点速度减四旋翼根节点速度 | 世界系，m/s |
| `15` | 已完成对准状态 | 0/1 |
| `16:19` | 甲板法向 | 机体系，单位向量 |
| `19:22` | 甲板角速度减四旋翼角速度 | 机体系，rad/s |

### 3.2 4 维连续动作

策略输出

\[
\mathbf{a}_t=[a_T,a_{\tau_x},a_{\tau_y},a_{\tau_z}]^\top,
\qquad \mathbf{a}_t\in[-1,1]^4.
\]

动作在送入物理引擎前被显式裁剪到 \([-1,1]\)。总推力和三轴力矩映射为

\[
F_z^B=1.9\,mg\frac{a_T+1}{2},
\]

\[
\boldsymbol{\tau}^B=0.01
\begin{bmatrix}
a_{\tau_x}\\a_{\tau_y}\\a_{\tau_z}
\end{bmatrix}.
\]

因此 \(a_T=-1\) 对应零推力，\(a_T=1\) 对应约 1.9 倍机体重力的总推力。

## 4 专家数据构建

### 4.1 专家策略

专家为冻结的 P6C PPO checkpoint。采集时使用策略高斯分布的 deterministic mean action，并在环境执行前再次裁剪到 \([-1,1]\)。只有最终满足 `settled_landing` 的完整 episode 被写入正式数据集；hard contact、deck miss、ground crash 和 timeout 回合被拒绝。

### 4.2 Transition 结构

每条 transition 至少保存：

\[
\left(
\text{episode id},
\text{step id},
\text{seed},
\mathbf{o}_t,
\mathbf{a}_t^E,
r_t,
\text{terminated},
\text{timeout},
\text{phase},
\text{deck parameters},
\text{terminal contact metrics}
\right).
\]

数据使用压缩 NPZ 分片和 manifest 管理。每个 shard 记录 SHA256，加载时检查 dtype、shape、NaN/Inf、动作范围、episode 连续性和 manifest 哈希。

### 4.3 按完整 episode 划分

设所有成功 episode 集合为 \(\mathcal{E}\)。数据划分在 episode 层面进行：

\[
\mathcal{E}=\mathcal{E}_{\mathrm{train}}\cup
\mathcal{E}_{\mathrm{val}}\cup
\mathcal{E}_{\mathrm{test}},
\]

并满足任意两者交集为空。当前划分为 80%/10%/10%：

| 子集 | Episodes | Transitions |
|---|---:|---:|
| Train | 3180 | 434288 |
| Validation | 397 | 54132 |
| Test | 399 | 51901 |

这种划分避免同一轨迹的相邻状态同时出现在训练集和测试集中。

### 4.4 飞行阶段标签

代码依据实时 landing terms 将 transition 划分为四个阶段：

1. `approach`：尚未进入对准区域；
2. `align`：进入对准候选区域或接近对准高度区间；
3. `descent`：`can_land=True` 且尚未真实接触；
4. `contact_settle`：ContactSensor 检测到 deck contact。

阶段优先级按上述顺序覆盖，接触阶段优先级最高。正式数据集中阶段占比为：approach 14.64%、align 34.75%、descent 46.52%、contact/settle 4.09%。

## 5 行为克隆方法

### 5.1 输入归一化

为保证 BC 与 PPO teacher 的输入语义一致，BC 不重新估计数据集均值和方差，而是直接读取 teacher checkpoint 中的 RL-Games `running_mean`、`running_var` 和 `count`。对原始观测 \(\mathbf{o}\) 的变换为

\[
\hat{\mathbf{o}}=
\operatorname{clip}\left(
\frac{\mathbf{o}-\boldsymbol{\mu}}
{\sqrt{\boldsymbol{\sigma}^2+10^{-5}}},
-5,5
\right).
\]

归一化统计在 BC 训练和推理期间保持冻结。

### 5.2 网络结构

BC actor 与 PPO actor 采用相同结构：

\[
\mathbf{h}_1=\operatorname{ELU}(W_1\hat{\mathbf{o}}+\mathbf{b}_1),
\]

\[
\mathbf{h}_2=\operatorname{ELU}(W_2\mathbf{h}_1+\mathbf{b}_2),
\]

\[
\tilde{\mathbf{a}}=W_3\mathbf{h}_2+\mathbf{b}_3,
\]

其中隐藏层宽度均为 64，最终确定性动作是

\[
\pi_{\mathrm{BC}}(\mathbf{o})=
\operatorname{clip}(\tilde{\mathbf{a}},-1,1).
\]

### 5.3 阶段逆频率权重

设训练集中共有 \(N\) 条 transition，出现的阶段数为 \(K\)，阶段 \(k\) 的样本数为 \(n_k\)。原始逆频率权重定义为

\[
\tilde w_k=\frac{N}{K n_k}.
\]

代码将权重上限设置为 8，再对所有样本权重归一化，使平均权重为 1：

\[
w_k=\frac{\min(\tilde w_k,8)}
{\frac{1}{N}\sum_{i=1}^{N}\min(\tilde w_{y_i},8)}.
\]

该设计提升 contact/settle 少数阶段在损失中的贡献，同时避免极端权重导致训练不稳定。

### 5.4 加权行为克隆目标

对一个 batch \(\mathcal{B}\)，单样本动作误差为

\[
\ell_i=\frac{1}{4}
\left\|\pi_{\mathrm{BC}}(\mathbf{o}_i)-\mathbf{a}_i^E\right\|_2^2.
\]

训练目标为

\[
\mathcal{L}_{\mathrm{BC}}=
\frac{\sum_{i\in\mathcal{B}}w_{y_i}\ell_i}
{\sum_{i\in\mathcal{B}}w_{y_i}}.
\]

优化器为 Adam，正式训练 learning rate 为 \(10^{-3}\)，batch size 为 4096，最多训练 50 epochs，并按 validation weighted MSE 保存最优 checkpoint。梯度范数裁剪上限为 10。

## 6 BC 到 PPO 的参数迁移

### 6.1 Actor 与归一化迁移

迁移工具将 BC 的两层 MLP 和 `mu` 输出头逐项复制到 RL-Games checkpoint，并复制 observation running mean、variance 和 count。迁移完成后，在同一批原始观测上验证

\[
\epsilon_{\mathrm{parity}}=
\max_i\left\|
\pi_{\mathrm{BC}}(\mathbf{o}_i)-
\pi_{\mathrm{RL}}(\mathbf{o}_i)
\right\|_\infty.
\]

当前实测 \(\epsilon_{\mathrm{parity}}=0\)，低于要求的 \(10^{-5}\)。

### 6.2 Fresh PPO 状态

为避免把 teacher 的训练历史误当作 BC 训练结果，迁移时执行以下重置：

- PPO optimizer moments 清空；
- epoch、frame、rolling reward 清零；
- environment state 清空；
- value normalization 置为均值 0、方差 1、count 1；
- scalar value head 使用固定随机种子重新初始化；
- fixed sigma 保留自统一 PPO 配置。

RL-Games 当前使用共享特征网络，BC actor MLP 也作为 critic 的输入特征，但 value 输出头是随机初始化的。这意味着在线训练开始时 actor 已较强，而 critic 仍处于 cold start。

## 7 PPO 微调目标

PPO 使用 clipped surrogate objective。记概率比为

\[
r_t(\theta)=
\frac{\pi_\theta(a_t|o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|o_t)},
\]

优势估计为 \(\hat A_t\)，则 actor 目标为

\[
\mathcal{L}_{\mathrm{clip}}=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat A_t,
 \operatorname{clip}(r_t(\theta),1-0.2,1+0.2)\hat A_t
\right)
\right].
\]

当前配置还包含 critic loss、动作边界损失和梯度裁剪。主要参数为：

| 参数 | 数值 |
|---|---:|
| \(\gamma\) | 0.99 |
| GAE \(\lambda\) (`tau`) | 0.95 |
| 初始 learning rate | \(10^{-4}\) |
| PPO clip | 0.2 |
| horizon length | 24 |
| minibatch size | 384 |
| mini epochs | 5 |
| critic coefficient | 2 |
| entropy coefficient | 0 |
| fixed sigma | True |

PPO-from-scratch 和 BC+PPO 使用同一任务、训练种子、并行环境数、PPO 配置、交互预算、checkpoint 选择规则和正式评估器。主实验唯一差异是 actor 和输入归一化的初始化来源。

## 8 实验设计

### 8.1 数据与训练规模

专家数据集包含 3976 个成功 episode 和 540321 条 transition，采集 seeds 为 42、43、44。BC 正式训练使用 seed 42。

PPO 对比实验统一使用：

- 256 个并行环境；
- 每个 epoch 交互 \(256\times24=6144\) steps；
- 最多 200 epochs；
- 每个训练 seed 最多 1,228,800 online environment steps；
- 训练 seeds 42、43、44。

### 8.2 闭环评估协议

每个正式方法使用 seeds 42、43、44，每个 seed 评估 256 个完整 episode，共 768 个 episode。聚合稳定降落率定义为

\[
R_{\mathrm{settle}}=
\frac{N_{\mathrm{settled}}}{N_{\mathrm{episodes}}}.
\]

同时报告 contact success、hard contact、ground crash、deck miss、timeout、首次接触位置误差和 touchdown distance。离线 action MSE 只用于检查监督拟合，不作为闭环成功依据。

## 9 实验结果

### 9.1 离线拟合

BC 最优 epoch 为 50，test split weighted action MSE 为

\[
1.66546\times10^{-4}.
\]

四个动作维度的 MSE 分别为

\[
[4.47829,\ 0.699266,\ 0.538782,\ 0.945484]\times10^{-4}.
\]

### 9.2 正式闭环结果

| 方法 | Settled landing | Contact success | Hard contact | Deck miss | Timeout |
|---|---:|---:|---:|---:|---:|
| Frozen PPO teacher | **94.66% ± 0.49%** | 99.87% | 0.13% | 5.34% | 0.00% |
| PPO from scratch | **0.91% ± 0.66%** | 10.55% | 5.73% | 21.61% | 73.57% |
| BC only | **88.28% ± 0.55%** | 99.74% | 0.26% | 11.72% | 0.00% |
| BC initialized PPO | **76.69% ± 9.16%** | 95.05% | 0.91% | 23.05% | 0.00% |

BC-only 超过预设的 80% 闭环验收线，并显著优于相同在线预算内的 PPO-from-scratch。BC+PPO 未达到 92% 目标，也未超过 BC-only。

### 9.3 成功回合精度

| 方法 | First-contact xy mean | Touchdown distance mean |
|---|---:|---:|
| Teacher | 0.0559 m | 0.0576 m |
| PPO scratch | 0.0967 m | 0.0979 m |
| BC only | 0.0561 m | 0.0561 m |
| BC+PPO | 0.0650 m | 0.0683 m |

BC 成功回合的落点精度接近 teacher，但总体 deck miss 更高，说明主要问题不是已成功轨迹内的局部控制精度，而是偏离专家状态分布后的恢复能力。

## 10 讨论

### 10.1 行为克隆为何有效

BC 直接学习 teacher 在完整 approach-align-descent-contact 链路中的动作映射，绕过了随机策略在稀疏稳定降落事件下的早期探索难题。冻结 teacher 的归一化统计和相同 actor 结构还消除了输入尺度和网络容量不一致带来的额外误差。

### 10.2 BC 与 teacher 的剩余差距

BC 只拟合训练分布中的单步动作。闭环执行时，小误差会改变后续状态分布，策略可能进入数据集中覆盖较少的区域。由于正式 demonstrations 仅保留成功回合，off-distribution recovery 样本不足，因此 contact success 很高，但 deck miss 仍高于 teacher。

### 10.3 BC+PPO 退化原因

以下原因是基于代码结构和实验现象的工程推断，而不是已被单独消融完全证明的因果结论：

1. value head 随机初始化，早期 advantage 估计噪声可能较大；
2. PPO 优化的是 shaped reward，而正式选择指标是 settled landing，两者并非严格同构；
3. 公共训练循环按 rolling mean episode reward 保存 best checkpoint，不能保证选择 settled landing 最优模型；
4. actor 与 critic 共享特征，critic 更新可能改变 BC 已学到的 actor 表征；
5. 成功 demonstrations 缺少接触失败、偏航和越界后的恢复数据。

一次将初始 learning rate 从 \(10^{-4}\) 降至 \(10^{-5}\) 的诊断重跑得到 63.67% ± 18.02%，未解决退化问题，因此该结果作为负实验保留。

### 10.4 后续可验证改进

后续方案应作为新实验独立验证，不能写成当前已完成结果：

- 使用独立 validation rollout 按 settled landing 选择 checkpoint；
- critic warm-up 期间冻结或限制 actor 更新；
- 对 actor 引入与 BC 权重的 KL/L2 保持项；
- 收集 DAgger 风格纠偏和失败恢复数据；
- 对 contact/settle 和 deck miss 场景进行定向重采样；
- 将 actor 与 critic 改为 separate network，降低共享特征干扰。

## 11 局限性

1. 当前策略是 state-based，不是视觉端到端降落；
2. 甲板运动仅包含 xy 匀速、正弦 heave、最大 ±5° roll/pitch；
3. 不包含 yaw、随机波谱、水动力或真实船舶完整六自由度运动；
4. demonstrations 仅保留成功 episode，恢复行为覆盖不足；
5. scratch 结论仅适用于当前 1.23M steps/seed 预算；
6. 当前保存的是数值 rollout 轨迹，不等同于人工 GUI 目视验收。

## 12 结论

本文在冻结实体运动甲板环境上完成了可复现的 PPO teacher、专家数据集、Behavior Cloning 和 BC 初始化 PPO 对比。纯 BC 在不进行任何在线 PPO 更新时达到 88.28% 的稳定降落率，证明成功专家轨迹可以有效迁移复杂接触降落行为；但直接 PPO 微调导致性能下降，说明强 actor 与冷启动 critic 的联合更新、reward 与最终指标不一致以及恢复数据不足是下一阶段需要重点验证的问题。

## 13 实现可追溯性

| 理论/实验内容 | 代码或证据来源 |
|---|---|
| 甲板绝对时间运动、角速度和接触点运动学 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck_attitude/quadrotor_ship_landing_physical_deck_attitude_env.py` |
| 推力/力矩动作映射及基础奖励 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing/quadrotor_ship_landing_env.py` |
| 实体接触与稳定降落阈值 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck/quadrotor_ship_landing_physical_deck_env.py` |
| 数据 schema、episode split、阶段权重 | `source/quadcopter_waypoint/quadcopter_waypoint/imitation/dataset.py` |
| BC 网络与归一化 | `source/quadcopter_waypoint/quadcopter_waypoint/imitation/policy.py` |
| BC 加权 MSE 训练 | `scripts/imitation/train_bc.py` |
| BC 到 RL-Games checkpoint 迁移 | `source/quadcopter_waypoint/quadcopter_waypoint/imitation/checkpoint.py` |
| PPO 参数 | `source/quadcopter_waypoint/quadcopter_waypoint/tasks/direct/quadrotor_ship_landing_physical_deck_attitude/agents/rl_games_ppo_cfg.yaml` |
| 正式数据与结果 | `benchmarks/phase7_imitation_hybrid/summary.json` |
| 完整复现实验命令 | `benchmarks/phase7_imitation_hybrid/commands.txt` |

本文顶部 `CODE_SYNC` 块由 `tests/test_p7_documentation_sync.py` 与实际代码参数进行一致性检查。后续修改网络、动作缩放、甲板运动范围、落地阈值或 PPO 配置时，测试会提示同步更新本文。

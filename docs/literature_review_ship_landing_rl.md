# 面向波浪扰动运动甲板的四旋翼自主降落：强化学习、视觉感知与 Sim-to-Real 研究综述

> 文档定位：面向本项目 `quadcopter_waypoint` 的毕业论文文献综述/Related Work 初稿。  
> 检索与整理日期：2026-08-11。  
> 当前项目对应任务：`Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0`。  
> 当前项目阶段：actor-preserving PPO 已完成正式 validation、独立 test 与可复现实验包。  
> 注意：本文将正式发表论文与 arXiv 预印本/在审工作明确区分，后续正式论文引用时应再次核对出版状态。

## 摘要

无人机在海上运动平台上的自主降落同时涉及目标相对运动估计、飞行器轨迹与姿态控制、降落时机决策、动态接触安全以及仿真到实机迁移等问题。与地面匀速移动平台相比，船舶甲板受到波浪和风扰影响，会出现平移、升沉以及横滚、俯仰等多自由度运动，使得无人机必须在有限甲板区域内同时满足位置、速度、姿态和触地冲击约束。传统方法通常采用状态估计、轨迹预测、模型预测控制或鲁棒控制构成模块化系统，具有较强可解释性；强化学习方法则通过与环境交互学习复杂状态到动作或高层决策映射，在未知平台运动、强扰动以及高维耦合条件下展现出潜力。近年来的研究进一步从普通移动平台扩展到真实海上场景，并逐步引入波浪谱建模、六自由度甲板运动、视觉相对定位、域随机化和 Sim-to-Real。

本文围绕运动甲板自主降落的模型方法、强化学习方法、海上船舶场景、视觉感知、Sim-to-Real、模仿学习及策略保守微调展开综述，并结合本项目基于 Isaac Lab 的状态策略进行分析。本项目当前采用 22 维状态观测和 4 维连续动作，在具有实体碰撞、升沉及小幅横滚/俯仰的运动甲板环境中完成 PPO 专家策略、行为克隆以及 actor-preserving PPO。actor-preserving PPO 独立测试的稳定降落率达到 96.74%，说明当前状态策略在冻结任务分布内已经具有较高性能。与现有研究相比，后续真正有研究价值的增量并非继续在同一分布上追求更高成功率，而是提高运动模型真实性与感知不确定性，逐步引入真实海况、风扰、延迟、视觉估计和 Sim-to-Real，并使用模型方法作为可靠 baseline。综述最终给出与当前工程直接对应的文献优先级和可复用开源项目，为后续毕业论文、实验设计与系统扩展提供依据。

**关键词：** 四旋翼；自主降落；运动平台；船舶甲板；强化学习；PPO；模仿学习；视觉定位；Sim-to-Real；Isaac Lab

---

## 1. 研究背景与问题定义

多旋翼无人机具有垂直起降、悬停和高机动性等特点，但续航能力仍显著受限。若无人机能够在车辆、无人船或大型舰船等移动平台上自主回收、补能和再次起飞，其任务半径与持续作业能力可以明显提升。因此，移动平台降落长期以来是无人机自主系统的重要研究方向。

普通静态着陆主要要求无人机将位置和速度收敛到固定目标附近；运动平台降落则需要跟踪不断变化的参考状态。对海上平台而言，问题进一步复杂化。船舶的状态可表示为

\[
\mathbf{x}_D = [\mathbf{p}_D,\mathbf{R}_D,\mathbf{v}_D,\boldsymbol{\omega}_D],
\]

其中位置、姿态、线速度和角速度均随时间变化。理想的降落控制器不能只最小化无人机和甲板中心之间的位置误差，而应同时考虑甲板表面接触点处的相对速度。若甲板上的候选接触点为 \(\mathbf{p}_S\)，则其刚体速度为

\[
\mathbf{v}_S = \mathbf{v}_D + \boldsymbol{\omega}_D \times (\mathbf{p}_S-\mathbf{p}_D).
\]

因此，即使甲板中心线速度较小，横滚或俯仰仍可能使远离旋转中心的局部接触点产生较大的瞬时速度。这说明海上降落实质上是一个兼具相对运动跟踪、时机选择和接触安全约束的连续决策问题。

从方法上，现有工作大致可以分为四类：

1. **模型驱动方法**：状态估计 + 轨迹预测/规划 + PID、滑模、MPC/NMPC 等控制；
2. **端到端或连续控制强化学习**：策略直接由相对状态产生推力、姿态、速度或其他控制量；
3. **分层强化学习**：强化学习只决定降落时机或高层参考，低层稳定仍由传统控制器负责；
4. **视觉—控制联合方法**：策略从视觉特征或视觉估计状态中完成移动甲板跟踪与降落。

对于实际系统，以上方法并非互斥。近年的高质量工作反而越来越倾向于混合架构，即保留可靠的低层飞控，同时使用学习方法处理复杂、不确定且难以显式建模的高层决策。

---

## 2. 模型驱动的运动平台降落

### 2.1 视觉估计、在线规划与鲁棒控制

Paris、Lopez 和 How 在 ICRA 2020 工作 *Dynamic Landing of an Autonomous Quadrotor on a Moving Platform in Turbulent Wind Conditions* 中构建了一套完整的动态降落系统。该方法使用扩展卡尔曼滤波估计移动平台的位置、姿态和速度，在远距离阶段依赖 GPS 类测量，在近距离阶段切换至视觉标志物；在线规划采用 receding-horizon 方式，控制器则采用具有扰动鲁棒性的 boundary-layer sliding control。与大量“先悬停到平台上方再垂直下降”的方案不同，该工作强调快速、直接的动态降落，并显式考虑湍流扰动。

这类方法的优势在于系统结构清晰、稳定性分析较强，并可以显式加入控制约束。然而，当平台运动模型、空气动力学或风场误差显著增加时，预测精度和优化实时性会成为关键瓶颈。对于本项目，该工作非常适合作为“为什么 RL 不是唯一方案”的传统方法基线文献。后续若进行算法对比，至少应设计固定下降/PID 或 MPC 类 baseline，而不能只在不同 PPO 变体之间比较。

### 2.2 ArUco 相对定位与自主状态机

Wang 等人在 *Quadrotor Autonomous Landing on Moving Platform* 中提出一个包含 ArUco 视觉定位、局部轨迹规划以及自主状态机的完整系统。其关键价值在于将视觉感知与飞行控制清晰解耦：相机首先通过 ArUco 得到无人机相对于降落平台的位姿，再由规划和控制模块完成跟踪和着陆。作者同时进行了仿真、室内和室外实验。

这一思路与本项目后续计划非常契合。当前策略是纯 state-based policy，因此第一版视觉扩展没有必要把原始 RGB 图像直接输入 PPO。更合理的工程路线是

\[
\text{multi-ArUco image}
\rightarrow
\text{relative pose/velocity estimator}
\rightarrow
\text{22D-compatible state}
\rightarrow
\pi_{\theta}.
\]

这样既能保持现有策略和训练基础，也可以单独评估视觉测量噪声、丢帧和延迟对闭环降落性能的影响。

### 2.3 NMPC 与安全约束

Batool 等人在 2025 年提出 *NMPC-Lander: Nonlinear MPC with Barrier Function for UAV Landing on a Mobile Platform*，将非线性模型预测控制与控制屏障函数结合，用于静态和移动平台的精确、安全着陆，并进行了真实硬件实验。该工作说明，在模型和状态估计充分可靠的条件下，NMPC 仍然是动态降落非常有竞争力的方案。

对本项目而言，NMPC-Lander 的意义主要有两点。第一，它可以作为强化学习方案的强 baseline；第二，它提示后续评价指标应超越“是否成功”，增加最终位置误差、相对速度、触地冲击、控制平滑性和安全约束违背次数等指标。仅报告 success rate 无法充分说明学习策略相对于模型方法的优势。

---

## 3. 强化学习在移动平台降落中的发展

### 3.1 早期深度强化学习连续降落

Rodriguez-Ramos 等人的 *A Deep Reinforcement Learning Strategy for UAV Autonomous Landing on a Moving Platform* 是这一方向较有代表性的早期工作。论文使用 DDPG 处理连续状态和连续动作，并构建 Gazebo 强化学习框架，将策略从仿真部署到真实飞行。该研究的重要历史意义在于证明了深度强化学习可以学习完整的连续移动平台降落机动，而不只是离散的“左/右/下降”决策。

但早期 DDPG 方法通常对超参数、探索噪声和训练稳定性较敏感。随着 PPO、SAC 等算法以及高并行 GPU 仿真工具的发展，当前研究更关注高样本吞吐、鲁棒性和 Sim-to-Real，而不是单纯证明“RL 能否完成移动平台降落”。本项目使用 Isaac Lab + PPO，采用了这一技术演进中较成熟的并行训练路线。

### 3.2 基于运动学结构的 curriculum 与真实部署

Goldschmid 和 Ahmad 的 *Reinforcement Learning based Autonomous Multi-Rotor Landing on Moving Platforms* 后续发表于 *Autonomous Robots*。该工作没有简单地将问题交给大网络端到端学习，而是利用移动平台运动学结构设计状态空间离散方式，并通过 sequential curriculum 和 transfer learning 将简单任务逐步扩展到更困难场景。作者还在真实硬件上完成了部署，并公开了完整代码。

它对本项目最重要的启示不是具体使用 Double Q-Learning，而是**课程式任务增长**。本项目从早期静态/移动目标逐渐发展到升沉甲板、实体碰撞、roll/pitch 甲板，再到 BC 与 PPO 微调，本质上已经形成一条 curriculum。毕业论文中应将这种阶段化设计明确写出，而不是把 heave-precision task、physical-deck task、physical-deck-attitude task 当作互不相关的工程版本。

### 3.3 部分可观测条件下的记忆策略

Aikins、Jagtap 和 Nguyen 在 2024 年的 *A Robust Strategy for UAV Autonomous Landing on a Moving Platform under Partial Observability* 中将问题建模为部分可观测场景，并使用包含 LSTM 的强化学习策略处理传感器 flicker、噪声和缺失测量。其实验基于 NVIDIA Isaac Gym，并比较了 PPO 与经典控制方法。

这一工作对本项目的后续研究非常关键。目前 22 维状态由仿真直接提供，马尔可夫性较强；一旦换成 ArUco 或真实视觉，相对位姿会存在噪声、延迟、遮挡和偶发丢失。此时仅对当前帧 observation 加高斯噪声只是最低层次的随机化，更真实的问题会逐渐变成 POMDP。是否需要 frame stack、状态估计器或 recurrent policy，可以通过该方向的文献与实验进一步判断。

---

## 4. 面向船舶与波浪扰动平台的强化学习

普通地面移动平台往往只考虑二维平移或已知轨迹，而海上平台受到波浪激励后存在明显的 heave、roll、pitch，甚至完整 6-DoF 运动。因此，真正与本项目毕业课题最接近的是以下几项海上降落研究。

### 4.1 Robust RL for Vision-based Ship Landing

Saj 等人在 *Robust Reinforcement Learning Algorithm for Vision-based Ship Landing of UAVs* 中研究了单目视觉条件下的 VTOL 无人机船舶降落。其船舶平台具有六自由度甲板运动，并考虑风阵等对抗性环境条件。系统首先从单目图像估计 UAV 相对船舶参考结构的位置，再通过鲁棒强化学习策略完成控制。作者在 Gazebo 和真实 Parrot ANAFI + 缩比船舶平台上进行了实验。

这篇工作最值得本项目借鉴的是**域随机化的对象**。真实迁移误差不只来自甲板运动参数，而是同时来自：

- 船舶运动和波浪条件变化；
- 风扰与近甲板气动效应；
- 视觉定位噪声；
- 测量和控制延迟；
- 无人机质量、惯量和执行器误差；
- 接触模型与真实起落架差异。

因此，本项目若进入 Sim-to-Real 阶段，domain randomization 不应仅随机正弦振幅和频率。

### 4.2 Offshore Docking：PPO、JONSWAP 与 Sim-to-Real

Ali、Gupta 和 Hashim 在 *Applied Soft Computing* 2024 发表 *Deep Reinforcement Learning for Sim-to-Real Policy Transfer of VTOL-UAVs Offshore Docking Operations*。该方法将任务拆成 approach 和 landing 两阶段：靠近阶段使用模型控制，最终 docking 阶段使用 DRL，并比较 DQN 与 PPO。论文使用 JONSWAP（Joint North Sea Wave Project）谱为每个 episode 创建随机波浪条件，以提升泛化和 Sim-to-Real 能力，同时公开了包含 PPO 和 DQN 的代码。

这项工作与本项目当前“正弦升沉 + roll/pitch”环境具有直接递进关系。当前正弦模型适合构造可控 benchmark，但它不能充分代表真实随机海况。后续可以保留 physical-deck-attitude task 作为 deterministic/synthetic benchmark，再新增一个不破坏旧实验合同的海况扩展任务：

\[
\text{JONSWAP sea state}
\rightarrow
\text{wave elevation / vessel response}
\rightarrow
\{z_D,\phi_D,\theta_D\}
\rightarrow
\text{landing task}.
\]

这样可以保持 physical-deck-attitude task/actor-preserving PPO 可复现性，又能形成新的研究增量。

### 4.3 WaveLander：将 RL 集中于降落时机决策

2026 年预印本 *WaveLander: A Generalizable Hierarchical Control Framework for UAV Landing on Wave-Disturbed Platforms via Reinforcement Learning* 与本项目课题高度相关。需要注意，截至 2026-08-11，该论文处于投稿/在审状态，其官方 GitHub 也标注代码尚待公开，不能按已正式发表论文描述。

WaveLander 的核心思想是分层控制：传统低层飞控负责姿态稳定、横向和速度跟踪；RL 策略只接收紧凑的平台相对状态，例如相对高度、垂直速度、平台倾角及倾角速度，并输出标量垂向速度参考。策略由此学习“何时下降、何时保持、何时撤回”，将复杂海上动态降落转化为低维、时间敏感的决策问题。

它与本项目形成非常有价值的结构对照：

- 本项目当前 actor 输出 4 维连续控制动作，更接近低层连续控制；
- WaveLander 将 RL 限制在高层垂向 landing decision；
- 本项目强调实体碰撞后的 settled landing；
- WaveLander 强调波浪扰动下的 touchdown timing 与层级可部署性。

因此，WaveLander 不应被简单“照搬”，而应作为后续研究中**分层 RL baseline/alternative architecture**。特别是在实机阶段，如果直接输出总推力和力矩的策略难以迁移，改为 RL 输出速度参考、PX4/传统飞控负责低层控制，是明显更低风险的路线。

### 4.4 Vision-Based Agile Landing on Turbulent Waters

Angelis、Bauersfeld、Scaramuzza 和 Boukas 在 2026 年预印本 *Vision-Based Agile Landing on Turbulent Waters* 中进一步研究了不显式依赖平台状态的海上动态降落。策略使用多旋翼自身状态和来自降落表面的局部视觉特征（keypoints 与 descriptors）输出姿态和推力命令，并由常规低层控制器跟踪。作者报告了在逼真仿真和大量真实降落实验中的结果，并在“Very Rough”级别的平台运动条件下与 MPC baseline 进行了比较。

其真正重要之处在于研究路线从

\[
\text{ground-truth platform state}
\]

发展为

\[
\text{explicit visual pose estimate}
\]

再进一步发展为

\[
\text{visual features} \rightarrow \text{policy}.
\]

本项目当前处于第一层。对毕业设计而言，没有必要立即跨到第三层；采用 multi-ArUco 完成第二层，会更符合工程进度和可解释性。但该工作可以作为论文展望中的前沿参考，说明最终可以弱化对显式相对位姿估计的依赖。

---

## 5. Sim-to-Real 与无人机强化学习基础设施

### 5.1 OmniDrones

OmniDrones 是面向多旋翼强化学习的 Isaac Sim 开源平台，提供多种无人机、传感器、控制模式和 benchmark，并强调 GPU 并行训练。其设计与 Isaac Lab/Orbit 有较强联系，因此对本项目在任务组织、批量环境、控制接口和传感器扩展方面具有参考价值。

但其官方仓库已经明确说明当前版本较难继续维护，主版本基于 Isaac Sim 4.1.0，而本项目运行在较新的 Isaac Lab/Isaac Sim 环境。因此，不建议将当前工程迁移到 OmniDrones；更合理的方式是只阅读其环境组织、无人机建模、控制抽象和视觉任务实现。

### 5.2 Aerial Gym Simulator

Aerial Gym Simulator 是 NTNU Autonomous Robots Lab 开源的高并行无人机学习仿真平台，提供多种多旋翼模型、GPU 几何控制器、高速 depth/segmentation 传感器与 RL 接口。其 2025 年框架论文发表于 IEEE Robotics and Automation Letters。当前主线仍以 Isaac Gym 为基础，官方说明 Isaac Lab/Isaac Sim 支持处于开发过程中。

对于本项目，它最有价值的不是替换 Isaac Lab，而是两个方面：

1. 参考其在 GPU 上实现无人机动力学和低层几何控制的分层方式；
2. 参考其大规模并行视觉传感和 domain randomization 设计。

### 5.3 SimpleFlight 与系统辨识/选择性域随机化

`thu-uav/SimpleFlight` 对应的研究关注“哪些因素真正决定零样本 Sim-to-Real 四旋翼 RL 控制”。其整体思路是先做系统辨识，再进行选择性 domain randomization，最终训练可部署的低层策略。这类工作说明，Sim-to-Real 不应等价于“把所有参数都大范围随机化”；过度随机化可能增加训练难度甚至降低策略精度。更合理的是先根据真实无人机确定主要不确定项，再有针对性地设计随机化范围。

这一原则对于本项目以后从 Isaac Lab 迁移到 PX4/真实四旋翼非常重要。

---

## 6. 模仿学习、PPO 微调与策略保持

本项目 imitation-learning benchmark/checkpoint-selection analysis/actor-preserving PPO 形成了一个与传统“从零 PPO”不同的研究问题：已经存在高质量 teacher 和成功轨迹后，如何利用 expert data，同时避免在线 PPO 微调破坏已有策略。

### 6.1 Behavior Cloning 的优势与局限

Behavior Cloning 通过最小化专家状态动作对上的监督误差学习策略，训练稳定且不需要在线交互，但存在经典的 distribution shift/covariate shift 问题：部署后一个小动作误差可能使系统进入专家数据未覆盖的状态，随后误差继续累积。对运动甲板降落而言，这一问题在最后接触阶段尤其突出，因为厘米级位置误差或较小的相对速度偏差都可能改变接触结果。

本项目 imitation-learning benchmark 的结果也体现了这一特征：BC-only 可以获得较高稳定降落率，但仍低于冻结 teacher；普通 BC 初始化 PPO 又会出现明显 policy drift，说明“先 BC 再 RL”并不会天然带来提升。

### 6.2 Kickstarting 与教师策略约束

Schmitt 等人的 *Kickstarting Deep Reinforcement Learning* 使用已有 teacher 对 student 的训练进行策略蒸馏式约束，并允许 student 随训练逐渐摆脱 teacher、最终超过 teacher。虽然该工作并非无人机降落，但其思想与 actor-preserving PPO 的 actor anchor 非常接近：在线学习过程中不能只优化环境回报，还需要显式约束策略偏离已有高质量行为的速度。

### 6.3 Actor-Critic Pretraining for PPO

Kernbach 等人在 2026 年预印本 *Actor-Critic Pretraining for Proximal Policy Optimization* 中指出，许多 expert-data + PPO 方法只预训练 actor，而忽略 critic 初始化；他们进一步同时预训练 actor 和 critic，并在多个机器人任务中报告更好的样本效率。

这与本项目 actor-preserving PPO 的经验形成有趣对照。actor-preserving PPO 没有直接使用专家数据训练 critic，而是采用 separate actor/critic、critic-only warm-up、冻结 observation RMS 和 BC mean-action L2 anchor，使 critic 在不改变 actor 的前提下先适应当前任务，再开放 PPO actor 更新。两者都说明：**高质量 actor + 冷启动 critic 的直接 PPO 微调可能并不稳定，critic 初始化/适应过程值得单独处理。**

因此，actor-preserving PPO 不应只被描述为“调 PPO 参数”，而可以放在更一般的 expert-initialized actor-critic optimization 问题下讨论。

---

## 7. 代表性工作横向比较

| 工作 | 年份/状态 | 平台运动 | 感知 | 学习/控制方法 | 实机 | 开源 | 对本项目价值 |
|---|---|---|---|---|---|---|---|
| Rodriguez-Ramos et al. | 2019，期刊 | 地面移动平台 | 相对状态/系统框架 | DDPG | 是 | 是 | 早期 DRL 连续降落代表作 |
| Paris et al. | 2020，ICRA | 移动平台 + 湍流 | EKF + visual fiducial | 在线规划 + sliding control | 是 | 部分 | 强传统 baseline |
| Wang et al. | 2022，预印本/系统论文 | 移动平台 | ArUco | 局部规划 + FSM | 是 | 以论文系统为主 | multi-ArUco 路线直接参考 |
| Saj et al. | 2022，论文 | 6-DoF 船舶 | 单目视觉 | Robust RL | 是 | - | 船舶、风扰、视觉、域随机化 |
| Goldschmid & Ahmad | 2024，Autonomous Robots | 2D 移动平台 | 状态 | Double Q + curriculum | 是 | **是** | curriculum、评估、实机代码 |
| Ali et al. | 2024，Applied Soft Computing | offshore wave docking | 状态 | PPO / DQN + 分阶段控制 | 仿真/Sim2Real 导向 | **是** | JONSWAP、PPO、海况随机化 |
| Aikins et al. | 2024，Drones | 移动平台 | noisy/missing state | RPO-LSTM/PPO | 仿真 | - | POMDP、丢帧/噪声 |
| NMPC-Lander | 2025，预印本 | mobile platform | 状态 | NMPC + CBF | 是 | - | 强模型 baseline、安全约束 |
| WaveLander | 2026，在审预印本 | wave-disturbed marine platform | compact relative state | hierarchical RL | 仿真/SITL/代表性实测 | 仓库已建，代码待放出 | 与课题高度同类；高层降落时机 |
| Vision-Based Agile Landing | 2026，预印本 | turbulent maritime platform | local visual features | RL + low-level controller | **是，300+ trials 报告** | 需关注作者发布 | 最前沿视觉海上降落参考 |
| **本项目 actor-preserving PPO** | 2026，当前工程 | XY + heave + roll/pitch 实体甲板 | **22D 仿真状态** | BC + actor-preserving PPO | 否 | 当前项目 | 接触稳定、策略保持、可复现实验 |

该表说明，本项目在“冻结任务分布内的 state-based landing policy”方面已经达到很强的闭环成功率，但与最先进海上自主降落研究相比仍有两个明显缺口：**环境真实性**和**感知真实性**。这两个方向比继续微调同一 PPO 的收益更大。

---

## 8. 开源项目评估与可复用内容

### 8.1 `robot-perception-group/rl_multi_rotor_landing`

**推荐级别：S，建议完整阅读。**

优势：

- 对应正式发表的 *Autonomous Robots* 论文；
- 包含仿真、训练、真实硬件、GCS 和实验分析；
- 可观察完整科研项目如何从算法走向实机；
- curriculum 和性能统计设计值得直接参考。

不建议直接迁移其 RL 算法，因为其方法体系和当前 Isaac Lab + PPO 差异较大。应主要借鉴实验组织、课程学习、状态构造和实机评估协议。

### 8.2 `phoenixrider12/drone_docking`

**推荐级别：S，建议重点阅读 PPO 分支和波浪建模。**

优势：

- 与 offshore docking 直接相关；
- 论文发表于 *Applied Soft Computing*；
- 提供 DQN 和 PPO 两套代码；
- JONSWAP 海况生成与本项目下一步高度相关。

建议重点提取其海浪谱、episode 随机化及 PPO 输入输出设计，不建议整体替换当前训练框架。

### 8.3 `JackyLi-HKUST/WaveLander`

**推荐级别：S-，持续跟踪。**

截至 2026-08-11 官方 README 明确写明 `Code release coming soon`。目前适合作为架构和最新相关工作参考，暂时不具备代码复用价值。待公开后优先检查：

- observation 定义；
- wave-platform 生成方式；
- attitude-aware reward；
- vertical decision action；
- MuJoCo → Isaac Sim SITL 接口。

### 8.4 `alejodosr/drl-landing`

**推荐级别：A-，经典代码阅读。**

这是 Rodriguez-Ramos 等早期 DRL moving-platform landing 工作对应的代码。由于依赖 Gazebo/ROS 等旧栈，不建议迁移到本项目，但可用于理解早期 moving-platform RL 的 observation/action/reward 和仿真—实机工作流。

### 8.5 `btx0424/OmniDrones`

**推荐级别：A，参考架构，不迁移。**

最值得看：

- 多旋翼动力学与控制抽象；
- GPU parallel env 组织；
- sensor/task 接口；
- state-based 与 vision-based RL 任务如何统一。

由于版本与维护状态原因，不建议把当前 Isaac Lab 5.x 项目搬到 OmniDrones。

### 8.6 `ntnu-arl/aerial_gym_simulator`

**推荐级别：A，重点看控制器、传感器和随机化。**

尤其值得借鉴其 GPU 几何控制器和高速传感器设计。若后续将本项目动作从底层总推力/力矩改为高层 body-rate/thrust 或 velocity reference，该项目的分层控制接口具有参考价值。

### 8.7 `thu-uav/SimpleFlight`

**推荐级别：A，实机前必读。**

重点不是 landing task，而是 quadrotor RL 的系统辨识、控制接口和 selective domain randomization。进入真实无人机部署之前，其价值可能高于继续寻找新的移动平台 demo。

---

## 9. 本项目在现有研究中的定位

当前仓库已经形成以下技术链条：

\[
\text{physical moving deck}
\rightarrow
\text{PPO teacher}
\rightarrow
\text{successful expert trajectories}
\rightarrow
\text{BC}
\rightarrow
\text{ordinary PPO drift diagnosis}
\rightarrow
\text{actor-preserving PPO}.
\]

当前主任务采用：

- 22 维 state observation；
- 4 维 continuous action；
- 64-64 ELU actor；
- 实体甲板 ContactSensor；
- XY 平移 + heave + roll/pitch；
- 真实碰撞后的 settled landing 判定；
- teacher、BC、PPO scratch、普通 BC+PPO、checkpoint-selection analysis、actor-preserving PPO 的独立实验链。

actor-preserving PPO metric-selected 在独立 formal test 上取得 96.74% settled landing，已经超过 frozen teacher 的 94.66%。因此，从科研角度继续围绕同一冻结分布反复调 PPO，使成功率从 96.7% 提升到 97% 或 98%，边际贡献有限。后续研究应该把问题从“优化器还能否提升一点”转向“策略在更真实海上条件下是否仍然成立”。

结合文献，可以将本项目的后续科学问题定义为：

> **在保留已学到的安全降落技能的前提下，如何使四旋翼策略从理想状态输入和规则周期甲板运动，逐步泛化到随机海况、测量不确定性和真实视觉观测？**

这一定义可以自然连接 actor-preserving PPO 的 actor preservation 与后续 Sim-to-Real，而不是重新开一个完全无关的新方向。

---

## 10. 文献驱动的后续研究路线

### 10.1 第一优先级：海况真实性，而不是继续改 PPO

保留 physical-deck-attitude task/actor-preserving PPO 作为 frozen benchmark，新建独立任务版本，不修改历史实验语义。新增：

- JONSWAP/PM 类随机波浪谱；
- 更丰富 heave/roll/pitch 组合；
- 可选 surge/sway/yaw；
- 不同海况等级的 train/test split；
- unseen sea-state generalization。

对应主要文献：Ali et al. 2024、WaveLander 2026、Vision-Based Agile Landing 2026。

### 10.2 第二优先级：Domain Randomization 与系统辨识

逐项随机：

- UAV mass/inertia；
- thrust coefficient / actuator response；
- wind disturbance；
- observation noise；
- latency；
- deck friction/restitution/contact parameters。

不建议一次性大范围全随机。应参考 SimpleFlight 采用“系统辨识 + 选择性随机化”，并通过消融实验确定每类随机化的实际作用。

### 10.3 第三优先级：multi-ArUco 状态感知

第一版视觉系统仍然输出显式相对状态：

\[
\text{multi-ArUco}
\rightarrow
\hat{\mathbf{p}}_{D/B},\hat{\mathbf{R}}_{D/B}
\rightarrow
\text{filter / finite difference}
\rightarrow
\hat{\mathbf{v}},\hat{\boldsymbol{\omega}}
\rightarrow
\text{policy}.
\]

训练阶段先对现有状态注入符合视觉测量统计的噪声、延迟和 dropout；再替换为真实视觉模块。这样可以隔离“控制策略问题”和“感知问题”。

对应主要文献：Wang et al. 2022、Saj et al. 2022、Aikins et al. 2024。

### 10.4 第四优先级：分层 RL 与传统低层飞控对比

当实机需要 PX4 或已有姿态控制器时，可以增加一个 WaveLander 风格 baseline：

\[
\text{RL} \rightarrow v_z^{ref}\ \text{or}\ [\mathbf{v}^{ref},\psi^{ref}]
\rightarrow
\text{PX4 / geometric controller}.
\]

与当前 4D 低层动作策略进行比较：

- sim performance；
- disturbance robustness；
- real-world transfer effort；
- safety；
- sample efficiency。

这会比仅比较 PPO 超参数更有论文贡献。

### 10.5 第五优先级：强 baseline

至少保留以下非学习方法之一：

- relative-state PID + phase/state machine；
- fixed-descent + tracking controller；
- MPC/NMPC landing controller。

然后比较：

\[
\text{classical baseline}
\leftrightarrow
\text{end-to-end PPO}
\leftrightarrow
\text{hierarchical RL}
\leftrightarrow
\text{actor-preserving / expert-initialized RL}.
\]

这样论文才能回答“强化学习究竟在哪种不确定性下产生收益”。

---

## 11. 建议优先精读顺序

如果当前只投入时间精读 8 篇，建议顺序如下：

1. **Ali et al., 2024 — Offshore Docking + PPO + JONSWAP**：与你当前下一步最直接；
2. **Saj et al., 2022 — Robust RL for Vision-based Ship Landing**：真实船舶、6-DoF、视觉和风扰；
3. **WaveLander, 2026 — Hierarchical RL for Wave-Disturbed Platforms**：与你课题最接近的最新架构；
4. **Angelis et al., 2026 — Vision-Based Agile Landing on Turbulent Waters**：当前前沿目标形态；
5. **Goldschmid & Ahmad, 2024 — RL Multi-Rotor Landing**：curriculum、真实部署和开放代码；
6. **Paris et al., 2020 — Dynamic Landing in Turbulent Wind**：传统方法强 baseline；
7. **Wang et al., 2022 — ArUco Moving Platform Landing**：后续视觉落地直接参考；
8. **Kernbach et al., 2026 — Actor-Critic Pretraining for PPO**：解释 imitation-learning benchmark/actor-preserving PPO 的策略初始化与 critic 问题。

如果准备毕业论文 related work，可以再补 Rodriguez-Ramos 2019 作为早期 DRL 代表、Aikins 2024 作为 POMDP/丢帧方向，以及 NMPC-Lander 作为近期模型方法。

---

## 12. 总结

移动平台自主降落研究正在经历明显的演进：早期工作主要解决二维移动目标上的视觉跟踪和连续降落；随后强化学习开始取代部分手工轨迹和控制策略；近年来研究重点转向海上随机运动、六自由度平台、风扰、视觉不确定性以及 Sim-to-Real。最新趋势并不是完全抛弃传统控制，而是采用更加模块化的学习控制架构，使 RL 集中处理难以建模的时机选择、非线性决策和不确定性，低层飞行稳定则继续依赖成熟控制器。

本项目目前的优势在于：已经具有 Isaac Lab 高并行训练环境、实体甲板碰撞、明确的 settled landing 指标、完整的专家数据与 BC/PPO 实验链，以及 actor-preserving PPO 的独立测试结果。这些基础使项目不需要重新从“能否降落”开始。后续最有价值的研究工作应围绕 **海况真实性、视觉状态估计、域随机化和 Sim-to-Real** 展开，并保持 physical-deck-attitude task/actor-preserving PPO 作为冻结 benchmark 进行可追溯对照。

从毕业论文叙事上，可以形成一条清晰主线：

> **从规则运动甲板上的强化学习降落出发，通过专家轨迹和 actor-preserving PPO 获得稳定策略；进一步引入随机海况与视觉感知不确定性，研究策略在波浪扰动船舶甲板上的鲁棒泛化与自主降落。**

该路线与现有高质量工作既存在充分联系，又保留了本项目自身的研究问题，不需要为了追逐最新论文而推翻当前已经完成的 moving-deck environment–actor-preserving pipeline 实验基础。

---

## 参考文献

[1] Rodriguez-Ramos, A., Sampedro, C., Bavle, H., de la Puente, P., Campoy, P. **A Deep Reinforcement Learning Strategy for UAV Autonomous Landing on a Moving Platform.** *Journal of Intelligent & Robotic Systems*, 93, 351–366, 2019. DOI: 10.1007/s10846-018-0891-8.  
论文存档：https://oa.upm.es/67140/  
代码：https://github.com/alejodosr/drl-landing

[2] Paris, A., Lopez, B. T., How, J. P. **Dynamic Landing of an Autonomous Quadrotor on a Moving Platform in Turbulent Wind Conditions.** ICRA 2020. arXiv:1909.11071.  
https://arxiv.org/abs/1909.11071

[3] Wang, P., Wang, C., Wang, J., Meng, M. Q.-H. **Quadrotor Autonomous Landing on Moving Platform.** arXiv:2208.05201, 2022.  
https://arxiv.org/abs/2208.05201

[4] Saj, V., Lee, B., Kalathil, D., Benedict, M. **Robust Reinforcement Learning Algorithm for Vision-based Ship Landing of UAVs.** arXiv:2209.08381, 2022.  
https://arxiv.org/abs/2209.08381

[5] Goldschmid, P., Ahmad, A. **Reinforcement Learning based Autonomous Multi-Rotor Landing on Moving Platforms.** *Autonomous Robots*, 2024. DOI: 10.1007/s10514-024-10162-8.  
预印本：https://arxiv.org/abs/2302.13192  
代码：https://github.com/robot-perception-group/rl_multi_rotor_landing

[6] Ali, A. M., Gupta, A., Hashim, H. A. **Deep Reinforcement Learning for Sim-to-Real Policy Transfer of VTOL-UAVs Offshore Docking Operations.** *Applied Soft Computing*, 162:111843, 2024. DOI: 10.1016/j.asoc.2024.111843.  
预印本：https://arxiv.org/abs/2406.00887  
代码：https://github.com/phoenixrider12/drone_docking

[7] Aikins, G., Jagtap, S., Nguyen, K.-D. **A Robust Strategy for UAV Autonomous Landing on a Moving Platform under Partial Observability.** *Drones*, 8(6):232, 2024. DOI: 10.3390/drones8060232.  
https://www.mdpi.com/2504-446X/8/6/232

[8] Batool, A., Batool, F., Khan, R. A., Mustafa, M. A., Fedoseev, A., Tsetserukou, D. **NMPC-Lander: Nonlinear MPC with Barrier Function for UAV Landing on a Mobile Platform.** arXiv:2505.03931, 2025.  
https://arxiv.org/abs/2505.03931

[9] Li, C.-K., Sit, I. L., Siu, M. F., Kui, K. Y., Lin, H. W., Wang, P., Shi, L. **WaveLander: A Generalizable Hierarchical Control Framework for UAV Landing on Wave-Disturbed Platforms via Reinforcement Learning.** arXiv:2607.01281, 2026. 截至 2026-08-11 为预印本/投稿在审工作。  
https://arxiv.org/abs/2607.01281  
官方仓库：https://github.com/JackyLi-HKUST/WaveLander

[10] Angelis, D., Bauersfeld, L., Scaramuzza, D., Boukas, E. **Vision-Based Agile Landing on Turbulent Waters.** arXiv:2605.23717, 2026.  
https://arxiv.org/abs/2605.23717

[11] Schmitt, S., Hudson, J. J., Zidek, A., et al. **Kickstarting Deep Reinforcement Learning.** arXiv:1803.03835, 2018.  
https://arxiv.org/abs/1803.03835

[12] Kernbach, A., Elsheikh, A., Grupp, N., Nagel, R., Huber, M. F. **Actor-Critic Pretraining for Proximal Policy Optimization.** arXiv:2602.23804, 2026.  
https://arxiv.org/abs/2602.23804

[13] Xu, B., Gao, F., Yu, C., Zhang, R., Wu, Y., Wang, Y. **OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control.** arXiv:2309.12825, 2023.  
代码：https://github.com/btx0424/OmniDrones

[14] Kulkarni, M., Rehberg, W., Alexis, K. **Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots.** *IEEE Robotics and Automation Letters*, 10(4):4093–4100, 2025. DOI: 10.1109/LRA.2025.3548507.  
代码：https://github.com/ntnu-arl/aerial_gym_simulator

[15] Chen, J., Yu, C., Xie, Y., et al. **What Matters in Learning A Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study.** 相关官方代码仓库 `thu-uav/SimpleFlight`.  
代码：https://github.com/thu-uav/SimpleFlight

---

## 开源项目快速入口

| 项目 | 地址 | 当前建议 |
|---|---|---|
| RL Multi-Rotor Landing | https://github.com/robot-perception-group/rl_multi_rotor_landing | 完整阅读，重点 curriculum/实机评估 |
| Offshore Drone Docking | https://github.com/phoenixrider12/drone_docking | 重点阅读 PPO 与 JONSWAP |
| WaveLander | https://github.com/JackyLi-HKUST/WaveLander | 跟踪代码发布 |
| DRL Landing | https://github.com/alejodosr/drl-landing | 阅读经典实现，不迁移旧栈 |
| OmniDrones | https://github.com/btx0424/OmniDrones | 参考架构与传感器设计 |
| Aerial Gym Simulator | https://github.com/ntnu-arl/aerial_gym_simulator | 参考 GPU 控制器/视觉/随机化 |
| SimpleFlight | https://github.com/thu-uav/SimpleFlight | Sim-to-Real 前重点阅读 |

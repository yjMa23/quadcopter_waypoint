# 项目文档索引

| 文档 | 用途 |
|---|---|
| `literature_review_ship_landing_rl.md` | 面向本项目的运动/船舶甲板无人机自主降落文献综述，覆盖 RL、传统 baseline、视觉、Sim-to-Real、模仿学习与高质量开源项目 |
| `literature_comparison_matrix.md` | 逐篇核对核心文献的 observation、action、reward、平台运动、控制频率、算法、成功定义、Sim-to-Real、实机结果和开源状态，并与当前 actor-preserving PPO 严格横向比较 |
| `heave_precision_theory.md` | heave-precision task 升沉甲板精确降落、代理成功语义与冻结结果 |
| `physical_deck_theory.md` | physical-deck task 实体水平甲板接触、稳定保持和失败分类 |
| `physical_deck_attitude_theory.md` | physical-deck-attitude task 倾斜实体甲板、刚体表面点运动学和正式验收 |
| `imitation_hybrid_paper.md` | imitation-learning benchmark 专家数据、BC、共享 actor/critic PPO 与负结果解释 |
| `checkpoint_selection_and_policy_drift.md` | checkpoint-selection analysis 周期 checkpoint 选模、独立测试和 policy drift 理论说明 |
| `actor_preserving_ppo.md` | actor-preserving PPO 设计、预注册、实现映射与结果回填主文档 |
| `sea_state_benchmark.md` | 独立 stochastic Sea-State benchmark：JONSWAP、surrogate vessel response、解析运动、安全 envelope、compatibility regression 与 zero-shot protocol |
| `px4_compatible_hierarchical_rl_theory.md` | Fixed-Stage PX4-compatible hierarchical RL 的历史理论与实现门禁：3D deck-relative velocity、contact-point feedforward、ENU/NED、vectorized PX4-like controller、Offboard velocity deployment 与 smoke protocol |
| `first_innovation_hierarchical_landing_plan.md` | 毕业论文第一创新点长期研究合同：Fixed-Stage baseline 冻结、3D relative velocity + continuous stage + terminal attitude guidance、S0–S15 entry/PASS/FAIL/stop gates、ablation/benchmark 与论文 claim 边界 |
| `continuous_stage_terminal_attitude_theory.md` | 第一创新点 S1 Theory Gate：continuous stage mapping/filter/envelopes/slew、contact-point kinematics、deck-heading yaw、SO(3)/quaternion terminal attitude guidance、relative angular velocity touchdown contract、Route A/B PX4 boundary 与 S2 pure-math API/tests |
| `RL_LONG_TERM_ROADMAP.md` | RL 研究长期路线：冻结基线、Sea-State 鲁棒边界、distribution-shift adaptation、感知退化、Continuous-Stage PX4-compatible 第一创新点、selective dynamics randomization、PX4/HIL 与论文正式指标 |
| `imitation_hybrid_interview_evidence.md` | imitation-learning benchmark 面试表述、数据证据、失败归因和可使用结论 |
| `../benchmarks/checkpoint_selection/README.md` | checkpoint-selection analysis 原始 benchmark、命令与图表入口 |
| `../benchmarks/px4_hierarchical_smoke/README.md` | PX4-compatible hierarchical action/controller 的 1/16-env 确定性 smoke、命令与首轮 PASS 证据；不是训练后策略结果 |
| `../benchmarks/px4_hierarchical_training/README.md` | M2 PPO 证据链：evaluator diagnostics、deterministic zero-relative-action baseline、sanity/candidate training、validation 与 checkpoint hash |
| `runtime_display_troubleshooting.md` | Isaac Sim 默认 display、GUI、SSH、tmux、Docker 和 headless 视频说明 |

## 文档与代码同步

imitation-learning benchmark 文档顶部包含 `CODE_SYNC` 参数快照；actor-preserving PPO 文档包含唯一的 `actor_preserving_preregistered_config` YAML 参数块。执行：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_imitation_documentation_sync.py \
  tests/test_actor_preserving_documentation_sync.py
```

测试会从当前源码、PPO YAML 和冻结 checkpoint-selection analysis 协议读取网络、归一化、动作维度、PPO 参数、seed 角色和评估规模，并与理论文档参数块比较。修改相关代码或正式配置时必须同步更新理论文档。

完整项目测试：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

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
| `imitation_hybrid_interview_evidence.md` | imitation-learning benchmark 面试表述、数据证据、失败归因和可使用结论 |
| `../benchmarks/checkpoint_selection/README.md` | checkpoint-selection analysis 原始 benchmark、命令与图表入口 |
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

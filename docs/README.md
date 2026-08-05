# 项目文档索引

| 文档 | 用途 |
|---|---|
| `p6a_heave_precision_theory.md` | P6A 升沉甲板精确降落、代理成功语义与冻结结果 |
| `p6b_physical_deck_theory.md` | P6B 实体水平甲板接触、稳定保持和失败分类 |
| `p6c_physical_deck_attitude_theory.md` | P6C 倾斜实体甲板、刚体表面点运动学和正式验收 |
| `p7_imitation_hybrid_paper.md` | P7 专家数据、BC、共享 actor/critic PPO 与负结果解释 |
| `p8a_checkpoint_selection_and_policy_drift.md` | P8A 周期 checkpoint 选模、独立测试和 policy drift 理论说明 |
| `p8b_actor_preserving_ppo.md` | P8B actor-preserving PPO 设计、预注册、实现映射与结果回填主文档 |
| `interview_p7_evidence.md` | P7 面试表述、数据证据、失败归因和可使用结论 |
| `../benchmarks/phase8a_checkpoint_selection/README.md` | P8A 原始 benchmark、命令与图表入口 |
| `runtime_display_troubleshooting.md` | Isaac Sim 默认 display、GUI、SSH、tmux、Docker 和 headless 视频说明 |

## 文档与代码同步

P7 文档顶部包含 `CODE_SYNC` 参数快照；P8B 文档包含唯一的 `p8b_preregistered_config` YAML 参数块。执行：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_p7_documentation_sync.py \
  tests/test_p8b_documentation_sync.py
```

测试会从当前源码、PPO YAML 和冻结 P8A 协议读取网络、归一化、动作维度、PPO 参数、seed 角色和评估规模，并与理论文档参数块比较。修改相关代码或正式配置时必须同步更新理论文档。

完整项目测试：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

# 项目文档索引

| 文档 | 用途 |
|---|---|
| `p7_imitation_hybrid_paper.md` | P7 论文式理论、公式、算法、实验与代码可追溯说明 |
| `interview_p7_evidence.md` | P7 面试表述、数据证据、失败归因和可使用结论 |
| `runtime_display_troubleshooting.md` | Isaac Sim 默认 display、GUI、SSH、tmux、Docker 和 headless 视频说明 |

## 文档与代码同步

`p7_imitation_hybrid_paper.md` 顶部包含 `CODE_SYNC` 参数快照。执行：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_p7_documentation_sync.py
```

测试会从当前源码读取 BC 网络、归一化、动作缩放、甲板范围、落地阈值和 PPO 参数，并与理论文档中的同步块比较。修改相关代码时必须同时更新理论文档。

完整项目测试：

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

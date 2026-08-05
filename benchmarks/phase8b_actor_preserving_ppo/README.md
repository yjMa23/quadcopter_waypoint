# P8B Actor-Preserving PPO 正式 Benchmark

P8B 保持冻结的 P6C 环境语义，采用 separate actor/critic、epoch 1–10 critic-only warm-up、冻结 observation RMS，以及 pilot 预先选定的 `bc_anchor_coefficient=50`。

## 冻结协议

- Training seeds：42、43、44
- Validation seeds：145、146、147；每 checkpoint/seed 128 episodes
- Formal test seeds：245、246、247；每 checkpoint/seed 256 episodes
- Formal test 未参与 checkpoint selection
- Validation-selected epochs：seed42=91、seed43=30、seed44=51
- Reward-selected epochs：seed42=91、seed43=128、seed44=51
- Last checkpoints：三个 training seed 均为 epoch200

## 正式结果

| 方法 | settled landing | deck miss | hard contact |
|---|---:|---:|---:|
| Frozen teacher | 94.66% | 5.34% | 0.13% |
| BC epoch0 | 86.20% | 13.80% | 0.00% |
| P8A metric-selected | 91.67% | 8.33% | 0.30% |
| **P8B metric-selected** | **96.74%** | **3.17%** | **0.09%** |
| P8B reward-selected | 95.40% | 4.51% | 0.13% |
| P8B epoch200 last | 92.40% | 7.55% | 0.04% |

P8B metric-selected 为 2229/2304，Wilson 95% CI 为 [95.94%, 97.40%]。达到 90%：`True`；达到 92%：`True`。

必须保留的负面结果：P8B metric-selected 的 hard contact 为 2/2304，而 BC 为 0/768；因此不能声称所有安全指标都严格改善。epoch200 last 低于 validation-selected，reward-selected 也低于 metric-selected。

## 证据索引

- `summary.json`：机器可解析总览和关键判定
- `comparison.csv` / `comparison.md`：teacher、BC、P7、P8A、P8B 对比
- `validation_results.csv` / `validation_aggregate.csv` / `validation_selection.json`：只用 validation 的选模证据
- `formal_results.csv` / `formal_aggregate.csv`：独立 test 聚合与 Wilson CI
- `prediction_verification.json`：七项预注册预测 verdict
- `policy_drift.csv` / `policy_drift.json`：actor、critic、action、RMS drift
- `failure_distribution.csv`：失败类型分布
- `checkpoint_hashes.json` / `checkpoint_inventory.json`：checkpoint、actor、critic、RMS 哈希
- `videos/video_manifest.json`：真实 `settled_landing` 与 `deck_miss` 视频/轨迹哈希
- `commands.txt`：完整复现命令

原始逐 episode CSV、checkpoint 和 TensorBoard 保留在 `logs/`。`video_generation_completed=True`，`human_review_completed=False`；自动 headless 视频不等于人工 GUI 目视验收。

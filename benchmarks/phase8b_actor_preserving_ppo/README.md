# P8B Actor-Preserving PPO Benchmark

本目录保存 P8B 的可提交小型 benchmark、配置快照、聚合表、图表与视频 manifest。大型 checkpoint、完整 TensorBoard、原始逐回合 CSV 和大视频保留在 `logs/`，这里只记录路径、哈希和生成命令。

## 当前状态

- 理论预注册提交：`05b1a95`
- warm-up scheduler 设计修正提交：`349a9f4`
- 实现与 smoke 验收提交：`6c1ebc4`
- pilot：已完成，validation seeds 145/146/147，每 seed 128 episodes
- pilot 选择：`bc_anchor_coefficient=50.0`
- formal training/test：待本阶段后续命令完成后由同一生成脚本回填

## Pilot 结论

| coefficient | 最佳 epoch | Settled landing | Deck miss | Hard contact | Touchdown distance |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 88.0208% | 11.9792% | 0.2604% | 0.05908 m |
| 10 | 20 | 79.4271% | 20.3125% | 1.3021% | 0.06609 m |
| 50 | 30 | **94.5312%** | **5.4688%** | 0.7812% | **0.05098 m** |

选择规则和逐 checkpoint 结果见 `pilot_summary.json` 与 `pilot_results.csv`。formal test seeds 245/246/247 未参与 pilot。

## 文件

- `preregistered_config.yaml`：正式实验冻结配置与 seed 角色。
- `environment_manifest.json`：生成时 Git 和运行环境摘要。
- `pilot_results.csv`：从 resumable validation manifest 自动聚合的逐 checkpoint 结果。
- `pilot_summary.json`：每个 coefficient 的最佳 checkpoint、唯一选择和规则。
- `commands.txt`：训练、validation、drift、聚合和后续正式实验命令。

所有数字由 `scripts/imitation/build_phase8b_benchmark.py` 从 raw manifest/CSV 自动生成，不手工填入聚合逻辑。

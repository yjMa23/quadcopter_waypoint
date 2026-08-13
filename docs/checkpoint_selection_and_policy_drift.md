# checkpoint-selection analysis：周期 Checkpoint 指标选模与 Policy Drift 诊断

> 状态：历史冻结说明。checkpoint-selection analysis 不重新训练、不修改 physical-deck-attitude task 环境，只对 imitation-learning benchmark 既有 checkpoint 做分层评估、去重、独立测试和离线漂移诊断。

## 1. 问题定义

imitation-learning benchmark 的 BC-only 已有较强闭环能力，但普通 BC 初始化 PPO 的 reward-selected checkpoint 明显退化。RL-Games 默认保存的“best”由 rolling mean episode reward 决定，而正式目标是实体接触后的 `settled_landing`，二者并不等价。checkpoint-selection analysis 检验：周期 checkpoint 中是否存在比 reward-selected 更好的任务指标模型，以及 actor、observation normalization 与确定性动作如何随 epoch 漂移。

## 2. 冻结任务和策略定义

任务、22 维观测、4 维动作、动力学、reward、termination、ContactSensor、hard contact、deck miss、ground crash、settle hold 和正式评估器全部继承 physical-deck-attitude task/imitation-learning benchmark。策略是 state-based `[64,64]` ELU 高斯 actor；checkpoint-selection analysis 不改变网络或 optimizer。

`training reward` 是训练窗口中的 shaped return；`contact_success` 是发生过 deck contact；`settled_landing` 是首次接触精度合格且 safe contact 连续保持；其余失败类别按 physical-deck-attitude task 定义。checkpoint-selection analysis 只用 settled landing 作为首要 checkpoint 指标，不把 reward 高解释为任务成功率高。

## 3. Screening、validation 和 independent test

冻结三层协议：

1. Screening：seed 145，每 checkpoint 64 回合，覆盖 BC epoch0、三个训练 seed 的周期 checkpoint 与 reward-selected checkpoint。
2. Validation：seeds 145/146/147，每 seed 128 回合；只评估 screening Top-K、reward-selected 和 BC。
3. Independent formal test：seeds 245/246/247，每 seed 256 回合；仅在 validation 选模完成后评估 teacher、BC、metric-selected、reward-selected。

validation seeds 参与 checkpoint 选择；formal test seeds 绝不参与 screening、排序或 tie-break。该分离防止用最终测试结果反向选择模型。

## 4. Checkpoint 去重和哈希

文件哈希定义为

\[
H_{ckpt}=\operatorname{SHA256}(\text{checkpoint file bytes}),
\]

它对 optimizer、epoch、frame、metadata 和序列化差异敏感。

checkpoint-selection analysis actor hash 对以下有序 tensor 的键名、dtype、shape 和 bytes 计算 SHA256：actor 两层 MLP、mu head、observation running mean、variance、count。因而

\[
H_{actor}=\operatorname{SHA256}(\theta_{actor},m_{obs},v_{obs},n_{obs}).
\]

`checkpoint SHA` 相同必然表示文件完全相同；`actor SHA` 相同只表示确定性 actor 与输入统计相同，checkpoint 仍可能因 critic、optimizer 或 metadata 不同而不同。周期 checkpoint 按同一 train seed、epoch、actor SHA 去重，重复文件保留 inventory 和 `duplicate_of`，不删除物理文件。

## 5. Checkpoint 选择规则

代码真实排序键为：

1. 最大化 validation settled landing；
2. 最小化 deck miss；
3. 最小化 hard contact；
4. 最小化 touchdown distance mean；
5. 选择更早 epoch。

这比早期文案中“hard contact 在 deck miss 前”更精确；本文以 `checkpoint_sweep.selection_sort_key` 为唯一真实逻辑。排序不使用 training reward 或 formal test。

## 6. Deterministic action drift

固定 observation batch 来自专家数据集 test split，共 51,901 transitions。对每个 checkpoint 使用其自身 RL-Games observation normalization，再计算 actor mean 并裁剪到 \([-1,1]\)。相对 BC 的 action drift 为

\[
D_{action}(\theta,\theta_{BC})=
\frac{1}{N}\sum_{i=1}^{N}
\|\mu_\theta(o_i)-\mu_{BC}(o_i)\|_2^2/4.
\]

源码实现是所有 batch/action 元素的均方，因此等价于上式除以动作维数 4。另报告每动作维 MSE。该量只描述固定观测集上的函数差异，不等价于闭环性能。

## 7. Actor 参数与 RMS drift

actor 相对参数漂移为

\[
D_{param}(\theta,\theta_{BC})=
\frac{\|\theta-\theta_{BC}\|_2}{\max(\|\theta_{BC}\|_2,10^{-12})},
\]

其中只含 actor MLP 和 mu head，不含 critic 与 sigma。

observation RMS drift 为

\[
D_m=\|m-m_{BC}\|_2,\qquad
D_v=\|v-v_{BC}\|_2,\qquad
D_n=n-n_{BC},
\]

并同时报告 mean/variance MSE。actor 权重 hash 不变并不必然保证 policy function 不变；若 RMS 更新，归一化输入改变，确定性动作仍可漂移。checkpoint-selection analysis actor hash 将 RMS 包含在内，另行保留参数 drift 以区分两者。

## 8. 真实结果

### 8.1 Validation 选择

三个训练 seed 的 metric-selected checkpoint 全部为 epoch 10：

| Train seed | Metric epoch | Validation settled | Reward-selected epoch |
|---:|---:|---:|---:|
| 42 | 10 | 90.1042% | 130 |
| 43 | 10 | 91.4062% | 21 |
| 44 | 10 | 90.6250% | 75 |

reward-selected actor 与 metric-selected actor 在三个 seed 上均不同。

### 8.2 Independent formal test

| 方法 | Settled landing | Contact | Hard | Deck miss | Timeout | Touchdown mean |
|---|---:|---:|---:|---:|---:|---:|
| BC epoch 0 | 86.1979% ±1.29% | 99.48% | 0.00% | 13.80% | 0.00% | 0.05876 m |
| Metric-selected BC+PPO | **91.6667% ±3.03%** | 99.35% | 0.3038% | 8.3333% | 0.00% | 0.05689 m |
| Reward-selected BC+PPO | 78.0816% ±8.66% | 96.18% | 0.9983% | 21.5712% | 0.0868% | 0.06837 m |
| Frozen PPO teacher | 94.6615% ±2.58% | 100% | 0.1302% | 5.3385% | 0.00% | 0.05848 m |

metric-selected 相对该 formal protocol 的 BC epoch0 提高 5.46875 个百分点，达到 90%，但没有达到 92%。imitation-learning benchmark 的 BC-only 88.28% 使用另一组正式 seeds；跨协议比较必须注明 seed 集，不能直接把差异归因于方法。

## 9. 漂移结果和可解释边界

最早保存 checkpoint 是 epoch 10。相对 BC 的 action MSE：seed42 `6.1230e-4`、seed43 `6.4974e-4`、seed44 `9.1777e-4`，说明到最早可见快照时策略函数已改变；它不能定位第一次 gradient update 的变化时刻。

64 个 paired snapshots 上，action drift 与 settled landing 的 Pearson 相关为 -0.4986，与 deck miss 为 +0.5080，与 hard contact 为 -0.1339。该结果属于中等统计关联，benchmark 的预注册判定没有把它称为“clear statistical association”，更不能据此证明 early drift 是退化的唯一因果原因。

## 10. 方案选择原因、候选方案和局限

checkpoint-selection analysis 选择重用既有 checkpoint，是因为可在不增加训练交互和不修改任务语义的条件下检验选模偏差。未在 checkpoint-selection analysis 修改 learning rate、网络、critic 或 reward，以避免把选模和训练方法混为一体。

局限：最早快照为 epoch10；固定 observation batch 主要来自成功 demonstration；相关性不等于因果；validation 和 formal seeds 数量有限。checkpoint-selection analysis 结论是“任务指标选模显著优于 reward-only”，不是“checkpoint 选择已稳定达到 92%”。

## 11. 理论—代码—配置—测试—证据映射

| 理论内容 | 代码位置/函数 | 配置键 | 单元测试 | 实验证据 |
|---|---|---|---|---|
| inventory 与 actor/checkpoint SHA | `source/.../imitation/checkpoint_sweep.py::inspect_checkpoint/actor_state_sha256/discover_checkpoints` | checkpoint glob、include reward | `tests/test_checkpoint_sweep.py` | `checkpoint_inventory.json` |
| resumable sweep | `scripts/imitation/evaluate_checkpoint_sweep.py` | task/seed/episodes/num_envs | 同上 resume tests | `screening_sweep_manifest.json` 等 |
| metric selection | `checkpoint_sweep.py::selection_sort_key/select_validation_best` | top-k、seed sets | tie-break test | `validation_selection.json` |
| action/parameter/RMS drift | `checkpoint_sweep.py::compute_drift_metrics`；`analyze_checkpoint_drift.py` | fixed dataset split | drift tests | `checkpoint_drift.csv/json` |
| independent test 聚合 | `scripts/imitation/build_checkpoint_selection_benchmark.py::_finalize_mode` | formal seeds 245/246/247 | aggregation tests | `formal_results.csv`, `summary.json` |
| 图表 | 同 benchmark builder | CSV inputs | regeneration smoke | `benchmarks/checkpoint_selection/*.png` |

# M2 S1 PPO Sanity Result

## Verdict

```text
S1 SANITY FAIL
```

S1 changed exactly one behavioral variable relative to S0:

```text
sigma_init.val: 0 -> -1.0
initial sigma:   1.0 -> exp(-1) ~= 0.368
```

The 64-env / seed-42 / 30-iteration run completed normally with no runtime NaN/Inf and no controller explosion. However, `rewards/iter` still degraded persistently and settled landing remained 0%, so the preregistered sanity gate is not satisfied even though deterministic landing intermediates improved.

C0 was not run.

## Reproducibility

S1 preregistration commit:

```text
21d08da experiment: calibrate M2 exploration for S1
```

Training command:

```bash
python scripts/rl_games/train.py \
  --task=Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0 \
  --num_envs=64 \
  --seed=42 \
  --headless \
  --max_iterations=30
```

Run directory:

```text
logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-17-58
```

Training completed normally in 27.36 s.

## Training trend

| Iteration | Reward | Align batch | Settled | Deck miss batch | Ref saturation | Controller tracking error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -2.7377 | 8.33% | n/a | n/a | n/a | n/a |
| 10 | -28.9095 | 2.08% | 0% | 16.67% | 0.12% | 0.1951 m/s |
| 20 | -53.9612 | 45.83% | 0% | 54.17% | 3.52% | 0.2331 m/s |
| 30 | -73.8551 | 8.33% | 0% | 37.50% | 0.30% | 0.2167 m/s |

The reward trajectory remained persistently degrading:

```text
-2.74 -> -28.91 -> -53.96 -> -73.86
```

The completed-episode training batches show no sustained align trend and no settled landing. At the same time, reference saturation stayed low and all five controller saturation ratios were 0 throughout training.

Training action statistics at the checkpoint iterations were:

| Iteration | t1 mean/std | t2 mean/std | normal mean/std |
| ---: | ---: | ---: | ---: |
| 10 | -0.0037 / 0.3743 | +0.0747 / 0.3930 | -0.1249 / 0.4021 |
| 20 | -0.1398 / 0.4246 | +0.1108 / 0.4083 | -0.0607 / 0.4683 |
| 30 | +0.0718 / 0.4101 | +0.0659 / 0.3918 | -0.0908 / 0.4798 |

## Actual PPO sigma

The checkpoint state confirms that the exploration scale stayed near the intended S1 regime instead of returning to S0 scale:

| Checkpoint | sigma parameter | exp(sigma) |
| --- | --- | --- |
| ep10 | `[-0.952354, -1.038738, -0.906445]` | `[0.385832, 0.353901, 0.403958]` |
| ep20 | `[-0.876632, -0.989035, -0.924444]` | `[0.416182, 0.371935, 0.396752]` |
| ep30 | `[-0.903659, -1.009942, -0.777285]` | `[0.405085, 0.364240, 0.459653]` |

This closes the specific S0 concern that exploration standard deviation remained of order one normalized action unit.

## Fixed-seed deterministic diagnosis

All checkpoints were evaluated with:

```text
seed = 145
num_envs = 64
episodes = 64
```

| Metric | ep10 | ep20 | ep30 |
| --- | ---: | ---: | ---: |
| align success | 7.81% | 15.62% | 31.25% |
| settled landing | 0% | 0% | 0% |
| contact success | 3.12% | 0% | 0% |
| crash | 100% | 64.06% | 51.56% |
| deck miss | 75.00% | 45.31% | 28.12% |
| hard contact | 0% | 0% | 0% |
| ground crash | 0% | 0% | 0% |
| timeout | 0% | 35.94% | 51.56% |
| residual non-deck/non-contact crash* | 25.00% | 18.75% | 23.44% |
| ref norm mean | 0.3373 m/s | 0.2031 m/s | 0.1667 m/s |
| ref saturation | 0% | 0.47% | 0% |
| controller tracking error mean | 0.0974 m/s | 0.0756 m/s | 0.0763 m/s |
| acceleration saturation | 0% | 0% | 0% |
| tilt saturation | 0% | 0% | 0% |
| thrust saturation | 0% | 0% | 0% |
| body-rate saturation | 0% | 0% | 0% |
| moment saturation | 0% | 0% | 0% |
| action t1 mean/std/abs max | -0.2430 / 0.1621 / 0.8127 | -0.0151 / 0.0852 / 0.9329 | +0.0875 / 0.0885 / 0.4469 |
| action t2 mean/std/abs max | +0.2026 / 0.0637 / 0.6254 | +0.0961 / 0.0700 / 0.8780 | +0.1181 / 0.0620 / 0.5543 |
| action normal mean/std/abs max | -0.3316 / 0.1544 / 0.7991 | -0.0693 / 0.1348 / 1.0000 | -0.1416 / 0.0955 / 0.7989 |

`*` The current evaluator does not publish a dedicated workspace-crash latch. With hard-contact and ground-crash rates both zero, the residual is reported as `crash - deck_miss - hard_contact - ground_crash`; it is an audit aid, not a new failure-contract metric.

The deterministic evidence is materially better than S0 in three important ways:

1. reference saturation is essentially eliminated;
2. crash rate falls from 100% to 51.56% by ep30;
3. align rises monotonically across checkpoints while deck miss falls monotonically.

Despite those improvements, settled landing remains 0% and the training reward continues to deteriorate.

## Checkpoint SHA-256

```text
84c5831e427be4f22840f6d438ed0ee71e0f2f088c13047de98d736b85056219  ep10
04c89f98389a2cbd244038ebbc1d79cc825d15ad669ee1a22ad8abc10638e477  ep20
084ddc1106a5bf5fe62a3cce9533b50cf7501e4ab8318cba4382a2db07c0f6b1  ep30
```

## Gate decision

S1 satisfies the stability/saturation parts of the gate:

```text
runtime NaN/Inf = 0
controller stable = yes
all controller saturation = 0
reference saturation = strongly improved vs S0
landing intermediates = deterministic align/crash/deck-miss improved
```

But it fails the required reward condition:

```text
reward no longer shows persistent monotonic degradation = false
```

Therefore:

```text
S1 SANITY FAIL
```

## Root-cause classification and next step

The S1 evidence no longer supports Case C as the primary bottleneck. The exploration scale is now reasonable, deterministic action/reference saturation is low, controller tracking is bounded, and controller saturation is zero.

At the same time, task behavior and training reward are partially anti-correlated: deterministic crash/deck-miss/align improve while `rewards/iter` worsens, and the policy increasingly reaches timeout instead of immediate crash without ever reaching settled landing. This is sufficient evidence to advance the diagnosis order to:

```text
Case D — reward compatibility audit
```

Do not run S2 (`sigma_init=-1.5`) automatically: the condition for S2 was persistent excessive action/reference saturation, and that condition is not present in S1.

The single next investigation is the M2 reward compatibility audit, theory first and with no reward modification yet. The audit should determine whether the inherited per-step reward terms and sparse landing bonus create an episode-length / safe-survival bias for velocity-reference control, then preregister exactly one minimal M2-only reward change if and only if the audit proves a mismatch.

M0/M1, controller, action semantics/range, contact/success/failure contracts, network, learning rate, and all other PPO hyperparameters remain frozen.

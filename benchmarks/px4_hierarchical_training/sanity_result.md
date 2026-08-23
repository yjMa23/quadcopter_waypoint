# M2 PPO Sanity Result

## Verdict

```text
SANITY FAIL
```

The gate failed because the 30-iteration run did not produce a stable learning trend. This is not a failure of the PX4-like controller: all five controller saturation ratios stayed at zero in both training diagnostics and deterministic checkpoint evaluation.

## Reproducibility

Training baseline commit:

```text
78d3be71449634a380b83f2f39afbace4e6fdab9
```

Action-interface baseline commit:

```text
ca974ee5118f8742af69a698a1c47a96aa7d0a9f
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
logs/rl_games/quadcopter_ship_landing_px4_hierarchical/2026-08-23_12-00-25
```

Training completed normally in 25.65 s with no NaN/Inf exception or controller explosion.

## Training trend

`rewards/iter` became worse rather than better:

| Iteration | Reward |
| ---: | ---: |
| 1 | -2.9475 |
| 10 | -37.4504 |
| 20 | -50.5182 |
| 25 | -64.1223 |
| 30 | -60.9177 |

Landing intermediates were unstable rather than convergent:

- `landing_success_rate = 0` for all 30 iterations.
- `m2_settled_landing_rate = 0` for every logged completed-episode batch.
- align success briefly rose to `0.9583` at iteration 15, then collapsed; iteration 30 was `0.0` in the training batch.
- ground-crash and hard-contact rates remained zero, so the instability was not controller-induced impact violence.
- deck-miss rate was highly non-monotonic and did not converge.

Reference/controller diagnostics remained finite:

- training reference-saturation ratio varied roughly from `0.0059` to `0.3006`, ending at `0.0720` for the iteration-30 completed-episode batch;
- controller acceleration/tilt/thrust/body-rate/moment saturation ratios were all `0` throughout the run;
- mean controller velocity tracking error stayed roughly `0.24–0.30 m/s` in training episode batches.

## PPO action-distribution diagnosis

The M2 config inherited the Direct PPO exploration setting:

```yaml
model:
  name: continuous_a2c_logstd
...
sigma_init:
  name: const_initializer
  val: 0
fixed_sigma: True
```

For `continuous_a2c_logstd`, the initial log standard deviation of zero corresponds to an exploration standard deviation of approximately one normalized action unit. The checkpoint state confirms that the state-independent exploration scale did not shrink:

| Checkpoint | sigma parameter | exp(sigma) |
| --- | --- | --- |
| ep10 | `[0.1287, -0.0303, 0.0941]` | `[1.1374, 0.9701, 1.0987]` |
| ep20 | `[0.1942, -0.0254, 0.1841]` | `[1.2144, 0.9749, 1.2022]` |
| ep30 | `[0.1918, 0.0360, 0.0865]` | `[1.2115, 1.0367, 1.0904]` |

This exploration scale is of the same order as the entire normalized action interval `[-1, 1]`. For the new velocity-reference semantics, one normalized unit maps to up to `0.8 m/s` tangential motion and `0.4/0.3 m/s` normal descent/ascent before slew limiting. That makes the inherited exploration distribution much more aggressive physically than a small perturbation around a velocity reference.

## Fixed-seed deterministic checkpoint evaluation

All checkpoints were evaluated with:

```text
seed = 145
num_envs = 64
episodes = 64
```

| Metric | ep10 | ep20 | ep30 |
| --- | ---: | ---: | ---: |
| align success | 25.00% | 26.56% | 28.12% |
| settled landing | 0% | 0% | 0% |
| crash | 100% | 100% | 100% |
| hard contact | 0% | 0% | 0% |
| ground crash | 0% | 0% | 0% |
| deck miss | 48.44% | 26.56% | 89.06% |
| timeout | 0% | 0% | 0% |
| mean final distance | 1.7204 m | 2.2169 m | 1.6672 m |
| mean minimum distance | 0.7205 m | 0.7992 m | 0.7602 m |
| ref norm mean | 0.4640 m/s | 0.4898 m/s | 0.6588 m/s |
| ref saturation | 25.20% | 17.30% | 48.80% |
| controller tracking error mean | 0.3298 m/s | 0.1407 m/s | 0.2042 m/s |
| acceleration saturation | 0% | 0% | 0% |
| tilt saturation | 0% | 0% | 0% |
| thrust saturation | 0% | 0% | 0% |
| body-rate saturation | 0% | 0% | 0% |
| moment saturation | 0% | 0% | 0% |
| max desired tilt | 10.36 deg | 10.78 deg | 10.62 deg |
| deterministic action mean t1 | -0.1782 | -0.1504 | -0.5185 |
| deterministic action mean t2 | 0.1496 | 0.0958 | -0.6156 |
| deterministic action mean normal | -0.0301 | +0.1603 | -0.5669 |

Per-episode evidence:

```text
sanity_ep10_seed145.csv
sanity_ep20_seed145.csv
sanity_ep30_seed145.csv
```

The ep30 policy moved strongly toward action limits: all three axes reached `abs(action)=1.0` during evaluation and reference saturation rose to `48.8%`. Yet the controller still had zero saturation. This is the first failure mode in the prescribed diagnosis order that is supported by evidence.

## Diagnosis classification

```text
Case A: no primary evidence that the physical reference bounds/frame/slew implementation is invalid.
Case B: rejected as the primary cause; controller saturation is 0 and tracking remains bounded.
Case C: SUPPORTED; policy/action distribution drifts toward normalized-action bounds and deterministic reference saturation rises to 48.8% by ep30.
Case D: not audited/modified yet because the required diagnosis order stops at Case C first.
```

The `crash=100%` result is primarily non-ground/non-hard-contact failure: the physical failure taxonomy reports ground crash and hard contact at zero, while deck miss plus inherited workspace-exit failure account for termination. This is consistent with an aggressive/unstable reference policy rather than a low-level controller explosion.

## Minimal next modification

Do **not** change reward, controller gains, action range, network, or learning rate yet.

The next one-variable-at-a-time experiment should change only the M2 PPO exploration initialization:

```yaml
sigma_init:
  name: const_initializer
  val: -1.0
```

This changes the initial state-independent exploration standard deviation from approximately `1.0` to `exp(-1) ≈ 0.368` normalized action units while preserving the existing 3D action bounds, physical scaling, controller, reward, success/contact contracts, and PPO architecture.

Run the same `seed=42 / 64 env / 30 iteration` sanity protocol again. Only if the repeated sanity shows a stable reward or landing-intermediate improvement with bounded saturation may M2 advance to 100–200 iteration candidate training.

No reward change is justified by the current gate because Case C must be addressed first.

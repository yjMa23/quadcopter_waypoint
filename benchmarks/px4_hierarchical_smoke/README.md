# PX4-Compatible Hierarchical RL Smoke Benchmark

> Date: 2026-08-23  
> Status: **PASS**  
> Scope: deterministic reference-adapter/controller smoke only. This is **not** a trained PPO policy result.

## Purpose

This benchmark validates the new deployable action path before PPO tuning:

```text
3-D deck-relative velocity action
→ PX4-compatible Reference Adapter
→ 25 Hz held world velocity reference
→ 100 Hz VectorizedPx4LikeController
→ thrust / moment
→ Isaac Lab rigid-body dynamics
```

The benchmark does not alter or replace any frozen Direct RL result.

## Reproduction

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint
conda activate env_isaaclab
export PYTHONPATH=source/quadcopter_waypoint

python scripts/rl_games/check_px4_hierarchical_smoke.py \
  --num_envs 1 --headless

python scripts/rl_games/check_px4_hierarchical_smoke.py \
  --num_envs 16 --headless
```

The committed numeric evidence is `smoke_16env.json`.

## 16-environment result

| Case | Max relative-position drift | Max velocity tracking error | Ref saturation | Controller saturation | Ground crash | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|
| static hover | 0.03825 m | 0.08865 m/s | 0% | 0% | 0 | 0 |
| constant XY deck | 0.000056 m | 0.00000022 m/s | 0% | 0% | 0 | 0 |
| heave deck | 0.03021 m | 0.01151 m/s | 0% | 0% | 0 | 0 |
| physical-deck-attitude | 0.02275 m | 0.00863 m/s | 0% | 0% | 0 | 0 |

Slow deck-normal descent/contact:

```text
contact seen                    = true
first-contact |normal speed|    = 0.01495 m/s
first-contact tangential speed  = 0.01723 m/s
hard contact                    = false
ground crash                    = false
NaN/Inf                         = false
```

All smoke gates passed:

```text
no_nan_inf             = PASS
basic_ground_crash_zero= PASS
tracking_stable        = PASS
contact_safe           = PASS
```

## Runtime note

The synchronized 16-env controller calls were approximately 1.77–1.95 ms mean in these cases. The static-hover run recorded a 61.1 ms cold/outlier maximum; this smoke timing includes explicit CUDA synchronization and is **not** a formal optimized latency benchmark. Policy inference latency is not measured here because no PPO policy is being run.

## Interpretation

This evidence is sufficient to open the next gate—training the independent PX4-compatible hierarchical policy—but it does not establish that the learned method will meet the final `settled_landing >= 95%` target. That requires a separate trained-policy benchmark and later PX4 SITL validation.

# Sea-State Benchmark v1 evidence

This directory is independent from `benchmarks/physical_deck_attitude/` and `benchmarks/actor_preserving_ppo/`.
It records the frozen Sea-State v1 health checks and the factor-isolated zero-shot robustness boundary study.

## Frozen contracts

- `Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0` is not modified by this study.
- Sea-State v1 keeps its JONSWAP finite spectrum, surrogate second-order response, analytic pose/velocity, 8 deg roll/pitch envelope, 0.12 m heave envelope, and conservative coefficient scaling unchanged.
- `scripts/rl_games/eval_metrics.py` remains the success/failure contract.
- The frozen actor-preserving deterministic formal result remains 2229/2304 = 96.74% settled landing.

## Current evidence files

- `profiles.yaml`: factor-isolated engineering benchmark profiles and severity ordering.
- `profile_realization_summary.csv/json`: offline realized-motion and spectral-scaling audit.
- `physics_1env.json`, `physics_16env.json`: frozen nominal Sea-State v1 physics checks.
- `physics_frequency_boundary_16env.json`: fastest accepted frequency-profile physics check.
- `physics_combined_boundary_16env.json`: strongest accepted combined-profile physics check.
- `pilot_raw/`: per-episode teacher CSV/log evidence produced by the frozen evaluator.
- `pilot_results.csv`: per-policy/profile aggregate metrics.
- `pilot_summary.json`: machine-readable pilot summary.
- `robustness_curves.csv`: realized angular-speed, tilt, heave-velocity, Tp and Hs buckets.
- `boundary_candidates.json`: automatic profile-level boundary output.
- `failure_analysis.csv`: outcome-conditioned realized-motion/contact statistics.
- `formal_protocol.json`: blocked formal/adaptation protocol and selected checkpoint paths.
- `commands.txt`: reproduction commands.

Legacy `lightweight_*` files remain as evidence from the earlier mild/shifted implementation stage; they are not the primary result of the factor-isolated study.

## Factor-isolated teacher pilot

The current study contains 1536 completed frozen-teacher episodes aggregated into 23 profile rows. Most profiles use seed 245, 32 parallel environments and 64 episodes. The nearest frequency transition probe (`frequency_shift_tp1p6_2p0`) was repeated with seed 246 for 128 episodes total.

Representative results:

| Profile | Episodes | Settled | Deck miss | Hard contact |
|---|---:|---:|---:|---:|
| nominal stochastic | 64 | 98.44% | 1.56% | 0.00% |
| frequency Tp=2.0..2.5 s | 64 | 95.31% | 4.69% | 0.00% |
| frequency Tp=1.6..2.0 s | 128 | 96.88% | 3.12% | 0.00% |
| tilt target 5 deg | 64 | 96.88% | 3.12% | 0.00% |
| heave-rate medium-high | 64 | 95.31% | 4.69% | 0.00% |
| combined high tilt + very high rate | 64 | 98.44% | 1.56% | 1.56% |

The automatic profile-level result is therefore:

```text
frequency_shift: no robustness boundary found
tilt_shift:      no robustness boundary found
heave_rate_shift:no robustness boundary found
combined_shift:  no robustness boundary found
```

No PPO training was started, and the metric-selected actor-preserving checkpoints were not evaluated on a shifted candidate because the teacher gate did not open. Their actor inputs were nevertheless verified as 22-D and their paths are recorded in `formal_protocol.json`.

## Strongest realized-motion signal

Pooling the factor-isolated teacher episodes by what the policy actually experiences gives the strongest current signal at deck angular-speed max `0.08..0.12 rad/s`:

```text
n = 196
settled landing = 93.88%
deck miss       = 6.12%
hard contact    = 0.51%
```

The corresponding heave-velocity `0.08..0.12 m/s` bucket has 172 episodes and 94.77% settled landing. Neither curve is monotonic at higher buckets because those buckets mix different controlled profile families, so these are hypothesis-generating target ranges rather than a claimed causal boundary.

## Gate conclusion

The next step is not PPO. A subsequent robustness study should construct a narrow controlled distribution concentrated around the realized-motion transition ranges and verify the signal across multiple teacher seeds. Only after a repeatable profile-level condition reaches roughly 75-90% settled landing (or shows another clear safety degradation) should the 3-checkpoint x 3-seed x 256-episode formal protocol and adaptation comparison be activated.

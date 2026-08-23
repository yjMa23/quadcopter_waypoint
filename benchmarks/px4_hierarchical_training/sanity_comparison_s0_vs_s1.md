# M2 Sanity Comparison: S0 vs S1

S0 and S1 use the same task, seed, environment count, iteration budget, controller, reward, action semantics/range, and PPO hyperparameters. The only behavioral difference is:

```text
S0 sigma_init.val = 0.0
S1 sigma_init.val = -1.0
```

## Fixed-seed deterministic comparison

Validation protocol for every checkpoint:

```text
seed = 145
episodes = 64
num_envs = 64
```

| Metric | S0 ep10 | S0 ep20 | S0 ep30 | S1 ep10 | S1 ep20 | S1 ep30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training reward | -37.45 | -50.52 | -60.92 | -28.91 | -53.96 | -73.86 |
| align success | 25.00% | 26.56% | 28.12% | 7.81% | 15.62% | 31.25% |
| settled landing | 0% | 0% | 0% | 0% | 0% | 0% |
| contact success | 4.69% | 0% | 0% | 3.12% | 0% | 0% |
| crash | 100% | 100% | 100% | 100% | 64.06% | 51.56% |
| deck miss | 48.44% | 26.56% | 89.06% | 75.00% | 45.31% | 28.12% |
| timeout | 0% | 0% | 0% | 0% | 35.94% | 51.56% |
| reference saturation | 25.20% | 17.30% | 48.80% | 0% | 0.47% | 0% |
| controller tracking error mean | 0.3298 m/s | 0.1407 m/s | 0.2042 m/s | 0.0974 m/s | 0.0756 m/s | 0.0763 m/s |
| mean episode action std, 3-axis average* | 0.2728 | 0.1989 | 0.3447 | 0.1267 | 0.0967 | 0.0820 |
| any deterministic action abs max | 1.0000 | 1.0000 | 1.0000 | 0.8127 | 1.0000 | 0.7989 |
| controller acceleration saturation | 0% | 0% | 0% | 0% | 0% | 0% |
| controller tilt saturation | 0% | 0% | 0% | 0% | 0% | 0% |
| controller thrust saturation | 0% | 0% | 0% | 0% | 0% | 0% |
| controller body-rate saturation | 0% | 0% | 0% | 0% | 0% | 0% |
| controller moment saturation | 0% | 0% | 0% | 0% | 0% | 0% |

`*` Average of the per-episode `t1/t2/normal` action standard deviations in the saved evaluation CSV.

## Interpretation

S1 strongly validates the exploration-scale hypothesis in a narrow sense: reducing initial PPO sigma eliminates the large reference-saturation drift seen in S0 and produces much smaller deterministic action variance. Controller tracking also improves while all controller saturation metrics remain zero.

However, this does not restore a passing learning signal. S1 still has zero settled landing, and `rewards/iter` degrades more strongly by ep30 even though deterministic align, crash, and deck-miss metrics improve. This decoupling is the key reason S1 cannot be declared PASS.

The evidence therefore changes the diagnosis ordering:

```text
Case C: substantially mitigated / no longer primary
Case D: next audit target
```

S2 (`sigma_init=-1.5`) is not justified by the current evidence because action/reference saturation is no longer excessive. C0 remains blocked because S1 failed the preregistered sanity gate.

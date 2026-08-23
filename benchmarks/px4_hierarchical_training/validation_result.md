# M2 Validation Result

Formal candidate validation is **NOT STARTED** because the mandatory 64-env / seed-42 / 30-iteration sanity gate failed.

A diagnostic fixed-seed comparison was performed only to determine whether the sanity run showed a stable learning trend:

```text
validation seed = 145
64 completed episodes per checkpoint
checkpoints = ep10, ep20, ep30
```

Results:

| Checkpoint | Align | Settled | Crash | Ground crash | Hard contact | Deck miss | Ref saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep10 | 25.00% | 0% | 100% | 0% | 0% | 48.44% | 25.20% |
| ep20 | 26.56% | 0% | 100% | 0% | 0% | 26.56% | 17.30% |
| ep30 | 28.12% | 0% | 100% | 0% | 0% | 89.06% | 48.80% |

These runs are sanity diagnostics, **not** validation-selected formal M2 candidates and not a nominal M0/M1 comparison.

The prescribed fixed-seed `145/146/147` candidate ranking and deterministic PhysicalDeckAttitude benchmark remain blocked until a future sanity run passes.

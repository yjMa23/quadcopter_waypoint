# M2 Candidate Table

| ID | Scope | Status | Result / reason |
| --- | --- | --- | --- |
| S0 | Current action/controller/reward, seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reward worsened, settled landing stayed 0%, deterministic checkpoints crashed 100%, and ep30 reference saturation reached 48.8%. Controller saturation stayed 0%. |
| C0 | Current action/controller/reward, 256 env, 100–200 iterations | **NOT STARTED** | Blocked by failed sanity gate. Increasing iterations is explicitly prohibited after sanity failure. |
| C1 | One-variable exploration candidate: M2 `sigma_init.val: 0 -> -1.0`; all action/controller/reward/PPO architecture otherwise frozen | **PROPOSED NEXT SANITY ONLY** | Addresses Case C evidence first. Must repeat 64-env / seed-42 / 30-iteration sanity before any candidate-length training. |
| C2 | Any action-range/controller/reward alternative | **NOT DEFINED** | No second variable is justified until C1 sanity evidence exists. |

No 100–200 iteration candidate training was run in this round.

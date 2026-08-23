# M2 Candidate Table

| ID | Scope | Status | Result / reason |
| --- | --- | --- | --- |
| S0 | M2 baseline exploration: `sigma_init.val = 0` (`sigma ≈ 1.0`), seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reward worsened, settled landing stayed 0%, deterministic checkpoints crashed 100%, and ep30 reference saturation reached 48.8%. Controller saturation stayed 0%. |
| S1 | One-variable exploration candidate: M2 `sigma_init.val = -1.0` (`sigma ≈ 0.368`), seed 42, 64 env, 30 iterations | **PREREGISTERED** | Only changed variable is exploration sigma initialization. Action/controller/reward/task contracts and all other PPO hyperparameters remain frozen. |
| C0 | S1 configuration, seed 42, 256 env, 150 iterations | **BLOCKED** | May run only if S1 receives an explicit `S1 SANITY PASS`. |
| S2 | One-variable fallback: M2 `sigma_init.val = -1.5` (`sigma ≈ 0.223`) | **NOT STARTED** | Diagnostic recommendation only if S1 fails with Case C evidence still dominant; do not run automatically. |

S1 is the only active sanity candidate. No C0/S2 training is authorized before the S1 gate decision.

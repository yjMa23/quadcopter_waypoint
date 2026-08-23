# M2 Candidate Table

| ID | Scope | Status | Result / reason |
| --- | --- | --- | --- |
| S0 | M2 baseline exploration: `sigma_init.val = 0` (`sigma ≈ 1.0`), seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reward worsened, settled landing stayed 0%, deterministic checkpoints crashed 100%, and ep30 reference saturation reached 48.8%. Controller saturation stayed 0%. |
| S1 | One-variable exploration candidate: M2 `sigma_init.val = -1.0` (`sigma ≈ 0.368`), seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reference saturation collapsed to 0/0.47/0% and deterministic crash/deck-miss improved, but reward still degraded persistently and settled landing remained 0%. |
| C0 | S1 configuration, seed 42, 256 env, 150 iterations | **BLOCKED** | Not run because S1 failed the preregistered sanity gate. |
| S2 | One-variable fallback: M2 `sigma_init.val = -1.5` (`sigma ≈ 0.223`) | **NOT JUSTIFIED** | S1 no longer shows excessive action/reference saturation, so the preregistered condition for S2 is absent. |
| D0 | M2 reward compatibility audit, theory/diagnostics only | **SUPPORTED** | `D0-B LATCHED DESCENT-PHASE REWARD MISMATCH SUPPORTED`: S1 reward/behavior ranking mismatch persists after length normalization and latched alignment frequently ends outside the instantaneous recovery envelope. |
| D1 | One-variable M2 descent-reward eligibility gate; same S1 `sigma_init=-1.0`, seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reward compatibility improves strongly (`ep30 reward/step=-0.2133`, phase-sensitive share 15.76%), and ep30 align/crash/deck-miss improve to 48.44%/1.56%/0%; however controller tracking mean degrades to 0.4425 m/s, timeout reaches 98.44%, and post-latch horizontal recovery violates preregistered outside-radius gates. |

C0 and S2 were not run. D1 does not authorize 100-200 iteration candidate training or PX4 SITL. The next step is theory-first diagnosis of post-latch horizontal recovery, within-bound action/reference variation, and controller tracking rather than longer training or another sigma reduction.

# M2 Candidate Table

| ID | Scope | Status | Result / reason |
| --- | --- | --- | --- |
| S0 | M2 baseline exploration: `sigma_init.val = 0` (`sigma ≈ 1.0`), seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reward worsened, settled landing stayed 0%, deterministic checkpoints crashed 100%, and ep30 reference saturation reached 48.8%. Controller saturation stayed 0%. |
| S1 | One-variable exploration candidate: M2 `sigma_init.val = -1.0` (`sigma ≈ 0.368`), seed 42, 64 env, 30 iterations | **SANITY FAIL** | Reference saturation collapsed to 0/0.47/0% and deterministic crash/deck-miss improved, but reward still degraded persistently and settled landing remained 0%. |
| C0 | S1 configuration, seed 42, 256 env, 150 iterations | **BLOCKED** | Not run because S1 failed the preregistered sanity gate. |
| S2 | One-variable fallback: M2 `sigma_init.val = -1.5` (`sigma ≈ 0.223`) | **NOT JUSTIFIED** | S1 no longer shows excessive action/reference saturation, so the preregistered condition for S2 is absent. |
| D0 | M2 reward compatibility audit, theory/diagnostics only | **NEXT** | Audit whether the inherited reward is anti-correlated with safe survival/alignment under velocity-reference control. No reward change is authorized until one minimal M2-only change is theoretically justified. |

C0 and S2 were not run. The next step is Case D reward compatibility audit, not longer training or another sigma reduction.

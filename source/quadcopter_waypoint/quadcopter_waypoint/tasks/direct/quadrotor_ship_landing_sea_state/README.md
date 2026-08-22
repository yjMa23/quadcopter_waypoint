# stochastic sea-state ship-landing task

Task ID:

```text
Isaac-Quadcopter-ShipLanding-SeaState-Direct-v0
```

This task inherits `QuadcopterShipLandingPhysicalDeckAttitudeEnv` and intentionally changes only the
deck-motion generator. The 22-D observation, 4-D action, reward, deck-frame contact kinematics,
termination, hard-contact logic, deck-miss logic, and settled-landing contract remain inherited.

## Motion architecture

```text
Hs / Tp / gamma / heading
        ↓
finite JONSWAP spectrum
        ↓
random component phases
        ↓
second-order surrogate vessel response
        ↓
heave + roll + pitch
        ↓
analytic pose + linear/angular velocity
```

The response layer is a configurable engineering benchmark surrogate. It is **not** an identified or
measured vessel RAO. It can later be replaced by measured/tabulated RAOs without changing the landing
task contract.

The finite JONSWAP spectrum is normalized so the actual discretized spectral moment satisfies
`m0 = (Hs / 4)^2`. Runtime motion is a finite cosine sum and all derivatives are analytical.

Roll and pitch are not copied from wave elevation. They are generated through separate frequency
responses with heading projection. The world-frame angular velocity uses the same exact XYZ Euler-rate
mapping as the frozen PhysicalDeckAttitude task.

## Safety envelope

Each response DOF computes the phase-independent bound:

```text
sum(abs(component_amplitude))
```

If necessary, all coefficients for that DOF are uniformly scaled once at episode reset. Runtime clipping
of roll/pitch/heave is not used, so the spectrum and analytical derivatives remain consistent. The final
heave/roll/pitch envelopes also pass the existing conservative deck-bottom/ground-clearance check.

## Compatibility mode

Set:

```text
env.sea_state_mode=compatibility
```

The task then delegates motion generation to the parent PhysicalDeckAttitude implementation. This mode
exists only for regression/evaluation checks; it is not a new training distribution.

## Engineering benchmark presets

The default config is the `nominal` engineering benchmark. Recommended first zero-shot comparisons are:

### compatibility

```text
env.sea_state_mode=compatibility
```

### mild stochastic

```text
env.sea_state_mode=stochastic
env.sea_state_benchmark_profile=mild
env.sea_state_hs_min_m=0.14
env.sea_state_hs_max_m=0.22
env.sea_state_tp_min_s=4.5
env.sea_state_tp_max_s=6.0
env.sea_state_gamma_min=2.5
env.sea_state_gamma_max=3.5
env.sea_state_roll_gain_deg_per_m=35.0
env.sea_state_pitch_gain_deg_per_m=35.0
```

### nominal stochastic

Use the task defaults:

```text
Hs = 0.18..0.30 m
Tp = 3.8..5.8 s
gamma = 2.5..4.0
heading = -180..180 deg
```

### shifted stochastic

```text
env.sea_state_mode=stochastic
env.sea_state_benchmark_profile=shifted
env.sea_state_hs_min_m=0.22
env.sea_state_hs_max_m=0.34
env.sea_state_tp_min_s=2.8
env.sea_state_tp_max_s=4.2
env.sea_state_gamma_min=3.0
env.sea_state_gamma_max=5.0
env.sea_state_heave_natural_frequency_hz=0.30
env.sea_state_roll_natural_frequency_hz=0.17
env.sea_state_pitch_natural_frequency_hz=0.17
```

These ranges are project-specific engineering stress tests. They must not be described as validated
real-vessel `Sea State 3/4` conditions.

## Validation

Pure math tests:

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests/test_sea_state_motion.py
```

Physics diagnostic:

```bash
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/check_sea_state_physics.py \
  --num_envs=1 --motion_steps=500 --headless
```

Zero-shot evaluation reuses `scripts/rl_games/eval_metrics.py`; no separate success definition is
introduced.

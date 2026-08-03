# P6C-PhysicalDeckAttitude

Task ID:

```text
Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0
```

rl_games experiment:

```text
quadcopter_ship_landing_physical_deck_attitude
```

P6C is an independent successor to the frozen P6B physical-deck task. It keeps the real kinematic
`Deck`, real `GroundSlab`, and two filtered `ContactSensor` instances, then adds absolute-time roll and
pitch motion. P6B files are not modified to implement the new dynamics.

## P7 use of this frozen task

P7 does not change this environment's observation, reward, contact, termination, or motion semantics. It
uses the frozen P6C checkpoint as a deterministic expert, then compares PPO-from-scratch, Behavior
Cloning, and BC-initialized PPO under the same task and formal evaluator.

Formal P7 evidence:

```text
benchmarks/phase7_imitation_hybrid/summary.json
benchmarks/phase7_imitation_hybrid/formal_evaluations/
docs/interview_p7_evidence.md
```

P7 collected 3976 successful episodes / 540321 transitions. BC-only achieved 88.28% settled landing
across seeds 42/43/44 (256 episodes each), while the fair BC+PPO run achieved 76.69% and exhibited policy
drift. These are retained as measured results; the environment remains frozen at tag
`p6c-physical-deck-attitude-v1`.

The policy is state based and does not contain camera images or real visual projection inputs.

## Motion contract

For episode time `t`, the deck command is computed directly from the sampled initial state:

```text
x = x0 + vx * t
y = y0 + vy * t
z = z0 + Az * sin(wz * t + phase_z)
roll  = Ar * sin(wr * t + phase_r)
pitch = Ap * sin(wp * t + phase_p)
yaw   = 0
```

The quaternion uses Isaac Lab's `(w, x, y, z)` convention. Angular velocity is expressed in the world
frame using the exact XYZ Euler-rate mapping. With zero yaw this is:

```text
omega_w = [roll_dot * cos(pitch), pitch_dot, -roll_dot * sin(pitch)]
```

It is not valid to write `[roll_dot, pitch_dot, 0]` directly once pitch is non-zero. Pure-Python tests
compare this analytic mapping with a neighboring-quaternion delta/log-map calculation.

Both pose and velocity are written before the decimated physics loop. Runtime diagnostics compare the
simulator's deck root pose and velocity with the previous command, detecting double integration,
orientation sign errors, or kinematic jitter.

## Ground-clearance safety

Initialization rejects configurations whose conservative minimum bottom-corner height is not above:

```text
ground slab top + safety margin
```

The bound includes deck half-length, half-width, half-thickness, maximum heave, maximum roll, and maximum
pitch. The default validated envelope is ±8 degrees with a 0.04 m safety margin; the frozen P6C
curriculum uses up to ±5 degrees.

## Deck-frame contact kinematics

State truth comes from the physical deck root pose, root linear velocity, root angular velocity, and the
filtered contact sensors. The visual marker is not used as truth.

The robot bottom point and corresponding deck top-surface point are transformed into deck coordinates.
The surface point velocity is:

```text
v_surface = v_deck_center + omega_deck x (surface_point - deck_center)
v_relative = v_robot_bottom_point - v_surface
v_normal = dot(v_relative, deck_normal)
v_tangent = norm(v_relative - v_normal * deck_normal)
```

Signed clearance is the robot bottom point's deck-frame z coordinate minus deck half-thickness. XY error
is the norm of its deck-frame x/y coordinates.

A settled landing requires real deck contact, no ground contact, valid first/current deck-frame contact
position, low normal and tangential relative speed, low body angular speed, body-z/deck-normal alignment,
world uprightness, sustained contact, no hard contact, and bounded penetration.

## Observation contract (22 dimensions)

| Slice | Quantity | Frame / unit |
|---|---|---|
| `0:3` | robot root linear velocity | robot body, m/s |
| `3:6` | robot root angular velocity | robot body, rad/s |
| `6:9` | projected gravity | robot body, unit vector |
| `9:12` | deck-center position relative to robot | robot body, m |
| `12:15` | deck surface-point velocity minus robot root velocity | world, m/s |
| `15` | align state | scalar 0/1 |
| `16:19` | deck normal | robot body, unit vector |
| `19:22` | deck angular velocity minus robot angular velocity | robot body, rad/s |

The first 16 columns preserve P6B zero-attitude semantics. No full quaternion is exposed to the policy.

## Checkpoint migration

Use:

```bash
PYTHONPATH=source/quadcopter_waypoint python scripts/rl_games/expand_checkpoint_observation.py \
  --input <P6B_CHECKPOINT.pth> \
  --output <P6C_22D_CHECKPOINT.pth>
```

The tool copies the first 16 policy-input columns, initializes the six new columns to zero, extends
observation running mean with zeros and variance with ones, preserves counts/value normalization, and
extends the corresponding Adam moments. It refuses unexpected shapes and existing output paths, and
writes SHA256 metadata beside the output checkpoint.

## Curriculum overrides

Stage A, zero tilt:

```text
env.deck_roll_amplitude_max_deg=0.0
env.deck_pitch_amplitude_max_deg=0.0
```

Stage B, roll ±2 degrees:

```text
env.deck_roll_amplitude_max_deg=2.0
env.deck_pitch_amplitude_max_deg=0.0
env.deck_roll_frequency_min=0.06
env.deck_roll_frequency_max=0.10
```

Stage C, roll/pitch ±3 degrees:

```text
env.deck_roll_amplitude_max_deg=3.0
env.deck_pitch_amplitude_max_deg=3.0
env.deck_roll_frequency_min=0.06
env.deck_roll_frequency_max=0.12
env.deck_pitch_frequency_min=0.06
env.deck_pitch_frequency_max=0.12
```

Stage D, frozen P6C range:

```text
env.deck_roll_amplitude_max_deg=5.0
env.deck_pitch_amplitude_max_deg=5.0
env.deck_roll_frequency_min=0.08
env.deck_roll_frequency_max=0.15
env.deck_pitch_frequency_min=0.08
env.deck_pitch_frequency_max=0.15
```

Yaw oscillation, random wave spectra, hydrodynamics, and coupled six-degree-of-freedom ship motion are
not part of P6C.

## Validation commands

Pure-Python tests:

```bash
PYTHONPATH=source/quadcopter_waypoint python -m pytest -q tests
```

Independent motion/contact check:

```bash
PYTHONPATH=source/quadcopter_waypoint python \
  scripts/rl_games/check_physical_deck_attitude_physics.py \
  --num_envs=16 --motion_steps=500 --headless
```

Formal evaluation uses 64 environments, 256 episodes per seed, and seeds 42/43/44. See
`benchmarks/phase6c_physical_deck_attitude/summary.json` for the frozen checkpoint, exact commands,
per-seed metrics, tilt buckets, angular-speed buckets, and acceptance result.

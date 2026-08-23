# PX4-Compatible Hierarchical Ship Landing

This task is an independent action-interface method built on the frozen PhysicalDeckAttitude landing contract.

```text
Task ID:
Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0

observation: 22-D PhysicalDeckAttitude-compatible state
action: 3-D normalized deck-relative velocity reference
         [v_t1_rel, v_t2_rel, v_n_rel]
policy/reference update: 25 Hz
physics + PX4-like training controller: 100 Hz
```

The task intentionally inherits the existing observation, reward, contact model, failure taxonomy, and settled-landing success semantics. It does **not** modify or deprecate the frozen Direct RL tasks or checkpoints.

## Control chain

```text
22-D observation
→ policy
→ normalized 3-D action
→ PX4 Reference Adapter
→ deck contact-point feedforward + deck-relative velocity
→ world/ENU velocity reference
→ VectorizedPx4LikeController
→ thrust + body moment
→ Isaac Lab dynamics
```

Deployment is intentionally a different backend:

```text
same exported policy
→ same reference math
→ ENU→NED velocity
→ ROS2
→ OffboardControlMode.velocity=true
→ TrajectorySetpoint.velocity
→ real PX4 flight stack
```

`VectorizedPx4LikeController` is a training surrogate and is **not an exact PX4 implementation**.

## Theory and safety contract

The authoritative design document is:

```text
docs/px4_compatible_hierarchical_rl_theory.md
```

Important first-version choices:

- yaw is deterministic and is not a learned action;
- normalized zero action means exactly zero deck-relative velocity;
- rigid contact-point velocity uses `v_deck + omega × r` from the existing PhysicalDeckAttitude math;
- velocity, acceleration/slew, tilt, thrust, body-rate, and moment limits are explicit;
- training has no ROS2 or PX4 runtime dependency;
- existing 22D→4D checkpoints cannot be loaded as this 22D→3D policy.

## Validation order

```text
Reference Adapter unit tests
→ vectorized controller unit tests
→ task-contract regression
→ minimal Isaac Lab smoke
→ only after smoke PASS: PPO tuning/training
```

No formal trained-policy benchmark is claimed by this README until a dedicated benchmark package is produced.

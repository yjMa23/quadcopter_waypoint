#!/usr/bin/env python3
"""Run and record a deterministic full touchdown/settle demo for the Continuous-Stage PX4 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
parser.add_argument("--initial_clearance", type=float, default=0.25)
parser.add_argument("--max_policy_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.tasks.direct.quadrotor_ship_landing_px4_continuous_stage.quadrotor_ship_landing_px4_continuous_stage_env import (
    QuadcopterShipLandingPx4ContinuousStageEnvCfg,
)
from quadcopter_waypoint.utils.physical_deck_attitude_math import quat_apply

TASK_ID = "Isaac-Quadcopter-ShipLanding-Px4ContinuousStage-Direct-v0"
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_quat(task) -> torch.Tensor:
    quat = torch.zeros(task.num_envs, 4, device=task.device)
    quat[:, 0] = 1.0
    return quat


def _set_static_flat_deck(task) -> None:
    env_ids = task._robot._ALL_INDICES
    task._deck_motion_time.zero_()
    task._deck_origin_xy_w[:] = task._terrain.env_origins[:, :2]
    task._deck_xy_velocity_w.zero_()
    task._pad_heave_amp.zero_()
    task._pad_heave_omega.zero_()
    task._deck_heave_phase0.zero_()
    task._deck_roll_amp.zero_()
    task._deck_pitch_amp.zero_()
    task._deck_roll_omega.zero_()
    task._deck_pitch_omega.zero_()
    task._deck_roll_phase0.zero_()
    task._deck_pitch_phase0.zero_()
    task._write_absolute_deck_state(env_ids)


def _place_robot(task, clearance: float) -> None:
    env_ids = task._robot._ALL_INDICES
    deck_pose = task._deck_pose_command_w.clone()
    local_offset = task._target_contact_point_d.clone()
    local_offset[:, 2] += task.cfg.robot_landing_surface_offset + clearance
    robot_pos = deck_pose[:, :3] + quat_apply(deck_pose[:, 3:7], local_offset)
    pose = torch.cat((robot_pos, _identity_quat(task)), dim=-1)
    velocity = torch.zeros(task.num_envs, 6, device=task.device)
    task._robot.write_root_pose_to_sim(pose)
    task._robot.write_root_velocity_to_sim(velocity)
    task._thrust.zero_()
    task._moment.zero_()
    task.scene.write_data_to_sim()
    task.sim.step(render=False)
    task.scene.update(dt=task.physics_dt)
    task._reset_continuous_reference_state(env_ids)
    task.episode_length_buf[:] = 1


def _action(task, normal: float, stage_action: float) -> torch.Tensor:
    action = torch.zeros(task.num_envs, 4, device=task.device)
    action[:, 2] = normal
    action[:, 3] = stage_action
    return action


def _snapshot(task, policy_step: int, phase: str) -> dict:
    terms = task._compute_landing_terms()
    return {
        "policy_step": policy_step,
        "phase": phase,
        "stage": float(task._landing_stage[0]),
        "stage_raw": float(task._stage_raw[0]),
        "clearance_m": float(terms["landing_surface_clearance"][0]),
        "horizontal_error_m": float(terms["horizontal_error"][0]),
        "normal_reference_mps": float(task._relative_velocity_ref_d[0, 2]),
        "normal_relative_speed_mps": float(terms["normal_rel_speed"][0]),
        "tangential_relative_speed_mps": float(terms["tangential_rel_speed"][0]),
        "terminal_alpha": float(task._terminal_alpha[0]),
        "relative_angular_speed_radps": float(terms["relative_ang_vel_norm"][0]),
        "deck_contact": bool(terms["deck_contact"][0]),
        "safe_contact": bool(terms["safe_contact"][0]),
        "settle_hold_steps": int(task._settle_hold_steps[0]),
        "landing_success": bool(task._landing_success[0]),
        "hard_contact": bool(terms["hard_contact"][0]),
        "ground_crash": bool(terms["ground_crash"][0]),
        "contact_force_n": float(terms["deck_force"][0]),
        "robot_position_w": [float(value) for value in task._robot.data.root_pos_w[0].detach().cpu().tolist()],
        "deck_position_w": [float(value) for value in task._deck.data.root_pos_w[0].detach().cpu().tolist()],
    }


def _render_frame(env, snapshot: dict, width: int, height: int) -> np.ndarray:
    rendered = env.render()
    if rendered is None:
        raise RuntimeError("headless render returned no frame; use --enable_cameras")
    frame = np.asarray(rendered)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f"unexpected render frame shape: {frame.shape}")
    frame = frame[:, :, :3].astype(np.uint8, copy=False)
    image = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    lines = [
        "Continuous-Stage Full Landing Demo",
        f"phase={snapshot['phase']}  step={snapshot['policy_step']}",
        f"stage={snapshot['stage']:.3f}  alpha={snapshot['terminal_alpha']:.3f}",
        f"clearance={snapshot['clearance_m']:+.3f} m  xy_err={snapshot['horizontal_error_m']:.3f} m",
        f"v_n_ref={snapshot['normal_reference_mps']:+.3f} m/s  v_n_rel={snapshot['normal_relative_speed_mps']:+.3f} m/s",
        f"contact={snapshot['deck_contact']}  safe={snapshot['safe_contact']}  settle={snapshot['settle_hold_steps']}",
        f"hard={snapshot['hard_contact']}  ground={snapshot['ground_crash']}  SUCCESS={snapshot['landing_success']}",
    ]
    y = 14
    for line in lines:
        box = draw.textbbox((14, y), line)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(0, 0, 0))
        draw.text((14, y), line, fill=(255, 255, 255))
        y += 22
    return np.asarray(image, dtype=np.uint8)


def _advance_policy_step(task, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    task._pre_physics_step(action)
    for _ in range(task.cfg.decimation):
        task._apply_action()
        task.scene.write_data_to_sim()
        task.sim.step(render=False)
        task.scene.update(dt=task.physics_dt)
    task.episode_length_buf += 1
    return task._get_dones()


def main() -> None:
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    metadata_path = output.with_suffix(".json")
    if metadata_path.exists():
        metadata_path.unlink()

    cfg = QuadcopterShipLandingPx4ContinuousStageEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = SEED
    cfg.debug_vis = False
    cfg.strict_deck_motion_consistency = False
    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
    task = env.unwrapped
    env.reset(seed=SEED)
    _set_static_flat_deck(task)
    _place_robot(task, args.initial_clearance)
    try:
        task.sim.set_camera_view(eye=[1.7, 1.7, 1.35], target=[0.0, 0.0, 0.35])
    except Exception:
        pass

    frames: list[np.ndarray] = []
    history: list[dict] = []
    first_contact: dict | None = None
    contact_latched = False
    result = "max_steps"

    for policy_step in range(args.max_policy_steps):
        terms_before = task._compute_landing_terms()
        clearance = float(terms_before["landing_surface_clearance"][0])
        deck_contact = bool(terms_before["deck_contact"][0])

        if policy_step < 20:
            phase = "initial_hover"
            stage_action = -1.0
            normal_action = 0.0
        elif policy_step < 50:
            phase = "stage_commit"
            fraction = (policy_step - 20) / 29.0
            stage_action = -1.0 + 2.0 * fraction
            normal_action = -0.55
        elif contact_latched:
            phase = "contact_settle"
            stage_action = 1.0
            # Once first contact is observed, keep a seating reference even if the force-based
            # contact bit flickers for a frame. The success thresholds themselves remain unchanged.
            normal_action = -1.00
        else:
            phase = "descent"
            stage_action = 1.0
            if clearance > 0.22:
                normal_action = -0.60
            elif clearance > 0.10:
                normal_action = -0.45
            elif clearance > 0.04:
                normal_action = -0.28
            else:
                normal_action = -0.12

        terminated, time_out = _advance_policy_step(task, _action(task, normal_action, stage_action))
        snapshot = _snapshot(task, policy_step, phase)
        history.append(snapshot)
        frames.append(_render_frame(env, snapshot, args.video_width, args.video_height))

        if snapshot["deck_contact"]:
            contact_latched = True
            if first_contact is None:
                first_contact = dict(snapshot)
        if snapshot["landing_success"]:
            result = "settled_landing"
            # Hold the final successful view for readability without advancing/resetting the physics.
            for _ in range(args.video_fps):
                frames.append(_render_frame(env, snapshot, args.video_width, args.video_height))
            break
        if snapshot["ground_crash"]:
            result = "ground_crash"
            break
        if snapshot["hard_contact"]:
            result = "hard_contact"
            break
        if bool(terminated[0]) or bool(time_out[0]):
            result = "terminated"
            break

    if not frames:
        env.close()
        app.close()
        raise RuntimeError("no video frames captured")

    imageio.mimwrite(
        output,
        frames,
        fps=args.video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("video encoder produced an empty file")

    final = history[-1]
    metadata = {
        "task_id": TASK_ID,
        "seed": SEED,
        "scenario": "deterministic static-level-deck full touchdown and settle demonstration",
        "initial_clearance_m": args.initial_clearance,
        "result": result,
        "settled_landing": result == "settled_landing",
        "policy_steps": len(history),
        "physics_hz": round(1.0 / task.physics_dt),
        "policy_hz": round(1.0 / task.step_dt),
        "video": {
            "path": str(output),
            "fps": args.video_fps,
            "resolution": [args.video_width, args.video_height],
            "frames": len(frames),
            "duration_seconds": len(frames) / args.video_fps,
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
        },
        "first_contact": first_contact,
        "final": final,
        "safety_thresholds": {
            "landing_success_radius_m": task.cfg.landing_success_radius,
            "safe_contact_normal_speed_mps": task.cfg.safe_contact_normal_speed,
            "safe_contact_tangential_speed_mps": task.cfg.safe_contact_tangential_speed,
            "safe_contact_relative_ang_speed_radps": task.cfg.safe_contact_ang_vel,
            "hard_contact_normal_speed_mps": task.cfg.hard_contact_normal_speed,
            "settle_hold_steps": task.cfg.settle_hold_steps,
        },
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "history"}, indent=2), flush=True)
    env.close()
    app.close()
    if result != "settled_landing":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

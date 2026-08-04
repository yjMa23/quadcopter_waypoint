# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Record one deterministic success or representative failure as an objective P7 state trajectory."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--target", choices=("success", "failure"), required=True)
parser.add_argument(
    "--failure_type",
    choices=("any", "hard_contact", "ground_crash", "deck_miss", "timeout"),
    default="any",
)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadcopter_waypoint.tasks  # noqa: F401
from quadcopter_waypoint.imitation.dataset import sha256_file


def _phase(task_env) -> torch.Tensor:
    terms = task_env._compute_landing_terms()
    phase = torch.zeros(task_env.num_envs, dtype=torch.int8, device=task_env.device)
    align = terms["align_candidate"] | (
        (terms["horizontal_error"] < 1.5 * task_env.cfg.align_radius)
        & (terms["robot_height_above_pad"] < task_env.cfg.align_height_max)
    )
    phase[align] = 1
    phase[terms["can_land"] & (~terms["deck_contact"])] = 2
    phase[terms["deck_contact"]] = 3
    return phase


def _new_buffer() -> dict[str, list[np.ndarray | float | int]]:
    return {
        "step_id": [],
        "raw_observation": [],
        "action": [],
        "reward": [],
        "flight_phase": [],
        "robot_position_w": [],
        "robot_quaternion_w": [],
        "robot_linear_velocity_w": [],
        "robot_angular_velocity_w": [],
        "deck_position_w": [],
        "deck_linear_velocity_w": [],
    }


def _failure_type(task_env, env_id: int) -> str:
    if bool(task_env._last_hard_contact[env_id].item()):
        return "hard_contact"
    if bool(task_env._last_ground_crash[env_id].item()):
        return "ground_crash"
    if bool(task_env._last_deck_miss[env_id].item()):
        return "deck_miss"
    if bool(task_env._terminal_time_out[env_id].item()):
        return "timeout"
    return "other"


def _matches(task_env, env_id: int) -> tuple[bool, str]:
    settled = bool(task_env._last_successful_settle[env_id].item())
    if args_cli.target == "success":
        return settled, "settled_landing"
    kind = _failure_type(task_env, env_id)
    return (not settled and (args_cli.failure_type == "any" or kind == args_cli.failure_type)), kind


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    output = Path(args_cli.output).resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite rollout case: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    task_env = raw_env.unwrapped
    env = RlGamesVecEnvWrapper(raw_env, rl_device, clip_obs, clip_actions, obs_groups, concatenate)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = task_env.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()
    agent.is_deterministic = True

    observation = env.reset()
    if isinstance(observation, dict):
        observation = observation["obs"]
    task_env.episode_length_buf.zero_()
    _ = agent.get_batch_size(observation, 1)
    if agent.is_rnn:
        agent.init_rnn()
    buffers = [_new_buffer() for _ in range(task_env.num_envs)]
    result: tuple[int, str] | None = None

    for _ in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            raw_observation = agent.obs_to_torch(observation)
            action = torch.clamp(agent.get_action(raw_observation, is_deterministic=True), -1.0, 1.0)
            phase = _phase(task_env)
            snapshots = {
                "raw_observation": raw_observation.detach().cpu().numpy(),
                "action": action.detach().cpu().numpy(),
                "flight_phase": phase.detach().cpu().numpy(),
                "robot_position_w": task_env._robot.data.root_pos_w.detach().cpu().numpy(),
                "robot_quaternion_w": task_env._robot.data.root_quat_w.detach().cpu().numpy(),
                "robot_linear_velocity_w": task_env._robot.data.root_lin_vel_w.detach().cpu().numpy(),
                "robot_angular_velocity_w": task_env._robot.data.root_ang_vel_w.detach().cpu().numpy(),
                "deck_position_w": task_env._pad_pos_w.detach().cpu().numpy(),
                "deck_linear_velocity_w": task_env._pad_vel_w.detach().cpu().numpy(),
            }
            next_observation, reward, done, _ = env.step(action)
            rewards = torch.as_tensor(reward).detach().cpu().numpy()
            done_ids = torch.nonzero(torch.as_tensor(done, dtype=torch.bool, device=task_env.device), as_tuple=False).squeeze(-1)

        for env_id, buffer in enumerate(buffers):
            buffer["step_id"].append(len(buffer["step_id"]))
            buffer["reward"].append(float(rewards[env_id]))
            for name, values in snapshots.items():
                value = values[env_id]
                buffer[name].append(value.copy() if np.ndim(value) else int(value))

        if done_ids.numel() > 0:
            for env_id in done_ids.detach().cpu().tolist():
                if not bool(task_env._terminal_state_valid[env_id].item()):
                    raise RuntimeError("missing exact terminal-state latch")
                matched, outcome = _matches(task_env, env_id)
                if matched:
                    result = (env_id, outcome)
                    break
                buffers[env_id] = _new_buffer()
        if result is not None:
            break
        observation = next_observation["obs"] if isinstance(next_observation, dict) else next_observation

    if result is None:
        env.close()
        raise RuntimeError(f"no matching {args_cli.target} rollout found within {args_cli.max_steps} steps")
    env_id, outcome = result
    buffer = buffers[env_id]
    arrays = {
        "step_id": np.asarray(buffer["step_id"], dtype=np.int32),
        "raw_observation": np.asarray(buffer["raw_observation"], dtype=np.float32),
        "action": np.asarray(buffer["action"], dtype=np.float32),
        "reward": np.asarray(buffer["reward"], dtype=np.float32),
        "flight_phase": np.asarray(buffer["flight_phase"], dtype=np.int8),
        "robot_position_w": np.asarray(buffer["robot_position_w"], dtype=np.float32),
        "robot_quaternion_w": np.asarray(buffer["robot_quaternion_w"], dtype=np.float32),
        "robot_linear_velocity_w": np.asarray(buffer["robot_linear_velocity_w"], dtype=np.float32),
        "robot_angular_velocity_w": np.asarray(buffer["robot_angular_velocity_w"], dtype=np.float32),
        "deck_position_w": np.asarray(buffer["deck_position_w"], dtype=np.float32),
        "deck_linear_velocity_w": np.asarray(buffer["deck_linear_velocity_w"], dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.floating)):
        raise RuntimeError("rollout trajectory contains NaN or Inf")
    np.savez_compressed(output, **arrays)
    metadata = {
        "task_id": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args_cli.seed,
        "parallel_envs": args_cli.num_envs,
        "selected_env_id": env_id,
        "target": args_cli.target,
        "outcome": outcome,
        "steps": len(buffer["step_id"]),
        "trajectory": output.name,
        "trajectory_sha256": sha256_file(output),
        "phase_names": {"0": "approach", "1": "align", "2": "descent", "3": "contact_settle"},
        "terminal": {
            "settled_landing": bool(task_env._last_successful_settle[env_id].item()),
            "deck_contact": bool(task_env._last_deck_contact[env_id].item()),
            "hard_contact": bool(task_env._last_hard_contact[env_id].item()),
            "ground_crash": bool(task_env._last_ground_crash[env_id].item()),
            "deck_miss": bool(task_env._last_deck_miss[env_id].item()),
            "time_out": bool(task_env._terminal_time_out[env_id].item()),
            "touchdown_distance_m": float(task_env._last_landing_touchdown_distance[env_id].item()),
            "first_contact_xy_error_m": float(task_env._last_first_contact_xy_error[env_id].item()),
            "first_contact_normal_relative_speed_mps": float(task_env._last_first_contact_normal_rel_speed[env_id].item()),
            "first_contact_tangential_relative_speed_mps": float(task_env._last_first_contact_tangential_rel_speed[env_id].item()),
            "first_contact_body_deck_normal_angle_rad": float(task_env._last_first_contact_body_deck_normal_angle[env_id].item()),
            "maximum_penetration_m": float(task_env._last_maximum_penetration[env_id].item()),
            "robot_position_w": task_env._terminal_robot_pos_w[env_id].detach().cpu().tolist(),
            "deck_position_w": task_env._terminal_pad_pos_w[env_id].detach().cpu().tolist(),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_generated": False,
        "video_note": (
            "No interactive display was available. This script records an objective state/action trajectory only; "
            "headless offscreen video would require a separate render-enabled recorder."
        ),
    }
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

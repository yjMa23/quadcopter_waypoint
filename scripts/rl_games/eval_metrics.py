# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an rl_games quadcopter checkpoint without modifying the training environment.

This script is intentionally separate from the training environment. It computes success and hover metrics from
observations/state during play, so evaluation does not affect PPO training trajectories.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a quadcopter rl_games checkpoint and report hover metrics.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments for evaluation.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--episodes", type=int, default=256, help="Number of completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=20000, help="Safety limit on total simulation steps.")
parser.add_argument("--success_radius", type=float, default=0.5, help="Success distance threshold in meters.")
parser.add_argument("--strict_success_radius", type=float, default=0.2, help="Strict success distance threshold in meters.")
parser.add_argument("--stable_radius", type=float, default=0.3, help="Stable hover distance threshold in meters.")
parser.add_argument("--stable_lin_vel", type=float, default=0.25, help="Stable hover linear velocity threshold in m/s.")
parser.add_argument("--stable_ang_vel", type=float, default=0.8, help="Stable hover angular velocity threshold in rad/s.")
parser.add_argument("--csv", type=str, default=None, help="Optional path to save per-episode metrics as CSV.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadcopter_waypoint.tasks  # noqa: F401


def _tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    """Convert a 1-D tensor to a Python float list."""
    return [float(x) for x in tensor.detach().cpu().tolist()]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Evaluate an RL-Games agent."""

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    raw_env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    task_env = raw_env.unwrapped

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(raw_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    num_envs = task_env.num_envs
    device = task_env.device
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_strict_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_stable_hover = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_min_distance = torch.full((num_envs,), float("inf"), dtype=torch.float, device=device)

    completed: list[dict[str, float | bool | int]] = []
    step = 0

    while simulation_app.is_running() and len(completed) < args_cli.episodes and step < args_cli.max_steps:
        with torch.inference_mode():
            # Capture state before env.step(). The base DirectRLEnv resets terminated environments inside step(),
            # so these values are the last stable per-env metrics before any automatic reset happens.
            distance_to_goal = torch.linalg.norm(task_env._desired_pos_w - task_env._robot.data.root_pos_w, dim=1)
            lin_vel = torch.linalg.norm(task_env._robot.data.root_lin_vel_b, dim=1)
            ang_vel = torch.linalg.norm(task_env._robot.data.root_ang_vel_b, dim=1)

            episode_success |= distance_to_goal < args_cli.success_radius
            episode_strict_success |= distance_to_goal < args_cli.strict_success_radius
            episode_min_distance = torch.minimum(episode_min_distance, distance_to_goal)
            stable_hover = torch.logical_and(
                distance_to_goal < args_cli.stable_radius,
                torch.logical_and(lin_vel < args_cli.stable_lin_vel, ang_vel < args_cli.stable_ang_vel),
            )
            episode_stable_hover |= stable_hover

            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device)
            done_ids = torch.nonzero(dones_tensor, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                final_distances = _tensor_to_float_list(distance_to_goal[done_ids])
                final_lin_vels = _tensor_to_float_list(lin_vel[done_ids])
                final_ang_vels = _tensor_to_float_list(ang_vel[done_ids])
                min_distances = _tensor_to_float_list(episode_min_distance[done_ids])
                successes = episode_success[done_ids].detach().cpu().tolist()
                strict_successes = episode_strict_success[done_ids].detach().cpu().tolist()
                stable_hovers = episode_stable_hover[done_ids].detach().cpu().tolist()
                final_stables = stable_hover[done_ids].detach().cpu().tolist()
                terminated = task_env.reset_terminated[done_ids].detach().cpu().tolist()
                timed_out = task_env.reset_time_outs[done_ids].detach().cpu().tolist()

                for local_idx, env_id in enumerate(done_ids.detach().cpu().tolist()):
                    if len(completed) >= args_cli.episodes:
                        break
                    completed.append(
                        {
                            "episode": len(completed),
                            "env_id": int(env_id),
                            "final_distance": final_distances[local_idx],
                            "final_lin_vel": final_lin_vels[local_idx],
                            "final_ang_vel": final_ang_vels[local_idx],
                            "min_distance": min_distances[local_idx],
                            "success": bool(successes[local_idx]),
                            "strict_success": bool(strict_successes[local_idx]),
                            "stable_hover": bool(stable_hovers[local_idx]),
                            "final_stable_hover": bool(final_stables[local_idx]),
                            "terminated": bool(terminated[local_idx]),
                            "time_out": bool(timed_out[local_idx]),
                        }
                    )

                episode_success[done_ids] = False
                episode_strict_success[done_ids] = False
                episode_stable_hover[done_ids] = False
                episode_min_distance[done_ids] = float("inf")

                if agent.is_rnn and agent.states is not None:
                    for state in agent.states:
                        state[:, dones_tensor, :] = 0.0

        step += 1

    env.close()

    if not completed:
        print("[WARN] No completed episodes were collected. Increase --max_steps or check the task/checkpoint pair.")
        return

    def mean(key: str) -> float:
        return sum(float(ep[key]) for ep in completed) / len(completed)

    def rate(key: str) -> float:
        return sum(1.0 for ep in completed if bool(ep[key])) / len(completed)

    print("\n========== Evaluation Summary ==========")
    print(f"task: {args_cli.task}")
    print(f"checkpoint: {resume_path}")
    print(f"episodes: {len(completed)}")
    print(f"num_envs: {num_envs}")
    print(f"steps: {step}")
    print(f"success_rate@{args_cli.success_radius:.2f}m: {rate('success'):.4f}")
    print(f"strict_success_rate@{args_cli.strict_success_radius:.2f}m: {rate('strict_success'):.4f}")
    print(f"stable_hover_rate: {rate('stable_hover'):.4f}")
    print(f"final_stable_hover_rate: {rate('final_stable_hover'):.4f}")
    print(f"termination_rate: {rate('terminated'):.4f}")
    print(f"timeout_rate: {rate('time_out'):.4f}")
    print(f"mean_final_distance: {mean('final_distance'):.4f} m")
    print(f"mean_min_distance: {mean('min_distance'):.4f} m")
    print(f"mean_final_lin_vel: {mean('final_lin_vel'):.4f} m/s")
    print(f"mean_final_ang_vel: {mean('final_ang_vel'):.4f} rad/s")

    if args_cli.csv is not None:
        csv_path = Path(args_cli.csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(completed[0].keys()))
            writer.writeheader()
            writer.writerows(completed)
        print(f"[INFO] Saved per-episode metrics to: {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

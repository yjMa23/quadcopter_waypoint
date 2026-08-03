# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate fixed-target, continuous-waypoint, and ship-landing rl_games quadcopter checkpoints.

Fixed-target tasks use state-derived hover metrics. Continuous-waypoint tasks consume exact one-step reach events
published by the environment, so waypoint switches cannot hide successful arrivals from the evaluator.
Ship-landing tasks consume terminal landing/crash events published by the environment.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from eval_metrics_utils import (
    DECK_ANGULAR_SPEED_BUCKETS,
    DECK_TILT_BUCKETS,
    PAD_SPEED_BUCKETS,
    deck_angular_speed_bucket,
    deck_tilt_bucket,
    mean_or_nan,
    pad_speed_bucket,
    percentile_or_nan,
)
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
    # Training staggers the initial episode counters to smooth reset spikes. Evaluation needs complete episodes.
    task_env.episode_length_buf.zero_()
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    num_envs = task_env.num_envs
    device = task_env.device
    continuous_waypoint = all(
        hasattr(task_env, name)
        for name in ("_waypoint_reached", "_waypoint_reach_distance", "_waypoint_reach_lin_vel")
    )
    ship_landing = all(
        hasattr(task_env, name)
        for name in ("_landing_success", "_landing_touchdown_distance", "_landing_touchdown_rel_vel", "_crash")
    )
    physical_deck = ship_landing and all(
        hasattr(task_env, name)
        for name in (
            "_last_successful_settle",
            "_last_deck_contact",
            "_last_hard_contact",
            "_last_ground_crash",
            "_last_deck_miss",
        )
    )
    physical_deck_attitude = physical_deck and all(
        hasattr(task_env, name)
        for name in (
            "_last_first_contact_deck_roll",
            "_last_first_contact_deck_pitch",
            "_last_first_contact_deck_tilt",
            "_last_first_contact_deck_angular_speed",
            "_last_first_contact_body_deck_normal_angle",
            "_last_terminal_body_deck_normal_angle",
            "_last_terminal_normal_relative_speed",
            "_last_terminal_tangential_relative_speed",
            "_last_max_contact_impulse",
        )
    )
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_strict_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_stable_hover = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_min_distance = torch.full((num_envs,), float("inf"), dtype=torch.float, device=device)
    episode_waypoint_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_waypoint_reach_distance_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_waypoint_reach_lin_vel_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_descent_speed_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_max_descent_speed = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_horizontal_speed_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_max_horizontal_speed = torch.zeros(num_envs, dtype=torch.float, device=device)
    episode_pad_speed = torch.zeros(num_envs, dtype=torch.float, device=device)
    if ship_landing:
        episode_pad_speed.copy_(torch.linalg.norm(task_env._pad_vel_w[:, :2], dim=1))

    completed: list[dict[str, float | bool | int | str]] = []
    step = 0

    while simulation_app.is_running() and len(completed) < args_cli.episodes and step < args_cli.max_steps:
        with torch.inference_mode():
            # Capture state before env.step(). The base DirectRLEnv resets terminated environments inside step(),
            # so these values are the last stable per-env metrics before any automatic reset happens.
            if ship_landing:
                distance = torch.linalg.norm(task_env._pad_pos_w - task_env._robot.data.root_pos_w, dim=1)
            else:
                distance = torch.linalg.norm(task_env._desired_pos_w - task_env._robot.data.root_pos_w, dim=1)
            lin_vel = torch.linalg.norm(task_env._robot.data.root_lin_vel_b, dim=1)
            ang_vel = torch.linalg.norm(task_env._robot.data.root_ang_vel_b, dim=1)
            if ship_landing:
                root_lin_vel_w = task_env._robot.data.root_lin_vel_w
                descent_speed = torch.clamp(-root_lin_vel_w[:, 2], min=0.0)
                horizontal_speed = torch.linalg.norm(root_lin_vel_w[:, :2] - task_env._pad_vel_w[:, :2], dim=1)
                episode_step_count += 1
                episode_descent_speed_sum += descent_speed
                episode_max_descent_speed = torch.maximum(episode_max_descent_speed, descent_speed)
                episode_horizontal_speed_sum += horizontal_speed
                episode_max_horizontal_speed = torch.maximum(episode_max_horizontal_speed, horizontal_speed)

            episode_min_distance = torch.minimum(episode_min_distance, distance)
            if not ship_landing:
                stable_hover = torch.logical_and(
                    distance < args_cli.stable_radius,
                    torch.logical_and(lin_vel < args_cli.stable_lin_vel, ang_vel < args_cli.stable_ang_vel),
                )
                if not continuous_waypoint:
                    episode_success |= distance < args_cli.success_radius
                    episode_strict_success |= distance < args_cli.strict_success_radius
                    episode_stable_hover |= stable_hover

            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            if continuous_waypoint:
                waypoint_reached = task_env._waypoint_reached
                episode_waypoint_count += waypoint_reached.long()
                episode_waypoint_reach_distance_sum += task_env._waypoint_reach_distance
                episode_waypoint_reach_lin_vel_sum += task_env._waypoint_reach_lin_vel
                episode_success |= waypoint_reached
                episode_strict_success |= torch.logical_and(
                    waypoint_reached, task_env._waypoint_reach_distance < args_cli.strict_success_radius
                )

            dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device)
            done_ids = torch.nonzero(dones_tensor, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                if ship_landing:
                    terminal_valid = task_env._terminal_state_valid[done_ids]
                    if not torch.all(terminal_valid):
                        raise RuntimeError("Ship-landing task did not publish an exact terminal-state latch before reset.")

                    terminal_robot_pos_w = task_env._terminal_robot_pos_w[done_ids]
                    terminal_robot_lin_vel_w = task_env._terminal_robot_lin_vel_w[done_ids]
                    terminal_pad_pos_w = task_env._terminal_pad_pos_w[done_ids]
                    terminal_pad_vel_w = task_env._terminal_pad_vel_w[done_ids]
                    terminal_relative_vel_w = task_env._terminal_relative_vel_w[done_ids]
                    terminal_distances = torch.linalg.norm(terminal_pad_pos_w - terminal_robot_pos_w, dim=1)
                    final_distances = _tensor_to_float_list(terminal_distances)
                    min_distances = _tensor_to_float_list(
                        torch.minimum(episode_min_distance[done_ids], terminal_distances)
                    )
                    terminated = task_env._terminal_terminated[done_ids].detach().cpu().tolist()
                    timed_out = task_env._terminal_time_out[done_ids].detach().cpu().tolist()
                    align_successes = getattr(task_env, "_last_align_success", task_env._landing_success)[
                        done_ids
                    ].detach().cpu().tolist()
                    successes = getattr(task_env, "_last_landing_success", task_env._landing_success)[
                        done_ids
                    ].detach().cpu().tolist()
                    touchdown_distances = _tensor_to_float_list(
                        getattr(
                            task_env, "_last_landing_touchdown_distance", task_env._landing_touchdown_distance
                        )[done_ids]
                    )
                    touchdown_rel_vels = _tensor_to_float_list(
                        getattr(task_env, "_last_landing_touchdown_rel_vel", task_env._landing_touchdown_rel_vel)[
                            done_ids
                        ]
                    )
                    crashes = getattr(task_env, "_last_crash", task_env._crash)[done_ids].detach().cpu().tolist()
                    step_counts = episode_step_count[done_ids].clamp_min(1)
                    landing_times = _tensor_to_float_list(step_counts.float() * task_env.step_dt)
                    max_descent_speeds = _tensor_to_float_list(episode_max_descent_speed[done_ids])
                    mean_descent_speeds = _tensor_to_float_list(episode_descent_speed_sum[done_ids] / step_counts)
                    max_horizontal_speeds = _tensor_to_float_list(episode_max_horizontal_speed[done_ids])
                    mean_horizontal_speeds = _tensor_to_float_list(episode_horizontal_speed_sum[done_ids] / step_counts)
                    final_vertical_speeds = _tensor_to_float_list(terminal_robot_lin_vel_w[:, 2])
                    final_horizontal_speeds = _tensor_to_float_list(
                        torch.linalg.norm(terminal_relative_vel_w[:, :2], dim=1)
                    )
                    terminal_vertical_relative_speeds = _tensor_to_float_list(terminal_relative_vel_w[:, 2])
                    terminal_relative_speeds = _tensor_to_float_list(torch.linalg.norm(terminal_relative_vel_w, dim=1))
                    terminal_surface_clearances = _tensor_to_float_list(
                        task_env._terminal_surface_clearance[done_ids]
                    )
                    terminal_horizontal_errors = _tensor_to_float_list(
                        task_env._terminal_horizontal_error[done_ids]
                    )
                    terminal_pad_vertical_speeds = _tensor_to_float_list(terminal_pad_vel_w[:, 2])
                    pad_speeds = _tensor_to_float_list(torch.linalg.norm(terminal_pad_vel_w[:, :2], dim=1))
                    pad_speed_buckets = [pad_speed_bucket(speed) for speed in pad_speeds]
                    if physical_deck:
                        contact_successes = task_env._last_deck_contact[done_ids].detach().cpu().tolist()
                        settled_landings = task_env._last_successful_settle[done_ids].detach().cpu().tolist()
                        hard_contacts = task_env._last_hard_contact[done_ids].detach().cpu().tolist()
                        ground_crashes = task_env._last_ground_crash[done_ids].detach().cpu().tolist()
                        deck_misses = task_env._last_deck_miss[done_ids].detach().cpu().tolist()
                        first_contact_seen = task_env._last_first_contact_seen[done_ids].detach().cpu().tolist()
                        first_contact_xy_errors = _tensor_to_float_list(
                            task_env._last_first_contact_xy_error[done_ids]
                        )
                        first_contact_normal_rel_speeds = _tensor_to_float_list(
                            task_env._last_first_contact_normal_rel_speed[done_ids]
                        )
                        first_contact_tangential_rel_speeds = _tensor_to_float_list(
                            task_env._last_first_contact_tangential_rel_speed[done_ids]
                        )
                        first_contact_forces = _tensor_to_float_list(task_env._last_first_contact_force[done_ids])
                        max_contact_forces = _tensor_to_float_list(task_env._last_max_contact_force[done_ids])
                        settle_times = _tensor_to_float_list(task_env._last_settle_time[done_ids])
                        minimum_surface_clearances = _tensor_to_float_list(
                            task_env._last_minimum_surface_clearance[done_ids]
                        )
                        maximum_penetrations = _tensor_to_float_list(task_env._last_maximum_penetration[done_ids])
                        if physical_deck_attitude:
                            first_contact_deck_rolls = _tensor_to_float_list(
                                task_env._last_first_contact_deck_roll[done_ids]
                            )
                            first_contact_deck_pitches = _tensor_to_float_list(
                                task_env._last_first_contact_deck_pitch[done_ids]
                            )
                            first_contact_deck_tilts = _tensor_to_float_list(
                                task_env._last_first_contact_deck_tilt[done_ids]
                            )
                            first_contact_deck_angular_speeds = _tensor_to_float_list(
                                task_env._last_first_contact_deck_angular_speed[done_ids]
                            )
                            first_contact_body_deck_angles = _tensor_to_float_list(
                                task_env._last_first_contact_body_deck_normal_angle[done_ids]
                            )
                            terminal_body_deck_angles = _tensor_to_float_list(
                                task_env._last_terminal_body_deck_normal_angle[done_ids]
                            )
                            terminal_normal_rel_speeds = _tensor_to_float_list(
                                task_env._last_terminal_normal_relative_speed[done_ids]
                            )
                            terminal_tangential_rel_speeds = _tensor_to_float_list(
                                task_env._last_terminal_tangential_relative_speed[done_ids]
                            )
                            max_contact_impulses = _tensor_to_float_list(task_env._last_max_contact_impulse[done_ids])
                            terminal_deck_rolls = _tensor_to_float_list(task_env._last_terminal_deck_roll[done_ids])
                            terminal_deck_pitches = _tensor_to_float_list(task_env._last_terminal_deck_pitch[done_ids])
                            terminal_deck_tilts = _tensor_to_float_list(task_env._last_terminal_deck_tilt[done_ids])
                            terminal_deck_angular_speeds = _tensor_to_float_list(
                                task_env._last_terminal_deck_angular_speed[done_ids]
                            )
                            max_deck_position_errors = _tensor_to_float_list(
                                task_env._last_max_deck_position_consistency_error[done_ids]
                            )
                            max_deck_orientation_errors = _tensor_to_float_list(
                                task_env._last_max_deck_orientation_consistency_error[done_ids]
                            )
                            max_deck_linear_velocity_errors = _tensor_to_float_list(
                                task_env._last_max_deck_linear_velocity_consistency_error[done_ids]
                            )
                            max_deck_angular_velocity_errors = _tensor_to_float_list(
                                task_env._last_max_deck_angular_velocity_consistency_error[done_ids]
                            )
                            deck_tilt_buckets = [deck_tilt_bucket(value) for value in first_contact_deck_tilts]
                            deck_angular_speed_buckets = [
                                deck_angular_speed_bucket(value) for value in first_contact_deck_angular_speeds
                            ]
                else:
                    final_distances = _tensor_to_float_list(distance[done_ids])
                    min_distances = _tensor_to_float_list(episode_min_distance[done_ids])
                    terminated = task_env.reset_terminated[done_ids].detach().cpu().tolist()
                    timed_out = task_env.reset_time_outs[done_ids].detach().cpu().tolist()
                    final_lin_vels = _tensor_to_float_list(lin_vel[done_ids])
                    final_ang_vels = _tensor_to_float_list(ang_vel[done_ids])
                    successes = episode_success[done_ids].detach().cpu().tolist()
                    strict_successes = episode_strict_success[done_ids].detach().cpu().tolist()
                    stable_hovers = episode_stable_hover[done_ids].detach().cpu().tolist()
                    final_stables = stable_hover[done_ids].detach().cpu().tolist()
                    waypoint_counts = episode_waypoint_count[done_ids]
                    mean_waypoint_reach_distances = torch.where(
                        waypoint_counts > 0,
                        episode_waypoint_reach_distance_sum[done_ids] / waypoint_counts.clamp_min(1),
                        torch.zeros_like(episode_waypoint_reach_distance_sum[done_ids]),
                    )
                    mean_waypoint_reach_lin_vels = torch.where(
                        waypoint_counts > 0,
                        episode_waypoint_reach_lin_vel_sum[done_ids] / waypoint_counts.clamp_min(1),
                        torch.zeros_like(episode_waypoint_reach_lin_vel_sum[done_ids]),
                    )
                    waypoint_counts_list = waypoint_counts.detach().cpu().tolist()
                    mean_waypoint_reach_distances_list = _tensor_to_float_list(mean_waypoint_reach_distances)
                    mean_waypoint_reach_lin_vels_list = _tensor_to_float_list(mean_waypoint_reach_lin_vels)

                for local_idx, env_id in enumerate(done_ids.detach().cpu().tolist()):
                    if len(completed) >= args_cli.episodes:
                        break
                    if ship_landing:
                        completed.append(
                            {
                                "episode": len(completed),
                                "env_id": int(env_id),
                                "align_success": bool(align_successes[local_idx]),
                                "success": bool(successes[local_idx]),
                                "final_distance": final_distances[local_idx],
                                "min_distance": min_distances[local_idx],
                                "touchdown_distance": touchdown_distances[local_idx],
                                "touchdown_rel_vel": touchdown_rel_vels[local_idx],
                                "landing_time": landing_times[local_idx],
                                "max_descent_speed": max_descent_speeds[local_idx],
                                "mean_descent_speed": mean_descent_speeds[local_idx],
                                # final_vertical_speed is the robot's terminal world-z velocity. Relative fields
                                # are computed against the terminal pad velocity in the world/deck-normal frame.
                                "final_vertical_speed": final_vertical_speeds[local_idx],
                                "terminal_vertical_relative_speed": terminal_vertical_relative_speeds[local_idx],
                                "terminal_relative_speed": terminal_relative_speeds[local_idx],
                                "terminal_horizontal_error": terminal_horizontal_errors[local_idx],
                                "terminal_surface_clearance": terminal_surface_clearances[local_idx],
                                "terminal_pad_vertical_speed": terminal_pad_vertical_speeds[local_idx],
                                "max_horizontal_speed": max_horizontal_speeds[local_idx],
                                "mean_horizontal_speed": mean_horizontal_speeds[local_idx],
                                "final_horizontal_speed": final_horizontal_speeds[local_idx],
                                "pad_speed": pad_speeds[local_idx],
                                "pad_speed_bucket": pad_speed_buckets[local_idx],
                                "crash": bool(crashes[local_idx]),
                                "terminated": bool(terminated[local_idx]),
                                "time_out": bool(timed_out[local_idx]),
                            }
                        )
                        if physical_deck:
                            completed[-1].update(
                                {
                                    "contact_success": bool(contact_successes[local_idx]),
                                    "settled_landing": bool(settled_landings[local_idx]),
                                    "hard_contact": bool(hard_contacts[local_idx]),
                                    "ground_crash": bool(ground_crashes[local_idx]),
                                    "deck_miss": bool(deck_misses[local_idx]),
                                    "first_contact_seen": bool(first_contact_seen[local_idx]),
                                    "first_contact_xy_error_deck_frame": first_contact_xy_errors[local_idx],
                                    "first_contact_normal_rel_speed": first_contact_normal_rel_speeds[local_idx],
                                    "first_contact_tangential_rel_speed": first_contact_tangential_rel_speeds[local_idx],
                                    "first_contact_force": first_contact_forces[local_idx],
                                    "max_contact_force": max_contact_forces[local_idx],
                                    "settle_time": settle_times[local_idx],
                                    "terminal_xy_error": terminal_horizontal_errors[local_idx],
                                    "minimum_surface_clearance": minimum_surface_clearances[local_idx],
                                    "maximum_penetration": maximum_penetrations[local_idx],
                                }
                            )
                            if physical_deck_attitude:
                                completed[-1].update(
                                    {
                                        "first_contact_deck_roll": first_contact_deck_rolls[local_idx],
                                        "first_contact_deck_pitch": first_contact_deck_pitches[local_idx],
                                        "first_contact_deck_tilt": first_contact_deck_tilts[local_idx],
                                        "first_contact_deck_angular_speed": first_contact_deck_angular_speeds[local_idx],
                                        "first_contact_body_deck_normal_angle": first_contact_body_deck_angles[local_idx],
                                        "terminal_body_deck_normal_angle": terminal_body_deck_angles[local_idx],
                                        "terminal_normal_relative_speed": terminal_normal_rel_speeds[local_idx],
                                        "terminal_tangential_relative_speed": terminal_tangential_rel_speeds[local_idx],
                                        "max_contact_impulse": max_contact_impulses[local_idx],
                                        "terminal_deck_roll": terminal_deck_rolls[local_idx],
                                        "terminal_deck_pitch": terminal_deck_pitches[local_idx],
                                        "terminal_deck_tilt": terminal_deck_tilts[local_idx],
                                        "terminal_deck_angular_speed": terminal_deck_angular_speeds[local_idx],
                                        "deck_tilt_bucket": deck_tilt_buckets[local_idx],
                                        "deck_angular_speed_bucket": deck_angular_speed_buckets[local_idx],
                                        "max_deck_position_consistency_error": max_deck_position_errors[local_idx],
                                        "max_deck_orientation_consistency_error": max_deck_orientation_errors[local_idx],
                                        "max_deck_linear_velocity_consistency_error": max_deck_linear_velocity_errors[local_idx],
                                        "max_deck_angular_velocity_consistency_error": max_deck_angular_velocity_errors[local_idx],
                                    }
                                )
                    else:
                        completed.append({
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
                            "waypoint_count": int(waypoint_counts_list[local_idx]),
                            "mean_waypoint_reach_distance": mean_waypoint_reach_distances_list[local_idx],
                            "mean_waypoint_reach_lin_vel": mean_waypoint_reach_lin_vels_list[local_idx],
                            "terminated": bool(terminated[local_idx]),
                            "time_out": bool(timed_out[local_idx]),
                        })

                episode_success[done_ids] = False
                episode_strict_success[done_ids] = False
                episode_stable_hover[done_ids] = False
                episode_min_distance[done_ids] = float("inf")
                episode_waypoint_count[done_ids] = 0
                episode_waypoint_reach_distance_sum[done_ids] = 0.0
                episode_waypoint_reach_lin_vel_sum[done_ids] = 0.0
                episode_step_count[done_ids] = 0
                episode_descent_speed_sum[done_ids] = 0.0
                episode_max_descent_speed[done_ids] = 0.0
                episode_horizontal_speed_sum[done_ids] = 0.0
                episode_max_horizontal_speed[done_ids] = 0.0
                if ship_landing:
                    episode_pad_speed[done_ids] = torch.linalg.norm(task_env._pad_vel_w[done_ids, :2], dim=1)

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

    def success_values(key: str) -> list[float]:
        return [float(ep[key]) for ep in completed if bool(ep["success"])]

    def success_mean(key: str) -> float:
        return mean_or_nan(success_values(key))

    def waypoint_event_mean(key: str) -> float:
        waypoint_count = sum(int(ep["waypoint_count"]) for ep in completed)
        if waypoint_count == 0:
            return float("nan")
        return sum(float(ep[key]) * int(ep["waypoint_count"]) for ep in completed) / waypoint_count

    print("\n========== Evaluation Summary ==========")
    print(f"task: {args_cli.task}")
    print(f"checkpoint: {resume_path}")
    print(f"episodes: {len(completed)}")
    print(f"num_envs: {num_envs}")
    print(f"steps: {step}")
    if ship_landing:
        print(f"align_success_rate: {rate('align_success'):.4f}")
        print(f"landing_success_rate: {rate('success'):.4f}")
        print(f"mean_final_distance: {mean('final_distance'):.4f} m")
        print(f"mean_min_distance: {mean('min_distance'):.4f} m")
        touchdown_distances = success_values("touchdown_distance")
        print(f"mean_touchdown_distance: {mean_or_nan(touchdown_distances):.4f} m")
        print(f"touchdown_distance_P50: {percentile_or_nan(touchdown_distances, 50.0):.4f} m")
        print(f"touchdown_distance_P90: {percentile_or_nan(touchdown_distances, 90.0):.4f} m")
        print(f"touchdown_distance_P95: {percentile_or_nan(touchdown_distances, 95.0):.4f} m")
        print(f"mean_touchdown_rel_vel: {success_mean('touchdown_rel_vel'):.4f} m/s")
        print(f"mean_landing_time: {success_mean('landing_time'):.4f} s")
        print(f"mean_max_descent_speed: {success_mean('max_descent_speed'):.4f} m/s")
        print(f"mean_descent_speed: {success_mean('mean_descent_speed'):.4f} m/s")
        print(f"mean_final_vertical_speed: {success_mean('final_vertical_speed'):.4f} m/s")
        print(
            f"mean_terminal_vertical_relative_speed: "
            f"{success_mean('terminal_vertical_relative_speed'):.4f} m/s"
        )
        print(f"mean_terminal_relative_speed: {success_mean('terminal_relative_speed'):.4f} m/s")
        print(f"mean_terminal_surface_clearance: {success_mean('terminal_surface_clearance'):.4f} m")
        print(f"mean_max_horizontal_speed: {success_mean('max_horizontal_speed'):.4f} m/s")
        print(f"mean_horizontal_speed: {success_mean('mean_horizontal_speed'):.4f} m/s")
        print(f"mean_final_horizontal_speed: {success_mean('final_horizontal_speed'):.4f} m/s")
        print(f"mean_pad_speed: {mean('pad_speed'):.4f} m/s")
        print(f"crash_rate: {rate('crash'):.4f}")
        if physical_deck:
            successful_first_contact_xy = [
                float(ep["first_contact_xy_error_deck_frame"])
                for ep in completed
                if bool(ep["success"]) and bool(ep["first_contact_seen"])
            ]
            all_first_contact_xy = [
                float(ep["first_contact_xy_error_deck_frame"])
                for ep in completed
                if bool(ep["first_contact_seen"])
            ]
            print(f"contact_success_rate: {rate('contact_success'):.4f}")
            print(f"settled_landing_rate: {rate('settled_landing'):.4f}")
            print(f"hard_contact_rate: {rate('hard_contact'):.4f}")
            print(f"ground_crash_rate: {rate('ground_crash'):.4f}")
            print(f"deck_miss_rate: {rate('deck_miss'):.4f}")
            print(
                f"successful_first_contact_xy_error_deck_frame_P95: "
                f"{percentile_or_nan(successful_first_contact_xy, 95.0):.4f} m"
            )
            print(
                f"all_contact_first_contact_xy_error_deck_frame_P95: "
                f"{percentile_or_nan(all_first_contact_xy, 95.0):.4f} m"
            )
            print(f"mean_first_contact_normal_rel_speed: {success_mean('first_contact_normal_rel_speed'):.4f} m/s")
            print(
                f"mean_first_contact_tangential_rel_speed: "
                f"{success_mean('first_contact_tangential_rel_speed'):.4f} m/s"
            )
            print(f"mean_max_contact_force: {success_mean('max_contact_force'):.4f} N")
            print(f"mean_settle_time: {success_mean('settle_time'):.4f} s")
            print(f"mean_maximum_penetration: {success_mean('maximum_penetration'):.4f} m")
            print(f"maximum_penetration_max: {max(float(ep['maximum_penetration']) for ep in completed):.4f} m")
            if physical_deck_attitude:
                successful_normal_speeds = [
                    abs(float(ep["first_contact_normal_rel_speed"]))
                    for ep in completed
                    if bool(ep["success"]) and bool(ep["first_contact_seen"])
                ]
                successful_body_deck_angles = [
                    math.degrees(float(ep["first_contact_body_deck_normal_angle"]))
                    for ep in completed
                    if bool(ep["success"]) and bool(ep["first_contact_seen"])
                ]
                print(
                    "successful_first_contact_normal_relative_speed_P95: "
                    f"{percentile_or_nan(successful_normal_speeds, 95.0):.4f} m/s"
                )
                print(
                    "successful_first_contact_body_deck_normal_angle_P95: "
                    f"{percentile_or_nan(successful_body_deck_angles, 95.0):.4f} deg"
                )
                print(f"mean_max_contact_impulse: {success_mean('max_contact_impulse'):.6f} N*s")
                print(
                    "max_deck_pose_consistency_error: "
                    f"{max(float(ep['max_deck_position_consistency_error']) for ep in completed):.6f} m, "
                    f"{math.degrees(max(float(ep['max_deck_orientation_consistency_error']) for ep in completed)):.6f} deg"
                )
                print(
                    "max_deck_velocity_consistency_error: "
                    f"{max(float(ep['max_deck_linear_velocity_consistency_error']) for ep in completed):.6f} m/s, "
                    f"{max(float(ep['max_deck_angular_velocity_consistency_error']) for ep in completed):.6f} rad/s"
                )
                for label, bucket_names, field_name in (
                    ("deck_tilt_buckets", DECK_TILT_BUCKETS, "deck_tilt_bucket"),
                    (
                        "deck_angular_speed_buckets",
                        DECK_ANGULAR_SPEED_BUCKETS,
                        "deck_angular_speed_bucket",
                    ),
                ):
                    print(f"{label}:")
                    for bucket in bucket_names:
                        bucket_eps = [ep for ep in completed if ep.get(field_name) == bucket]
                        if not bucket_eps:
                            continue
                        bucket_successes = [ep for ep in bucket_eps if bool(ep["success"])]
                        print(
                            f"  {bucket}: n={len(bucket_eps)}, "
                            f"settled_landing_rate={len(bucket_successes) / len(bucket_eps):.4f}, "
                            f"hard_contact_rate={sum(bool(ep['hard_contact']) for ep in bucket_eps) / len(bucket_eps):.4f}, "
                            f"ground_crash_rate={sum(bool(ep['ground_crash']) for ep in bucket_eps) / len(bucket_eps):.4f}, "
                            f"timeout_rate={sum(bool(ep['time_out']) for ep in bucket_eps) / len(bucket_eps):.4f}"
                        )
        print("pad_speed_buckets:")
        for bucket in PAD_SPEED_BUCKETS:
            bucket_eps = [ep for ep in completed if ep.get("pad_speed_bucket") == bucket]
            if not bucket_eps:
                continue
            bucket_successes = [ep for ep in bucket_eps if bool(ep["success"])]
            bucket_success_rate = len(bucket_successes) / len(bucket_eps)
            if bucket_successes:
                bucket_touchdown_distance = sum(float(ep["touchdown_distance"]) for ep in bucket_successes) / len(
                    bucket_successes
                )
                bucket_touchdown_rel_vel = sum(float(ep["touchdown_rel_vel"]) for ep in bucket_successes) / len(
                    bucket_successes
                )
            else:
                bucket_touchdown_distance = float("nan")
                bucket_touchdown_rel_vel = float("nan")
            print(
                f"  {bucket}: n={len(bucket_eps)}, success_rate={bucket_success_rate:.4f}, "
                f"mean_touchdown_distance={bucket_touchdown_distance:.4f} m, "
                f"mean_touchdown_rel_vel={bucket_touchdown_rel_vel:.4f} m/s"
            )
    elif continuous_waypoint:
        print(f"waypoint_episode_success_rate: {rate('success'):.4f}")
        print(f"mean_waypoints_per_episode: {mean('waypoint_count'):.4f}")
        print(f"mean_waypoint_reach_distance: {waypoint_event_mean('mean_waypoint_reach_distance'):.4f} m")
        print(f"mean_waypoint_reach_lin_vel: {waypoint_event_mean('mean_waypoint_reach_lin_vel'):.4f} m/s")
    else:
        print(f"success_rate@{args_cli.success_radius:.2f}m: {rate('success'):.4f}")
        print(f"strict_success_rate@{args_cli.strict_success_radius:.2f}m: {rate('strict_success'):.4f}")
        print(f"stable_hover_rate: {rate('stable_hover'):.4f}")
        print(f"final_stable_hover_rate: {rate('final_stable_hover'):.4f}")
        print(f"mean_final_distance: {mean('final_distance'):.4f} m")
        print(f"mean_min_distance: {mean('min_distance'):.4f} m")
        print(f"mean_final_lin_vel: {mean('final_lin_vel'):.4f} m/s")
        print(f"mean_final_ang_vel: {mean('final_ang_vel'):.4f} rad/s")
    print(f"termination_rate: {rate('terminated'):.4f}")
    print(f"timeout_rate: {rate('time_out'):.4f}")

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

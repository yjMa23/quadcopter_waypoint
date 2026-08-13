# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Collect successful deterministic physical-deck-attitude task teacher episodes into resumable compressed shards."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

ORIGINAL_ARGV = sys.argv.copy()

parser = argparse.ArgumentParser(description="Collect imitation-learning benchmark expert trajectories from the frozen physical-deck-attitude task teacher.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--successful_episodes", type=int, default=700)
parser.add_argument("--transitions", type=int, default=180000)
parser.add_argument("--episodes_per_shard", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=200000)
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
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
from quadcopter_waypoint.imitation.dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    save_shard,
    sha256_file,
    validate_shard_arrays,
    write_manifest,
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _new_episode_buffer() -> dict[str, list[Any]]:
    fields = (
        "step_id",
        "raw_observation",
        "teacher_action",
        "reward",
        "flight_phase",
        "deck_xy_velocity",
        "deck_heave_amplitude",
        "deck_heave_omega",
        "deck_roll_amplitude",
        "deck_roll_omega",
        "deck_pitch_amplitude",
        "deck_pitch_omega",
    )
    return {name: [] for name in fields}


def _phase_from_terms(task_env) -> torch.Tensor:
    terms = task_env._compute_landing_terms()
    phase = torch.zeros(task_env.num_envs, dtype=torch.int8, device=task_env.device)
    align_region = terms["align_candidate"] | (
        (terms["horizontal_error"] < 1.5 * task_env.cfg.align_radius)
        & (terms["robot_height_above_pad"] < task_env.cfg.align_height_max)
    )
    phase[align_region] = 1
    phase[terms["can_land"] & (~terms["deck_contact"])] = 2
    phase[terms["deck_contact"]] = 3
    return phase


def _episode_arrays(
    buffer: dict[str, list[Any]],
    episode_id: int,
    seed: int,
    terminated: bool,
    time_out: bool,
    outcome: dict[str, Any],
) -> dict[str, np.ndarray]:
    length = len(buffer["step_id"])
    arrays: dict[str, np.ndarray] = {
        "episode_id": np.full(length, episode_id, dtype=np.int64),
        "step_id": np.asarray(buffer["step_id"], dtype=np.int32),
        "seed": np.full(length, seed, dtype=np.int32),
        "raw_observation": np.asarray(buffer["raw_observation"], dtype=np.float32),
        "teacher_action": np.asarray(buffer["teacher_action"], dtype=np.float32),
        "reward": np.asarray(buffer["reward"], dtype=np.float32),
        "terminated": np.zeros(length, dtype=np.bool_),
        "time_out": np.zeros(length, dtype=np.bool_),
        "flight_phase": np.asarray(buffer["flight_phase"], dtype=np.int8),
        "deck_xy_velocity": np.asarray(buffer["deck_xy_velocity"], dtype=np.float32),
        "deck_heave_amplitude": np.asarray(buffer["deck_heave_amplitude"], dtype=np.float32),
        "deck_heave_omega": np.asarray(buffer["deck_heave_omega"], dtype=np.float32),
        "deck_roll_amplitude": np.asarray(buffer["deck_roll_amplitude"], dtype=np.float32),
        "deck_roll_omega": np.asarray(buffer["deck_roll_omega"], dtype=np.float32),
        "deck_pitch_amplitude": np.asarray(buffer["deck_pitch_amplitude"], dtype=np.float32),
        "deck_pitch_omega": np.asarray(buffer["deck_pitch_omega"], dtype=np.float32),
    }
    arrays["terminated"][-1] = terminated
    arrays["time_out"][-1] = time_out
    boolean_fields = ("contact_success", "settled_landing", "hard_contact", "ground_crash", "deck_miss")
    float_fields = (
        "touchdown_distance",
        "first_contact_xy_error",
        "first_contact_normal_relative_speed",
        "first_contact_tangential_relative_speed",
        "first_contact_body_deck_normal_angle",
        "maximum_penetration",
    )
    for name in boolean_fields:
        arrays[name] = np.full(length, bool(outcome[name]), dtype=np.bool_)
    for name in float_fields:
        arrays[name] = np.full(length, float(outcome[name]), dtype=np.float32)
    validate_shard_arrays(arrays)
    return arrays


def _merge_episode_arrays(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([episode[name] for episode in episodes], axis=0) for name in episodes[0]}


def _load_partial_manifest(output_dir: Path, checkpoint_sha256: str) -> dict[str, Any]:
    partial_path = output_dir / "partial_manifest.json"
    if not partial_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": args_cli.task,
            "teacher_checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "teacher_checkpoint_sha256": checkpoint_sha256,
            "seed": args_cli.seed,
            "observation_shape": [OBSERVATION_DIM],
            "action_shape": [ACTION_DIM],
            "observation_dtype": "float32",
            "action_dtype": "float32",
            "action_semantics": "deterministic RL-Games mean action clamped to [-1, 1] before env.step",
            "phase_names": {"0": "approach", "1": "align", "2": "descent", "3": "contact_settle"},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "collection_command": " ".join(shlex.quote(value) for value in ORIGINAL_ARGV),
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "isaaclab": _package_version("isaaclab"),
                "isaacsim": _package_version("isaacsim"),
                "rl_games": _package_version("rl-games"),
            },
            "shards": [],
            "successful_episode_count": 0,
            "transition_count": 0,
            "rejected_episode_count": 0,
        }
    manifest = json.loads(partial_path.read_text(encoding="utf-8"))
    if manifest["teacher_checkpoint_sha256"] != checkpoint_sha256 or int(manifest["seed"]) != args_cli.seed:
        raise RuntimeError("existing partial manifest belongs to a different teacher checkpoint or seed")
    for record in manifest["shards"]:
        shard_path = output_dir / record["path"]
        if not shard_path.is_file() or sha256_file(shard_path) != record["sha256"]:
            raise RuntimeError(f"existing shard is missing or corrupt: {shard_path}")
    return manifest


def _write_partial_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(output_dir / "partial_manifest.json", manifest, overwrite=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    manifest = _load_partial_manifest(output_dir, checkpoint_sha256)
    if (
        int(manifest["successful_episode_count"]) >= args_cli.successful_episodes
        and int(manifest["transition_count"]) >= args_cli.transitions
    ):
        print("[INFO] Requested collection target already satisfied; verified existing shards and exiting.")
        return

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    task_env = raw_env.unwrapped
    env = RlGamesVecEnvWrapper(raw_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint_path)
    agent_cfg["params"]["config"]["num_actors"] = task_env.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint_path))
    agent.reset()
    agent.is_deterministic = True

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    task_env.episode_length_buf.zero_()
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    buffers = [_new_episode_buffer() for _ in range(task_env.num_envs)]
    pending_successes: list[dict[str, np.ndarray]] = []
    next_local_episode = int(manifest["successful_episode_count"])
    rejected = int(manifest.get("rejected_episode_count", 0))
    total_steps = 0

    def flush_pending() -> None:
        nonlocal pending_successes
        if not pending_successes:
            return
        shard_index = len(manifest["shards"])
        shard_path = output_dir / f"shard_{shard_index:05d}.npz"
        arrays = _merge_episode_arrays(pending_successes)
        record = save_shard(shard_path, arrays)
        manifest["shards"].append(record)
        manifest["successful_episode_count"] = int(manifest["successful_episode_count"]) + record["episodes"]
        manifest["transition_count"] = int(manifest["transition_count"]) + record["transitions"]
        manifest["rejected_episode_count"] = rejected
        _write_partial_manifest(output_dir, manifest)
        print(
            f"[INFO] Saved {record['path']}: total_success={manifest['successful_episode_count']}, "
            f"total_transitions={manifest['transition_count']}, rejected={rejected}"
        )
        pending_successes = []

    while simulation_app.is_running() and total_steps < args_cli.max_steps:
        if (
            int(manifest["successful_episode_count"]) + len(pending_successes) >= args_cli.successful_episodes
            and int(manifest["transition_count"]) + sum(len(ep["step_id"]) for ep in pending_successes) >= args_cli.transitions
        ):
            break
        with torch.inference_mode():
            raw_obs_tensor = agent.obs_to_torch(obs)
            if raw_obs_tensor.shape[-1] != OBSERVATION_DIM:
                raise RuntimeError(f"teacher observation dimension is {raw_obs_tensor.shape[-1]}, expected {OBSERVATION_DIM}")
            action_tensor = agent.get_action(raw_obs_tensor, is_deterministic=True)
            action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            phase_tensor = _phase_from_terms(task_env)
            metadata_tensors = {
                "deck_xy_velocity": task_env._deck_xy_velocity_w,
                "deck_heave_amplitude": task_env._pad_heave_amp,
                "deck_heave_omega": task_env._pad_heave_omega,
                "deck_roll_amplitude": task_env._deck_roll_amp,
                "deck_roll_omega": task_env._deck_roll_omega,
                "deck_pitch_amplitude": task_env._deck_pitch_amp,
                "deck_pitch_omega": task_env._deck_pitch_omega,
            }
            obs_cpu = raw_obs_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            action_cpu = action_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            phase_cpu = phase_tensor.detach().cpu().numpy().astype(np.int8, copy=False)
            metadata_cpu = {name: value.detach().cpu().numpy() for name, value in metadata_tensors.items()}
            next_obs, rewards, dones, _ = env.step(action_tensor)
            rewards_cpu = torch.as_tensor(rewards).detach().cpu().numpy().astype(np.float32, copy=False)
            dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=task_env.device)
            done_ids = torch.nonzero(dones_tensor, as_tuple=False).squeeze(-1)

        for env_id in range(task_env.num_envs):
            buffer = buffers[env_id]
            buffer["step_id"].append(len(buffer["step_id"]))
            buffer["raw_observation"].append(obs_cpu[env_id].copy())
            buffer["teacher_action"].append(action_cpu[env_id].copy())
            buffer["reward"].append(float(rewards_cpu[env_id]))
            buffer["flight_phase"].append(int(phase_cpu[env_id]))
            for name in metadata_cpu:
                value = metadata_cpu[name][env_id]
                buffer[name].append(value.copy() if np.ndim(value) else float(value))

        if done_ids.numel() > 0:
            for env_id in done_ids.detach().cpu().tolist():
                terminal_valid = bool(task_env._terminal_state_valid[env_id].item())
                if not terminal_valid:
                    raise RuntimeError("task did not publish terminal-state latch before automatic reset")
                settled = bool(task_env._last_successful_settle[env_id].item())
                if settled:
                    episode_id = args_cli.seed * 1_000_000_000 + next_local_episode
                    outcome = {
                        "contact_success": bool(task_env._last_deck_contact[env_id].item()),
                        "settled_landing": settled,
                        "hard_contact": bool(task_env._last_hard_contact[env_id].item()),
                        "ground_crash": bool(task_env._last_ground_crash[env_id].item()),
                        "deck_miss": bool(task_env._last_deck_miss[env_id].item()),
                        "touchdown_distance": float(task_env._last_landing_touchdown_distance[env_id].item()),
                        "first_contact_xy_error": float(task_env._last_first_contact_xy_error[env_id].item()),
                        "first_contact_normal_relative_speed": float(
                            task_env._last_first_contact_normal_rel_speed[env_id].item()
                        ),
                        "first_contact_tangential_relative_speed": float(
                            task_env._last_first_contact_tangential_rel_speed[env_id].item()
                        ),
                        "first_contact_body_deck_normal_angle": float(
                            task_env._last_first_contact_body_deck_normal_angle[env_id].item()
                        ),
                        "maximum_penetration": float(task_env._last_maximum_penetration[env_id].item()),
                    }
                    pending_successes.append(
                        _episode_arrays(
                            buffers[env_id],
                            episode_id,
                            args_cli.seed,
                            bool(task_env._terminal_terminated[env_id].item()),
                            bool(task_env._terminal_time_out[env_id].item()),
                            outcome,
                        )
                    )
                    next_local_episode += 1
                else:
                    rejected += 1
                buffers[env_id] = _new_episode_buffer()
            if len(pending_successes) >= args_cli.episodes_per_shard:
                flush_pending()
        obs = next_obs["obs"] if isinstance(next_obs, dict) else next_obs
        total_steps += 1

    flush_pending()
    manifest["rejected_episode_count"] = rejected
    _write_partial_manifest(output_dir, manifest)
    if (
        int(manifest["successful_episode_count"]) < args_cli.successful_episodes
        or int(manifest["transition_count"]) < args_cli.transitions
    ):
        raise RuntimeError(
            f"collection stopped before target: successes={manifest['successful_episode_count']}/"
            f"{args_cli.successful_episodes}, transitions={manifest['transition_count']}/{args_cli.transitions}"
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

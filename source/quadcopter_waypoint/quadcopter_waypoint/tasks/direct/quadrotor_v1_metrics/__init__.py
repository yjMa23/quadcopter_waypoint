"""Waypoint v1 task registration."""

import gymnasium as gym

from ..quadrotor_official_clone import quadrotor_official_clone_env
from . import agents

gym.register(
    id="Isaac-Quadcopter-WaypointV1-Direct-v0",
    entry_point=f"{quadrotor_official_clone_env.__name__}:QuadcopterOfficialCloneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{quadrotor_official_clone_env.__name__}:QuadcopterOfficialCloneEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

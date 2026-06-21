"""Continuous waypoint v2 quadcopter task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Quadcopter-WaypointV2-Direct-v0",
    entry_point=f"{__name__}.quadrotor_waypoint_v2_env:QuadcopterWaypointV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrotor_waypoint_v2_env:QuadcopterWaypointV2EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

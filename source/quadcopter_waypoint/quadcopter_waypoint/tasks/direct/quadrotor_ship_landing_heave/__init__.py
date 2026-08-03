"""Heave ship landing quadcopter task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Quadcopter-ShipLanding-Heave-Direct-v0",
    entry_point=f"{__name__}.quadrotor_ship_landing_heave_env:QuadcopterShipLandingHeaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrotor_ship_landing_heave_env:QuadcopterShipLandingHeaveEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

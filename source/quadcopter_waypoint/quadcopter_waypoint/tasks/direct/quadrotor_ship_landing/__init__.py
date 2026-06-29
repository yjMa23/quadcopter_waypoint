"""Ship landing quadcopter task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Quadcopter-ShipLanding-Direct-v0",
    entry_point=f"{__name__}.quadrotor_ship_landing_env:QuadcopterShipLandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrotor_ship_landing_env:QuadcopterShipLandingEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

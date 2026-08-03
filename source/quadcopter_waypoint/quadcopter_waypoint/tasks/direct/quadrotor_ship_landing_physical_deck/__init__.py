"""Physical-deck ship landing quadcopter task."""

import gymnasium as gym


gym.register(
    id="Isaac-Quadcopter-ShipLanding-PhysicalDeck-Direct-v0",
    entry_point=f"{__name__}.quadrotor_ship_landing_physical_deck_env:QuadcopterShipLandingPhysicalDeckEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.quadrotor_ship_landing_physical_deck_env:"
            "QuadcopterShipLandingPhysicalDeckEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

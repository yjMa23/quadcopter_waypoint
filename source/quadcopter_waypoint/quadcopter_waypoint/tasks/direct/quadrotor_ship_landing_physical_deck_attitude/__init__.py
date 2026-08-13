"""Physical-deck attitude ship-landing quadcopter task."""

import gymnasium as gym


gym.register(
    id="Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0",
    entry_point=(
        f"{__name__}.quadrotor_ship_landing_physical_deck_attitude_env:"
        "QuadcopterShipLandingPhysicalDeckAttitudeEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.quadrotor_ship_landing_physical_deck_attitude_env:"
            "QuadcopterShipLandingPhysicalDeckAttitudeEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
        "rl_games_actor_preserving_cfg_entry_point": f"{__name__}.agents:rl_games_actor_preserving_ppo_cfg.yaml",
    },
)

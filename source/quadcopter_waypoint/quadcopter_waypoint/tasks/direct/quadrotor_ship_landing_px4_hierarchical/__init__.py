"""PX4-compatible hierarchical ship-landing quadcopter task."""

import gymnasium as gym


gym.register(
    id="Isaac-Quadcopter-ShipLanding-Px4Hierarchical-Direct-v0",
    entry_point=(
        f"{__name__}.quadrotor_ship_landing_px4_hierarchical_env:"
        "QuadcopterShipLandingPx4HierarchicalEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.quadrotor_ship_landing_px4_hierarchical_env:"
            "QuadcopterShipLandingPx4HierarchicalEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

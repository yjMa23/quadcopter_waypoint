"""Official-logic clone of Isaac Lab's quadcopter Direct RL task."""

import gymnasium as gym

from isaaclab_tasks.direct.quadcopter import agents as quadcopter_agents

gym.register(
    id="Isaac-Quadcopter-OfficialClone-Direct-v0",
    entry_point=f"{__name__}.quadrotor_official_clone_env:QuadcopterOfficialCloneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrotor_official_clone_env:QuadcopterOfficialCloneEnvCfg",
        "rl_games_cfg_entry_point": f"{quadcopter_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

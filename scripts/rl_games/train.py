"""Train rl_games agents for the quadcopter_waypoint external project.

The official Isaac Lab training script must launch Isaac Sim before importing task modules that depend on USD/pxr.
This wrapper injects the external task registration at Isaac Lab's extension placeholder, preserving the official launch order.
"""

from pathlib import Path

ISAACLAB_ROOT = Path.home() / "IsaacLab"
OFFICIAL_SCRIPT = ISAACLAB_ROOT / "scripts/reinforcement_learning/rl_games/train.py"

source = OFFICIAL_SCRIPT.read_text()
source = source.replace(
    "# PLACEHOLDER: Extension template (do not remove this comment)",
    "import quadcopter_waypoint.tasks  # noqa: F401",
)
source = source.replace(
    "runner.load(agent_cfg)",
    "from quadcopter_waypoint.imitation.actor_preserving_agent import register_actor_preserving_runner\n"
    "    register_actor_preserving_runner(runner)\n"
    "    runner.load(agent_cfg)",
)

code = compile(source, str(OFFICIAL_SCRIPT), "exec")
exec_globals = {"__name__": "__main__", "__file__": str(OFFICIAL_SCRIPT)}
exec(code, exec_globals)

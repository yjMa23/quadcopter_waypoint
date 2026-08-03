"""External Isaac Lab project for quadcopter waypoint RL tasks.

Task registration is intentionally explicit through ``import quadcopter_waypoint.tasks`` after AppLauncher starts.
Keeping the top-level package import side-effect free lets pure-Python utilities run without loading USD/PhysX modules.
"""

"""Isaac Lab backend (M6 scene smoke + M7 Reach/Grasp ManagerBased env).

Import heavy isaaclab modules only after AppLauncher / SimulationApp is running.
"""

from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD, USD_DIR

__all__ = ["PROXY_URDF", "ROBOT_USD", "USD_DIR", "make_reach_grasp_env_cfg"]


def make_reach_grasp_env_cfg(*args, **kwargs):
    """Lazy re-export — import after SimulationApp is running."""
    from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import (
        make_reach_grasp_env_cfg as _make,
    )

    return _make(*args, **kwargs)

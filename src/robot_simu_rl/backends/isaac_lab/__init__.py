"""Isaac Lab backend (M6 first slice: proxy URDF/USD + scene smoke).

Import heavy isaaclab modules only after AppLauncher / SimulationApp is running.
"""

from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD, USD_DIR

__all__ = ["PROXY_URDF", "ROBOT_USD", "USD_DIR"]

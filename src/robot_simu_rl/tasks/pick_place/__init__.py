"""Platform-agnostic pick-place task specs (obs/act/reward; M1 Reach first)."""

from robot_simu_rl.tasks.pick_place.rewards import ReachConfig, reach_distance, reach_reward, reach_target

__all__ = ["ReachConfig", "reach_distance", "reach_reward", "reach_target"]

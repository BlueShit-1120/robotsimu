"""MuJoCo-aligned action bounds for SB3 on ManagerBased Reach/Grasp envs."""

from __future__ import annotations

import numpy as np


def configure_sb3_action_space(env) -> None:
    """Set arm Δq ∈ [-1,1]^7 and gripper ∈ [0,1] on ``env`` action spaces.

    Isaac Lab's ActionManager defaults to unbounded Box; SB3 then samples
    gripper ~N(0,·) which clamps to open. MuJoCo PickPlaceMujocoEnv uses the
    bounded space below.
    """
    from gymnasium import spaces

    low = np.concatenate([np.full(7, -1.0, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    high = np.concatenate([np.full(7, 1.0, dtype=np.float32), np.array([1.0], dtype=np.float32)])
    env.single_action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    n = int(env.num_envs)
    env.action_space = spaces.Box(
        low=np.broadcast_to(low, (n, 8)).copy(),
        high=np.broadcast_to(high, (n, 8)).copy(),
        dtype=np.float32,
    )

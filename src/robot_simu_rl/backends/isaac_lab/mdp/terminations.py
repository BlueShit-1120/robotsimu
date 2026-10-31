"""Termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Re-export Lab's time_out for cfg convenience.
from isaaclab.envs.mdp import time_out  # noqa: F401


def early_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when reach/grasp success flag was set this step (extras log)."""
    log = env.extras.get("log", {})
    phase = getattr(env.cfg, "phase", "reach")
    if phase == "grasp":
        flag = log.get("grasp_success")
    else:
        flag = log.get("reach_success")
    if flag is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return flag.bool()

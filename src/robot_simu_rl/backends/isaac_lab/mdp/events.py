"""Reset helpers for Reach/Grasp env."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Match MuJoCo M1 spawn / table height.
_BOX_XY = (0.70, -0.20)
_BOX_Z = 0.765
_BOX_Z0 = 0.75  # table top (approx; half-height 0.015 → center 0.765)


def reset_aux_buffers(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Allocate / clear progress + success bookkeeping used by rewards."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = env.num_envs
    if not hasattr(env, "_prev_task_dist"):
        env._prev_task_dist = torch.zeros(n, device=env.device)
        env._reached_once = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._lifted_once = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._box_z0 = torch.full((n,), _BOX_Z0, device=env.device)
    env._prev_task_dist[env_ids] = 0.0
    env._reached_once[env_ids] = False
    env._lifted_once[env_ids] = False
    env._box_z0[env_ids] = _BOX_Z0


def reset_scene_and_workpiece(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("workpiece"),
    randomize_box: bool = False,
    xy_half_range: float = 0.04,
) -> None:
    """Reset robot/object to defaults, then place workpiece at MuJoCo spawn."""
    del robot_cfg  # defaults handle robot via reset_scene_to_default
    reset_scene_to_default(env, env_ids, reset_joint_targets=True)

    obj = env.scene[object_cfg.name]
    n = len(env_ids)
    pos = torch.zeros(n, 3, device=env.device)
    pos[:, 0] = _BOX_XY[0]
    pos[:, 1] = _BOX_XY[1]
    pos[:, 2] = _BOX_Z
    if randomize_box:
        jitter = (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * xy_half_range
        pos[:, 0:2] += jitter
    pos = pos + env.scene.env_origins[env_ids]
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # wxyz identity
    pose = torch.cat([pos, quat], dim=-1)
    vel = torch.zeros(n, 6, device=env.device)
    obj.write_root_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(vel, env_ids=env_ids)

    reset_aux_buffers(env, env_ids)

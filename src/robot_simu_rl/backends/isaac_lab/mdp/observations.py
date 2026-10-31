"""Observations aligned with MuJoCo PickPlaceMujocoEnv (31-dim)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from robot_simu_rl.common.joints import (
    ARM_JOINT_LIMITS,
    GRIPPER_JOINT_LIMIT,
    URDF_RIGHT_ARM_JOINTS,
    URDF_RIGHT_GRIPPER_JOINT,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Place goal XY (same as MuJoCo M3 default) — included in obs for parity.
_PLACE_GOAL_XY = (0.70, -0.12)

_ARM_LOW = torch.tensor([lo for lo, _ in ARM_JOINT_LIMITS], dtype=torch.float32)
_ARM_HIGH = torch.tensor([hi for _, hi in ARM_JOINT_LIMITS], dtype=torch.float32)


def mujoco_style_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("workpiece"),
    ee_body_name: str = "right_arm_ee_link",
) -> torch.Tensor:
    """Return MuJoCo-parity obs: qn(7)+dqn(7)+grip(1)+ee(7)+box(7)+goal_xy(2)=31."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    device = env.device

    # Resolve joint / body ids once and cache on env.
    if not hasattr(env, "_obs_arm_ids"):
        arm_ids, _ = robot.find_joints(list(URDF_RIGHT_ARM_JOINTS), preserve_order=True)
        grip_ids, _ = robot.find_joints([URDF_RIGHT_GRIPPER_JOINT], preserve_order=True)
        ee_ids, _ = robot.find_bodies([ee_body_name], preserve_order=True)
        env._obs_arm_ids = arm_ids
        env._obs_grip_id = grip_ids[0]
        env._obs_ee_id = ee_ids[0]
        env._obs_arm_low = _ARM_LOW.to(device)
        env._obs_arm_high = _ARM_HIGH.to(device)
        env._obs_goal_xy = torch.tensor(_PLACE_GOAL_XY, device=device, dtype=torch.float32)

    q = robot.data.joint_pos[:, env._obs_arm_ids]
    dq = robot.data.joint_vel[:, env._obs_arm_ids]
    mid = 0.5 * (env._obs_arm_low + env._obs_arm_high)
    half = 0.5 * (env._obs_arm_high - env._obs_arm_low)
    qn = (q - mid) / torch.clamp(half, min=1e-6)
    dqn = dq / torch.pi

    grip_lo, grip_hi = GRIPPER_JOINT_LIMIT
    grip_q = robot.data.joint_pos[:, env._obs_grip_id : env._obs_grip_id + 1]
    grip = (grip_q - grip_lo) / (grip_hi - grip_lo + 1e-9)

    ee_pos = robot.data.body_pos_w[:, env._obs_ee_id]
    ee_quat = robot.data.body_quat_w[:, env._obs_ee_id]  # wxyz
    # Express EE relative to env origin (world for single-env / env origins subtracted).
    ee_pos = ee_pos - env.scene.env_origins

    box_pos = obj.data.root_pos_w - env.scene.env_origins
    box_quat = obj.data.root_quat_w

    goal = env._obs_goal_xy.unsqueeze(0).expand(env.num_envs, -1)
    return torch.cat([qn, dqn, grip, ee_pos, ee_quat, box_pos, box_quat, goal], dim=-1)

"""Reach / Grasp rewards (MuJoCo-inspired dense shaping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from robot_simu_rl.common.joints import GRIPPER_JOINT_LIMIT, URDF_RIGHT_GRIPPER_JOINT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_ids(
    env: ManagerBasedRLEnv,
    robot,
    ee_body_name: str,
    tip_l_name: str,
    tip_r_name: str,
) -> None:
    if hasattr(env, "_rew_ee_id"):
        return
    ee_ids, _ = robot.find_bodies([ee_body_name], preserve_order=True)
    tip_l_ids, _ = robot.find_bodies([tip_l_name], preserve_order=True)
    tip_r_ids, _ = robot.find_bodies([tip_r_name], preserve_order=True)
    grip_ids, _ = robot.find_joints([URDF_RIGHT_GRIPPER_JOINT], preserve_order=True)
    env._rew_ee_id = ee_ids[0]
    env._rew_tip_l_id = tip_l_ids[0]
    env._rew_tip_r_id = tip_r_ids[0]
    env._rew_grip_id = grip_ids[0]


def _tip_mid(robot, env) -> torch.Tensor:
    p_l = robot.data.body_pos_w[:, env._rew_tip_l_id]
    p_r = robot.data.body_pos_w[:, env._rew_tip_r_id]
    return 0.5 * (p_l + p_r)


def _sensor_force_norm(sens, n: int) -> torch.Tensor:
    """Max contact-force magnitude from net and (if present) filtered force matrix."""
    device = sens.data.net_forces_w.device
    mag = torch.zeros(n, device=device)
    # net_forces_w: (N, B, 3) — any contact on the tip body
    mag = torch.maximum(mag, torch.norm(sens.data.net_forces_w, dim=-1).amax(dim=-1))
    # force_matrix_w: (N, B, M, 3) when filter_prim_paths_expr is set (tip↔Workpiece)
    fmat = getattr(sens.data, "force_matrix_w", None)
    if fmat is not None:
        mag = torch.maximum(mag, torch.norm(fmat.reshape(n, -1, 3), dim=-1).amax(dim=-1))
    # Prefer recent history peak when available (brief contacts).
    hist = getattr(sens.data, "net_forces_w_history", None)
    if hist is not None:
        # (N, T, B, 3)
        mag = torch.maximum(mag, torch.norm(hist, dim=-1).amax(dim=(-1, -2)))
    fhist = getattr(sens.data, "force_matrix_w_history", None)
    if fhist is not None:
        mag = torch.maximum(mag, torch.norm(fhist.reshape(n, -1, 3), dim=-1).amax(dim=-1))
    return mag


def _finger_contacts(
    env: ManagerBasedRLEnv, threshold: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (left_hit, right_hit, left_force, right_force)."""
    n = env.num_envs
    device = env.device
    left_f = torch.zeros(n, device=device)
    right_f = torch.zeros(n, device=device)
    if "tip_l_contact" in env.scene.keys():
        left_f = _sensor_force_norm(env.scene["tip_l_contact"], n)
    if "tip_r_contact" in env.scene.keys():
        right_f = _sensor_force_norm(env.scene["tip_r_contact"], n)
    return left_f > threshold, right_f > threshold, left_f, right_f


def _geometric_pinch(
    robot,
    env: ManagerBasedRLEnv,
    box: torch.Tensor,
    closed: torch.Tensor,
    tip_mid_dist: torch.Tensor,
) -> torch.Tensor:
    """Heuristic pinch when ContactSensor is silent: tips opposite + sep spans box."""
    p_l = robot.data.body_pos_w[:, env._rew_tip_l_id]
    p_r = robot.data.body_pos_w[:, env._rew_tip_r_id]
    sep = torch.norm(p_l - p_r, dim=-1)
    # Workpiece size (0.05, 0.04, 0.03); closed tip sep should cover ~box width.
    d_l = p_l - box
    d_r = p_r - box
    opposite = (d_l[:, 0] * d_r[:, 0] + d_l[:, 1] * d_r[:, 1]) < 0.0
    return (
        closed
        & (tip_mid_dist < 0.055)
        & opposite
        & (sep < 0.08)
        & (sep > 0.02)
    )


def _grip_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last raw gripper command in [0,1] if available, else from joint."""
    am = env.action_manager
    if "gripper_action" in am._terms:
        return am._terms["gripper_action"].raw_actions.squeeze(-1).clamp(0.0, 1.0)
    return torch.zeros(env.num_envs, device=env.device)


def reach_reward(
    env: ManagerBasedRLEnv,
    hover_offset: float = 0.30,
    success_dist: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("workpiece"),
    ee_body_name: str = "right_arm_ee_link",
    tip_l_name: str = "right_robotiq_85_left_finger_tip_link",
    tip_r_name: str = "right_robotiq_85_right_finger_tip_link",
) -> torch.Tensor:
    """MuJoCo-like reach: −d + progress + success bonuses toward hover above box."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _ensure_ids(env, robot, ee_body_name, tip_l_name, tip_r_name)

    ee = robot.data.body_pos_w[:, env._rew_ee_id]
    box = obj.data.root_pos_w
    target = box.clone()
    target[:, 2] = box[:, 2] + hover_offset
    dist = torch.norm(ee - target, dim=-1)

    if not hasattr(env, "_prev_task_dist"):
        from robot_simu_rl.backends.isaac_lab.mdp.events import reset_aux_buffers

        reset_aux_buffers(env, None)

    # First step after reset: seed prev without huge progress spike.
    seed = env._prev_task_dist <= 0.0
    progress = env._prev_task_dist - dist
    progress = torch.where(seed, torch.zeros_like(progress), progress)
    env._prev_task_dist[:] = dist

    r = -dist + 2.0 * progress
    success = dist < success_dist
    r = r + torch.where(success, torch.full_like(r, 5.0), torch.zeros_like(r))
    first = success & ~env._reached_once
    r = r + torch.where(first, torch.full_like(r, 10.0), torch.zeros_like(r))
    env._reached_once |= success

    # Expose metrics for smoke / logging.
    env.extras["log"] = env.extras.get("log", {})
    env.extras["log"]["reach_dist"] = dist
    env.extras["log"]["reach_success"] = success.float()
    return r


def grasp_reward(
    env: ManagerBasedRLEnv,
    lift_height: float = 0.03,
    contact_threshold: float = 0.05,
    hold_steps: int = 15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("workpiece"),
    ee_body_name: str = "right_arm_ee_link",
    tip_l_name: str = "right_robotiq_85_left_finger_tip_link",
    tip_r_name: str = "right_robotiq_85_right_finger_tip_link",
) -> torch.Tensor:
    """MuJoCo-like grasp shaping: tip-mid→box, close, contact, *sustained* lift.

    No kinematic latch / contact_hold welding — PhysX contact + geometric pinch.
    ``grasp_success`` requires holding with lift for ``hold_steps`` control ticks
    (~0.3 s at 50 Hz) so peak bounce alone does not count.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _ensure_ids(env, robot, ee_body_name, tip_l_name, tip_r_name)

    ee = robot.data.body_pos_w[:, env._rew_ee_id]
    box = obj.data.root_pos_w
    tip_mid = _tip_mid(robot, env)
    dist = torch.norm(tip_mid - box, dim=-1)
    dist_box = torch.norm(ee - box, dim=-1)

    if not hasattr(env, "_prev_task_dist"):
        from robot_simu_rl.backends.isaac_lab.mdp.events import reset_aux_buffers

        reset_aux_buffers(env, None)

    seed = env._prev_task_dist <= 0.0
    progress = env._prev_task_dist - dist
    progress = torch.where(seed, torch.zeros_like(progress), progress)
    env._prev_task_dist[:] = dist

    grip_cmd = _grip_cmd(env)
    grip_lo, grip_hi = GRIPPER_JOINT_LIMIT
    grip_q = robot.data.joint_pos[:, env._rew_grip_id]
    grip_norm = (grip_q - grip_lo) / (grip_hi - grip_lo + 1e-9)

    left_hit, right_hit, left_f, right_f = _finger_contacts(
        env, threshold=float(contact_threshold)
    )
    pinch_sensor = left_hit & right_hit
    near = dist < 0.08
    close_tip = dist < 0.05
    closed = grip_norm > 0.30
    closed_soft = grip_norm > 0.18
    pinch_geom = _geometric_pinch(robot, env, box, closed, dist)
    pinch = pinch_sensor | pinch_geom
    # Holding: dual-tip pinch (sensor force is ground truth even if knuckle joint
    # lags) OR tip-near+closed with ≥1 tip force. Never open-finger tip-mid alone.
    one_tip = left_hit | right_hit
    holding = pinch | (close_tip & closed & one_tip)

    r = -0.8 * dist + 1.2 * progress - 0.05
    r = r - 0.15 * torch.clamp(dist_box - 0.25, min=0.0)

    # Close curriculum — punish open-finger tip-mid farming (M14 failure mode).
    r = r + torch.where(near, 10.0 * grip_cmd + 8.0 * grip_norm, torch.zeros_like(r))
    r = r - torch.where(close_tip & ~closed, 20.0 * (1.0 - grip_cmd), torch.zeros_like(r))
    r = r + torch.where(close_tip & closed, 15.0 * grip_norm, torch.zeros_like(r))
    r = r + torch.where(near & one_tip, torch.full_like(r, 12.0), 0.0)
    r = r + torch.where(pinch, torch.full_like(r, 30.0), 0.0)
    r = r - torch.where(~near, 1.5 * grip_cmd, torch.zeros_like(r))
    # Penalize tip-near while fingers stay open (dominant M14 local optimum).
    r = r - torch.where(close_tip & (grip_norm < 0.2), torch.full_like(r, 15.0), 0.0)

    if not hasattr(env, "_box_z0"):
        env._box_z0 = torch.full((env.num_envs,), 0.75, device=env.device)
    lift = box[:, 2] - env._box_z0
    r = r + torch.where(
        holding & ~env._reached_once, torch.full_like(r, 15.0), torch.zeros_like(r)
    )
    env._reached_once |= holding

    # Lift reward only while actually holding (no open-finger bounce farming).
    r = r + torch.where(
        holding, 80.0 * torch.clamp(lift, max=lift_height + 0.08).clamp(min=0.0), 0.0
    )
    r = r + torch.where(
        holding, 10.0 * torch.clamp(ee[:, 2] - (env._box_z0 + 0.05), min=0.0), 0.0
    )
    r = r + torch.where(holding, 8.0 * grip_cmd, torch.zeros_like(r))

    # Sustained dual-pinch lift (not single-frame bounce / one-tip bump).
    if not hasattr(env, "_hold_lift_steps"):
        env._hold_lift_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cand = pinch & closed_soft & (lift > lift_height)
    env._hold_lift_steps = torch.where(
        cand, env._hold_lift_steps + 1, torch.zeros_like(env._hold_lift_steps)
    )
    grasp_ok = env._hold_lift_steps >= int(hold_steps)
    r = r + torch.where(cand, torch.full_like(r, 5.0), 0.0)  # per-step sustain bonus
    r = r + torch.where(grasp_ok, torch.full_like(r, 100.0), 0.0)
    first_lift = grasp_ok & ~env._lifted_once
    r = r + torch.where(first_lift, torch.full_like(r, 200.0), 0.0)
    env._lifted_once |= grasp_ok

    env.extras["log"] = env.extras.get("log", {})
    env.extras["log"]["grasp_dist"] = dist
    env.extras["log"]["box_lift"] = lift
    env.extras["log"]["box_z"] = box[:, 2]
    env.extras["log"]["holding"] = holding.float()
    env.extras["log"]["pinch"] = pinch.float()
    env.extras["log"]["pinch_sensor"] = pinch_sensor.float()
    env.extras["log"]["pinch_geom"] = pinch_geom.float()
    env.extras["log"]["grasp_success"] = grasp_ok.float()
    env.extras["log"]["tip_l_contact"] = left_hit.float()
    env.extras["log"]["tip_r_contact"] = right_hit.float()
    env.extras["log"]["tip_l_force"] = left_f
    env.extras["log"]["tip_r_force"] = right_f
    env.extras["log"]["hold_lift_steps"] = env._hold_lift_steps.float()
    return r

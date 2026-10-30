"""Shared pick-place reward / success helpers (C0 Reach → C1 Grasp)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReachConfig:
    """C0 Reach: EE approaches a pre-grasp hover pose above the box."""

    # Pre-grasp hover above box center. Default 0.30 m keeps target near arm home
    # EE height (~1.19 m) so short PPO can close the gap; lower toward 0.08 for grasp prep.
    hover_offset_m: float = 0.30
    success_dist_m: float = 0.03  # plan exit: ~2–3 cm
    success_bonus: float = 10.0
    action_l2_coef: float = 0.005
    # Potential-based progress: w * (prev_dist - dist)
    progress_coef: float = 2.0
    # Terminate early on sustained reach success (speeds PPO)
    success_hold_steps: int = 5


@dataclass(frozen=True)
class GraspConfig:
    """C1 Grasp: approach low hover, close fingers, lift or firm hold."""

    hover_offset_m: float = 0.10  # M2 target band ~0.08–0.15
    approach_dist_m: float = 0.06  # "near" gate for close/lift shaping
    success_dist_m: float = 0.08  # EE–box distance for hold/lift gates
    lift_height_m: float = 0.05  # box center above spawn → lift success
    grip_closed_min: float = 0.35  # knuckle q (0 open … 0.8 closed)
    success_bonus: float = 20.0
    contact_bonus: float = 1.0
    close_coef: float = 0.5  # reward grip when near
    lift_coef: float = 8.0  # reward box lift while gripped+near
    action_l2_coef: float = 0.005
    progress_coef: float = 2.0
    success_hold_steps: int = 8  # sustained grasp before terminate


def reach_target(box_pos: np.ndarray, hover_offset_m: float) -> np.ndarray:
    """World-frame pre-grasp point: above box center."""
    t = np.asarray(box_pos, dtype=np.float64).reshape(3).copy()
    t[2] += float(hover_offset_m)
    return t


def reach_distance(ee_pos: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(target)))


def reach_reward(
    ee_pos: np.ndarray,
    target: np.ndarray,
    action: np.ndarray | None = None,
    cfg: ReachConfig | None = None,
    prev_dist: float | None = None,
) -> tuple[float, float, bool]:
    """Dense reach shaping + progress + sparse success bonus.

    Returns (reward, distance, success_this_step).
    """
    cfg = cfg or ReachConfig()
    dist = reach_distance(ee_pos, target)
    success = dist < cfg.success_dist_m
    r = -dist
    if prev_dist is not None and cfg.progress_coef > 0:
        r += cfg.progress_coef * (float(prev_dist) - dist)
    if success:
        r += cfg.success_bonus
    if action is not None and cfg.action_l2_coef > 0:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        # Prefer penalizing arm Δq only (ignore held gripper cmd)
        r -= cfg.action_l2_coef * float(np.dot(a[:7], a[:7]))
    return float(r), dist, bool(success)


def grasp_reward(
    ee_pos: np.ndarray,
    box_pos: np.ndarray,
    grip_q: float,
    box_spawn_z: float,
    fingertip_contact: bool,
    action: np.ndarray | None = None,
    cfg: GraspConfig | None = None,
    prev_approach_dist: float | None = None,
) -> tuple[float, float, bool, bool]:
    """Dense grasp shaping: approach → close → contact → lift / firm hold.

    Returns (reward, approach_dist, grasp_success, lifted).
    """
    cfg = cfg or GraspConfig()
    target = reach_target(box_pos, cfg.hover_offset_m)
    approach = reach_distance(ee_pos, target)
    ee_box = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(box_pos)))
    grip = float(grip_q)
    near = approach < cfg.approach_dist_m or ee_box < cfg.success_dist_m
    lift_dz = float(box_pos[2] - box_spawn_z)
    gripped = grip >= cfg.grip_closed_min
    lifted = bool(gripped and ee_box < cfg.success_dist_m and lift_dz >= cfg.lift_height_m)
    firm = bool(
        gripped
        and fingertip_contact
        and ee_box < cfg.success_dist_m
        and lift_dz > -0.02  # still roughly on/above table
    )
    success = lifted or firm

    r = -approach
    if prev_approach_dist is not None and cfg.progress_coef > 0:
        r += cfg.progress_coef * (float(prev_approach_dist) - approach)
    if near:
        # Encourage closing when near hover / box
        r += cfg.close_coef * min(grip / 0.8, 1.0)
    if fingertip_contact:
        r += cfg.contact_bonus
    if gripped and near:
        r += cfg.lift_coef * max(lift_dz, 0.0)
    if success:
        r += cfg.success_bonus
    if action is not None and cfg.action_l2_coef > 0:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        r -= cfg.action_l2_coef * float(np.dot(a[:7], a[:7]))
    return float(r), float(approach), bool(success), bool(lifted)

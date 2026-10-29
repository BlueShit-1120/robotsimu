"""Shared pick-place reward / success helpers (curriculum C0 Reach first)."""

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

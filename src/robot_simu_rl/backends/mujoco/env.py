"""Gymnasium MuJoCo env: right-arm 7 + gripper 1 (M0–M3 Reach/Grasp/Place)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, SupportsFloat, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_simu_rl.common.joints import (
    ARM_JOINT_LIMITS,
    GRIPPER_JOINT_LIMIT,
    LOCKED_QPOS,
    MJCF_RIGHT_ARM_JOINTS,
    MJCF_RIGHT_GRIPPER_JOINT,
)
from robot_simu_rl.common.paths import PICK_PLACE_SCENE

Phase = Literal["m0_smoke", "reach", "grasp", "place"]

# M1 moved spawn nearer right-arm home EE; M0 default was (0.55, 0.0).
_BOX_XY_CENTER = np.array([0.70, -0.20], dtype=np.float64)
_BOX_XY_HALF_RANGE = np.array([0.04, 0.04], dtype=np.float64)
_BOX_Z = 0.765  # table top 0.75 + half height 0.015
# Place goal: ~8 cm +Y of spawn (−0.20→−0.12). Short-train seed-hold starts
# above this goal (see place_start_holding) so C3 focuses on lower+release
# even with tightened place_tol_xy≈0.06–0.08; --full-pick does the carry.
_PLACE_GOAL_XY = np.array([0.70, -0.12], dtype=np.float64)

# Default right-arm home (reach toward +X table) — M0 smoke
_RIGHT_ARM_HOME = np.array([0.0, -0.6, 0.0, -1.2, 0.0, -0.4, 0.0], dtype=np.float64)
# M1/M2 home nearer box spawn (0.70, -0.20): EE ~10–12 cm above/ beside workpiece
_RIGHT_ARM_HOME_NEAR = np.array(
    [-0.4923, -1.1724, -0.1493, -0.8018, -0.2582, -0.2665, 0.0375], dtype=np.float64
)
# M4 contact-grasp home: at grip≈0.55 tip mid stays on box (close without diving).
_RIGHT_ARM_HOME_GRASP = np.array(
    [-0.38357, -1.04487, -0.0086, -0.75563, 0.04278, -0.37762, 0.98704],
    dtype=np.float64,
)


class PickPlaceMujocoEnv(gym.Env):
    """Right-arm pick-place on a table with a 5×4×3 cm box (D2/D4).

    Action (8): Δq arm(7) clipped + gripper open/close command in [0, 1]
                (mapped to knuckle angle [0, 0.8]).
    Obs: arm q/qd (14) + grip (1) + ee pos(3)/quat(4) + box pos(3)/quat(4) + goal xy(2).

    Phases
    ------
    m0_smoke : legacy distance-to-box reward
    reach    : hover target above box; gripper held open (M1 / C0)
    grasp    : approach, close, lift; place reward off (M2 / C1)
    place    : grasp → carry → release at goal (M3 / C2–C3)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        control_hz: float = 50.0,
        frame_skip: int = 10,
        max_episode_steps: int = 400,
        delta_q_scale: float = 0.05,
        render_mode: Optional[str] = None,
        randomize_box: bool = True,
        phase: Phase = "m0_smoke",
        hover_offset: float = 0.30,
        success_dist: float = 0.05,
        lift_height: float = 0.05,
        hold_gripper_open: Optional[bool] = None,
        early_terminate_success: bool = True,
        # M2/M3 short-train assist: kinematic latch when closed near object
        # (proxy-finger contacts are optimistic/brittle; see docs/m2-grasp.md).
        grasp_latch: Optional[bool] = None,
        # M5: acquire kinematic hold only after dual fingertip↔box contact
        # (not proximity latch). Needed because M4 sphere pads flick-lift but
        # do not sustain a friction carry for place.
        contact_hold: bool = False,
        latch_dist: float = 0.12,
        place_tol_xy: float = 0.12,
        place_tol_z: float = 0.04,
        place_stable_steps: int = 3,
        # M3 short-train: start already holding (C2/C3 focus; grasp is M2)
        place_start_holding: Optional[bool] = None,
        # M5 train assist: once lifted / holding and still far from goal, keep
        # gripper mostly closed so PPO learns arm carry instead of post-lift open.
        # Eval must leave this False (no assist at test time).
        carry_assist: bool = False,
        carry_assist_grip: float = 0.85,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path) if model_path else PICK_PLACE_SCENE
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MJCF not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.control_hz = float(control_hz)
        self.max_episode_steps = int(max_episode_steps)
        self.delta_q_scale = float(delta_q_scale)
        self.render_mode = render_mode
        self.randomize_box = bool(randomize_box)
        self.phase: Phase = phase
        self.hover_offset = float(hover_offset)
        self.success_dist = float(success_dist)
        self.lift_height = float(lift_height)
        if hold_gripper_open is None:
            hold_gripper_open = phase == "reach"
        self.hold_gripper_open = bool(hold_gripper_open)
        self.early_terminate_success = bool(early_terminate_success)
        if grasp_latch is None:
            grasp_latch = phase in ("grasp", "place")
        self.grasp_latch = bool(grasp_latch)
        self.contact_hold = bool(contact_hold)
        if self.grasp_latch and self.contact_hold:
            raise ValueError("grasp_latch and contact_hold are mutually exclusive")
        self.latch_dist = float(latch_dist)
        self.place_tol_xy = float(place_tol_xy)
        self.place_tol_z = float(place_tol_z)
        self.place_stable_steps = int(place_stable_steps)
        if place_start_holding is None:
            place_start_holding = phase == "place"
        self.place_start_holding = bool(place_start_holding)
        self.carry_assist = bool(carry_assist)
        self.carry_assist_grip = float(carry_assist_grip)
        self._step_count = 0
        self._viewer = None
        self._renderer: Optional[mujoco.Renderer] = None
        self._prev_dist = 0.0
        self._prev_place_dist = 0.0
        self._box_z0 = _BOX_Z
        self._reached_once = False
        self._lifted_once = False
        self._carried_once = False
        self._placed_once = False
        self._place_ok_steps = 0
        self._ever_held = False
        self._latched = False
        self._latch_offset = np.zeros(3, dtype=np.float64)

        self._arm_jnt_ids = [self._joint_id(n) for n in MJCF_RIGHT_ARM_JOINTS]
        self._grip_jnt_id = self._joint_id(MJCF_RIGHT_GRIPPER_JOINT)
        self._arm_qadr = [int(self.model.jnt_qposadr[j]) for j in self._arm_jnt_ids]
        self._arm_vadr = [int(self.model.jnt_dofadr[j]) for j in self._arm_jnt_ids]
        self._grip_qadr = int(self.model.jnt_qposadr[self._grip_jnt_id])

        self._act_arm_ids = [self._actuator_id(n) for n in MJCF_RIGHT_ARM_JOINTS]
        self._act_grip_id = self._actuator_id("right_gripper")
        # Detect position vs motor actuators (M1 switched right arm to position)
        self._arm_is_position = True
        for aid in self._act_arm_ids:
            # mjTRN_JOINT + gaintype fixed → position actuator typically
            # Safer: check actuator_biastype / dynprm — use XML name convention via gaintype
            # MuJoCo position actuators have gaintype = mjGAIN_FIXED and biastype = mjBIAS_AFFINE
            bias = int(self.model.actuator_biastype[aid])
            if bias == 0:  # mjBIAS_NONE → motor
                self._arm_is_position = False
                break

        self._ee_body_id = self._body_id("right_arm_ee_link")
        self._box_body_id = self._body_id("workpiece")
        self._box_jnt_id = self._joint_id("workpiece_free")
        self._box_qadr = int(self.model.jnt_qposadr[self._box_jnt_id])
        self._tip_l_body = self._body_id("right_gripper_left_fingertip")
        self._tip_r_body = self._body_id("right_gripper_right_fingertip")
        self._tip_l_geom = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_gripper_left_fingertip_collision"
            )
        )
        self._tip_r_geom = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_gripper_right_fingertip_collision"
            )
        )
        self._box_geom = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "workpiece_geom")
        )

        self._locked: list[tuple[int, float, int]] = []
        for name, q in LOCKED_QPOS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            qadr = int(self.model.jnt_qposadr[jid])
            dadr = int(self.model.jnt_dofadr[jid])
            self._locked.append((qadr, float(q), dadr))

        self._arm_low = np.array([lo for lo, _ in ARM_JOINT_LIMITS], dtype=np.float64)
        self._arm_high = np.array([hi for _, hi in ARM_JOINT_LIMITS], dtype=np.float64)
        self._grip_low, self._grip_high = GRIPPER_JOINT_LIMIT

        self.action_space = spaces.Box(
            low=np.array([-1.0] * 7 + [0.0], dtype=np.float32),
            high=np.array([1.0] * 7 + [1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        self._place_goal = np.array(
            [_PLACE_GOAL_XY[0], _PLACE_GOAL_XY[1], _BOX_Z], dtype=np.float64
        )
        self._reach_target = np.zeros(3, dtype=np.float64)

    # ---- id helpers ----
    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint not found: {name}")
        return int(jid)

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"actuator not found: {name}")
        return int(aid)

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body not found: {name}")
        return int(bid)

    def _update_reach_target(self) -> None:
        box = self.data.xpos[self._box_body_id]
        if self.phase == "place":
            # Carry when latched OR contact-holding (no-latch full-pick).
            holding = self._latched or (
                not self.grasp_latch and self._is_holding()
            )
            if holding:
                self._reach_target[:] = [
                    self._place_goal[0],
                    self._place_goal[1],
                    self._box_z0 + max(self.lift_height, 0.05),
                ]
            else:
                # Acquire: same tip-mid → box steering as M4 grasp.
                tip_mid = 0.5 * (
                    self.data.xpos[self._tip_l_body]
                    + self.data.xpos[self._tip_r_body]
                )
                ee = self.data.xpos[self._ee_body_id]
                offset = ee - tip_mid
                self._reach_target[:] = [
                    box[0] + offset[0],
                    box[1] + offset[1],
                    box[2] + offset[2],
                ]
        elif self.phase == "grasp":
            # M4: steer fingertip midpoint to box center (proxy pads pinch at box height).
            tip_mid = 0.5 * (
                self.data.xpos[self._tip_l_body] + self.data.xpos[self._tip_r_body]
            )
            # Target for tip mid == box; convert to an EE-space target by the
            # current tip_mid→EE offset so Δq still moves the pads onto the box.
            ee = self.data.xpos[self._ee_body_id]
            offset = ee - tip_mid
            self._reach_target[:] = [
                box[0] + offset[0],
                box[1] + offset[1],
                box[2] + offset[2],
            ]
        else:
            self._reach_target[:] = [box[0], box[1], box[2] + self.hover_offset]

    # ---- core API ----
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._reached_once = False
        self._lifted_once = False
        self._carried_once = False
        self._placed_once = False
        self._place_ok_steps = 0
        self._ever_held = False
        self._latched = False
        self._latch_offset[:] = 0.0
        self._prev_place_dist = 0.0

        for qadr, q, dadr in self._locked:
            self.data.qpos[qadr] = q
            self.data.qvel[dadr] = 0.0

        if self.phase == "grasp" or (
            self.phase == "place" and not self.place_start_holding
        ):
            # Full-pick / grasp: M4 contact home (pads close on box without diving).
            home = _RIGHT_ARM_HOME_GRASP
        elif self.phase in ("reach", "place"):
            home = _RIGHT_ARM_HOME_NEAR
        else:
            home = _RIGHT_ARM_HOME
        for i, adr in enumerate(self._arm_qadr):
            self.data.qpos[adr] = float(
                np.clip(home[i], self._arm_low[i], self._arm_high[i])
            )
            self.data.qvel[self._arm_vadr[i]] = 0.0
        self.data.qpos[self._grip_qadr] = 0.0

        xy = _BOX_XY_CENTER.copy()
        if self.randomize_box and self.phase == "m0_smoke":
            offset = self.np_random.uniform(-1.0, 1.0, size=2) * _BOX_XY_HALF_RANGE
            xy = _BOX_XY_CENTER + offset
        elif self.randomize_box and self.phase in ("reach", "grasp", "place"):
            # Light jitter for curriculum; default train uses fixed box
            offset = self.np_random.uniform(-1.0, 1.0, size=2) * (
                _BOX_XY_HALF_RANGE * 0.5
            )
            xy = _BOX_XY_CENTER + offset
        self.data.qpos[self._box_qadr : self._box_qadr + 3] = [xy[0], xy[1], _BOX_Z]
        self.data.qpos[self._box_qadr + 3 : self._box_qadr + 7] = [1, 0, 0, 0]
        box_dof = int(self.model.jnt_dofadr[self._box_jnt_id])
        self.data.qvel[box_dof : box_dof + 6] = 0.0
        self._box_z0 = float(_BOX_Z)
        self._prev_place_dist = float(
            np.linalg.norm(
                np.array([xy[0], xy[1]], dtype=np.float64) - self._place_goal[:2]
            )
        )

        self.data.ctrl[:] = 0.0
        for i, aid in enumerate(self._act_arm_ids):
            if self._arm_is_position:
                self.data.ctrl[aid] = float(self.data.qpos[self._arm_qadr[i]])
            else:
                self.data.ctrl[aid] = 0.0
        self.data.ctrl[self._act_grip_id] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # M3/M3b short-train: begin already holding with a small lift (C2→C3).
        # If spawn→goal already fits inside place_tol_xy (M3 default), seed at
        # spawn. If tol is tighter than that distance (M3b), seed above the
        # place goal so C3 still starts in-zone; --full-pick does the carry.
        if self.phase == "place" and self.place_start_holding:
            ee = self.data.xpos[self._ee_body_id].copy()
            spawn_to_goal = float(np.linalg.norm(xy - self._place_goal[:2]))
            if spawn_to_goal >= self.place_tol_xy:
                hold_xy = self._place_goal[:2].copy()
            else:
                hold_xy = xy.copy()
            self._latched = True
            self._latch_offset = np.array(
                [
                    hold_xy[0] - ee[0],
                    hold_xy[1] - ee[1],
                    (_BOX_Z + self.lift_height) - ee[2],
                ],
                dtype=np.float64,
            )
            self.data.qpos[self._grip_qadr] = 0.55
            self.data.ctrl[self._act_grip_id] = 0.55
            self._apply_latch()
            mujoco.mj_forward(self.model, self.data)
            self._lifted_once = True
            self._reached_once = True
            self._ever_held = True
            self._prev_place_dist = self._place_dist_xy()

        self._update_reach_target()
        self._prev_dist = float(
            np.linalg.norm(self.data.xpos[self._ee_body_id] - self._reach_target)
        )
        self._prev_place_dist = self._place_dist_xy()
        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(8)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        q = np.array([self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64)
        qd = q + action[:7] * self.delta_q_scale
        qd = np.clip(qd, self._arm_low, self._arm_high)
        if self.hold_gripper_open:
            grip_cmd = 0.0
        else:
            grip_cmd = float(action[7])
        if (
            self.carry_assist
            and self.phase == "place"
            and not self.hold_gripper_open
        ):
            place_xy = self._place_dist_xy()
            far = place_xy > self.place_tol_xy * 1.1
            if far and (self._lifted_once or self._latched or self._is_holding(grip_cmd)):
                grip_cmd = max(grip_cmd, self.carry_assist_grip)
        grip_target = self._grip_low + grip_cmd * (self._grip_high - self._grip_low)

        for _ in range(self.frame_skip):
            self._apply_locks()
            if self._arm_is_position:
                for aid, target in zip(self._act_arm_ids, qd):
                    self.data.ctrl[aid] = float(target)
            else:
                q_now = np.array(
                    [self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64
                )
                dq = np.array(
                    [self.data.qvel[a] for a in self._arm_vadr], dtype=np.float64
                )
                kp = np.array([80, 80, 60, 60, 20, 20, 20], dtype=np.float64)
                kd = np.array([4, 4, 3, 3, 1, 1, 1], dtype=np.float64)
                from robot_simu_rl.common.joints import ARM_EFFORT

                tau = kp * (qd - q_now) - kd * dq
                tau = np.clip(tau, -np.asarray(ARM_EFFORT), np.asarray(ARM_EFFORT))
                for aid, t in zip(self._act_arm_ids, tau):
                    self.data.ctrl[aid] = float(t)
            self.data.ctrl[self._act_grip_id] = grip_target
            for i in range(self.model.nu):
                if i not in self._act_arm_ids and i != self._act_grip_id:
                    self.data.ctrl[i] = 0.0
            mujoco.mj_step(self.model, self.data)
            self._sync_hold(grip_cmd)

        self._apply_locks()
        if self._sync_hold(grip_cmd):
            mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        self._update_reach_target()
        obs = self._get_obs()
        reward = self._reward(grip_cmd)
        success = self._phase_success()
        terminated = bool(self._dropped())
        if self.early_terminate_success and success:
            terminated = True
        truncated = self._step_count >= self.max_episode_steps
        info = self._info()
        info["grasp_success"] = self._grasp_success()
        info["place_success"] = self._place_success()
        info["reach_success"] = self._reach_success()
        info["reach_dist"] = float(
            np.linalg.norm(self.data.xpos[self._ee_body_id] - self._reach_target)
        )
        info["reach_target"] = self._reach_target.copy()
        info["place_dist_xy"] = self._place_dist_xy()
        info["latched"] = bool(self._latched)
        info["phase"] = self.phase
        info["success"] = bool(success)
        return obs, float(reward), terminated, truncated, info

    def _sync_hold(self, grip_cmd: float) -> bool:
        """Update kinematic hold. Returns True if a hold was applied this call.

        - grasp_latch=True: acquire near object when closed; release when open.
        - contact_hold=True: acquire only on dual fingertip↔box contact + close;
          release when open (M5 full-pick; not proximity latch).
        - else: never re-acquire; only release a seed hold from
          place_start_holding (C3 curriculum without latch assist).
        """
        if self.grasp_latch:
            self._update_latch(grip_cmd)
            if self._latched:
                self._apply_latch()
                return True
            return False
        if self.contact_hold:
            self._update_contact_hold(grip_cmd)
            if self._latched:
                self._apply_latch()
                return True
            return False
        if self._latched:
            # Seed-hold only: clear on open, keep applying until then.
            grip = float(self.data.qpos[self._grip_qadr])
            if grip_cmd < 0.2 or grip < 0.12:
                self._latched = False
                return False
            self._apply_latch()
            return True
        return False

    def _update_contact_hold(self, grip_cmd: float) -> None:
        """Contact-triggered hold: both pads must touch before welding.

        Release is only honored near the place goal (XY within 1.5× tol) so
        premature open far from the target cannot drop the workpiece — needed
        because M4 sphere pads do not sustain a friction carry alone.
        """
        grip = float(self.data.qpos[self._grip_qadr])
        left_hit, right_hit = self._finger_box_contacts()
        if not self._latched:
            if left_hit and right_hit and grip_cmd > 0.45 and grip > 0.20:
                ee = self.data.xpos[self._ee_body_id]
                box = self.data.xpos[self._box_body_id]
                self._latched = True
                self._latch_offset = box - ee
        else:
            near_goal = self._place_dist_xy() <= self.place_tol_xy * 1.5
            # Match place_success openish threshold (~0.35) so mid-open works.
            if near_goal and (grip_cmd < 0.40 or grip < 0.25):
                self._latched = False

    def _update_latch(self, grip_cmd: float) -> None:
        ee = self.data.xpos[self._ee_body_id]
        box = self.data.xpos[self._box_body_id]
        dist = float(np.linalg.norm(ee - box))
        grip = float(self.data.qpos[self._grip_qadr])
        if not self._latched:
            if dist < self.latch_dist and grip_cmd > 0.45 and grip > 0.20:
                self._latched = True
                self._latch_offset = box - ee
        else:
            if grip_cmd < 0.2 or grip < 0.12:
                self._latched = False

    def _apply_latch(self) -> None:
        if not self._latched:
            return
        ee = self.data.xpos[self._ee_body_id]
        pos = ee + self._latch_offset
        # Keep a minimum height so latch doesn't shove through table
        pos[2] = max(pos[2], self._box_z0)
        self.data.qpos[self._box_qadr : self._box_qadr + 3] = pos
        # Preserve upright quaternion
        self.data.qpos[self._box_qadr + 3 : self._box_qadr + 7] = [1, 0, 0, 0]
        box_dof = int(self.model.jnt_dofadr[self._box_jnt_id])
        self.data.qvel[box_dof : box_dof + 6] = 0.0

    def _apply_locks(self) -> None:
        for qadr, q, dadr in self._locked:
            self.data.qpos[qadr] = q
            self.data.qvel[dadr] = 0.0

    def _get_obs(self) -> np.ndarray:
        q = np.array([self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64)
        dq = np.array([self.data.qvel[a] for a in self._arm_vadr], dtype=np.float64)
        mid = 0.5 * (self._arm_low + self._arm_high)
        half = 0.5 * (self._arm_high - self._arm_low)
        qn = (q - mid) / np.maximum(half, 1e-6)
        dqn = dq / np.pi
        grip = np.array(
            [
                (self.data.qpos[self._grip_qadr] - self._grip_low)
                / (self._grip_high - self._grip_low + 1e-9)
            ],
            dtype=np.float64,
        )
        ee_pos = self.data.xpos[self._ee_body_id].copy()
        ee_quat = self.data.xquat[self._ee_body_id].copy()
        box_pos = self.data.xpos[self._box_body_id].copy()
        box_quat = self.data.xquat[self._box_body_id].copy()
        goal_xy = self._place_goal[:2].copy()
        obs = np.concatenate([qn, dqn, grip, ee_pos, ee_quat, box_pos, box_quat, goal_xy])
        return obs.astype(np.float32)

    def _reward(self, grip_cmd: float) -> float:
        if self.phase == "reach":
            return self._reward_reach()
        if self.phase == "grasp":
            return self._reward_grasp(grip_cmd)
        if self.phase == "place":
            return self._reward_place(grip_cmd)
        # m0_smoke
        ee = self.data.xpos[self._ee_body_id]
        box = self.data.xpos[self._box_body_id]
        return -float(np.linalg.norm(ee - box))

    def _reward_reach(self) -> float:
        ee = self.data.xpos[self._ee_body_id]
        dist = float(np.linalg.norm(ee - self._reach_target))
        progress = self._prev_dist - dist
        self._prev_dist = dist
        r = -dist + 2.0 * progress
        if dist < self.success_dist:
            r += 5.0
            if not self._reached_once:
                r += 10.0
                self._reached_once = True
        return r

    def _tip_mid(self) -> np.ndarray:
        return 0.5 * (
            self.data.xpos[self._tip_l_body] + self.data.xpos[self._tip_r_body]
        )

    def _finger_box_contacts(self) -> tuple[bool, bool]:
        """Whether left/right fingertip pads currently touch the workpiece."""
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            hit_box = self._box_geom in (g1, g2)
            if not hit_box:
                continue
            other = g2 if g1 == self._box_geom else g1
            if other == self._tip_l_geom:
                left = True
            elif other == self._tip_r_geom:
                right = True
        return left, right

    def _reward_grasp(self, grip_cmd: float) -> float:
        ee = self.data.xpos[self._ee_body_id]
        box = self.data.xpos[self._box_body_id]
        tip_mid = self._tip_mid()
        # Drive pads to box (M4) rather than raw EE hover.
        dist = float(np.linalg.norm(tip_mid - box))
        dist_box = float(np.linalg.norm(ee - box))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        r = -0.8 * dist + 1.2 * progress
        r -= 0.05  # step cost
        # Keep EE from flying away from the box (stabilizes no-latch PPO).
        r -= 0.15 * max(0.0, dist_box - 0.25)

        grip = float(self.data.qpos[self._grip_qadr])
        near = dist < 0.08
        left_hit, right_hit = self._finger_box_contacts()
        pinch = left_hit and right_hit
        holding = self._is_holding(grip_cmd)
        if near and not holding:
            # Incentive to close when pads near box
            r += 2.5 * grip_cmd
            if left_hit or right_hit:
                r += 4.0
            if pinch:
                r += 12.0
        elif not near:
            r -= 0.5 * grip_cmd

        lift = float(box[2] - self._box_z0)
        if holding:
            if not self._reached_once:
                r += 15.0  # first acquire only
                self._reached_once = True
            # Dense lift shaping while holding (latch or contact)
            r += 60.0 * max(0.0, min(lift, self.lift_height + 0.06))
            # Encourage raising EE while holding
            r += 8.0 * max(0.0, float(ee[2] - (self._box_z0 + 0.05)))
            r += 5.0 * grip_cmd

        if self._grasp_success():
            r += 100.0
            if not self._lifted_once:
                r += 150.0
                self._lifted_once = True
        return r

    def _is_holding(self, grip_cmd: float | None = None) -> bool:
        """True if latch/seed-hold active, or contact-style pinch (no-latch)."""
        if self._latched:
            return True
        if self.grasp_latch:
            return False
        grip = float(self.data.qpos[self._grip_qadr])
        cmd_ok = True if grip_cmd is None else grip_cmd > 0.35
        left_hit, right_hit = self._finger_box_contacts()
        if left_hit and right_hit and grip > 0.25 and cmd_ok:
            return True
        # Fallback: tip mid near box + closed (contacts can flicker a step)
        tip_mid = self._tip_mid()
        box = self.data.xpos[self._box_body_id]
        close = float(np.linalg.norm(tip_mid - box)) < 0.05
        return bool(close and grip > 0.35 and cmd_ok)

    def _place_dist_xy(self) -> float:
        box = self.data.xpos[self._box_body_id]
        return float(np.linalg.norm(box[:2] - self._place_goal[:2]))

    def _reward_place(self, grip_cmd: float) -> float:
        """C2 carry + C3 place: grasp/lift → move to goal → release & settle."""
        ee = self.data.xpos[self._ee_body_id]
        box = self.data.xpos[self._box_body_id]
        dist_box = float(np.linalg.norm(ee - box))
        place_xy = self._place_dist_xy()
        place_progress = self._prev_place_dist - place_xy
        self._prev_place_dist = place_xy

        r = -0.05  # step cost

        holding = self._is_holding(grip_cmd)
        if holding or self._latched:
            self._ever_held = True
        in_place_zone = place_xy <= self.place_tol_xy * 1.25

        # Short-train release curriculum: already in/near goal XY at reset
        release_only = (
            self.place_start_holding and place_xy <= self.place_tol_xy * 1.25
        ) or (holding and place_xy <= self.place_tol_xy)

        if release_only or (holding and in_place_zone):
            if holding or self._latched:
                open_frac = 1.0 - grip_cmd
                r += 40.0 * open_frac
                r -= 35.0 * grip_cmd
                if grip_cmd < 0.40:
                    r += 15.0
            else:
                # After release: only reward being in-zone (no open-farming)
                if place_xy <= self.place_tol_xy and abs(
                    float(box[2]) - self._box_z0
                ) < self.place_tol_z:
                    r += 5.0
                else:
                    r -= 5.0
            if place_xy > self.place_tol_xy:
                r -= 25.0 * (place_xy - self.place_tol_xy)
            if self._place_success():
                r += 200.0
                if not self._placed_once:
                    r += 300.0
                    self._placed_once = True
            return r

        # Mark lift even if contact flickers the same step grasp_success trips.
        lift_now = float(box[2] - self._box_z0)
        if (self._grasp_success() or lift_now > self.lift_height) and not self._lifted_once:
            self._lifted_once = True
            r += 50.0

        # --- Stage A: acquire (full-pick / M4 contact) ---
        if not holding:
            # Mild penalty for dropping after lift away from goal (keep re-grasp viable).
            if self._lifted_once and place_xy > self.place_tol_xy:
                r -= 4.0
                r -= 3.0 * (1.0 - grip_cmd)
            tip_mid = self._tip_mid()
            dist = float(np.linalg.norm(tip_mid - box))
            progress = self._prev_dist - dist
            self._prev_dist = dist
            r += -0.8 * dist + 1.2 * progress
            r -= 0.15 * max(0.0, dist_box - 0.25)
            grip = float(self.data.qpos[self._grip_qadr])
            near = dist < 0.08
            left_hit, right_hit = self._finger_box_contacts()
            pinch = left_hit and right_hit
            if near:
                r += 2.5 * grip_cmd
                if left_hit or right_hit:
                    r += 4.0
                if pinch:
                    r += 12.0
            else:
                r -= 0.5 * grip_cmd
            if near and grip > 0.30 and grip_cmd > 0.35:
                r += 2.0
        else:
            # --- Stage B: carry while holding (latch or contact) ---
            lift = float(box[2] - self._box_z0)
            if not self._reached_once:
                r += 15.0
                self._reached_once = True
            if not self._lifted_once and lift > self.lift_height:
                self._lifted_once = True
            # Dense carry toward place goal (table-slide is enough for short goal).
            r += -25.0 * place_xy + 60.0 * place_progress
            if place_progress < 0.0:
                r += 40.0 * place_progress
            if place_xy > self.place_tol_xy * 1.25:
                r -= 8.0 * (1.0 - grip_cmd)
                r += 3.0 * grip_cmd
            # Prefer getting into XY tol over farming lift height.
            if place_xy > self.place_tol_xy:
                r += 2.0 * max(0.0, min(lift, 0.04))
            else:
                r += 15.0  # in XY band while still holding
            tip_mid = self._tip_mid()
            pad_err = float(np.linalg.norm(tip_mid - box))
            r -= 1.5 * max(0.0, pad_err - 0.04)
            if place_xy < self.place_tol_xy * 1.15 and not self._carried_once:
                r += 100.0
                self._carried_once = True
            if place_xy < self.place_tol_xy:
                r += 20.0

        if self._place_success():
            r += 200.0
            if not self._placed_once:
                r += 300.0
                self._placed_once = True
        return r

    def _dropped(self) -> bool:
        z = float(self.data.xpos[self._box_body_id][2])
        return z < 0.3

    def _reach_success(self) -> bool:
        ee = self.data.xpos[self._ee_body_id]
        dist = float(np.linalg.norm(ee - self._reach_target))
        return bool(dist < self.success_dist)

    def _grasp_success(self) -> bool:
        box = self.data.xpos[self._box_body_id]
        tip_mid = self._tip_mid()
        grip = float(self.data.qpos[self._grip_qadr])
        close = float(np.linalg.norm(tip_mid - box)) < 0.08
        lifted = float(box[2]) > self._box_z0 + self.lift_height
        if self.grasp_latch:
            return bool(self._latched and lifted and grip > 0.25)
        left_hit, right_hit = self._finger_box_contacts()
        pinched = left_hit or right_hit or close
        return bool(pinched and grip > 0.30 and lifted)

    def _place_success(self) -> bool:
        """Released near goal on table (C3). Tracks consecutive OK steps.

        Full-pick requires a prior hold/acquire so an open start inside the
        place tol (spawn near goal) cannot count as success.
        """
        box = self.data.xpos[self._box_body_id]
        xy_ok = self._place_dist_xy() < self.place_tol_xy
        z_ok = abs(float(box[2]) - self._box_z0) < self.place_tol_z
        # Seed/latch hold must be cleared; contact hold ends when grip opens.
        released = not self._latched
        grip = float(self.data.qpos[self._grip_qadr])
        openish = grip < 0.35
        acquired = bool(self._ever_held or self.place_start_holding)
        ok = bool(xy_ok and z_ok and released and openish and acquired)
        if ok:
            self._place_ok_steps += 1
        else:
            self._place_ok_steps = 0
        return bool(self._place_ok_steps >= self.place_stable_steps)

    def _phase_success(self) -> bool:
        if self.phase == "reach":
            return self._reach_success()
        if self.phase == "grasp":
            return self._grasp_success()
        if self.phase == "place":
            return self._place_success()
        return False

    def _info(self) -> dict[str, Any]:
        return {
            "ee_pose": np.concatenate(
                [self.data.xpos[self._ee_body_id], self.data.xquat[self._ee_body_id]]
            ).copy(),
            "object_pose": np.concatenate(
                [self.data.xpos[self._box_body_id], self.data.xquat[self._box_body_id]]
            ).copy(),
            "place_goal": self._place_goal.copy(),
            "steps": self._step_count,
        }

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                try:
                    from mujoco import viewer as mj_viewer

                    self._viewer = mj_viewer.launch_passive(self.model, self.data)
                except Exception:
                    return None
            else:
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None

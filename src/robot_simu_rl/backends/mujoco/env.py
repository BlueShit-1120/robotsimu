"""Gymnasium MuJoCo env: right-arm 7 + gripper 1 pick-place (M0/M1)."""

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
from robot_simu_rl.tasks.pick_place.rewards import ReachConfig, reach_reward, reach_target

# Box spawn: table center-ish with light XY randomization (meters, world frame)
# Box XY near right-arm home EE projection (M1 reachability)
_BOX_XY_CENTER = np.array([0.70, -0.20], dtype=np.float64)
_BOX_XY_HALF_RANGE = np.array([0.05, 0.06], dtype=np.float64)  # ±
_BOX_Z = 0.765  # table top 0.75 + half height 0.015
_PLACE_GOAL_XY = np.array([0.70, 0.05], dtype=np.float64)

# Default right-arm home (reachable toward +X table). M1 hover is raised so
# the pre-grasp target sits near this EE height (see ReachConfig.hover_offset_m).
_RIGHT_ARM_HOME = np.array([0.0, -0.6, 0.0, -1.2, 0.0, -0.4, 0.0], dtype=np.float64)

CurriculumPhase = Literal["reach", "pick_place"]


class PickPlaceMujocoEnv(gym.Env):
    """Right-arm pick-place on a table with a 5×4×3 cm box (D2/D4).

    Action (8): Δq arm(7) clipped + gripper open/close command in [0, 1]
                (mapped to knuckle angle [0, 0.8]).
    Obs: arm q/qd (14) + grip (1) + ee pos(3)/quat(4) + box pos(3)/quat(4) + goal xy(2).

    Curriculum:
      - ``phase="reach"`` (M1/C0): reward EE→hover above box; optional open-gripper hold.
      - ``phase="pick_place"``: legacy M0 distance-to-box shaping.
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
        phase: CurriculumPhase = "pick_place",
        hold_gripper_open: Optional[bool] = None,
        reach_cfg: Optional[ReachConfig] = None,
        terminate_on_reach_success: bool = True,
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
        self.phase: CurriculumPhase = phase
        self.reach_cfg = reach_cfg or ReachConfig()
        # M1 default: hold gripper open so policy focuses on arm reach
        if hold_gripper_open is None:
            hold_gripper_open = phase == "reach"
        self.hold_gripper_open = bool(hold_gripper_open)
        self.terminate_on_reach_success = bool(terminate_on_reach_success)
        self._step_count = 0
        self._reach_success_streak = 0
        self._last_reach_dist = 0.0
        self._viewer = None
        self._renderer: Optional[mujoco.Renderer] = None

        self._arm_jnt_ids = [self._joint_id(n) for n in MJCF_RIGHT_ARM_JOINTS]
        self._grip_jnt_id = self._joint_id(MJCF_RIGHT_GRIPPER_JOINT)
        self._arm_qadr = [int(self.model.jnt_qposadr[j]) for j in self._arm_jnt_ids]
        self._arm_vadr = [int(self.model.jnt_dofadr[j]) for j in self._arm_jnt_ids]
        self._grip_qadr = int(self.model.jnt_qposadr[self._grip_jnt_id])

        self._act_arm_ids = [self._actuator_id(n) for n in MJCF_RIGHT_ARM_JOINTS]
        self._act_grip_id = self._actuator_id("right_gripper")

        self._ee_body_id = self._body_id("right_arm_ee_link")
        self._box_body_id = self._body_id("workpiece")
        self._box_jnt_id = self._joint_id("workpiece_free")
        self._box_qadr = int(self.model.jnt_qposadr[self._box_jnt_id])

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

        # Action: Δq_norm in [-1,1]^7 + grip cmd in [0,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0] * 7 + [0.0], dtype=np.float32),
            high=np.array([1.0] * 7 + [1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Obs dim: 14 + 1 + 3+4 + 3+4 + 2 + reach_rel(3) = 34
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32
        )

        self._place_goal = np.array(
            [_PLACE_GOAL_XY[0], _PLACE_GOAL_XY[1], _BOX_Z], dtype=np.float64
        )

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

    # ---- core API ----
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Lock frozen DOFs
        for qadr, q, dadr in self._locked:
            self.data.qpos[qadr] = q
            self.data.qvel[dadr] = 0.0

        self._reach_success_streak = 0

        # Right arm home + open gripper
        for i, adr in enumerate(self._arm_qadr):
            self.data.qpos[adr] = float(
                np.clip(_RIGHT_ARM_HOME[i], self._arm_low[i], self._arm_high[i])
            )
            self.data.qvel[self._arm_vadr[i]] = 0.0
        self.data.qpos[self._grip_qadr] = 0.0

        # Box spawn
        xy = _BOX_XY_CENTER.copy()
        if self.randomize_box:
            offset = self.np_random.uniform(-1.0, 1.0, size=2) * _BOX_XY_HALF_RANGE
            xy = _BOX_XY_CENTER + offset
        self.data.qpos[self._box_qadr : self._box_qadr + 3] = [xy[0], xy[1], _BOX_Z]
        self.data.qpos[self._box_qadr + 3 : self._box_qadr + 7] = [1, 0, 0, 0]
        box_dof = int(self.model.jnt_dofadr[self._box_jnt_id])
        self.data.qvel[box_dof : box_dof + 6] = 0.0

        # Clear non-right ctrl; arm position targets = home; grip open
        self.data.ctrl[:] = 0.0
        for i, aid in enumerate(self._act_arm_ids):
            self.data.ctrl[aid] = float(
                np.clip(_RIGHT_ARM_HOME[i], self._arm_low[i], self._arm_high[i])
            )
        self.data.ctrl[self._act_grip_id] = 0.0

        mujoco.mj_forward(self.model, self.data)
        # Seed potential for progress shaping
        self._last_reach_dist = float(
            np.linalg.norm(self.data.xpos[self._ee_body_id] - self._reach_target())
        )
        obs = self._get_obs()
        info = self._info()
        info["reach_success"] = False
        info["grasp_success"] = False
        info["place_success"] = False
        info["phase"] = self.phase
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(8)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.hold_gripper_open:
            action = action.copy()
            action[7] = 0.0

        # Desired arm targets via incremental position (MJCF position actuators)
        q = np.array([self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64)
        qd = q + action[:7] * self.delta_q_scale
        qd = np.clip(qd, self._arm_low, self._arm_high)
        grip_cmd = float(action[7])  # 0 open .. 1 closed
        grip_target = self._grip_low + grip_cmd * (self._grip_high - self._grip_low)

        for _ in range(self.frame_skip):
            self._apply_locks()
            for aid, target in zip(self._act_arm_ids, qd):
                self.data.ctrl[aid] = float(target)
            self.data.ctrl[self._act_grip_id] = grip_target
            # Zero other actuators (frozen DOFs held via qpos locks)
            for i in range(self.model.nu):
                if i not in self._act_arm_ids and i != self._act_grip_id:
                    self.data.ctrl[i] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._apply_locks()
        self._step_count += 1
        obs = self._get_obs()
        reward, reach_ok = self._compute_reward(action)
        dropped = bool(self._dropped())
        terminated = dropped
        if (
            self.phase == "reach"
            and self.terminate_on_reach_success
            and reach_ok
        ):
            self._reach_success_streak += 1
            if self._reach_success_streak >= self.reach_cfg.success_hold_steps:
                terminated = True
        else:
            self._reach_success_streak = 0
        truncated = self._step_count >= self.max_episode_steps
        info = self._info()
        info["grasp_success"] = self._grasp_success()
        info["place_success"] = False  # curriculum later (M3)
        info["reach_success"] = bool(reach_ok)
        info["phase"] = self.phase
        return obs, float(reward), terminated, truncated, info

    def _apply_locks(self) -> None:
        for qadr, q, dadr in self._locked:
            self.data.qpos[qadr] = q
            self.data.qvel[dadr] = 0.0

    def _get_obs(self) -> np.ndarray:
        q = np.array([self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64)
        dq = np.array([self.data.qvel[a] for a in self._arm_vadr], dtype=np.float64)
        # normalize arm to [-1,1]
        mid = 0.5 * (self._arm_low + self._arm_high)
        half = 0.5 * (self._arm_high - self._arm_low)
        qn = (q - mid) / np.maximum(half, 1e-6)
        dqn = dq / np.pi  # rough
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
        reach_rel = self._reach_target() - ee_pos
        obs = np.concatenate(
            [qn, dqn, grip, ee_pos, ee_quat, box_pos, box_quat, goal_xy, reach_rel]
        )
        return obs.astype(np.float32)

    def _reach_target(self) -> np.ndarray:
        box = self.data.xpos[self._box_body_id]
        return reach_target(box, self.reach_cfg.hover_offset_m)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        ee = self.data.xpos[self._ee_body_id]
        if self.phase == "reach":
            target = self._reach_target()
            prev = self._last_reach_dist
            r, dist, success = reach_reward(
                ee,
                target,
                action=action,
                cfg=self.reach_cfg,
                prev_dist=prev if self._step_count > 1 else None,
            )
            self._last_reach_dist = dist
            return r, success
        box = self.data.xpos[self._box_body_id]
        dist = float(np.linalg.norm(ee - box))
        self._last_reach_dist = dist
        return -dist, False

    def _reward(self) -> float:
        """Backward-compatible single-value reward (no action penalty)."""
        r, _ = self._compute_reward(np.zeros(8, dtype=np.float64))
        return r

    def _dropped(self) -> bool:
        z = float(self.data.xpos[self._box_body_id][2])
        return z < 0.3

    def _grasp_success(self) -> bool:
        box_z = float(self.data.xpos[self._box_body_id][2])
        grip = float(self.data.qpos[self._grip_qadr])
        ee = self.data.xpos[self._ee_body_id]
        box = self.data.xpos[self._box_body_id]
        close = float(np.linalg.norm(ee - box)) < 0.08
        return bool(close and grip > 0.35 and box_z > _BOX_Z + 0.03)

    def _info(self) -> dict[str, Any]:
        target = self._reach_target()
        ee = self.data.xpos[self._ee_body_id]
        dist = float(np.linalg.norm(ee - target))
        return {
            "ee_pose": np.concatenate(
                [self.data.xpos[self._ee_body_id], self.data.xquat[self._ee_body_id]]
            ).copy(),
            "object_pose": np.concatenate(
                [self.data.xpos[self._box_body_id], self.data.xquat[self._box_body_id]]
            ).copy(),
            "reach_target": target.copy(),
            "reach_dist": dist,
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

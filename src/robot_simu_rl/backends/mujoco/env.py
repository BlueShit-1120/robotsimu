"""Gymnasium MuJoCo env: right-arm 7 + gripper 1 pick-place (M0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_simu_rl.common.joints import (
    ARM_EFFORT,
    ARM_JOINT_LIMITS,
    GRIPPER_JOINT_LIMIT,
    LOCKED_QPOS,
    MJCF_RIGHT_ARM_JOINTS,
    MJCF_RIGHT_GRIPPER_JOINT,
)
from robot_simu_rl.common.paths import PICK_PLACE_SCENE

# Box spawn: table center-ish with light XY randomization (meters, world frame)
_BOX_XY_CENTER = np.array([0.55, 0.0], dtype=np.float64)
_BOX_XY_HALF_RANGE = np.array([0.06, 0.08], dtype=np.float64)  # ±
_BOX_Z = 0.765  # table top 0.75 + half height 0.015
_PLACE_GOAL_XY = np.array([0.55, 0.15], dtype=np.float64)

# Default right-arm home (reach toward +X table); tuned lightly for M0 smoke
_RIGHT_ARM_HOME = np.array([0.0, -0.6, 0.0, -1.2, 0.0, -0.4, 0.0], dtype=np.float64)


class PickPlaceMujocoEnv(gym.Env):
    """Right-arm pick-place on a table with a 5×4×3 cm box (D2/D4).

    Action (8): Δq arm(7) clipped + gripper open/close command in [0, 1]
                (mapped to knuckle angle [0, 0.8]).
    Obs: arm q/qd (14) + grip (1) + ee pos(3)/quat(4) + box pos(3)/quat(4) + goal xy(2).
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
        self._step_count = 0
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
        # Obs dim: 14 + 1 + 3+4 + 3+4 + 2 = 31
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
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

        # Clear non-right ctrl; hold grip open
        self.data.ctrl[:] = 0.0
        for i, aid in enumerate(self._act_arm_ids):
            # motor actuators: torque; start at 0
            self.data.ctrl[aid] = 0.0
        self.data.ctrl[self._act_grip_id] = 0.0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(8)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Desired arm targets via incremental position; PD via torque ≈ effort * err (simple)
        q = np.array([self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64)
        qd = q + action[:7] * self.delta_q_scale
        qd = np.clip(qd, self._arm_low, self._arm_high)
        grip_cmd = float(action[7])  # 0 open .. 1 closed
        grip_target = self._grip_low + grip_cmd * (self._grip_high - self._grip_low)

        for _ in range(self.frame_skip):
            self._apply_locks()
            q_now = np.array([self.data.qpos[a] for a in self._arm_qadr], dtype=np.float64)
            dq = np.array([self.data.qvel[a] for a in self._arm_vadr], dtype=np.float64)
            # Simple PD → torque, saturated by URDF effort
            kp = np.array([80, 80, 60, 60, 20, 20, 20], dtype=np.float64)
            kd = np.array([4, 4, 3, 3, 1, 1, 1], dtype=np.float64)
            tau = kp * (qd - q_now) - kd * dq
            tau = np.clip(tau, -np.asarray(ARM_EFFORT), np.asarray(ARM_EFFORT))
            for aid, t in zip(self._act_arm_ids, tau):
                self.data.ctrl[aid] = float(t)
            self.data.ctrl[self._act_grip_id] = grip_target
            # Zero other actuators
            for i in range(self.model.nu):
                if i not in self._act_arm_ids and i != self._act_grip_id:
                    self.data.ctrl[i] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._apply_locks()
        self._step_count += 1
        obs = self._get_obs()
        reward = self._reward()
        terminated = bool(self._dropped())
        truncated = self._step_count >= self.max_episode_steps
        info = self._info()
        info["grasp_success"] = self._grasp_success()
        info["place_success"] = False  # curriculum later (M3)
        info["phase"] = "m0_smoke"
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
        obs = np.concatenate([qn, dqn, grip, ee_pos, ee_quat, box_pos, box_quat, goal_xy])
        return obs.astype(np.float32)

    def _reward(self) -> float:
        ee = self.data.xpos[self._ee_body_id]
        box = self.data.xpos[self._box_body_id]
        dist = float(np.linalg.norm(ee - box))
        return -dist

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

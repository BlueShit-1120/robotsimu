"""Action terms: MuJoCo-style Δq arm + absolute gripper command."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from robot_simu_rl.common.joints import GRIPPER_JOINT_LIMIT, URDF_RIGHT_ARM_JOINTS, URDF_RIGHT_GRIPPER_JOINT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DeltaJointPositionAction(ActionTerm):
    """Latch ``q_target = q + scale * a`` once per env step (MuJoCo Δq parity).

    Unlike Lab's :class:`RelativeJointPositionAction`, the target is **not**
    re-added every physics substep inside ``decimation``.
    """

    cfg: "DeltaJointPositionActionCfg"

    def __init__(self, cfg: "DeltaJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Latched absolute targets: integrate Δq from reset pose instead of
        # ``q_des = q + Δq`` every step. PhysX implicit PD can lag gravity within
        # one control period; re-reading ``q`` then makes a=0 chase the drift
        # (MuJoCo's stiffer servos hide this). Latch matches ideal Δq semantics.
        self._q_des = torch.zeros_like(self._raw_actions)
        self._scale = float(self.cfg.scale)
        self._q_des_initialized = False

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        # MuJoCo clips arm Δq cmds to [-1, 1] before scaling.
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        if not self._q_des_initialized:
            self._q_des[:] = self._asset.data.joint_pos[:, self._joint_ids]
            self._q_des_initialized = True
        self._q_des = self._q_des + self._scale * self._raw_actions
        self._processed_actions = self._q_des

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._q_des[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._raw_actions[env_ids] = 0.0
            self._q_des[env_ids] = self._asset.data.joint_pos[:, self._joint_ids][env_ids]
        self._q_des_initialized = True


@configclass
class DeltaJointPositionActionCfg(ActionTermCfg):
    """Config for :class:`DeltaJointPositionAction`."""

    class_type: type[ActionTerm] = DeltaJointPositionAction
    joint_names: list[str] = MISSING
    scale: float = 0.05


class AbsoluteGripperAction(ActionTerm):
    """Map continuous grip cmd in ``[0, 1]`` → knuckle angle; sync Robotiq mimic set.

    Also forces open when ``hold_open=True`` (reach phase parity with MuJoCo).
    """

    cfg: "AbsoluteGripperActionCfg"

    # Mimic multipliers relative to left knuckle (URDF).
    _MIMIC = (
        ("right_robotiq_85_left_knuckle_joint", 1.0),
        ("right_robotiq_85_right_knuckle_joint", -1.0),
        ("right_robotiq_85_left_inner_knuckle_joint", 1.0),
        ("right_robotiq_85_right_inner_knuckle_joint", -1.0),
        ("right_robotiq_85_left_finger_tip_joint", -1.0),
        ("right_robotiq_85_right_finger_tip_joint", 1.0),
    )

    def __init__(self, cfg: "AbsoluteGripperActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._grip_lo, self._grip_hi = GRIPPER_JOINT_LIMIT
        names = [n for n, _ in self._MIMIC]
        ids, resolved = self._asset.find_joints(names, preserve_order=True)
        # Keep only joints that exist on this articulation.
        self._joint_ids: list[int] = []
        self._multipliers: list[float] = []
        name_to_mult = dict(self._MIMIC)
        for jid, name in zip(ids, resolved):
            self._joint_ids.append(int(jid))
            self._multipliers.append(float(name_to_mult.get(name, 1.0)))
        self._mult_t = torch.tensor(self._multipliers, device=self.device, dtype=torch.float32)
        self._drive_id = None
        for jid, name in zip(ids, resolved):
            if name == URDF_RIGHT_GRIPPER_JOINT:
                self._drive_id = int(jid)
                break
        self._targets = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets

    def process_actions(self, actions: torch.Tensor) -> None:
        cmd = actions.view(self.num_envs, -1)[:, :1].clamp(0.0, 1.0)
        if self.cfg.hold_open:
            cmd = torch.zeros_like(cmd)
        self._raw_actions[:] = cmd
        q = self._grip_lo + cmd * (self._grip_hi - self._grip_lo)  # (N,1)
        self._targets = q * self._mult_t.unsqueeze(0)

    def apply_actions(self) -> None:
        if not self._joint_ids:
            return
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class AbsoluteGripperActionCfg(ActionTermCfg):
    """Config for :class:`AbsoluteGripperAction`."""

    class_type: type[ActionTerm] = AbsoluteGripperAction
    hold_open: bool = False


# Re-export arm joint list for cfg convenience
DEFAULT_ARM_JOINT_NAMES = list(URDF_RIGHT_ARM_JOINTS)

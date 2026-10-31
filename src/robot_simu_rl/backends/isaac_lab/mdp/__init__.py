"""MDP terms for Isaac Lab Reach/Grasp (MuJoCo-aligned action/obs where practical)."""

from robot_simu_rl.backends.isaac_lab.mdp.actions import (
    AbsoluteGripperAction,
    AbsoluteGripperActionCfg,
    DeltaJointPositionAction,
    DeltaJointPositionActionCfg,
)
from robot_simu_rl.backends.isaac_lab.mdp.events import reset_aux_buffers, reset_scene_and_workpiece
from robot_simu_rl.backends.isaac_lab.mdp.observations import mujoco_style_obs
from robot_simu_rl.backends.isaac_lab.mdp.rewards import (
    grasp_reward,
    reach_reward,
)
from robot_simu_rl.backends.isaac_lab.mdp.terminations import early_success, time_out

__all__ = [
    "AbsoluteGripperAction",
    "AbsoluteGripperActionCfg",
    "DeltaJointPositionAction",
    "DeltaJointPositionActionCfg",
    "reset_aux_buffers",
    "reset_scene_and_workpiece",
    "mujoco_style_obs",
    "grasp_reward",
    "reach_reward",
    "early_success",
    "time_out",
]

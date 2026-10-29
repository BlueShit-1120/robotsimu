"""Joint name mapping: logical ↔ URDF ↔ MJCF (robotsimu DVT1).

URDF SoT uses `right_robotiq_85_*`; MJCF uses `right_gripper_*`.
Phase 0 (D4): right arm 7 + right gripper drive 1 = 8 DoF.
"""

from __future__ import annotations

# Logical names used by task / training code
LOGICAL_ARM = tuple(f"arm_r_j{i}" for i in range(1, 8))
LOGICAL_GRIPPER = "grip_r"

URDF_RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 8))
URDF_RIGHT_GRIPPER_JOINT = "right_robotiq_85_left_knuckle_joint"

MJCF_RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 8))
MJCF_RIGHT_GRIPPER_JOINT = "right_gripper_left_finger_joint"

LOGICAL_TO_URDF = {
    **{f"arm_r_j{i}": f"right_arm_joint{i}" for i in range(1, 8)},
    "grip_r": URDF_RIGHT_GRIPPER_JOINT,
}

LOGICAL_TO_MJCF = {
    **{f"arm_r_j{i}": f"right_arm_joint{i}" for i in range(1, 8)},
    "grip_r": MJCF_RIGHT_GRIPPER_JOINT,
}

# (lower, upper) rad — from robot_assemble.urdf / MJCF
ARM_JOINT_LIMITS = (
    (-2.9322, 2.9322),
    (-2.0595, 2.0595),
    (-2.9322, 2.9322),
    (-2.4086, 1.0123),
    (-2.9322, 2.9322),
    (-1.0123, 1.0123),
    (-1.5359, 1.5359),
)

GRIPPER_JOINT_LIMIT = (0.0, 0.8)  # 0 open-ish posture; 0.8 closed (2F-85 knuckle)

# Scene-fixed pose for locked DOFs (torso/waist common scene values)
LOCKED_QPOS = {
    "torso_lift_joint": 0.5,
    "waist_pitch_joint": 0.3,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    # Left arm stowed-ish (within limits)
    "left_arm_joint1": 0.0,
    "left_arm_joint2": 0.4,
    "left_arm_joint3": 0.0,
    "left_arm_joint4": -1.2,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": 0.0,
    "left_arm_joint7": 0.0,
    "left_gripper_left_finger_joint": 0.0,
    # Wheels / steering
    "front_left_steering_joint": 0.0,
    "front_right_steering_joint": 0.0,
    "rear_left_steering_joint": 0.0,
    "rear_right_steering_joint": 0.0,
    "front_left_wheel_joint": 0.0,
    "front_right_wheel_joint": 0.0,
    "rear_left_wheel_joint": 0.0,
    "rear_right_wheel_joint": 0.0,
}

ARM_EFFORT = (108.0, 108.0, 66.0, 66.0, 18.0, 18.0, 18.0)

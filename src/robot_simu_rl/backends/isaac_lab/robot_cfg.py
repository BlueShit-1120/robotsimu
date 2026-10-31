"""ArticulationCfg for DVT1 proxy URDF / USD (Phase 0: frozen base, right arm + grip).

Requires Isaac Lab / Isaac Sim imports — only import inside env_isaacsim.
"""

from __future__ import annotations

from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD
from robot_simu_rl.common.joints import LOCKED_QPOS, URDF_RIGHT_ARM_JOINTS, URDF_RIGHT_GRIPPER_JOINT

# Grasp-near home from MuJoCo M4 (rad); used as default arm pose for smoke/reach.
_RIGHT_ARM_HOME_GRASP = {
    "right_arm_joint1": -0.38357,
    "right_arm_joint2": -1.04487,
    "right_arm_joint3": -0.0086,
    "right_arm_joint4": -0.75563,
    "right_arm_joint5": 0.04278,
    "right_arm_joint6": -0.37762,
    "right_arm_joint7": 0.98704,
}


def build_spr_robot_cfg(
    *,
    prefer_usd: bool = True,
    fix_base: bool = True,
    activate_contact_sensors: bool = False,
):
    """Return ArticulationCfg spawning from USD (if present) or proxy URDF.

    Parameters
    ----------
    prefer_usd:
        Use pre-converted USD when ``ROBOT_USD`` exists; else spawn via UrdfFileCfg.
    fix_base:
        Weld ``base_link`` (D4).
    activate_contact_sensors:
        Enable PhysX contact reporters (needed for fingertip ContactSensor).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg

    # URDF joint names only (skip MJCF gripper aliases in LOCKED_QPOS).
    _lock_keys = (
        "torso_lift_joint",
        "waist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        *(f"left_arm_joint{i}" for i in range(1, 8)),
        "front_left_steering_joint",
        "front_right_steering_joint",
        "rear_left_steering_joint",
        "rear_right_steering_joint",
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    )
    joint_pos = {
        **{k: float(LOCKED_QPOS[k]) for k in _lock_keys if k in LOCKED_QPOS},
        "left_robotiq_85_left_knuckle_joint": 0.0,
        **_RIGHT_ARM_HOME_GRASP,
        URDF_RIGHT_GRIPPER_JOINT: 0.0,
    }

    if prefer_usd and ROBOT_USD.is_file():
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD.resolve()),
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        )
    else:
        if not PROXY_URDF.is_file():
            raise FileNotFoundError(
                f"Proxy URDF missing: {PROXY_URDF}. Run scripts/make_proxy_urdf.py first."
            )
        spawn = sim_utils.UrdfFileCfg(
            asset_path=str(PROXY_URDF.resolve()),
            fix_base=fix_base,
            force_usd_conversion=True,
            merge_fixed_joints=False,
            activate_contact_sensors=activate_contact_sensors,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=80.0,
                    damping=4.0,
                ),
                target_type="position",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        )

    return ArticulationCfg(
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=joint_pos,
        ),
        actuators={
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=list(URDF_RIGHT_ARM_JOINTS),
                effort_limit_sim=108.0,
                stiffness=80.0,
                damping=4.0,
            ),
            # One regex per joint — Lab forbids overlapping exprs for the same DOF.
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["right_robotiq_85_.*"],
                effort_limit_sim=50.0,
                stiffness=200.0,
                damping=10.0,
            ),
            # Lock non-trained DOFs with stiff position holds (D4).
            "locked": ImplicitActuatorCfg(
                joint_names_expr=[
                    "torso_lift_joint",
                    "waist_pitch_joint",
                    "head_.*",
                    "left_arm_joint.*",
                    "left_robotiq_85_.*",
                    ".*_steering_joint",
                    ".*_wheel_joint",
                ],
                effort_limit_sim=200.0,
                stiffness=1.0e4,
                damping=1.0e2,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


# Lazy name used by docs / smoke; call build_spr_robot_cfg() after AppLauncher.
SPR_ROBOT_CFG = None  # populated by smoke after SimulationApp starts

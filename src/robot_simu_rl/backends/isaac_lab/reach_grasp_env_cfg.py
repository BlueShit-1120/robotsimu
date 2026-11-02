"""ManagerBasedRLEnvCfg: Reach / Grasp phases with MuJoCo-aligned 8-DoF interface."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from robot_simu_rl.backends.isaac_lab import mdp
from robot_simu_rl.backends.isaac_lab.mdp.actions import DEFAULT_ARM_JOINT_NAMES
from robot_simu_rl.backends.isaac_lab.robot_cfg import build_spr_robot_cfg
from robot_simu_rl.common.joints import URDF_RIGHT_ARM_JOINTS

Phase = Literal["reach", "grasp"]


def _build_scene(
    num_envs: int,
    env_spacing: float,
    prefer_usd: bool,
    enable_contact: bool,
    arm_home: str = "grasp",
):
    robot_cfg = build_spr_robot_cfg(
        prefer_usd=prefer_usd,
        fix_base=True,
        activate_contact_sensors=enable_contact,
        arm_home=arm_home,
    )
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"

    table_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.65, 0.0, 0.71]),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 1.0, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.28)),
        ),
    )

    workpiece_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.70, -0.20, 0.765], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.04, 0.03),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.15)),
        ),
    )

    ground_cfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )
    light_cfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    tip_l = None
    tip_r = None
    if enable_contact:
        tip_l = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_robotiq_85_left_finger_tip_link",
            update_period=0.0,
            history_length=2,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece"],
        )
        tip_r = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_robotiq_85_right_finger_tip_link",
            update_period=0.0,
            history_length=2,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece"],
        )

    @configclass
    class ReachGraspSceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        table = table_cfg
        workpiece = workpiece_cfg
        ground = ground_cfg
        light = light_cfg
        tip_l_contact = tip_l
        tip_r_contact = tip_r

    return ReachGraspSceneCfg(
        num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True
    )


@configclass
class ActionsCfg:
    arm_action: mdp.DeltaJointPositionActionCfg = mdp.DeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=DEFAULT_ARM_JOINT_NAMES,
        scale=0.05,
    )
    gripper_action: mdp.AbsoluteGripperActionCfg = mdp.AbsoluteGripperActionCfg(
        asset_name="robot",
        hold_open=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.mujoco_style_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset = EventTerm(func=mdp.reset_scene_and_workpiece, mode="reset")


@configclass
class RewardsReachCfg:
    reach = RewTerm(func=mdp.reach_reward, weight=1.0, params={"hover_offset": 0.30, "success_dist": 0.05})


@configclass
class RewardsGraspCfg:
    grasp = RewTerm(func=mdp.grasp_reward, weight=1.0, params={"lift_height": 0.03})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.early_success)


@configclass
class ReachGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Right-arm Reach/Grasp ManagerBased env (frozen base, 8-DoF Δq+grip).

    Extra fields (not part of upstream Lab cfg schema but read by MDP terms):
    - ``phase``: ``\"reach\"`` | ``\"grasp\"``
    - ``delta_q_scale``, ``hover_offset``, ``success_dist``, ``lift_height``
    """

    phase: Phase = "reach"
    delta_q_scale: float = 0.05
    hover_offset: float = 0.30
    success_dist: float = 0.05
    lift_height: float = 0.03
    prefer_usd: bool = True
    enable_contact: bool = True
    early_terminate_success: bool = True

    scene: InteractiveSceneCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsReachCfg = RewardsReachCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None
    num_envs: int = 1
    env_spacing: float = 3.0

    def __post_init__(self):
        # Control 50 Hz (dt=0.01, decimation=2). Finer 500 Hz PhysX was ~6× slower
        # with little gain once Δq targets are latched (see mdp/actions.py).
        self.sim.dt = 0.01
        self.decimation = 2
        self.episode_length_s = 8.0  # 50 Hz control → 400 steps like MuJoCo default
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.8, -1.2, 1.4)
        self.viewer.lookat = (0.5, -0.1, 0.8)

        # Reach uses MuJoCo M1 near-home; Grasp keeps M4 grasp-home.
        arm_home = "near" if self.phase == "reach" else "grasp"
        self.scene = _build_scene(
            num_envs=int(self.num_envs),
            env_spacing=float(self.env_spacing),
            prefer_usd=self.prefer_usd,
            enable_contact=bool(self.enable_contact and self.phase == "grasp"),
            arm_home=arm_home,
        )

        self.actions.arm_action.scale = self.delta_q_scale
        self.actions.arm_action.joint_names = list(URDF_RIGHT_ARM_JOINTS)
        self.actions.gripper_action.hold_open = self.phase == "reach"

        # Isaac Lab RewardManager multiplies terms by step_dt (= sim.dt * decimation).
        # MuJoCo returns per-control-step rewards without that factor — compensate so
        # −dist / progress / success bonuses match M1 magnitudes for SB3 PPO.
        step_dt = float(self.sim.dt) * float(self.decimation)
        reward_dt_scale = 1.0 / max(step_dt, 1e-6)

        if self.phase == "grasp":
            self.rewards = RewardsGraspCfg()
            self.rewards.grasp.params["lift_height"] = self.lift_height
            self.rewards.grasp.weight = reward_dt_scale
        else:
            self.rewards = RewardsReachCfg()
            self.rewards.reach.params["hover_offset"] = self.hover_offset
            self.rewards.reach.params["success_dist"] = self.success_dist
            self.rewards.reach.weight = reward_dt_scale

        if not self.early_terminate_success:
            self.terminations.success = None


def make_reach_grasp_env_cfg(
    *,
    phase: Phase = "reach",
    num_envs: int = 1,
    prefer_usd: bool = True,
    delta_q_scale: float = 0.05,
    hover_offset: float = 0.30,
    success_dist: float = 0.05,
    lift_height: float = 0.03,
    enable_contact: bool = True,
    early_terminate_success: bool = False,
    max_episode_steps: int | None = None,
) -> ReachGraspEnvCfg:
    """Factory used by smoke / training entrypoints."""
    cfg = ReachGraspEnvCfg(
        phase=phase,
        num_envs=num_envs,
        prefer_usd=prefer_usd,
        delta_q_scale=delta_q_scale,
        hover_offset=hover_offset,
        success_dist=success_dist,
        lift_height=lift_height,
        enable_contact=enable_contact,
        early_terminate_success=early_terminate_success,
    )
    if max_episode_steps is not None:
        cfg.episode_length_s = float(max_episode_steps) * cfg.decimation * cfg.sim.dt
    return cfg

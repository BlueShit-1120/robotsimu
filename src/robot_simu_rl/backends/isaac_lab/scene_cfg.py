"""Minimal InteractiveScene for right-arm + table + box (Isaac Lab first slice)."""

from __future__ import annotations


def build_pick_place_scene_cfg(num_envs: int = 1, env_spacing: float = 3.0, prefer_usd: bool = True):
    """Build InteractiveSceneCfg with frozen-base SPR robot, cuboid table, and 5×4×3 cm box."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
    from isaaclab.utils import configclass

    from robot_simu_rl.backends.isaac_lab.robot_cfg import build_spr_robot_cfg

    robot_cfg = build_spr_robot_cfg(prefer_usd=prefer_usd, fix_base=True)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"

    # Table top ≈ z=0.75 like MuJoCo scene (box half-height 0.04 → center 0.71).
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

    # Workpiece: 5×4×3 cm, 0.15 kg (D2). Spawn above table at MuJoCo-ish XY.
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

    @configclass
    class PickPlaceSceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        table = table_cfg
        workpiece = workpiece_cfg
        ground = ground_cfg
        light = light_cfg

    return PickPlaceSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)

"""Reset helpers for Reach/Grasp env + M4 contact-stack startup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Match MuJoCo M1 spawn / table height.
_BOX_XY = (0.70, -0.20)
_BOX_Z = 0.765
_BOX_Z0 = 0.75  # table top (approx; half-height 0.015 → center 0.765)
# MuJoCo M3 place goal (≈8 cm from spawn toward +Y).
_PLACE_GOAL_XY = (0.70, -0.12)

# MuJoCo M4 right fingertip pads (sphere r=0.018, μ≈3.5, tip↔table filtered).
_RIGHT_TIP_BODY_NAMES = (
    "right_robotiq_85_left_finger_tip_link",
    "right_robotiq_85_right_finger_tip_link",
)


def reset_aux_buffers(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Allocate / clear progress + success bookkeeping used by rewards."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = env.num_envs
    if not hasattr(env, "_prev_task_dist"):
        env._prev_task_dist = torch.zeros(n, device=env.device)
        env._reached_once = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._lifted_once = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._hold_lift_steps = torch.zeros(n, dtype=torch.long, device=env.device)
        env._place_ok_steps = torch.zeros(n, dtype=torch.long, device=env.device)
        env._placed_once = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._ever_held = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._place_hold_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._prev_place_dist = torch.zeros(n, device=env.device)
        env._box_z0 = torch.full((n,), _BOX_Z0, device=env.device)
        env._place_goal_xy = torch.tensor(_PLACE_GOAL_XY, device=env.device, dtype=torch.float32)
    if not hasattr(env, "_hold_lift_steps"):
        env._hold_lift_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_place_ok_steps"):
        env._place_ok_steps = torch.zeros(n, dtype=torch.long, device=env.device)
        env._placed_once = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._ever_held = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._place_hold_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._prev_place_dist = torch.zeros(n, device=env.device)
        env._place_goal_xy = torch.tensor(_PLACE_GOAL_XY, device=env.device, dtype=torch.float32)
    env._prev_task_dist[env_ids] = 0.0
    env._reached_once[env_ids] = False
    env._lifted_once[env_ids] = False
    env._hold_lift_steps[env_ids] = 0
    env._place_ok_steps[env_ids] = 0
    env._placed_once[env_ids] = False
    env._ever_held[env_ids] = False
    env._place_hold_active[env_ids] = False
    env._prev_place_dist[env_ids] = 0.0
    env._box_z0[env_ids] = _BOX_Z0


def reset_scene_and_workpiece(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("workpiece"),
    randomize_box: bool = False,
    xy_half_range: float = 0.04,
) -> None:
    """Reset robot/object; Place phase may seed holding (M3 C3 release curriculum)."""
    reset_scene_to_default(env, env_ids, reset_joint_targets=True)

    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    n = len(env_ids)
    phase = getattr(env.cfg, "phase", "reach")
    place_start_holding = bool(getattr(env.cfg, "place_start_holding", phase == "place"))
    lift_height = float(getattr(env.cfg, "lift_height", 0.03))
    place_tol_xy = float(getattr(env.cfg, "place_tol_xy", 0.12))

    pos = torch.zeros(n, 3, device=env.device)
    pos[:, 0] = _BOX_XY[0]
    pos[:, 1] = _BOX_XY[1]
    pos[:, 2] = _BOX_Z
    # M3: if spawn→goal already inside tol, seed at spawn; else seed at goal.
    if phase == "place" and place_start_holding:
        spawn_xy = torch.tensor(_BOX_XY, device=env.device)
        goal_xy = torch.tensor(_PLACE_GOAL_XY, device=env.device)
        spawn_to_goal = torch.norm(spawn_xy - goal_xy)
        if float(spawn_to_goal) >= place_tol_xy:
            pos[:, 0] = _PLACE_GOAL_XY[0]
            pos[:, 1] = _PLACE_GOAL_XY[1]
        pos[:, 2] = _BOX_Z0 + lift_height
    if randomize_box:
        jitter = (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * xy_half_range
        pos[:, 0:2] += jitter
    pos = pos + env.scene.env_origins[env_ids]
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # wxyz identity
    pose = torch.cat([pos, quat], dim=-1)
    vel = torch.zeros(n, 6, device=env.device)
    obj.write_root_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(vel, env_ids=env_ids)

    reset_aux_buffers(env, env_ids)

    if phase == "place" and place_start_holding:
        # Close gripper + enable kinematic hold-assist (MuJoCo latch stand-in).
        from robot_simu_rl.common.joints import GRIPPER_JOINT_LIMIT, URDF_RIGHT_GRIPPER_JOINT

        grip_ids, _ = robot.find_joints([URDF_RIGHT_GRIPPER_JOINT], preserve_order=True)
        grip_id = int(grip_ids[0])
        grip_lo, grip_hi = GRIPPER_JOINT_LIMIT
        closed_q = grip_lo + 0.70 * (grip_hi - grip_lo)
        q = robot.data.joint_pos.clone()
        q[env_ids, grip_id] = closed_q
        # Mimic companions roughly (same magnitude, signs from URDF).
        for name, mult in (
            ("right_robotiq_85_right_knuckle_joint", -1.0),
            ("right_robotiq_85_left_inner_knuckle_joint", 1.0),
            ("right_robotiq_85_right_inner_knuckle_joint", -1.0),
            ("right_robotiq_85_left_finger_tip_joint", -1.0),
            ("right_robotiq_85_right_finger_tip_joint", 1.0),
        ):
            ids, _ = robot.find_joints([name], preserve_order=True)
            if ids:
                q[env_ids, int(ids[0])] = closed_q * mult
        robot.write_joint_position_to_sim(q[env_ids], env_ids=env_ids)
        # Seed AbsoluteGripperAction raw cmd closed if present.
        am = getattr(env, "action_manager", None)
        if am is not None and "gripper_action" in getattr(am, "_terms", {}):
            term = am._terms["gripper_action"]
            if hasattr(term, "_raw_actions"):
                term._raw_actions[env_ids] = 0.70
        env._place_hold_active[env_ids] = True
        env._ever_held[env_ids] = True
        env._lifted_once[env_ids] = True
        env._reached_once[env_ids] = True
        box_xy = pos[:, :2] - env.scene.env_origins[env_ids, :2]
        goal = env._place_goal_xy.unsqueeze(0).expand(n, -1)
        env._prev_place_dist[env_ids] = torch.norm(box_xy - goal, dim=-1)


def _num_shapes_per_body(robot) -> list[int]:
    """Shape counts per link (same pattern as Lab ``randomize_rigid_body_material``)."""
    counts: list[int] = []
    for link_path in robot.root_physx_view.link_paths[0]:
        link_view = robot._physics_sim_view.create_rigid_body_view(link_path)
        counts.append(int(link_view.max_shapes))
    return counts


def apply_m4_contact_stack(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tip_static_friction: float = 3.5,
    tip_dynamic_friction: float = 3.0,
    tip_restitution: float = 0.0,
    filter_table: bool = True,
) -> None:
    """Port MuJoCo M4 tip pads: high friction + tip–table FilteredPairs.

    Non-tip right-gripper collisions are stripped in ``make_proxy_urdf.py`` (and
    the regenerated USD). Do **not** toggle CollisionAPI after PhysX tensor views
    exist — that invalidates the sim view (detachShape / setTransforms failures).

    PhysX has no ``noslip_iterations``; high μ + more solver iters are the stand-in.
    """
    del env_ids  # apply to all envs / stage prims once
    import omni.usd
    from pxr import Sdf, UsdPhysics

    robot = env.scene[robot_cfg.name]
    stage = omni.usd.get_context().get_stage()
    env_arange = torch.arange(env.num_envs, device="cpu")

    tip_ids, tip_names = robot.find_bodies(list(_RIGHT_TIP_BODY_NAMES), preserve_order=True)
    tip_shape_ranges: list[tuple[int, int]] = []
    if tip_ids:
        materials = robot.root_physx_view.get_material_properties().clone()
        max_shapes = int(robot.root_physx_view.max_shapes)
        try:
            shapes_per_body = _num_shapes_per_body(robot)
            if sum(shapes_per_body) != max_shapes:
                shapes_per_body = None
        except Exception:  # noqa: BLE001
            shapes_per_body = None

        if shapes_per_body is not None:
            for bid in tip_ids:
                start = sum(shapes_per_body[: int(bid)])
                end = start + shapes_per_body[int(bid)]
                if start < end:
                    materials[env_arange, start:end, 0] = tip_static_friction
                    materials[env_arange, start:end, 1] = tip_dynamic_friction
                    materials[env_arange, start:end, 2] = tip_restitution
                    tip_shape_ranges.append((start, end))
        elif max_shapes == len(robot.body_names):
            for bid in tip_ids:
                materials[env_arange, int(bid), 0] = tip_static_friction
                materials[env_arange, int(bid), 1] = tip_dynamic_friction
                materials[env_arange, int(bid), 2] = tip_restitution
                tip_shape_ranges.append((int(bid), int(bid) + 1))
        else:
            materials[env_arange, :, 0] = tip_static_friction
            materials[env_arange, :, 1] = tip_dynamic_friction
            materials[env_arange, :, 2] = tip_restitution
            tip_shape_ranges.append((0, max_shapes))

        robot.root_physx_view.set_material_properties(materials, env_arange)
        env._m4_tip_friction = (tip_static_friction, tip_dynamic_friction)
        env._m4_tip_body_ids = [int(i) for i in tip_ids]
        env._m4_tip_body_names = list(tip_names)
        env._m4_tip_shape_ranges = tip_shape_ranges

    # Apply FilteredPairs on Table/Ground (unique per-env prims), NOT on tip links.
    # Tip links live under an instanceable robot USD; mutating them deletes shared
    # collision prototypes and invalidates PhysX tensor views (multi-env crash).
    if filter_table:
        tip_targets = [
            Sdf.Path(f"/World/envs/env_{env_i}/Robot/{tip_name}")
            for env_i in range(env.num_envs)
            for tip_name in _RIGHT_TIP_BODY_NAMES
        ]
        for env_i in range(env.num_envs):
            table_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/Table")
            if not table_prim.IsValid():
                continue
            api = UsdPhysics.FilteredPairsAPI.Apply(table_prim)
            rel = api.GetFilteredPairsRel()
            if not rel:
                rel = api.CreateFilteredPairsRel()
            rel.ClearTargets(True)
            for tip_name in _RIGHT_TIP_BODY_NAMES:
                tip_path = f"/World/envs/env_{env_i}/Robot/{tip_name}"
                if stage.GetPrimAtPath(tip_path).IsValid():
                    rel.AddTarget(Sdf.Path(tip_path))
        ground = stage.GetPrimAtPath("/World/GroundPlane")
        if ground.IsValid():
            api = UsdPhysics.FilteredPairsAPI.Apply(ground)
            rel = api.GetFilteredPairsRel()
            if not rel:
                rel = api.CreateFilteredPairsRel()
            rel.ClearTargets(True)
            for t in tip_targets:
                if stage.GetPrimAtPath(t).IsValid():
                    rel.AddTarget(t)
        env._m4_tip_table_filter = True

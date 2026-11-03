#!/usr/bin/env python3
"""Diagnose M4 contact stack on Isaac Grasp: tip friction, tip–table filter, oracle pinch-lift.

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/diag_isaac_grasp_contact.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Isaac M4 tip contact stack.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-urdf", action="store_true")
    parser.add_argument("--steps-close", type=int, default=40)
    parser.add_argument("--steps-lift", type=int, default=50)
    parser.add_argument("--out", type=str, default="")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    result: dict = {"ok": False}
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from pxr import UsdPhysics
        import omni.usd

        from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import make_reach_grasp_env_cfg

        prefer_usd = not bool(args.force_urdf)
        cfg = make_reach_grasp_env_cfg(
            phase="grasp",
            num_envs=args.num_envs,
            prefer_usd=prefer_usd,
            delta_q_scale=0.03,
            hover_offset=0.06,
            lift_height=0.03,
            enable_contact=True,
            early_terminate_success=False,
            max_episode_steps=160,
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()

        robot = env.scene["robot"]
        stage = omni.usd.get_context().get_stage()
        tip_names = [
            "right_robotiq_85_left_finger_tip_link",
            "right_robotiq_85_right_finger_tip_link",
        ]
        tip_ids, _ = robot.find_bodies(tip_names, preserve_order=True)

        # Friction sample at tip shape indices.
        mats = robot.root_physx_view.get_material_properties()
        tip_mu = []
        for bid in tip_ids:
            # materials may be indexed by shape; try body index first
            row = mats[0, int(bid)].tolist() if mats.shape[1] > int(bid) else []
            tip_mu.append({"body_id": int(bid), "material": row})

        # FilteredPairs live on Table/Ground (not tip links — instanceable-safe).
        filters = {}
        for path in ("/World/envs/env_0/Table", "/World/GroundPlane"):
            prim = stage.GetPrimAtPath(path)
            targets = []
            if prim.IsValid() and prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                rel = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
                if rel:
                    targets = [str(t) for t in rel.GetTargets()]
            filters[path] = targets

        result["prefer_usd"] = prefer_usd
        result["m4_tip_friction_attr"] = getattr(env, "_m4_tip_friction", None)
        result["m4_tip_table_filter_attr"] = getattr(env, "_m4_tip_table_filter", None)
        result["m4_tip_shape_ranges"] = getattr(env, "_m4_tip_shape_ranges", None)
        result["tip_material_sample"] = tip_mu
        result["filtered_pairs"] = filters
        if getattr(env, "_m4_tip_shape_ranges", None):
            mats0 = mats[0]
            shape_mu = []
            for start, end in env._m4_tip_shape_ranges:
                shape_mu.append(mats0[start:end].tolist())
            result["tip_shape_materials"] = shape_mu

        # Oracle: hold tip-mid near box, close grip, then lift joint2 a bit.
        obs, _ = env.reset()
        box = env.scene["workpiece"]
        z0 = float(box.data.root_pos_w[0, 2].item())
        act_dim = env.action_space.shape[-1]
        # action: 7 arm Δq + 1 grip
        close = torch.zeros(args.num_envs, act_dim, device=env.device)
        close[:, -1] = 1.0  # close gripper
        max_box_z = z0
        left_hits = 0
        right_hits = 0
        for _ in range(int(args.steps_close)):
            obs, rew, term, trunc, info = env.step(close)
            max_box_z = max(max_box_z, float(box.data.root_pos_w[0, 2].item()))
            log = info.get("log", {}) if isinstance(info, dict) else {}
            # ManagerBased may put extras on env
            extras = getattr(env, "extras", {}).get("log", {})
            if "tip_l_contact" in extras:
                left_hits += int(float(extras["tip_l_contact"].float().mean().item()) > 0.5)
            if "tip_r_contact" in extras:
                right_hits += int(float(extras["tip_r_contact"].float().mean().item()) > 0.5)

        lift = close.clone()
        # nudge shoulder/elbow upward (joint2 index 1 often raises tip for this arm)
        lift[:, 1] = -0.4
        lift[:, -1] = 1.0
        for _ in range(int(args.steps_lift)):
            env.step(lift)
            max_box_z = max(max_box_z, float(box.data.root_pos_w[0, 2].item()))

        z1 = float(box.data.root_pos_w[0, 2].item())
        result["box_z0"] = z0
        result["box_z_final"] = z1
        result["box_z_peak"] = max_box_z
        result["lift_peak"] = max_box_z - z0
        result["lift_final"] = z1 - z0
        result["close_left_hit_steps"] = left_hits
        result["close_right_hit_steps"] = right_hits
        result["ok"] = True
        print(json.dumps(result, indent=2, default=str))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        print(json.dumps(result, indent=2, default=str))
        raise
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe tip ContactSensor net/matrix forces vs Workpiece after forced close.

Usage::
    ./isaaclab.sh -p .../scripts/diag_isaac_tip_forces.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--out", type=str, default="/tmp/isaac-m15-tip-forces.json")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    result: dict = {"ok": False}
    try:
        import omni.usd
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from pxr import PhysxSchema

        from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import make_reach_grasp_env_cfg

        cfg = make_reach_grasp_env_cfg(
            phase="grasp",
            num_envs=1,
            prefer_usd=True,
            delta_q_scale=0.03,
            hover_offset=0.06,
            lift_height=0.03,
            enable_contact=True,
            early_terminate_success=False,
            max_episode_steps=160,
        )
        if args.no_filter:
            cfg.scene.tip_l_contact.filter_prim_paths_expr = []
            cfg.scene.tip_r_contact.filter_prim_paths_expr = []

        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        robot = env.scene["robot"]
        tip_l_id0 = robot.find_bodies(["right_robotiq_85_left_finger_tip_link"])[0][0]
        tip_r_id0 = robot.find_bodies(["right_robotiq_85_right_finger_tip_link"])[0][0]
        tip_mid0 = 0.5 * (
            robot.data.body_pos_w[:, tip_l_id0] + robot.data.body_pos_w[:, tip_r_id0]
        )
        wp_obj = env.scene["workpiece"]
        env_ids = torch.tensor([0], device=env.device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        # Nudge box slightly up so sphere pads sandwich it when closing.
        target = tip_mid0.clone()
        target[:, 2] += 0.01
        wp_obj.write_root_pose_to_sim(torch.cat([target, quat], dim=-1), env_ids=env_ids)
        wp_obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)
        env.scene.write_data_to_sim()
        for _ in range(4):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        result["placed_box_at_tip_mid"] = target[0].tolist()
        stage = omni.usd.get_context().get_stage()

        tip_paths = [
            "/World/envs/env_0/Robot/right_robotiq_85_left_finger_tip_link",
            "/World/envs/env_0/Robot/right_robotiq_85_right_finger_tip_link",
        ]
        wp = "/World/envs/env_0/Workpiece"
        reporters = {}
        for p in tip_paths + [wp]:
            prim = stage.GetPrimAtPath(p)
            has = bool(prim.IsValid() and prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
            reporters[p] = {"valid": prim.IsValid(), "contact_report_api": has}
            if prim.IsValid():
                kids = [str(c.GetPath()) for c in prim.GetChildren()]
                reporters[p]["children"] = kids[:12]

        sens_l = env.scene["tip_l_contact"]
        sens_r = env.scene["tip_r_contact"]
        result["sensor_body_names"] = {
            "tip_l": list(getattr(sens_l, "body_names", [])),
            "tip_r": list(getattr(sens_r, "body_names", [])),
        }
        result["filter_expr"] = {
            "tip_l": list(sens_l.cfg.filter_prim_paths_expr),
            "tip_r": list(sens_r.cfg.filter_prim_paths_expr),
        }
        result["reporters"] = reporters

        act = torch.zeros(1, env.action_space.shape[-1], device=env.device)
        act[:, -1] = 1.0
        peaks = {
            "net_l": 0.0,
            "net_r": 0.0,
            "mat_l": 0.0,
            "mat_r": 0.0,
            "holding": 0,
            "pinch": 0,
            "pinch_geom": 0,
            "box_z_peak": float(env.scene["workpiece"].data.root_pos_w[0, 2].item()),
        }
        for i in range(args.steps):
            if i == args.steps // 2:
                act[:, 1] = -0.5  # start lifting mid-way
            env.step(act)
            nl = float(torch.norm(sens_l.data.net_forces_w, dim=-1).amax().item())
            nr = float(torch.norm(sens_r.data.net_forces_w, dim=-1).amax().item())
            peaks["net_l"] = max(peaks["net_l"], nl)
            peaks["net_r"] = max(peaks["net_r"], nr)
            if sens_l.data.force_matrix_w is not None:
                peaks["mat_l"] = max(
                    peaks["mat_l"],
                    float(torch.norm(sens_l.data.force_matrix_w.reshape(1, -1, 3), dim=-1).amax().item()),
                )
            if sens_r.data.force_matrix_w is not None:
                peaks["mat_r"] = max(
                    peaks["mat_r"],
                    float(torch.norm(sens_r.data.force_matrix_w.reshape(1, -1, 3), dim=-1).amax().item()),
                )
            log = env.extras.get("log", {})
            if "holding" in log:
                peaks["holding"] += int(float(log["holding"].float().item()) > 0.5)
            if "pinch" in log:
                peaks["pinch"] += int(float(log["pinch"].float().item()) > 0.5)
            if "pinch_geom" in log:
                peaks["pinch_geom"] += int(float(log["pinch_geom"].float().item()) > 0.5)
            peaks["box_z_peak"] = max(
                peaks["box_z_peak"], float(env.scene["workpiece"].data.root_pos_w[0, 2].item())
            )

        # Tip / box poses at end
        robot = env.scene["robot"]
        tip_l_id = robot.find_bodies(["right_robotiq_85_left_finger_tip_link"])[0][0]
        tip_r_id = robot.find_bodies(["right_robotiq_85_right_finger_tip_link"])[0][0]
        pl = robot.data.body_pos_w[0, tip_l_id].tolist()
        pr = robot.data.body_pos_w[0, tip_r_id].tolist()
        box = env.scene["workpiece"].data.root_pos_w[0].tolist()
        result["poses"] = {"tip_l": pl, "tip_r": pr, "box": box}
        result["tip_sep"] = float(torch.norm(robot.data.body_pos_w[0, tip_l_id] - robot.data.body_pos_w[0, tip_r_id]).item())
        result["tip_mid_dist"] = float(
            torch.norm(0.5 * (robot.data.body_pos_w[0, tip_l_id] + robot.data.body_pos_w[0, tip_r_id]) - env.scene["workpiece"].data.root_pos_w[0]).item()
        )
        result["peaks"] = peaks
        result["no_filter"] = bool(args.no_filter)
        result["ok"] = True
        print(json.dumps(result, indent=2))
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        print(json.dumps(result, indent=2))
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        try:
            app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
